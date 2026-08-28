const express = require('express');
const app = express();
app.use(express.json());

// Handle malformed JSON bodies gracefully
app.use((err, req, res, next) => {
  if (err && err.type === 'entity.parse.failed') {
    return res.status(400).json({ error: 'INVALID_INPUT' });
  }
  next(err);
});

const TS_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?(Z|[+-]\d{2}:\d{2})$/;
const CANON_INT_RE = /^[1-9]\d*$/;

function isFiniteNum(x) {
  return typeof x === 'number' && Number.isFinite(x);
}
function inRange01(x) {
  return isFiniteNum(x) && x >= 0 && x <= 1;
}
function isNonNegSafeInt(x) {
  return typeof x === 'number' && Number.isInteger(x) && x >= 0 && x <= Number.MAX_SAFE_INTEGER;
}
function isNonEmptyString(x) {
  return typeof x === 'string' && x.length > 0;
}
function isCanonicalVersion(v) {
  return typeof v === 'string' && CANON_INT_RE.test(v);
}
function parseTimestamp(ts) {
  if (typeof ts !== 'string' || !TS_RE.test(ts)) return null;
  const ms = Date.parse(ts);
  if (Number.isNaN(ms)) return null;
  return ms;
}

function addCode(map, key, code) {
  if (!Object.prototype.hasOwnProperty.call(map, key) || !Array.isArray(map[key])) {
    map[key] = [];
  }
  if (!map[key].includes(code)) map[key].push(code);
}

function validatePolicy(policy, asOfMs) {
  if (asOfMs === null) return false;
  if (!policy || typeof policy !== 'object' || Array.isArray(policy)) return false;
  if (!isNonEmptyString(policy.datasetDigest)) return false;
  if (!isNonEmptyString(policy.schemaDigest)) return false;
  if (!isNonNegSafeInt(policy.maxAgeSeconds)) return false;
  if (!inRange01(policy.accuracyFloor)) return false;
  if (!isFiniteNum(policy.maxLatencyMs) || policy.maxLatencyMs < 0) return false;
  if (!isNonNegSafeInt(policy.maxSizeBytes)) return false;
  if (!inRange01(policy.minImprovement)) return false;

  if (
    !policy.requiredSlices ||
    typeof policy.requiredSlices !== 'object' ||
    Array.isArray(policy.requiredSlices)
  ) {
    return false;
  }
  for (const k of Object.keys(policy.requiredSlices)) {
    if (!inRange01(policy.requiredSlices[k])) return false;
  }

  return true;
}

function evaluateVersion(v, policy, asOfMs) {
  const codes = [];
  const ev = v.evaluation;

  if (!ev || typeof ev !== 'object' || Array.isArray(ev)) {
    return { codes: ['MISSING_EVALUATION'], eligible: false, ev: null };
  }

  const accFinite = isFiniteNum(ev.accuracy);
  const latFinite = isFiniteNum(ev.latencyMs);
  const sizeFinite = isFiniteNum(ev.sizeBytes);
  if (!accFinite || !latFinite || !sizeFinite) codes.push('NON_FINITE');

  if (accFinite && (ev.accuracy < 0 || ev.accuracy > 1)) codes.push('METRIC_RANGE');
  if (latFinite && ev.latencyMs < 0) codes.push('METRIC_RANGE');
  if (sizeFinite && ev.sizeBytes < 0) codes.push('METRIC_RANGE');

  const createdMs = parseTimestamp(ev.createdAt);
  if (createdMs === null) {
    codes.push('INVALID_TIMESTAMP');
  } else {
    if (createdMs > asOfMs) codes.push('FUTURE_EVALUATION');
    else if (createdMs < asOfMs - policy.maxAgeSeconds * 1000) codes.push('STALE_EVALUATION');
  }

  if (!isNonEmptyString(ev.artifactDigest) || ev.artifactDigest !== v.artifactDigest) {
    codes.push('ARTIFACT_MISMATCH');
  }
  if (!isNonEmptyString(ev.datasetDigest) || ev.datasetDigest !== policy.datasetDigest) {
    codes.push('DATASET_MISMATCH');
  }
  if (!isNonEmptyString(ev.schemaDigest) || ev.schemaDigest !== policy.schemaDigest) {
    codes.push('SCHEMA_MISMATCH');
  }

  if (accFinite && ev.accuracy < policy.accuracyFloor) codes.push('ACCURACY_FLOOR');
  if (latFinite && ev.latencyMs > policy.maxLatencyMs) codes.push('LATENCY_LIMIT');
  if (sizeFinite && ev.sizeBytes > policy.maxSizeBytes) codes.push('SIZE_LIMIT');

  const slices = (ev.slices && typeof ev.slices === 'object' && !Array.isArray(ev.slices)) ? ev.slices : {};
  const requiredSlices = policy.requiredSlices || {};
  for (const name of Object.keys(requiredSlices)) {
    const floor = requiredSlices[name];
    if (!(name in slices)) {
      codes.push(`MISSING_SLICE:${name}`);
      continue;
    }
    const val = slices[name];
    if (!inRange01(val)) {
      codes.push(`SLICE_RANGE:${name}`);
      continue;
    }
    if (val < floor) {
      codes.push(`SLICE_FLOOR:${name}`);
    }
  }

  const uniqueSorted = [...new Set(codes)].sort();
  return { codes: uniqueSorted, eligible: uniqueSorted.length === 0, ev };
}

app.post('/promote', (req, res) => {
  try {
    const body = req.body;

    if (
      !body || typeof body !== 'object' ||
      !body.policy || typeof body.policy !== 'object' || Array.isArray(body.policy) ||
      !Array.isArray(body.versions) ||
      typeof body.championVersion !== 'string'
    ) {
      return res.status(400).json({ error: 'INVALID_INPUT' });
    }

    const { policy, versions, championVersion } = body;
    const asOfMs = parseTimestamp(body.asOf);
    const failedGates = {};

    const counts = new Map();
    const canonicalList = [];

    versions.forEach((v, idx) => {
      const id = v && typeof v === 'object' && !Array.isArray(v) ? v.version : undefined;
      if (!isCanonicalVersion(id)) {
        const key = typeof id === 'string' && id.length > 0 ? id : `__invalid_${idx}`;
        addCode(failedGates, key, 'INVALID_VERSION');
        return;
      }
      counts.set(id, (counts.get(id) || 0) + 1);
      canonicalList.push({ id, v });
    });

    const versionMap = new Map();
    for (const { id, v } of canonicalList) {
      if (counts.get(id) > 1) {
        addCode(failedGates, id, 'DUPLICATE_VERSION');
        continue;
      }
      versionMap.set(id, v);
    }

    const policyValid = validatePolicy(policy, asOfMs);
    if (!policyValid) {
      for (const id of versionMap.keys()) {
        addCode(failedGates, id, 'INVALID_POLICY');
      }
      return res.json({
        action: 'block',
        championVersion,
        selectedVersion: null,
        eligibleVersions: [],
        failedGates,
        aliasMutation: null,
        evidence: null
      });
    }

    const results = new Map();
    for (const [id, v] of versionMap.entries()) {
      if (!Object.prototype.hasOwnProperty.call(failedGates, id)) {
        failedGates[id] = [];
      }
      const r = evaluateVersion(v, policy, asOfMs);
      results.set(id, r);
      if (r.codes.length) {
        for (const c of r.codes) addCode(failedGates, id, c);
      }
    }

    const eligibleIds = [...results.entries()].filter(([, r]) => r.eligible).map(([id]) => id);

    const ranked = eligibleIds
      .map(id => ({ id, ev: results.get(id).ev }))
      .sort((a, b) => {
        if (b.ev.accuracy !== a.ev.accuracy) return b.ev.accuracy - a.ev.accuracy;
        if (a.ev.latencyMs !== b.ev.latencyMs) return a.ev.latencyMs - b.ev.latencyMs;
        if (a.ev.sizeBytes !== b.ev.sizeBytes) return a.ev.sizeBytes - b.ev.sizeBytes;
        return Number(a.id) - Number(b.id);
      });

    const eligibleVersionsSorted = ranked.map(r => r.id);

    const championResult = versionMap.has(championVersion) ? results.get(championVersion) : null;
    const championValid = championResult && championResult.eligible;

    if (!championValid) {
      return res.json({
        action: 'block',
        championVersion,
        selectedVersion: null,
        eligibleVersions: eligibleVersionsSorted,
        failedGates,
        aliasMutation: null,
        evidence: null
      });
    }

    const winner = ranked[0];
    const championEv = results.get(championVersion).ev;

    let action, selectedVersion, evidence, aliasMutation;

    if (winner.id === championVersion) {
      action = 'retain';
      selectedVersion = championVersion;
      evidence = championEv;
      aliasMutation = null;
    } else {
      const diff = Math.round((winner.ev.accuracy - championEv.accuracy) * 1e12) / 1e12;
      if (diff >= policy.minImprovement) {
        action = 'promote';
        selectedVersion = winner.id;
        evidence = winner.ev;
        aliasMutation = { alias: 'champion', version: winner.id };
      } else {
        action = 'retain';
        selectedVersion = championVersion;
        evidence = championEv;
        aliasMutation = null;
      }
    }

    return res.json({
      action,
      championVersion,
      selectedVersion,
      eligibleVersions: eligibleVersionsSorted,
      failedGates,
      aliasMutation,
      evidence
    });
  } catch (err) {
    console.error(err);
    return res.status(400).json({ error: 'INVALID_INPUT' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Listening on ${PORT}`));
