const express = require('express');
const app = express();
app.use(express.json());

app.use((err, req, res, next) => {
  if (err && err.type === 'entity.parse.failed') {
    return res.status(400).json({ error: 'INVALID_INPUT' });
  }
  next(err);
});

// ---------------- shared helpers ----------------
function isFiniteNum(x) { return typeof x === 'number' && Number.isFinite(x); }
function inRange01(x) { return isFiniteNum(x) && x >= 0 && x <= 1; }
function isNonNegSafeInt(x) {
  return typeof x === 'number' && Number.isInteger(x) && x >= 0 && x <= Number.MAX_SAFE_INTEGER;
}
function isPosSafeInt(x) {
  return typeof x === 'number' && Number.isInteger(x) && x > 0 && x <= Number.MAX_SAFE_INTEGER;
}
function isNonEmptyString(x) { return typeof x === 'string' && x.length > 0; }
function sortUtf8(arr) {
  return [...arr].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}
function sortDedupeCodes(codes) {
  return sortUtf8([...new Set(codes)]);
}

const HEX40_RE = /^[0-9a-f]{40}$/;
const HEX64_RE = /^[0-9a-f]{64}$/;

// ================= CHOOSE =================
const INTERVENTION_ORDER = ['prompt_only', 'retrieval', 'lora', 'qlora'];

function validatePolicyChoose(policy) {
  if (!policy || typeof policy !== 'object' || Array.isArray(policy)) return false;
  if (!inRange01(policy.minQuality)) return false;
  if (typeof policy.freshnessRequired !== 'boolean') return false;
  if (!isFiniteNum(policy.maxLatencyMs) || policy.maxLatencyMs < 0) return false;
  if (!isFiniteNum(policy.maxMemoryMb) || policy.maxMemoryMb < 0) return false;
  if (!isNonNegSafeInt(policy.maxLabeledExamples)) return false;
  if (!isFiniteNum(policy.maxTotalCost) || policy.maxTotalCost < 0) return false;
  if (!isNonNegSafeInt(policy.horizonRequests)) return false;
  return true;
}

function validateCandidateShape(c) {
  if (!c || typeof c !== 'object' || Array.isArray(c)) return false;
  if (!INTERVENTION_ORDER.includes(c.name)) return false;
  if (typeof c.available !== 'boolean') return false;
  if (!inRange01(c.quality)) return false;
  if (typeof c.freshness !== 'boolean') return false;
  if (!isFiniteNum(c.latencyMs) || c.latencyMs < 0) return false;
  if (!isFiniteNum(c.memoryMb) || c.memoryMb < 0) return false;
  if (!isNonNegSafeInt(c.labeledExamples)) return false;
  if (!isFiniteNum(c.oneTimeCost) || c.oneTimeCost < 0) return false;
  if (!isFiniteNum(c.recurringCost) || c.recurringCost < 0) return false;
  return true;
}

function handleChoose(body, res) {
  const { policy, candidates } = body;

  if (!validatePolicyChoose(policy) || !Array.isArray(candidates)) {
    return res.status(400).json({ error: 'INVALID_INPUT' });
  }

  // Exactly one candidate per required name, no extras
  const byName = new Map();
  for (const c of candidates) {
    const name = c && typeof c === 'object' ? c.name : undefined;
    if (!INTERVENTION_ORDER.includes(name) || byName.has(name)) {
      return res.status(400).json({ error: 'INVALID_INPUT' });
    }
    byName.set(name, c);
  }
  if (byName.size !== INTERVENTION_ORDER.length) {
    return res.status(400).json({ error: 'INVALID_INPUT' });
  }

  const totalCosts = {};
  const reasonCodes = {};
  const eligibleFlags = {};

  for (const name of INTERVENTION_ORDER) {
    const c = byName.get(name);
    const codes = [];

    if (!validateCandidateShape(c)) {
      codes.push('INVALID_INPUT');
      reasonCodes[name] = sortDedupeCodes(codes);
      totalCosts[name] = 0;
      eligibleFlags[name] = false;
      continue;
    }

    const totalCost = Math.round(
      (c.oneTimeCost + policy.horizonRequests * c.recurringCost) * 1e12
    ) / 1e12;
    totalCosts[name] = totalCost;

    if (!c.available) codes.push('UNAVAILABLE');
    if (c.quality < policy.minQuality) codes.push('QUALITY_FLOOR');
    if (policy.freshnessRequired && !c.freshness) codes.push('FRESHNESS_REQUIRED');
    if (c.latencyMs > policy.maxLatencyMs) codes.push('LATENCY_LIMIT');
    if (c.memoryMb > policy.maxMemoryMb) codes.push('MEMORY_LIMIT');
    if (c.labeledExamples > policy.maxLabeledExamples) codes.push('DATA_LIMIT');
    if (totalCost > policy.maxTotalCost) codes.push('COST_LIMIT');

    reasonCodes[name] = sortDedupeCodes(codes);
    eligibleFlags[name] = codes.length === 0;
  }

  const eligible = INTERVENTION_ORDER.filter(name => eligibleFlags[name]);
  const selected = eligible.length > 0 ? eligible[0] : null;

  return res.json({ selected, eligible, totalCosts, reasonCodes });
}

// ================= REPAIR =================
const VALID_ROLES = new Set(['system', 'user', 'assistant']);

function computeLabels(tokens) {
  if (!Array.isArray(tokens) || tokens.length === 0) {
    return { labels: [], valid: false, code: 'INVALID_TOKEN' };
  }
  let allValid = true;
  for (const t of tokens) {
    if (
      !t || typeof t !== 'object' ||
      !isNonNegSafeInt(t.id) ||
      !VALID_ROLES.has(t.role) ||
      typeof t.padding !== 'boolean' ||
      typeof t.text !== 'string'
    ) {
      allValid = false;
      break;
    }
  }
  if (!allValid) {
    return { labels: tokens.map(() => -100), valid: false, code: 'INVALID_TOKEN' };
  }
  const labels = tokens.map(t => (t.role === 'assistant' && t.padding === false ? t.id : -100));
  return { labels, valid: true, code: null };
}

function checkTemplate(templateApplications) {
  return templateApplications === 1;
}

function checkParameters(parameters, allowedTargets) {
  const codes = [];
  const result = { trainableParams: [], trainableCount: 0, pass: false };

  const paramsOk = Array.isArray(parameters) && parameters.length > 0 &&
    parameters.every(p => p && typeof p === 'object' &&
      isNonEmptyString(p.name) && isNonEmptyString(p.target) && isPosSafeInt(p.numel));

  const namesUnique = paramsOk && new Set(parameters.map(p => p.name)).size === parameters.length;

  const targetsOk = Array.isArray(allowedTargets) && allowedTargets.length > 0 &&
    allowedTargets.every(t => isNonEmptyString(t)) &&
    new Set(allowedTargets).size === allowedTargets.length;

  if (!paramsOk || !namesUnique || !targetsOk) {
    codes.push('INVALID_PARAMETER');
    return { ...result, codes };
  }

  const allowedSet = new Set(allowedTargets);
  const trainable = parameters.filter(p =>
    allowedSet.has(p.target) &&
    (p.name.endsWith('.lora_A.weight') || p.name.endsWith('.lora_B.weight'))
  );

  if (trainable.length === 0) {
    codes.push('INVALID_PARAMETER');
    return { ...result, codes };
  }

  const sortedTrainable = sortUtf8(trainable.map(p => p.name));
  let sum = 0n;
  for (const p of trainable) sum += BigInt(p.numel);
  const trainableCount = sum <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(sum) : Number.MAX_SAFE_INTEGER;

  return {
    trainableParams: sortedTrainable,
    trainableCount,
    pass: true,
    codes: []
  };
}

function checkEvalIsolation(trainRowIds, evalRowIds) {
  const isValidIdArray = (arr) =>
    Array.isArray(arr) && arr.length > 0 &&
    arr.every(x => isNonEmptyString(x)) &&
    new Set(arr).size === arr.length;

  if (!isValidIdArray(trainRowIds) || !isValidIdArray(evalRowIds)) {
    return { pass: false, code: 'EVAL_LEAKAGE' };
  }
  const trainSet = new Set(trainRowIds);
  const overlap = evalRowIds.some(id => trainSet.has(id));
  if (overlap) {
    return { pass: false, code: 'EVAL_LEAKAGE' };
  }
  return { pass: true, code: null };
}

function checkArtifactFiles(artifactFiles) {
  const required = ['adapter_config.json', 'adapter_model.safetensors'];
  const codes = [];

  if (!Array.isArray(artifactFiles)) {
    codes.push('ADAPTER_FILE_SET');
    return { adapterFiles: [], codes };
  }

  const sorted = sortUtf8([...new Set(artifactFiles)]);
  const requiredSorted = sortUtf8(required);

  const hasDuplicates = new Set(artifactFiles).size !== artifactFiles.length;
  const exactMatch = !hasDuplicates &&
    sorted.length === requiredSorted.length &&
    sorted.every((f, i) => f === requiredSorted[i]);

  if (!exactMatch) {
    codes.push('ADAPTER_FILE_SET');
    // Flag full-model artifact if something outside the adapter set is present
    const hasExtra = artifactFiles.some(f => !required.includes(f));
    if (hasExtra) codes.push('FULL_MODEL_ARTIFACT');
  }

  return { adapterFiles: sorted, codes };
}

function checkLineage(baseRevision, datasetDigest, codeDigest, configDigest, expectedDigests) {
  const codes = [];

  if (!isNonEmptyString(baseRevision) || !HEX40_RE.test(baseRevision)) {
    codes.push('MUTABLE_BASE_REVISION');
  }

  const digestsFormatValid =
    isNonEmptyString(datasetDigest) && HEX64_RE.test(datasetDigest) &&
    isNonEmptyString(codeDigest) && HEX64_RE.test(codeDigest) &&
    isNonEmptyString(configDigest) && HEX64_RE.test(configDigest);

  if (!digestsFormatValid) {
    codes.push('LINEAGE_MISMATCH');
  } else if (expectedDigests && typeof expectedDigests === 'object') {
    const mismatches =
      (expectedDigests.datasetDigest !== undefined && expectedDigests.datasetDigest !== datasetDigest) ||
      (expectedDigests.codeDigest !== undefined && expectedDigests.codeDigest !== codeDigest) ||
      (expectedDigests.configDigest !== undefined && expectedDigests.configDigest !== configDigest);
    if (mismatches) codes.push('LINEAGE_MISMATCH');
  }

  return { pass: codes.length === 0, codes };
}

function checkBatch(microBatch, gradientAccumulation, replicas, expectedEffectiveBatch) {
  const allPos = [microBatch, gradientAccumulation, replicas, expectedEffectiveBatch].every(isPosSafeInt);
  if (!allPos) return false;
  return microBatch * gradientAccumulation * replicas === expectedEffectiveBatch;
}

function checkCheckpoint(checkpoint) {
  const required = ['model', 'optimizer', 'scheduler', 'step', 'rng', 'dataPosition'];
  if (!checkpoint || typeof checkpoint !== 'object' || Array.isArray(checkpoint)) return false;
  return required.every(k => Object.prototype.hasOwnProperty.call(checkpoint, k));
}

function checkResume(uninterruptedWeights, resumedWeights, resumeTolerance) {
  const isValidArr = (arr) =>
    Array.isArray(arr) && arr.length > 0 && arr.every(isFiniteNum);

  if (!isValidArr(uninterruptedWeights) || !isValidArr(resumedWeights)) return false;
  if (uninterruptedWeights.length !== resumedWeights.length) return false;
  if (!isFiniteNum(resumeTolerance) || resumeTolerance < 0) return false;

  for (let i = 0; i < uninterruptedWeights.length; i++) {
    if (Math.abs(uninterruptedWeights[i] - resumedWeights[i]) > resumeTolerance) return false;
  }
  return true;
}

function handleRepair(body, res) {
  const reasonCodes = [];

  // 1. Tokens
  const { labels, valid: tokensValid, code: tokenCode } = computeLabels(body.tokens);
  if (!tokensValid) reasonCodes.push(tokenCode);

  // 2. Template
  const templatePass = checkTemplate(body.templateApplications);
  if (!templatePass) reasonCodes.push('CHAT_TEMPLATE_COUNT');

  // 3. Parameters / PEFT config
  const paramResult = checkParameters(body.parameters, body.allowedTargets);
  if (paramResult.codes.length) reasonCodes.push(...paramResult.codes);
  const peftConfigPass = paramResult.pass;

  // 4. Inference mode / dropout -> evaluationDeterministic
  let evaluationDeterministic = true;
  if (body.inferenceMode !== false) {
    reasonCodes.push('INFERENCE_MODE');
    evaluationDeterministic = false;
  }
  if (body.dropoutActiveDuringEval !== false) {
    reasonCodes.push('EVAL_DROPOUT_ACTIVE');
    evaluationDeterministic = false;
  }

  // 5. Eval isolation
  const isoResult = checkEvalIsolation(body.trainRowIds, body.evalRowIds);
  if (!isoResult.pass) reasonCodes.push(isoResult.code);
  const evalIsolated = isoResult.pass;

  // 6. Artifact files
  const artifactResult = checkArtifactFiles(body.artifactFiles);
  if (artifactResult.codes.length) reasonCodes.push(...artifactResult.codes);

  // 7. Lineage
  const lineageResult = checkLineage(
    body.baseRevision, body.datasetDigest, body.codeDigest, body.configDigest, body.expectedDigests
  );
  if (lineageResult.codes.length) reasonCodes.push(...lineageResult.codes);

  // 8. Batch
  const batchOk = checkBatch(
    body.microBatch, body.gradientAccumulation, body.replicas, body.expectedEffectiveBatch
  );
  if (!batchOk) reasonCodes.push('EFFECTIVE_BATCH_MISMATCH');
  if (!batchOk) lineageResult.pass = false; // batch mismatch also invalidates lineage pass per spec grouping? kept separate below

  // 9. Checkpoint
  const checkpointComplete = checkCheckpoint(body.checkpoint);
  if (!checkpointComplete) reasonCodes.push('INCOMPLETE_CHECKPOINT');

  // 10. Resume
  const resumePass = checkResume(body.uninterruptedWeights, body.resumedWeights, body.resumeTolerance);
  if (!resumePass) reasonCodes.push('RESUME_DIVERGENCE');

  const lineagePass = lineageResult.pass && batchOk;

  return res.json({
    labels,
    templatePass,
    trainableParams: paramResult.trainableParams,
    trainableCount: paramResult.trainableCount,
    peftConfigPass,
    adapterFiles: artifactResult.adapterFiles,
    checkpointComplete,
    lineagePass,
    evalIsolated,
    evaluationDeterministic,
    resumePass,
    reasonCodes: sortDedupeCodes(reasonCodes)
  });
}

// ================= ROUTE =================
app.post('/adapt', (req, res) => {
  try {
    const body = req.body;
    if (!body || typeof body !== 'object' || Array.isArray(body)) {
      return res.status(400).json({ error: 'INVALID_INPUT' });
    }
    if (body.operation === 'choose') return handleChoose(body, res);
    if (body.operation === 'repair') return handleRepair(body, res);
    return res.status(400).json({ error: 'INVALID_INPUT' });
  } catch (err) {
    console.error(err);
    return res.status(400).json({ error: 'INVALID_INPUT' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Listening on ${PORT}`));
