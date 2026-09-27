// Token sampling on the CPU: repetition penalty, temperature, top-k, top-p, seeded RNG.

/** Small, fast, seedable PRNG (mulberry32). Returns floats in [0, 1). */
export function makeRng(seed = Date.now()) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function argmax(logits) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  return best;
}

/** Indices of the k largest values, sorted descending (binary min-heap, O(n log k)). */
export function topK(values, k) {
  k = Math.min(k, values.length);
  const heap = new Int32Array(k);
  let size = 0;
  const less = (a, b) => values[a] < values[b] || (values[a] === values[b] && a > b);
  const siftDown = (i) => {
    for (;;) {
      const l = 2 * i + 1;
      const r = l + 1;
      let m = i;
      if (l < size && less(heap[l], heap[m])) m = l;
      if (r < size && less(heap[r], heap[m])) m = r;
      if (m === i) return;
      [heap[i], heap[m]] = [heap[m], heap[i]];
      i = m;
    }
  };
  for (let i = 0; i < values.length; i++) {
    if (size < k) {
      heap[size] = i;
      let j = size++;
      while (j > 0) {
        const p = (j - 1) >> 1;
        if (!less(heap[j], heap[p])) break;
        [heap[j], heap[p]] = [heap[p], heap[j]];
        j = p;
      }
    } else if (less(heap[0], i)) {
      heap[0] = i;
      siftDown(0);
    }
  }
  return Array.from(heap.subarray(0, size)).sort((a, b) => values[b] - values[a] || a - b);
}

/**
 * Pick the next token.
 * @param {Float32Array} logits  modified in place by the repetition penalty
 * @param {{temperature?: number, topK?: number, topP?: number, repetitionPenalty?: number, history?: number[]}} opts
 * @param {() => number} rng
 */
export function sample(logits, opts = {}, rng = Math.random) {
  const { temperature = 0.7, topK: k = 40, topP = 0.9, repetitionPenalty = 1, history = [] } = opts;
  if (repetitionPenalty !== 1) {
    // transformers' RepetitionPenaltyLogitsProcessor: shrink the logits of tokens already seen
    for (const id of new Set(history)) {
      const v = logits[id];
      logits[id] = v > 0 ? v / repetitionPenalty : v * repetitionPenalty;
    }
  }
  if (!(temperature > 0)) return argmax(logits);
  const idx = topK(logits, Math.max(1, Math.min(k || 1000, 1000)));
  const top = logits[idx[0]];
  const probs = idx.map((i) => Math.exp((logits[i] - top) / temperature));
  const total = probs.reduce((a, b) => a + b, 0);
  let keep = probs.length;
  if (topP < 1) {
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i] / total;
      if (cum >= topP) {
        keep = i + 1;
        break;
      }
    }
  }
  let mass = 0;
  for (let i = 0; i < keep; i++) mass += probs[i];
  let r = rng() * mass;
  for (let i = 0; i < keep; i++) {
    r -= probs[i];
    if (r <= 0) return idx[i];
  }
  return idx[keep - 1];
}
