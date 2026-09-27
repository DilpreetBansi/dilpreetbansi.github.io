// WGSL compute kernels for single-token (decode) inference of Llama-architecture models.
//
// Every kernel is generated from a small template so that model dimensions become WGSL
// `const`s: the shader compiler sees fixed loop bounds and the kernels need no uniforms
// except the per-step position/token.
//
// Weight matrices stay in their checkpoint dtype on the GPU:
//   bf16 -> two values packed per u32, widened in registers with a shift/mask (no shader-f16 needed)
//   f16  -> two values per u32, widened with unpack2x16float
//   f32  -> plain vec4<f32>
// Activations, the KV cache and all accumulation are f32.

export const WG = 256; // workgroup size for the reduction kernels

// Per-step uniform: written with queue.writeBuffer before each token's command buffer.
const STEP_STRUCT = /* wgsl */ `
struct Step {
  pos: u32,       // position of the token being processed (0-based)
  emb_row: u32,   // row of the embedding chunk that holds this token
  _pad0: u32,
  _pad1: u32,
}`;

// Weight array element type and an 8-wide dot product for each storage dtype.
// `i` indexes 8-element chunks: chunk i of row r is (r * K / 8 + c).
function dot8(dtype) {
  switch (dtype) {
    case 'bf16':
      return /* wgsl */ `
fn dot8(i: u32, x0: vec4<f32>, x1: vec4<f32>) -> f32 {
  let w = W[i];
  // Little-endian: element 2j is the low half of word j, element 2j+1 the high half.
  let lo = bitcast<vec4<f32>>(w << vec4<u32>(16u));
  let hi = bitcast<vec4<f32>>(w & vec4<u32>(0xffff0000u));
  return dot(vec4<f32>(lo.x, hi.x, lo.y, hi.y), x0) + dot(vec4<f32>(lo.z, hi.z, lo.w, hi.w), x1);
}`;
    case 'f16':
      return /* wgsl */ `
fn dot8(i: u32, x0: vec4<f32>, x1: vec4<f32>) -> f32 {
  let w = W[i];
  return dot(vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)), x0)
       + dot(vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w)), x1);
}`;
    case 'f32':
      return /* wgsl */ `
fn dot8(i: u32, x0: vec4<f32>, x1: vec4<f32>) -> f32 {
  return dot(W[2u * i], x0) + dot(W[2u * i + 1u], x1);
}`;
    default:
      throw new Error(`unsupported weight dtype ${dtype}`);
  }
}

const weightType = (dtype) => (dtype === 'f32' ? 'vec4<f32>' : 'vec4<u32>');

// Threads per row for the matrix-vector kernels: a power of two in [4, 32], chosen so each thread
// handles at least eight 8-element chunks (enough work to amortise the reduction) while a row is
// still read with contiguous 16-byte loads. K=576 -> 8 threads/row, K=1536 -> 16, K=2048 -> 32.
export function threadsPerRow(K) {
  const k8 = K / 8;
  let t = 4;
  while (t < 32 && k8 / (t * 2) >= 8) t *= 2;
  return t;
}

/** Workgroup grid for a matvec/swiglu over `rows` output rows with inner dimension K. */
export function matvecGroups(rows, K) {
  const groups = Math.ceil(rows / (WG / threadsPerRow(K)));
  // Stay under maxComputeWorkgroupsPerDimension (65535) by folding into a 2D grid.
  if (groups <= 65535) return [groups, 1];
  const x = 65535;
  return [x, Math.ceil(groups / x)];
}

/**
 * y[off + r] (+)= dot(W[r, :], x) (+ b[r])  for r in [0, rows)
 * opts.residual: accumulate into y instead of overwriting (fuses the residual add).
 * opts.bias:     add a per-row f32 bias (Qwen2-style attention projections).
 */
export function matvec({ rows, K, dtype, residual = false, bias = false, outOffset = 0 }) {
  if (K % 8) throw new Error(`matvec: K=${K} must be a multiple of 8`);
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> W: array<${weightType(dtype)}>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
${bias ? '@group(0) @binding(3) var<storage, read> B: array<f32>;' : ''}
${dot8(dtype)}

const ROWS: u32 = ${rows}u;
const K8: u32 = ${K / 8}u;
const TPR: u32 = ${threadsPerRow(K)}u;
const RPW: u32 = ${WG / threadsPerRow(K)}u;
const OFF: u32 = ${outOffset}u;
var<workgroup> part: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let lane = li % TPR;
  let row = (wid.x + wid.y * nwg.x) * RPW + li / TPR;
  var acc = 0.0;
  if (row < ROWS) {
    let base = row * K8;
    for (var c = lane; c < K8; c = c + TPR) {
      acc = acc + dot8(base + c, X[2u * c], X[2u * c + 1u]);
    }
  }
  part[li] = acc;
  workgroupBarrier();
  for (var s = TPR / 2u; s > 0u; s = s >> 1u) {
    if (lane < s) { part[li] = part[li] + part[li + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && row < ROWS) {
    var y = part[li];
    ${bias ? 'y = y + B[row];' : ''}
    ${residual ? 'Y[OFF + row] = Y[OFF + row] + y;' : 'Y[OFF + row] = y;'}
  }
}`;
}

/**
 * Fused SwiGLU MLP input: W holds the gate rows [0, I) followed by the up rows [I, 2I).
 * y[r] = silu(gate_r . x) * (up_r . x). One read of x, one dispatch, no intermediate buffers.
 */
export function swiglu({ I, K, dtype }) {
  if (K % 8) throw new Error(`swiglu: K=${K} must be a multiple of 8`);
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> W: array<${weightType(dtype)}>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
${dot8(dtype)}

const I: u32 = ${I}u;
const K8: u32 = ${K / 8}u;
const TPR: u32 = ${threadsPerRow(K)}u;
const RPW: u32 = ${WG / threadsPerRow(K)}u;
var<workgroup> pg: array<f32, ${WG}>;
var<workgroup> pu: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let lane = li % TPR;
  let row = (wid.x + wid.y * nwg.x) * RPW + li / TPR;
  var g = 0.0;
  var u = 0.0;
  if (row < I) {
    let gb = row * K8;
    let ub = (row + I) * K8;
    for (var c = lane; c < K8; c = c + TPR) {
      let x0 = X[2u * c];
      let x1 = X[2u * c + 1u];
      g = g + dot8(gb + c, x0, x1);
      u = u + dot8(ub + c, x0, x1);
    }
  }
  pg[li] = g;
  pu[li] = u;
  workgroupBarrier();
  for (var s = TPR / 2u; s > 0u; s = s >> 1u) {
    if (lane < s) {
      pg[li] = pg[li] + pg[li + s];
      pu[li] = pu[li] + pu[li + s];
    }
    workgroupBarrier();
  }
  if (lane == 0u && row < I) {
    let gs = pg[li];
    Y[row] = gs / (1.0 + exp(-gs)) * pu[li];
  }
}`;
}

/** y = x * rsqrt(mean(x^2) + eps) * gamma. One workgroup; N is the hidden size. */
export function rmsnorm({ N, eps }) {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;

const N: u32 = ${N}u;
const EPS: f32 = ${f32Literal(eps)};
var<workgroup> red: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) li: u32) {
  var ss = 0.0;
  for (var i = li; i < N; i = i + ${WG}u) {
    let v = X[i];
    ss = ss + v * v;
  }
  red[li] = ss;
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s = s >> 1u) {
    if (li < s) { red[li] = red[li] + red[li + s]; }
    workgroupBarrier();
  }
  let inv = 1.0 / sqrt(red[0] / f32(N) + EPS);
  for (var i = li; i < N; i = i + ${WG}u) {
    Y[i] = G[i] * (X[i] * inv);
  }
}`;
}

/**
 * Rotary embeddings (HF "rotate_half" convention) applied in place to the query heads, and the
 * rotated keys plus the values written into this layer's KV cache at `step.pos`.
 * QKV layout: [q heads | k heads | v heads], each head D floats.
 * Cache layout: [position][kv head][D].
 */
export function ropeKv({ HQ, HKV, D }) {
  const half = D / 2;
  const nRot = (HQ + HKV) * half;
  const total = nRot + HKV * D;
  return {
    total,
    code: /* wgsl */ `
${STEP_STRUCT}
@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<storage, read_write> QKV: array<f32>;
@group(0) @binding(2) var<storage, read> COS: array<f32>;
@group(0) @binding(3) var<storage, read> SIN: array<f32>;
@group(0) @binding(4) var<storage, read_write> KC: array<f32>;
@group(0) @binding(5) var<storage, read_write> VC: array<f32>;

const HQ: u32 = ${HQ}u;
const HKV: u32 = ${HKV}u;
const D: u32 = ${D}u;
const HALF: u32 = ${half}u;
const NROT: u32 = ${nRot}u;
const TOTAL: u32 = ${total}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let p = step.pos;
  let row = p * HKV * D;
  if (i < NROT) {
    let head = i / HALF;
    let j = i % HALF;
    let c = COS[p * HALF + j];
    let s = SIN[p * HALF + j];
    let base = head * D;
    let a = QKV[base + j];
    let b = QKV[base + j + HALF];
    let ra = a * c - b * s;
    let rb = b * c + a * s;
    if (head < HQ) {
      QKV[base + j] = ra;
      QKV[base + j + HALF] = rb;
    } else {
      let kh = head - HQ;
      KC[row + kh * D + j] = ra;
      KC[row + kh * D + j + HALF] = rb;
    }
  } else if (i < TOTAL) {
    let v = i - NROT;
    VC[row + v] = QKV[(HQ + HKV) * D + v];
  }
}`,
  };
}

/**
 * Causal attention for one new query token over positions [0, pos], one workgroup per query head.
 * Grouped-query attention: query head h reads kv head h / (HQ / HKV).
 * Scores live in a [HQ][maxT] scratch buffer; the softmax is computed in f32 with max-subtraction.
 */
export function attention({ HQ, HKV, D, maxT }) {
  if (D % 4) throw new Error('head_dim must be a multiple of 4');
  if (WG % D) throw new Error(`head_dim ${D} must divide ${WG}`);
  const group = HQ / HKV;
  return /* wgsl */ `
${STEP_STRUCT}
@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> KC: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> VC: array<f32>;
@group(0) @binding(4) var<storage, read_write> ATT: array<f32>;
@group(0) @binding(5) var<storage, read_write> OUT: array<f32>;

const HKV: u32 = ${HKV}u;
const D: u32 = ${D}u;
const D4: u32 = ${D / 4}u;
const GROUP: u32 = ${group}u;
const MAXT: u32 = ${maxT}u;
const SCALE: f32 = ${f32Literal(1 / Math.sqrt(D))};
const G: u32 = ${WG / D}u; // position groups in the weighted-sum phase

var<workgroup> q: array<vec4<f32>, ${D / 4}>;
var<workgroup> red: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wid.x;
  let kvh = h / GROUP;
  let T = step.pos + 1u;
  let arow = h * MAXT;
  if (li < D4) {
    let o = h * D + 4u * li;
    q[li] = vec4<f32>(Q[o], Q[o + 1u], Q[o + 2u], Q[o + 3u]);
  }
  workgroupBarrier();

  // 1) scores s_t = q . k_t / sqrt(D), and the running max
  var m = -3.0e38;
  for (var t = li; t < T; t = t + ${WG}u) {
    let kb = (t * HKV + kvh) * D4;
    var s = 0.0;
    for (var d = 0u; d < D4; d = d + 1u) {
      s = s + dot(q[d], KC[kb + d]);
    }
    s = s * SCALE;
    ATT[arow + t] = s;
    m = max(m, s);
  }
  red[li] = m;
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s = s >> 1u) {
    if (li < s) { red[li] = max(red[li], red[li + s]); }
    workgroupBarrier();
  }
  m = red[0];
  workgroupBarrier();

  // 2) exponentiate (each thread revisits the positions it scored) and sum
  var sum = 0.0;
  for (var t = li; t < T; t = t + ${WG}u) {
    let e = exp(ATT[arow + t] - m);
    ATT[arow + t] = e;
    sum = sum + e;
  }
  red[li] = sum;
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s = s >> 1u) {
    if (li < s) { red[li] = red[li] + red[li + s]; }
    workgroupBarrier();
  }
  let total = red[0];
  storageBarrier(); // ATT writes from phase 2 must be visible to every thread below
  workgroupBarrier();

  // 3) out_d = sum_t p_t v_t[d]; thread (g, d) sums positions g, g+G, ...
  let d = li % D;
  let g = li / D;
  var acc = 0.0;
  for (var t = g; t < T; t = t + G) {
    acc = acc + ATT[arow + t] * VC[(t * HKV + kvh) * D + d];
  }
  red[li] = acc;
  workgroupBarrier();
  if (li < D) {
    var o = 0.0;
    for (var k = 0u; k < G; k = k + 1u) { o = o + red[k * D + li]; }
    OUT[h * D + li] = o / total;
  }
}`;
}

/** x = embedding[row] widened to f32. */
export function embed({ H, dtype }) {
  let load;
  if (dtype === 'bf16') {
    load = `let w = E[e >> 1u];
    X[i] = select(bitcast<f32>(w & 0xffff0000u), bitcast<f32>(w << 16u), (e & 1u) == 0u);`;
  } else if (dtype === 'f16') {
    load = `let w = unpack2x16float(E[e >> 1u]);
    X[i] = select(w.y, w.x, (e & 1u) == 0u);`;
  } else {
    load = 'X[i] = E[e];';
  }
  return /* wgsl */ `
${STEP_STRUCT}
@group(0) @binding(0) var<uniform> step: Step;
@group(0) @binding(1) var<storage, read> E: array<${dtype === 'f32' ? 'f32' : 'u32'}>;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
const H: u32 = ${H}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < H) {
    let e = step.emb_row * H + i;
    ${load}
  }
}`;
}

/** Greedy decoding helper: index of the largest logit, so only 4 bytes cross back to the CPU. */
export function argmax({ V }) {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> L: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT: array<u32>;
const V: u32 = ${V}u;
var<workgroup> bv: array<f32, ${WG}>;
var<workgroup> bi: array<u32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) li: u32) {
  var best = -3.4e38;
  var idx = 0u;
  for (var i = li; i < V; i = i + ${WG}u) {
    let v = L[i];
    if (v > best) { best = v; idx = i; }
  }
  bv[li] = best;
  bi[li] = idx;
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s = s >> 1u) {
    if (li < s) {
      let ov = bv[li + s];
      let oi = bi[li + s];
      // ties go to the lower index, matching torch.argmax
      if (ov > bv[li] || (ov == bv[li] && oi < bi[li])) { bv[li] = ov; bi[li] = oi; }
    }
    workgroupBarrier();
  }
  if (li == 0u) { OUT[0] = bi[0]; }
}`;
}

// Shortest round-tripping f32 literal for WGSL.
export function f32Literal(x) {
  const f = Math.fround(x);
  let s = String(f);
  if (!/[.eE]/.test(s)) s += '.0';
  return s;
}
