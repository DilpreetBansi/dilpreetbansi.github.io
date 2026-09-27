// Llama-architecture decoder running on WebGPU.
//
// One call to `step()` runs the whole network for one token and encodes it as a single compute
// pass of 8 dispatches per layer:
//
//   rmsnorm -> QKV matvec (fused, + bias) -> RoPE + KV-cache write -> attention
//   -> O matvec (+ residual) -> rmsnorm -> gate/up matvec + SiLU (fused) -> down matvec (+ residual)
//
// followed by the final norm, the LM head, and an optional GPU argmax. Weights stay in their
// checkpoint dtype (bf16 / f16 / f32) and are widened in registers; everything else is f32.

import * as K from './kernels.js';
import { readSafetensors, toFloat32 } from './safetensors.js';

const DTYPES = { BF16: 'bf16', F16: 'f16', F32: 'f32' };
const DTYPE_BYTES = { bf16: 2, f16: 2, f32: 4 };
const SUPPORTED = new Set(['llama', 'mistral', 'qwen2']);

/** Request an adapter and a device with the largest buffer limits the adapter allows. */
export async function createDevice() {
  if (!('gpu' in navigator)) throw new Error('WebGPU is not available in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter was found.');
  const requiredLimits = {};
  for (const k of ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxStorageBuffersPerShaderStage']) {
    requiredLimits[k] = adapter.limits[k];
  }
  const device = await adapter.requestDevice({ requiredLimits });
  return { adapter, device };
}

/** Normalise a Hugging Face config.json into the handful of numbers the engine needs. */
export function parseConfig(c) {
  if (!SUPPORTED.has(c.model_type)) {
    throw new Error(`model_type "${c.model_type}" is not supported (expected ${[...SUPPORTED].join(', ')})`);
  }
  if (c.hidden_act && c.hidden_act !== 'silu') throw new Error(`activation ${c.hidden_act} is not supported`);
  // transformers v5 moved rope_theta into rope_parameters; v4 configs use rope_theta/rope_scaling.
  const rope = c.rope_parameters ?? {};
  const ropeType = rope.rope_type ?? c.rope_scaling?.rope_type ?? c.rope_scaling?.type ?? 'default';
  if (ropeType !== 'default') throw new Error(`RoPE scaling "${ropeType}" is not supported yet`);
  const H = c.hidden_size;
  const HQ = c.num_attention_heads;
  const HKV = c.num_key_value_heads ?? HQ;
  const D = c.head_dim ?? H / HQ;
  const eos = Array.isArray(c.eos_token_id) ? c.eos_token_id : [c.eos_token_id].filter((x) => x != null);
  return {
    type: c.model_type,
    H, HQ, HKV, D,
    I: c.intermediate_size,
    L: c.num_hidden_layers,
    V: c.vocab_size,
    eps: c.rms_norm_eps ?? 1e-6,
    theta: rope.rope_theta ?? c.rope_theta ?? 10000,
    tied: c.tie_word_embeddings ?? false,
    // Sliding-window attention is identical to full attention inside the window, so cap the
    // context there instead of implementing the window.
    maxPos: Math.min(c.max_position_embeddings ?? 2048, c.use_sliding_window && c.sliding_window ? c.sliding_window : Infinity),
    bos: c.bos_token_id,
    eos,
  };
}

/**
 * RoPE tables matching transformers' float32 computation:
 * inv_freq = 1 / theta^(2j/D) in f32, angle = f32(pos * inv_freq), then cos/sin.
 */
export function ropeTables(D, theta, maxT) {
  const half = D / 2;
  const inv = new Float32Array(half);
  for (let j = 0; j < half; j++) inv[j] = 1 / Math.fround(theta ** Math.fround((2 * j) / D));
  const cos = new Float32Array(maxT * half);
  const sin = new Float32Array(maxT * half);
  for (let p = 0; p < maxT; p++) {
    for (let j = 0; j < half; j++) {
      const a = Math.fround(p * inv[j]);
      cos[p * half + j] = Math.cos(a);
      sin[p * half + j] = Math.sin(a);
    }
  }
  return { cos, sin };
}

function planChunks(rows, rowBytes, maxBytes) {
  let per = Math.floor(maxBytes / rowBytes);
  per -= per % 8;
  if (per < 8) throw new Error('a single weight row does not fit in a GPU buffer');
  const chunks = [];
  for (let start = 0; start < rows; start += per) chunks.push({ start, rows: Math.min(per, rows - start) });
  return chunks;
}

export class LlamaModel {
  /**
   * @param {GPUDevice} device
   * @param {object} config  parsed with parseConfig()
   * @param {{maxSeqLen?: number, kvBudgetBytes?: number}} opts
   */
  constructor(device, config, { maxSeqLen = 2048, kvBudgetBytes = 512 * 2 ** 20 } = {}) {
    this.device = device;
    this.cfg = config;
    const { H, HQ, HKV, D, I, L, V } = config;
    const kvPerToken = L * 2 * HKV * D * 4;
    this.maxT = Math.max(16, Math.min(maxSeqLen, config.maxPos, Math.floor(kvBudgetBytes / kvPerToken)));
    this.pos = 0;
    this.tokens = []; // ids whose keys/values are in the cache, in order
    this.bytes = { weights: 0, kv: 0, activations: 0 };
    this.dispatchesPerToken = 0;

    const S = GPUBufferUsage.STORAGE;
    const act = (n, extra = 0) => this._buffer(n * 4, S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extra, 'activations');
    this.nqkv = (HQ + 2 * HKV) * D;
    this.buf = {
      x: act(H),
      hn: act(H),
      qkv: act(this.nqkv),
      attn: act(HQ * D),
      act: act(I),
      logits: act(V),
      att: act(HQ * this.maxT),
      argmax: act(1),
    };
    this.step = this._buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'activations');
    this.readLogits = device.createBuffer({ size: V * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.readArgmax = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const { cos, sin } = ropeTables(D, config.theta, this.maxT);
    this.cos = this._upload(cos, 'activations');
    this.sin = this._upload(sin, 'activations');

    this.kc = [];
    this.vc = [];
    for (let l = 0; l < L; l++) {
      this.kc.push(this._buffer(this.maxT * HKV * D * 4, S | GPUBufferUsage.COPY_DST, 'kv'));
      this.vc.push(this._buffer(this.maxT * HKV * D * 4, S | GPUBufferUsage.COPY_DST, 'kv'));
    }
  }

  _buffer(size, usage, kind) {
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
    this.bytes[kind] += b.size;
    (this._owned ??= []).push(b);
    return b;
  }

  _upload(f32, kind = 'weights') {
    const b = this._buffer(f32.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, kind);
    this.device.queue.writeBuffer(b, 0, f32);
    return b;
  }

  /** Allocate weight buffers from the safetensors header (before any tensor data arrives). */
  _allocate(header) {
    const { H, HQ, HKV, D, I, L, V, tied } = this.cfg;
    const byName = new Map(header.tensors.map((t) => [t.name, t]));
    // Optional biases are detected from the checkpoint: Qwen2 has q/k/v biases, Llama with
    // attention_bias=true also has an o_proj bias.
    this.qkvBias = byName.has('model.layers.0.self_attn.q_proj.bias');
    this.oBias = byName.has('model.layers.0.self_attn.o_proj.bias');
    if (byName.has('model.layers.0.mlp.gate_proj.bias')) throw new Error('MLP biases are not supported');
    const bias = this.qkvBias;
    const need = (name) => {
      const t = byName.get(name);
      if (!t) throw new Error(`checkpoint is missing ${name}`);
      return t;
    };
    const dtypeOf = (...names) => {
      const ds = new Set(names.map((n) => need(n).dtype));
      if (ds.size !== 1) throw new Error(`mixed dtypes in ${names.join(', ')}`);
      const d = DTYPES[[...ds][0]];
      if (!d) throw new Error(`${names[0]}: dtype ${[...ds][0]} is not supported`);
      return d;
    };
    const W = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const maxBytes = Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize);
    const matrix = (rows, cols, dtype) => {
      const size = rows * cols * DTYPE_BYTES[dtype];
      if (size > maxBytes) throw new Error(`a ${rows}x${cols} weight exceeds this GPU's buffer limit`);
      return this._buffer(size, W, 'weights');
    };
    this.expected = new Set();
    this.layers = [];
    for (let l = 0; l < L; l++) {
      const p = `model.layers.${l}.`;
      const qkvNames = ['q', 'k', 'v'].map((x) => `${p}self_attn.${x}_proj.weight`);
      const mlpNames = [`${p}mlp.gate_proj.weight`, `${p}mlp.up_proj.weight`];
      const layer = {
        qkvDtype: dtypeOf(...qkvNames),
        oDtype: dtypeOf(`${p}self_attn.o_proj.weight`),
        mlpDtype: dtypeOf(...mlpNames),
        downDtype: dtypeOf(`${p}mlp.down_proj.weight`),
      };
      layer.qkv = matrix(this.nqkv, H, layer.qkvDtype);
      layer.o = matrix(H, HQ * D, layer.oDtype);
      layer.gateup = matrix(2 * I, H, layer.mlpDtype);
      layer.down = matrix(H, I, layer.downDtype);
      layer.ln1 = this._buffer(H * 4, W, 'weights');
      layer.ln2 = this._buffer(H * 4, W, 'weights');
      layer.bias = bias ? this._buffer(this.nqkv * 4, W, 'weights') : null;
      layer.oBias = this.oBias ? this._buffer(H * 4, W, 'weights') : null;
      this.layers.push(layer);
      for (const n of [...qkvNames, ...mlpNames, `${p}self_attn.o_proj.weight`, `${p}mlp.down_proj.weight`,
        `${p}input_layernorm.weight`, `${p}post_attention_layernorm.weight`]) this.expected.add(n);
      if (bias) for (const x of ['q', 'k', 'v']) this.expected.add(`${p}self_attn.${x}_proj.bias`);
      if (this.oBias) this.expected.add(`${p}self_attn.o_proj.bias`);
    }
    this.finalNorm = this._buffer(H * 4, W, 'weights');
    this.expected.add('model.norm.weight');

    const chunked = (name) => {
      const t = need(name);
      const dtype = dtypeOf(name);
      const rowBytes = H * DTYPE_BYTES[dtype];
      return {
        dtype,
        chunks: planChunks(V, rowBytes, maxBytes).map((c) => ({ ...c, buf: this._buffer(c.rows * rowBytes, W, 'weights') })),
        rowBytes,
        shape: t.shape,
      };
    };
    this.embedding = chunked('model.embed_tokens.weight');
    this.expected.add('model.embed_tokens.weight');
    if (tied || !byName.has('lm_head.weight')) {
      if (!tied) throw new Error('checkpoint has no lm_head.weight and tie_word_embeddings is false');
      this.head = this.embedding;
    } else {
      this.head = chunked('lm_head.weight');
      this.expected.add('lm_head.weight');
    }
  }

  /** Route one tensor's bytes to its place on the GPU. */
  _store(t, bytes) {
    const q = this.device.queue;
    const { H, HQ, HKV, D, I } = this.cfg;
    if (!this.expected.has(t.name)) return; // e.g. rotary inv_freq buffers, tied lm_head copies
    this.expected.delete(t.name);
    if (t.name === 'model.embed_tokens.weight' || t.name === 'lm_head.weight') {
      const target = t.name === 'lm_head.weight' ? this.head : this.embedding;
      for (const c of target.chunks) {
        q.writeBuffer(c.buf, 0, bytes, c.start * target.rowBytes, c.rows * target.rowBytes);
      }
      return;
    }
    if (t.name === 'model.norm.weight') {
      q.writeBuffer(this.finalNorm, 0, toFloat32(bytes, t.dtype));
      return;
    }
    const m = /^model\.layers\.(\d+)\.(.+)$/.exec(t.name);
    const layer = this.layers[+m[1]];
    const rowBytes = (dtype, cols) => cols * DTYPE_BYTES[dtype];
    const qRows = HQ * D;
    const kRows = HKV * D;
    switch (m[2]) {
      case 'self_attn.q_proj.weight': q.writeBuffer(layer.qkv, 0, bytes); break;
      case 'self_attn.k_proj.weight': q.writeBuffer(layer.qkv, qRows * rowBytes(layer.qkvDtype, H), bytes); break;
      case 'self_attn.v_proj.weight': q.writeBuffer(layer.qkv, (qRows + kRows) * rowBytes(layer.qkvDtype, H), bytes); break;
      case 'self_attn.q_proj.bias': q.writeBuffer(layer.bias, 0, toFloat32(bytes, t.dtype)); break;
      case 'self_attn.k_proj.bias': q.writeBuffer(layer.bias, qRows * 4, toFloat32(bytes, t.dtype)); break;
      case 'self_attn.v_proj.bias': q.writeBuffer(layer.bias, (qRows + kRows) * 4, toFloat32(bytes, t.dtype)); break;
      case 'self_attn.o_proj.weight': q.writeBuffer(layer.o, 0, bytes); break;
      case 'self_attn.o_proj.bias': q.writeBuffer(layer.oBias, 0, toFloat32(bytes, t.dtype)); break;
      case 'mlp.gate_proj.weight': q.writeBuffer(layer.gateup, 0, bytes); break;
      case 'mlp.up_proj.weight': q.writeBuffer(layer.gateup, I * rowBytes(layer.mlpDtype, H), bytes); break;
      case 'mlp.down_proj.weight': q.writeBuffer(layer.down, 0, bytes); break;
      case 'input_layernorm.weight': q.writeBuffer(layer.ln1, 0, toFloat32(bytes, t.dtype)); break;
      case 'post_attention_layernorm.weight': q.writeBuffer(layer.ln2, 0, toFloat32(bytes, t.dtype)); break;
      default: throw new Error(`unexpected tensor ${t.name}`);
    }
  }

  /**
   * Stream a single-file safetensors checkpoint into GPU memory, then compile the kernels.
   * @param {ReadableStream<Uint8Array>} stream
   * @param {{onBytes?: (n: number) => void}} opts
   */
  async loadWeights(stream, { onBytes } = {}) {
    await readSafetensors(stream, {
      onHeader: (h) => this._allocate(h),
      onTensor: (t, bytes) => this._store(t, bytes),
      onBytes,
    });
    if (this.expected.size) {
      throw new Error(`checkpoint is missing ${this.expected.size} tensors, e.g. ${[...this.expected][0]}`);
    }
    await this._build();
  }

  async _pipeline(code) {
    this._pipelines ??= new Map();
    let p = this._pipelines.get(code);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      this._pipelines.set(code, p);
    }
    return p;
  }

  _bind(pipeline, resources) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: resources.map((r, i) => ({ binding: i, resource: r instanceof GPUBuffer ? { buffer: r } : r })),
    });
  }

  async _build() {
    const { H, HQ, HKV, D, I, V, eps } = this.cfg;
    const bias = this.qkvBias;
    const B = this.buf;
    const norm = await this._pipeline(K.rmsnorm({ N: H, eps }));
    const rope = K.ropeKv({ HQ, HKV, D });
    const ropeP = await this._pipeline(rope.code);
    const attnP = await this._pipeline(K.attention({ HQ, HKV, D, maxT: this.maxT }));
    const qView = { buffer: B.qkv, offset: 0, size: HQ * D * 4 };
    const ops = []; // [pipeline, bindGroup, groupsX, groupsY, label]
    const add = (p, bg, gx, gy = 1, label = '') => ops.push([p, bg, gx, gy, label]);

    this.embedOps = await Promise.all(this.embedding.chunks.map(async (c) => {
      const p = await this._pipeline(K.embed({ H, dtype: this.embedding.dtype }));
      return [p, this._bind(p, [this.step, c.buf, B.x]), Math.ceil(H / 64), 1, 'embed'];
    }));

    for (let l = 0; l < this.layers.length; l++) {
      const w = this.layers[l];
      const qkvP = await this._pipeline(K.matvec({ rows: this.nqkv, K: H, dtype: w.qkvDtype, bias }));
      const oP = await this._pipeline(K.matvec({ rows: H, K: HQ * D, dtype: w.oDtype, residual: true, bias: this.oBias }));
      const mlpP = await this._pipeline(K.swiglu({ I, K: H, dtype: w.mlpDtype }));
      const downP = await this._pipeline(K.matvec({ rows: H, K: I, dtype: w.downDtype, residual: true }));
      add(norm, this._bind(norm, [B.x, w.ln1, B.hn]), 1, 1, 'rmsnorm');
      add(qkvP, this._bind(qkvP, bias ? [w.qkv, B.hn, B.qkv, w.bias] : [w.qkv, B.hn, B.qkv]), ...K.matvecGroups(this.nqkv, H), 'qkv');
      add(ropeP, this._bind(ropeP, [this.step, B.qkv, this.cos, this.sin, this.kc[l], this.vc[l]]), Math.ceil(rope.total / 64), 1, 'rope');
      add(attnP, this._bind(attnP, [this.step, qView, this.kc[l], this.vc[l], B.att, B.attn]), HQ, 1, 'attention');
      add(oP, this._bind(oP, this.oBias ? [w.o, B.attn, B.x, w.oBias] : [w.o, B.attn, B.x]), ...K.matvecGroups(H, HQ * D), 'o_proj');
      add(norm, this._bind(norm, [B.x, w.ln2, B.hn]), 1, 1, 'rmsnorm');
      add(mlpP, this._bind(mlpP, [w.gateup, B.hn, B.act]), ...K.matvecGroups(I, H), 'gate_up');
      add(downP, this._bind(downP, [w.down, B.act, B.x]), ...K.matvecGroups(H, I), 'down_proj');
    }
    add(norm, this._bind(norm, [B.x, this.finalNorm, B.hn]), 1, 1, 'rmsnorm');
    for (const c of this.head.chunks) {
      const p = await this._pipeline(K.matvec({ rows: c.rows, K: H, dtype: this.head.dtype, outOffset: c.start }));
      add(p, this._bind(p, [c.buf, B.hn, B.logits]), ...K.matvecGroups(c.rows, H), 'lm_head');
    }
    this.layerOps = ops;
    const am = await this._pipeline(K.argmax({ V }));
    this.argmaxOp = [am, this._bind(am, [B.logits, B.argmax]), 1, 1, 'argmax'];
    this.dispatchesPerToken = 1 + ops.length;
    this.ready = true;
  }

  _encode(token, readback) {
    const { V } = this.cfg;
    const chunkIdx = this.embedding.chunks.findIndex((c) => token >= c.start && token < c.start + c.rows);
    if (chunkIdx < 0) throw new Error(`token id ${token} is outside the vocabulary`);
    const row = token - this.embedding.chunks[chunkIdx].start;
    this.device.queue.writeBuffer(this.step, 0, new Uint32Array([this.pos, row, 0, 0]));
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    const run = ([p, bg, gx, gy]) => {
      pass.setPipeline(p);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(gx, gy);
    };
    run(this.embedOps[chunkIdx]);
    for (const op of this.layerOps) run(op);
    if (readback === 'argmax') run(this.argmaxOp);
    pass.end();
    if (readback === 'logits') enc.copyBufferToBuffer(this.buf.logits, 0, this.readLogits, 0, V * 4);
    if (readback === 'argmax') enc.copyBufferToBuffer(this.buf.argmax, 0, this.readArgmax, 0, 4);
    this.device.queue.submit([enc.finish()]);
    this.tokens[this.pos] = token;
    this.pos += 1;
    this.tokens.length = this.pos;
  }

  _checkRoom(n = 1) {
    if (!this.ready) throw new Error('weights are not loaded');
    if (this.pos + n > this.maxT) throw new Error(`context window full (${this.maxT} tokens)`);
  }

  /** Run one token; resolves to its next-token logits (Float32Array of vocab size). */
  async forward(token) {
    this._checkRoom();
    this._encode(token, 'logits');
    await this.readLogits.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.readLogits.getMappedRange().slice(0));
    this.readLogits.unmap();
    return out;
  }

  /** Run one token and return only the argmax of its logits (4 bytes read back). */
  async forwardGreedy(token) {
    this._checkRoom();
    this._encode(token, 'argmax');
    await this.readArgmax.mapAsync(GPUMapMode.READ);
    const id = new Uint32Array(this.readArgmax.getMappedRange())[0];
    this.readArgmax.unmap();
    return id;
  }

  /**
   * Make the cache hold exactly `ids` and return the logits after the last one.
   * Reuses the longest cached prefix (multi-turn chat only pays for the new tokens), then queues
   * the remaining tokens back to back with no CPU round trips except the final readback.
   */
  async prefill(ids, { greedy = false } = {}) {
    if (!ids.length) throw new Error('prefill needs at least one token');
    let common = 0;
    while (common < this.pos && common < ids.length && this.tokens[common] === ids[common]) common++;
    if (common === ids.length) common -= 1; // recompute the last token to get its logits
    this.pos = common;
    this.tokens.length = common;
    this._checkRoom(ids.length - common);
    for (let i = common; i < ids.length - 1; i++) this._encode(ids[i], 'none');
    const last = ids[ids.length - 1];
    const result = greedy ? await this.forwardGreedy(last) : await this.forward(last);
    return { result, reused: common, computed: ids.length - common };
  }

  reset() {
    this.pos = 0;
    this.tokens = [];
  }

  /** Read an activation buffer back (tests and the inspector use this). */
  async read(name, count) {
    const src = this.buf[name];
    const size = (count ?? src.size / 4) * 4;
    const dst = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, dst, 0, size);
    this.device.queue.submit([enc.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(dst.getMappedRange().slice(0));
    dst.destroy();
    return out;
  }

  /** Free every GPU buffer this model allocated (the device itself is left alone). */
  destroy() {
    for (const b of this._owned ?? []) b.destroy();
    this.readLogits.destroy();
    this.readArgmax.destroy();
    this._owned = [];
    this._pipelines?.clear();
    this.ready = false;
  }
}
