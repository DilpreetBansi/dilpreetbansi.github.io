// Streaming safetensors reader.
//
// A .safetensors file is: u64 little-endian header length N, N bytes of JSON describing each
// tensor ({dtype, shape, data_offsets: [begin, end]} relative to the data section), then the raw
// tensor bytes. We parse the header as soon as it arrives and hand each tensor to a callback the
// moment its bytes are complete, so weights go from the network to the GPU without ever holding
// the whole checkpoint in JavaScript memory.

const DTYPE_BYTES = { F32: 4, F16: 2, BF16: 2, I32: 4, I64: 8, U8: 1, I8: 1, BOOL: 1 };

/** Parse a header from its JSON bytes. Returns {tensors: [{name, dtype, shape, begin, end}], metadata}. */
export function parseHeader(jsonBytes) {
  const header = JSON.parse(new TextDecoder().decode(jsonBytes));
  const metadata = header.__metadata__ || {};
  delete header.__metadata__;
  const tensors = Object.entries(header).map(([name, t]) => {
    const [begin, end] = t.data_offsets;
    const numel = t.shape.reduce((a, b) => a * b, 1);
    const bytes = DTYPE_BYTES[t.dtype];
    if (!bytes) throw new Error(`${name}: unsupported dtype ${t.dtype}`);
    if (end - begin !== numel * bytes) throw new Error(`${name}: size mismatch`);
    return { name, dtype: t.dtype, shape: t.shape, begin, end };
  });
  tensors.sort((a, b) => a.begin - b.begin);
  return { tensors, metadata };
}

/**
 * Read a safetensors stream.
 * @param {ReadableStream<Uint8Array>} stream
 * @param {(header) => void|Promise<void>} onHeader   called once, before any tensor
 * @param {(tensor, bytes: Uint8Array) => void|Promise<void>} onTensor  bytes are owned by the callee
 * @param {(loadedBytes: number) => void} [onBytes]
 */
export async function readSafetensors(stream, { onHeader, onTensor, onBytes }) {
  const reader = stream.getReader();
  let pending = []; // chunks not yet consumed
  let pendingBytes = 0;
  let loaded = 0;
  let done = false;

  async function pull() {
    const r = await reader.read();
    if (r.done) {
      done = true;
      return;
    }
    pending.push(r.value);
    pendingBytes += r.value.byteLength;
    loaded += r.value.byteLength;
    if (onBytes) onBytes(loaded);
  }

  // Take exactly n bytes from the front of `pending` into dst (or a new array).
  function take(n, dst = new Uint8Array(n), dstOffset = 0) {
    let filled = 0;
    while (filled < n) {
      const head = pending[0];
      const k = Math.min(n - filled, head.byteLength);
      dst.set(head.subarray(0, k), dstOffset + filled);
      filled += k;
      if (k === head.byteLength) pending.shift();
      else pending[0] = head.subarray(k);
    }
    pendingBytes -= n;
    return dst;
  }

  async function need(n) {
    while (pendingBytes < n) {
      if (done) throw new Error('safetensors: unexpected end of stream');
      await pull();
    }
  }

  await need(8);
  const lenBytes = take(8);
  const view = new DataView(lenBytes.buffer);
  const headerLen = Number(view.getBigUint64(0, true));
  if (headerLen > 100_000_000) throw new Error('safetensors: header too large (not a safetensors file?)');
  await need(headerLen);
  const header = parseHeader(take(headerLen));
  if (onHeader) await onHeader(header);

  let offset = 0; // position within the data section
  for (const t of header.tensors) {
    if (t.begin > offset) {
      await need(t.begin - offset);
      take(t.begin - offset); // skip padding
      offset = t.begin;
    }
    const size = t.end - t.begin;
    // Fill the tensor incrementally so large tensors don't require the stream to buffer them twice.
    const bytes = new Uint8Array(size);
    let filled = 0;
    while (filled < size) {
      if (pendingBytes === 0) {
        if (done) throw new Error(`safetensors: stream ended inside ${t.name}`);
        await pull();
        continue;
      }
      const k = Math.min(size - filled, pendingBytes);
      take(k, bytes, filled);
      filled += k;
    }
    offset = t.end;
    await onTensor(t, bytes);
  }
  reader.cancel().catch(() => {});
}

/** Convert raw tensor bytes to a Float32Array (used for small tensors: norms and biases). */
export function toFloat32(bytes, dtype) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dtype === 'F32') {
    const out = new Float32Array(bytes.byteLength / 4);
    for (let i = 0; i < out.length; i++) out[i] = dv.getFloat32(4 * i, true);
    return out;
  }
  const n = bytes.byteLength / 2;
  const out = new Float32Array(n);
  if (dtype === 'BF16') {
    const u32 = new Uint32Array(1);
    const f32 = new Float32Array(u32.buffer);
    for (let i = 0; i < n; i++) {
      u32[0] = dv.getUint16(2 * i, true) << 16;
      out[i] = f32[0];
    }
    return out;
  }
  if (dtype === 'F16') {
    for (let i = 0; i < n; i++) out[i] = halfToFloat(dv.getUint16(2 * i, true));
    return out;
  }
  throw new Error(`toFloat32: unsupported dtype ${dtype}`);
}

export function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}
