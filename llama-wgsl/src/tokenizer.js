// Byte-level BPE tokenizer that reads a Hugging Face tokenizer.json (GPT-2 / SmolLM2 / Llama-3 style).
// Supports: added (special) tokens, the Digits pre-tokenizer, the ByteLevel pre-tokenizer with the
// GPT-2 regex, BPE merges by rank, and byte-level decoding.

const GPT2_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function bytesToUnicode() {
  // GPT-2's reversible mapping from bytes to printable unicode characters.
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const enc = new Map(), dec = new Map();
  bs.forEach((b, i) => { const ch = String.fromCodePoint(cs[i]); enc.set(b, ch); dec.set(ch, b); });
  return { enc, dec };
}

export class Tokenizer {
  constructor(json) {
    const model = json.model;
    if (model.type !== "BPE") throw new Error(`unsupported tokenizer model ${model.type}`);
    this.vocab = new Map(Object.entries(model.vocab));
    this.idToToken = [];
    for (const [tok, id] of this.vocab) this.idToToken[id] = tok;
    this.ranks = new Map();
    model.merges.forEach((m, i) => {
      const pair = Array.isArray(m) ? m.join(" ") : m;
      this.ranks.set(pair, i);
    });
    this.added = (json.added_tokens || []).map((t) => ({ id: t.id, content: t.content, special: t.special }));
    for (const t of this.added) this.idToToken[t.id] = t.content;
    this.specialIds = new Set(this.added.filter((t) => t.special).map((t) => t.id));
    this.addedByContent = new Map(this.added.map((t) => [t.content, t.id]));
    const escaped = this.added.map((t) => t.content)
      .sort((a, b) => b.length - a.length)
      .map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    this.addedRe = escaped.length ? new RegExp(`(${escaped.join("|")})`) : null;

    const pre = json.pre_tokenizer || {};
    const seq = pre.type === "Sequence" ? pre.pretokenizers : [pre];
    this.splitDigits = seq.some((p) => p && p.type === "Digits" && p.individual_digits);
    this.byteLevel = seq.find((p) => p && p.type === "ByteLevel") || null;
    this.addPrefixSpace = !!(this.byteLevel && this.byteLevel.add_prefix_space);
    const { enc, dec } = bytesToUnicode();
    this.byteEnc = enc;
    this.byteDec = dec;
    this.utf8 = new TextEncoder();
    this.cache = new Map();
  }

  static async fromUrl(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`tokenizer download failed: ${r.status}`);
    return new Tokenizer(await r.json());
  }

  preTokenize(text) {
    let pieces = [text];
    if (this.splitDigits) {
      pieces = pieces.flatMap((p) => p.split(/(\p{N})/u).filter((s) => s.length));
    }
    if (this.byteLevel && this.byteLevel.use_regex !== false) {
      pieces = pieces.flatMap((p) => p.match(GPT2_PATTERN) || []);
    }
    return pieces;
  }

  bpe(word) {
    const cached = this.cache.get(word);
    if (cached) return cached;
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      parts = [...parts.slice(0, best), parts[best] + parts[best + 1], ...parts.slice(best + 2)];
    }
    const ids = parts.map((p) => {
      const id = this.vocab.get(p);
      if (id === undefined) throw new Error(`token ${JSON.stringify(p)} not in vocab`);
      return id;
    });
    if (this.cache.size < 50000) this.cache.set(word, ids);
    return ids;
  }

  encode(text) {
    const ids = [];
    const segments = this.addedRe ? text.split(this.addedRe) : [text];
    for (const seg of segments) {
      if (!seg) continue;
      const added = this.addedByContent.get(seg);
      if (added !== undefined) { ids.push(added); continue; }
      const input = this.addPrefixSpace && !seg.startsWith(" ") ? " " + seg : seg;
      for (const piece of this.preTokenize(input)) {
        let mapped = "";
        for (const b of this.utf8.encode(piece)) mapped += this.byteEnc.get(b);
        ids.push(...this.bpe(mapped));
      }
    }
    return ids;
  }

  decode(ids, { skipSpecial = true } = {}) {
    const bytes = [];
    let out = "";
    const flush = () => { if (bytes.length) { out += new TextDecoder().decode(new Uint8Array(bytes)); bytes.length = 0; } };
    for (const id of ids) {
      if (this.specialIds.has(id)) {
        if (!skipSpecial) { flush(); out += this.idToToken[id]; }
        continue;
      }
      for (const ch of this.idToToken[id] || "") {
        const b = this.byteDec.get(ch);
        if (b !== undefined) bytes.push(b);
      }
    }
    flush();
    return out;
  }
}

// Incremental decoder for streaming: returns only complete UTF-8 text as tokens arrive.
// A character split across tokens (emoji, CJK) is held back until its last byte arrives.
export class StreamDecoder {
  constructor(tokenizer) { this.tok = tokenizer; this.pending = []; }
  push(id) {
    this.pending.push(id);
    const text = this.tok.decode(this.pending);
    // U+FFFD at the end means an incomplete UTF-8 sequence; give up waiting after a few tokens
    // so genuinely invalid bytes can't stall the stream.
    if (text.endsWith("�") && this.pending.length < 8) return "";
    this.pending = [];
    return text;
  }
  flush() {
    const text = this.pending.length ? this.tok.decode(this.pending) : "";
    this.pending = [];
    return text;
  }
}
