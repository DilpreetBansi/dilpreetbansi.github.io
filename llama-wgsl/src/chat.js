// Multi-turn chat on top of LlamaModel: ChatML formatting, streaming generation, KV-cache reuse.

import { StreamDecoder } from './tokenizer.js';
import { makeRng, sample } from './sampler.js';

/** ChatML, the template SmolLM2 (and Qwen2) instruct models are trained with. */
export function chatML(messages, { system, addGenerationPrompt = true } = {}) {
  let s = '';
  if (system && messages[0]?.role !== 'system') s += `<|im_start|>system\n${system}<|im_end|>\n`;
  for (const m of messages) s += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
  if (addGenerationPrompt) s += '<|im_start|>assistant\n';
  return s;
}

/**
 * Stream tokens for `promptIds`. Yields {id, text} per token and returns generation stats.
 * The KV cache keeps whatever prefix of promptIds it already holds, so follow-up turns only
 * compute the new tokens.
 */
export async function* generate(model, tokenizer, promptIds, opts = {}) {
  const {
    maxNewTokens = 256,
    temperature = 0.7,
    topK = 40,
    topP = 0.9,
    repetitionPenalty = 1.1,
    seed = Date.now(),
    stop = model.cfg.eos,
    signal,
  } = opts;
  const greedy = !(temperature > 0) && repetitionPenalty === 1;
  const room = model.maxT - promptIds.length;
  if (room <= 0) throw new Error(`the prompt (${promptIds.length} tokens) does not fit the ${model.maxT}-token context`);
  const budget = Math.min(maxNewTokens, room);
  const rng = makeRng(seed);
  const history = promptIds.slice();
  const decoder = new StreamDecoder(tokenizer);
  const stats = { promptTokens: promptIds.length, reused: 0, prefillMs: 0, tokens: 0, decodeMs: 0, stopReason: 'length' };

  const t0 = performance.now();
  const pre = await model.prefill(promptIds, { greedy });
  stats.prefillMs = performance.now() - t0;
  stats.reused = pre.reused;
  stats.prefilled = pre.computed;
  let next = greedy ? pre.result : sample(pre.result, { temperature, topK, topP, repetitionPenalty, history }, rng);

  const t1 = performance.now();
  for (let i = 0; i < budget; i++) {
    if (stop.includes(next)) {
      stats.stopReason = 'stop';
      break;
    }
    history.push(next);
    stats.tokens++;
    const text = decoder.push(next);
    yield { id: next, text };
    if (signal?.aborted) {
      stats.stopReason = 'aborted';
      break;
    }
    if (i === budget - 1) break; // the last token needs no forward pass
    if (greedy) {
      next = await model.forwardGreedy(next);
    } else {
      const logits = await model.forward(next);
      next = sample(logits, { temperature, topK, topP, repetitionPenalty, history }, rng);
    }
  }
  const tail = decoder.flush();
  if (tail) yield { id: -1, text: tail };
  stats.decodeMs = performance.now() - t1;
  return stats;
}

/** A conversation kept as token ids, so every turn extends the cached prefix exactly. */
export class ChatSession {
  constructor(model, tokenizer, { system } = {}) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.system = system;
    this.reset();
  }

  reset() {
    this.ids = this.system ? this.tokenizer.encode(`<|im_start|>system\n${this.system}<|im_end|>\n`) : [];
    this.messages = [];
  }

  /** Stream the assistant's reply to `text`; resolves the generator's return value to stats. */
  async *send(text, opts = {}) {
    const turn = this.tokenizer.encode(`<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n`);
    const prompt = [...this.ids, ...turn];
    const reply = [];
    let out = '';
    const gen = generate(this.model, this.tokenizer, prompt, opts);
    let r;
    while (!(r = await gen.next()).done) {
      if (r.value.id >= 0) reply.push(r.value.id);
      out += r.value.text;
      yield r.value;
    }
    const end = this.tokenizer.encode('<|im_end|>\n');
    this.ids = [...prompt, ...reply, ...end];
    this.messages.push({ role: 'user', content: text }, { role: 'assistant', content: out });
    return r.value;
  }
}
