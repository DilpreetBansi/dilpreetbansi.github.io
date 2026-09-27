// Demo UI: model picker, streaming chat, live speed stats, and a benchmark button.
import { createDevice } from './model.js';
import { MODELS, hubBase, loadModel } from './loader.js';
import { ChatSession, generate } from './chat.js';

const $ = (id) => document.getElementById(id);
const els = {
  messages: $('messages'), welcome: $('welcome'), composer: $('composer'), prompt: $('prompt'),
  send: $('send'), stop: $('stop'), select: $('model-select'), load: $('load'), newChat: $('new-chat'),
  progress: $('progress'), progressFill: $('progress-fill'), progressText: $('progress-text'),
  note: $('model-note'), badge: $('gpu-badge'), tps: $('tps'), spark: $('spark'), bench: $('bench'),
  benchOut: $('bench-out'), ctxFill: $('ctx-fill'),
};

const state = { device: null, adapter: null, model: null, tokenizer: null, session: null, busy: false, abort: null, loaded: null };
const MiB = (b) => `${(b / 2 ** 20).toFixed(0)} MiB`;
const fmtRate = (r) => (r >= 100 ? r.toFixed(0) : r.toFixed(1));
const fmtBytes = (b) => (b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(2)} GB` : `${(b / 2 ** 20).toFixed(0)} MB`);

for (const m of MODELS) {
  const o = document.createElement('option');
  o.value = m.id;
  o.textContent = `${m.id}  ·  ${fmtBytes(m.bytes)}`;
  els.select.append(o);
}
const params = new URLSearchParams(location.search);
if (params.get('model')) els.select.value = params.get('model');

// ---------- WebGPU availability ----------
async function initGpu() {
  try {
    const { adapter, device } = await createDevice();
    state.adapter = adapter;
    state.device = device;
    const info = adapter.info || {};
    const name = [info.vendor, info.architecture].filter(Boolean).join(' ') || 'WebGPU';
    els.badge.textContent = name;
    els.badge.classList.add('ok');
    $('st-adapter').textContent = name;
    $('st-adapter').title = info.description || name;
    device.lost.then((e) => {
      if (e.reason !== 'destroyed') showError(`The GPU device was lost (${e.message}). Reload the page to continue.`);
    });
  } catch (e) {
    els.badge.textContent = 'no WebGPU';
    els.badge.classList.add('err');
    els.load.disabled = true;
    els.note.innerHTML = 'This browser does not expose WebGPU. Use a recent Chrome or Edge on desktop, ' +
      'Safari 26+, or Firefox 141+ on Windows. <br><span class="muted">' + escapeHtml(e.message) + '</span>';
  }
}

// ---------- loading ----------
async function load() {
  const entry = MODELS.find((m) => m.id === els.select.value);
  if (!state.device || state.busy) return;
  setBusy(true);
  els.load.disabled = true;
  els.progress.hidden = false;
  els.note.textContent = '';
  if (state.model) {
    state.model.destroy();
    state.model = null;
  }
  const t0 = performance.now();
  let lastT = t0;
  let lastBytes = 0;
  let rate = 0;
  try {
    // ?base=/models/foo loads from any URL serving config.json, tokenizer.json, model.safetensors
    const base = params.get('base') || hubBase(entry.repo);
    const { model, tokenizer } = await loadModel(base, {
      device: state.device,
      maxSeqLen: 2048,
      onProgress: (p) => {
        if (p.phase === 'config') els.progressText.textContent = 'fetching config and tokenizer…';
        if (p.phase === 'weights') {
          const total = p.total || entry.bytes;
          const now = performance.now();
          if (now - lastT > 400) {
            rate = ((p.loaded - lastBytes) / (now - lastT)) * 1000;
            lastT = now;
            lastBytes = p.loaded;
          }
          els.progressFill.style.width = `${Math.min(100, (100 * p.loaded) / total).toFixed(1)}%`;
          const src = p.fromCache ? 'from browser cache' : rate ? `${fmtBytes(rate)}/s` : 'downloading';
          els.progressText.textContent = `${fmtBytes(p.loaded)} / ${fmtBytes(total)} · ${src}`;
        }
        if (p.phase === 'ready') els.progressText.textContent = 'compiling kernels…';
      },
    });
    state.model = model;
    state.tokenizer = tokenizer;
    state.loaded = entry;
    state.session = new ChatSession(model, tokenizer, { system: entry.system });
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    els.progressFill.style.width = '100%';
    els.progressText.textContent = `ready in ${secs} s`;
    els.note.textContent = `${entry.id}: ${model.cfg.L} layers, ${model.cfg.HQ} heads (${model.cfg.HKV} KV), context ${model.maxT} tokens.`;
    $('st-weights').textContent = MiB(model.bytes.weights);
    $('st-kv').textContent = MiB(model.bytes.kv);
    $('st-dispatch').textContent = `${model.dispatchesPerToken} / token`;
    updateContext();
    els.prompt.disabled = false;
    els.prompt.placeholder = `Message ${entry.id}…`;
    els.newChat.disabled = false;
    els.bench.disabled = false;
    els.prompt.focus();
  } catch (e) {
    console.error(e);
    els.progressText.textContent = 'failed';
    els.note.innerHTML = `<span style="color:var(--err)">${escapeHtml(e.message)}</span>`;
  } finally {
    els.load.disabled = false;
    setBusy(false);
  }
}

// ---------- chat ----------
function sampling() {
  return {
    temperature: +$('temp').value,
    topP: +$('topp').value,
    repetitionPenalty: +$('rep').value,
    maxNewTokens: +$('maxtok').value,
  };
}

async function send(text) {
  if (!state.session || state.busy || !text.trim()) return;
  els.welcome.hidden = true;
  addMessage('user', text);
  const bot = addMessage('bot', '');
  bot.body.classList.add('cursor');
  const abort = { aborted: false };
  state.abort = abort;
  setBusy(true);
  const times = [];
  let raw = '';
  let last = performance.now();
  let stats;
  try {
    const gen = state.session.send(text, { ...sampling(), signal: abort });
    let r;
    while (!(r = await gen.next()).done) {
      const now = performance.now();
      times.push(now - last);
      last = now;
      raw += r.value.text;
      bot.body.innerHTML = renderMarkdown(raw);
      scrollToEnd();
      if (times.length % 4 === 0) liveSpeed(times);
    }
    stats = r.value;
    bot.body.classList.remove('cursor');
    const decodeTps = stats.tokens > 1 ? (stats.tokens - 1) / (stats.decodeMs / 1000) : 0;
    const prefillTps = stats.prefilled / (stats.prefillMs / 1000);
    setTps(decodeTps);
    $('st-prefill').textContent = `${stats.prefilled} new tok · ${fmtRate(prefillTps)} tok/s` + (stats.reused ? ` · ${stats.reused} cached` : '');
    bot.meta.textContent = `${stats.tokens} tokens · ${fmtRate(decodeTps)} tok/s` + (stats.stopReason === 'length' ? ' · hit max tokens' : '');
    drawSpark(times.slice(1));
  } catch (e) {
    console.error(e);
    bot.body.classList.remove('cursor');
    bot.el.classList.add('error');
    bot.body.textContent = e.message.includes('context')
      ? `${e.message}. Start a new chat to free the KV cache.`
      : `Error: ${e.message}`;
  } finally {
    setBusy(false);
    updateContext();
  }
}

function liveSpeed(times) {
  const recent = times.slice(-16);
  const ms = recent.reduce((a, b) => a + b, 0) / recent.length;
  setTps(1000 / ms);
  drawSpark(times.slice(1));
}

function setTps(v) {
  els.tps.textContent = v > 0 ? fmtRate(v) : '—';
  els.tps.classList.toggle('empty', !(v > 0));
}

function updateContext() {
  const m = state.model;
  if (!m) return;
  $('st-ctx').textContent = `${m.pos} / ${m.maxT} tokens`;
  els.ctxFill.style.width = `${(100 * m.pos) / m.maxT}%`;
}

// ---------- benchmark ----------
async function bench() {
  if (!state.model || state.busy) return;
  setBusy(true);
  els.benchOut.hidden = false;
  els.benchOut.textContent = 'running: 64-token prompt, then 128 greedy tokens…';
  const m = state.model;
  const tok = state.tokenizer;
  try {
    const text = 'The history of the graphics processing unit begins in the 1970s, when arcade machines and home computers needed dedicated chips to draw sprites and text. ';
    let ids = tok.encode(text.repeat(4)).slice(0, 64);
    m.reset();
    // warm-up (pipeline caches, first-use allocations)
    await m.prefill(ids.slice(0, 4), { greedy: true });
    m.reset();
    const t0 = performance.now();
    let { result: next } = await m.prefill(ids, { greedy: true });
    const prefillMs = performance.now() - t0;
    const t1 = performance.now();
    let n = 0;
    for (; n < 128; n++) next = await m.forwardGreedy(next);
    const decodeMs = performance.now() - t1;
    const info = state.adapter.info || {};
    const line = `${state.loaded.id} | ${[info.vendor, info.architecture].filter(Boolean).join(' ')} | ` +
      `prefill ${(64 / (prefillMs / 1000)).toFixed(1)} tok/s | decode ${(n / (decodeMs / 1000)).toFixed(1)} tok/s`;
    els.benchOut.textContent = `${line}\n${navigator.userAgent}`;
    setTps(n / (decodeMs / 1000));
  } catch (e) {
    els.benchOut.textContent = `benchmark failed: ${e.message}`;
  } finally {
    // the benchmark overwrote the KV cache, so the conversation starts over
    m.reset();
    clearChat();
    setBusy(false);
  }
}

// ---------- rendering helpers ----------
function addMessage(role, text) {
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="who">${role === 'user' ? 'you' : 'AI'}</div><div><div class="body"></div><div class="meta"></div></div>`;
  const body = el.querySelector('.body');
  body.innerHTML = role === 'user' ? escapeHtml(text) : renderMarkdown(text);
  els.messages.append(el);
  scrollToEnd();
  return { el, body, meta: el.querySelector('.meta') };
}

function scrollToEnd() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// Just enough markdown for chat replies: fenced code, inline code, bold, italics.
function renderMarkdown(src) {
  const parts = src.split(/```/);
  return parts
    .map((p, i) => {
      if (i % 2 === 1) {
        const body = p.replace(/^[\w+-]*\n/, '');
        return `<pre><code>${escapeHtml(body)}</code></pre>`;
      }
      return escapeHtml(p)
        .replace(/`([^`\n]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>')
        .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<i>$2</i>');
    })
    .join('');
}

function drawSpark(times) {
  const c = els.spark;
  if (times.length < 2) return;
  c.hidden = false;
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth;
  const h = c.clientHeight;
  c.width = w * dpr;
  c.height = h * dpr;
  const g = c.getContext('2d');
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  if (times.length < 2) return;
  const tps = times.map((t) => 1000 / t);
  const max = Math.max(...tps) * 1.15;
  const step = w / (tps.length - 1);
  const y = (v) => h - 4 - (v / max) * (h - 8);
  const grad = g.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, '#8b5cf6');
  grad.addColorStop(1, '#22d3ee');
  g.beginPath();
  tps.forEach((v, i) => (i ? g.lineTo(i * step, y(v)) : g.moveTo(0, y(v))));
  g.strokeStyle = grad;
  g.lineWidth = 1.6;
  g.stroke();
  g.lineTo(w, h);
  g.lineTo(0, h);
  g.closePath();
  const fill = g.createLinearGradient(0, 0, 0, h);
  fill.addColorStop(0, 'rgba(139,92,246,0.25)');
  fill.addColorStop(1, 'rgba(34,211,238,0)');
  g.fillStyle = fill;
  g.fill();
}

function setBusy(b) {
  state.busy = b;
  els.send.disabled = b || !state.session;
  els.stop.hidden = !b || !state.abort;
  els.bench.disabled = b || !state.model;
  els.newChat.disabled = b || !state.session;
  els.select.disabled = b;
  document.querySelectorAll('#suggestions button').forEach((x) => (x.disabled = b));
  if (!b) state.abort = null;
}

function showError(msg) {
  const m = addMessage('bot', '');
  m.el.classList.add('error');
  m.body.textContent = msg;
}

// ---------- wiring ----------
els.load.addEventListener('click', load);
els.composer.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = els.prompt.value;
  els.prompt.value = '';
  autoGrow();
  send(text);
});
els.prompt.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});
const autoGrow = () => {
  els.prompt.style.height = 'auto';
  els.prompt.style.height = `${Math.min(180, els.prompt.scrollHeight)}px`;
};
els.prompt.addEventListener('input', autoGrow);
els.stop.addEventListener('click', () => {
  if (state.abort) state.abort.aborted = true;
});
function clearChat() {
  state.session?.reset();
  els.messages.querySelectorAll('.msg').forEach((m) => m.remove());
  els.welcome.hidden = false;
  updateContext();
}
els.newChat.addEventListener('click', clearChat);
els.bench.addEventListener('click', bench);
document.querySelectorAll('#suggestions button').forEach((b) =>
  b.addEventListener('click', async () => {
    if (!state.session) await load();
    if (state.session) send(b.textContent);
  }),
);
for (const [input, out] of [['temp', 'v-temp'], ['topp', 'v-topp'], ['rep', 'v-rep'], ['maxtok', 'v-max']]) {
  $(input).addEventListener('input', () => ($(out).textContent = $(input).value));
}

// exported for tests / console tinkering
window.llamaWgsl = { state, generate };
initGpu();
