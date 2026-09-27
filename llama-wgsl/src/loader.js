// Fetch a model (config, tokenizer, weights) from the Hugging Face Hub or any static URL, keep a
// copy in Cache Storage so the next visit loads from disk, and stream the weights to the GPU.

import { LlamaModel, parseConfig } from './model.js';
import { Tokenizer } from './tokenizer.js';

const CACHE_NAME = 'llama-wgsl-v1';

export const MODELS = [
  {
    id: 'SmolLM2-135M-Instruct',
    repo: 'HuggingFaceTB/SmolLM2-135M-Instruct',
    bytes: 269_060_552,
    system: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face',
  },
  {
    id: 'SmolLM2-360M-Instruct',
    repo: 'HuggingFaceTB/SmolLM2-360M-Instruct',
    bytes: 723_674_912,
    system: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face',
  },
  {
    id: 'SmolLM2-1.7B-Instruct',
    repo: 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
    bytes: 3_422_777_952,
    system: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face',
  },
];

export const hubBase = (repo, revision = 'main') => `https://huggingface.co/${repo}/resolve/${revision}`;

async function openCache() {
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null; // Cache Storage is unavailable (e.g. insecure context); just download
  }
}

/**
 * fetch() with a Cache Storage copy. For large bodies the response is tee'd so the download is
 * written to the cache while it streams to the caller.
 * @returns {Promise<{response: Response, total: number, fromCache: boolean}>}
 */
export async function cachedFetch(url, { cache = true } = {}) {
  const store = cache ? await openCache() : null;
  if (store) {
    const hit = await store.match(url);
    if (hit) return { response: hit, total: Number(hit.headers.get('content-length')) || 0, fromCache: true };
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get('content-length')) || 0;
  if (!store || !response.body) return { response, total, fromCache: false };
  const [forCaller, forCache] = response.body.tee();
  const headers = { 'content-type': response.headers.get('content-type') || 'application/octet-stream' };
  if (total) headers['content-length'] = String(total);
  store.put(url, new Response(forCache, { headers })).catch((e) => console.warn(`not cached: ${url}`, e));
  return { response: new Response(forCaller, { headers }), total, fromCache: false };
}

/** Remove every cached model file. */
export async function clearCache() {
  try {
    return await caches.delete(CACHE_NAME);
  } catch {
    return false;
  }
}

/**
 * Load config, tokenizer and weights from `base` (a Hub resolve URL or any directory URL that
 * serves config.json, tokenizer.json and model.safetensors).
 * @param {string} base
 * @param {{device: GPUDevice, maxSeqLen?: number, cache?: boolean,
 *          onProgress?: (p: {phase: string, loaded?: number, total?: number, fromCache?: boolean}) => void}} opts
 */
export async function loadModel(base, { device, maxSeqLen = 2048, cache = true, onProgress = () => {} } = {}) {
  onProgress({ phase: 'config' });
  const cfgJson = await (await cachedFetch(`${base}/config.json`, { cache })).response.json();
  const config = parseConfig(cfgJson);
  const tokJson = await (await cachedFetch(`${base}/tokenizer.json`, { cache })).response.json();
  const tokenizer = new Tokenizer(tokJson);

  const model = new LlamaModel(device, config, { maxSeqLen });
  const w = await cachedFetch(`${base}/model.safetensors`, { cache });
  onProgress({ phase: 'weights', loaded: 0, total: w.total, fromCache: w.fromCache });
  await model.loadWeights(w.response.body, {
    onBytes: (loaded) => onProgress({ phase: 'weights', loaded, total: w.total, fromCache: w.fromCache }),
  });
  await device.queue.onSubmittedWorkDone();
  onProgress({ phase: 'ready' });
  return { model, tokenizer, config };
}
