/* IHAB Thickness app: network-first for the page (edits show on refresh), cache fallback for offline. */
const CACHE = 'ihab-thk-v86';
self.addEventListener('install', e => { self.skipWaiting(); e.waitUntil(caches.open(CACHE).then(c => c.addAll(['./', 'index.html', 'card.html', 'manifest.json', 'icon-192.png', 'icon-512.png', 'icon-maskable-512.png']).catch(() => {}))); });
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE && k !== 'ihab-ai-v1').map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  // AI runtime + model (~35 MB): cache-first in their own cache that survives app updates
  // big AI files (runtime, OpenCV, models — never change under the same name): cache-first in a cache that survives updates.
  // Our own small .js (lens-ai.js, card-measure.js) go network-first below so fixes arrive.
  if ((/\/models\//.test(e.request.url) || /\/ai\/(transformers\.min\.js|opencv\.js|.*\.wasm|.*\.mjs)$/.test(e.request.url)) && e.request.method === 'GET'){
    e.respondWith(caches.open('ihab-ai-v1').then(c => c.match(e.request).then(hit => hit || fetch(e.request).then(res => { if (res.ok) c.put(e.request, res.clone()); return res; }))));
    return;
  }
  const req = e.request; if (req.method !== 'GET') return;
  const same = new URL(req.url).origin === location.origin;
  // the page gets cross-origin isolation (COOP/COEP) → the AI runs on several CPU threads (3–4× faster)
  const iso = res => { if (!res || req.mode !== 'navigate') return res; const h = new Headers(res.headers); h.set('Cross-Origin-Opener-Policy', 'same-origin'); h.set('Cross-Origin-Embedder-Policy', 'credentialless');
    return new Response(res.body, { status:res.status, statusText:res.statusText, headers:h }); };
  e.respondWith(fetch(same ? new Request(req, { cache:'no-cache' }) : req).then(res => { if (res && (res.ok || res.type === 'opaque')) { const cp = res.clone(); caches.open(CACHE).then(c => c.put(req, cp)); } return iso(res); })
    .catch(() => caches.match(req, { ignoreSearch: true }).then(iso)));
});
