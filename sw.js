/* IHAB Thickness app: network-first for the page (edits show on refresh), cache fallback for offline. */
const CACHE = 'ihab-thk-v18';
self.addEventListener('install', e => { self.skipWaiting(); e.waitUntil(caches.open(CACHE).then(c => c.addAll(['./', 'index.html', 'card.html', 'manifest.json', 'icon-192.png', 'icon-512.png', 'icon-maskable-512.png']).catch(() => {}))); });
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  const req = e.request; if (req.method !== 'GET') return;
  const same = new URL(req.url).origin === location.origin;
  e.respondWith(fetch(same ? new Request(req, { cache:'no-cache' }) : req).then(res => { if (res && (res.ok || res.type === 'opaque')) { const cp = res.clone(); caches.open(CACHE).then(c => c.put(req, cp)); } return res; })
    .catch(() => caches.match(req, { ignoreSearch: true })));
});
