/* On-device lens segmentation (SAM ViT-B on GPU/CPU, SlimSAM fallback) — runs offline in the browser.
   Everything (runtime + model, ~35 MB) is served from this site and cached after the first load.
   segmentLens(canvas, box, points) → closed outline of the lens opening in image pixels, or null. */
let T = null, model = null, processor = null, loading = null, embCache = { cv:null, emb:null, inp:null };
const base = new URL('..', import.meta.url).href;

export let modelName = '';
export function aiReady(){ return !!model; }
export function resetAI(){ model = null; processor = null; loading = null; modelName = ''; embCache = { cv:null, emb:null, inp:null }; }   // next loadAI() picks the model again
export async function loadAI(onProgress){
  if (model) return true;
  if (!loading) loading = (async () => {
    T = await import('./transformers.min.js');
    T.env.allowRemoteModels = false; T.env.allowLocalModels = true;
    T.env.localModelPath = base + 'models/';
    T.env.backends.onnx.wasm.wasmPaths = base + 'ai/';
    T.env.backends.onnx.wasm.numThreads = (self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1);
    // files over GitHub's 100 MB limit are stored in parts (models/split.json) → joined back here, each part cached by the SW
    let split = {}; try{ split = await (await fetch(base + 'models/split.json')).json(); }catch(e){}
    T.env.useCustomCache = true;
    T.env.customCache = { put:async () => {}, match:async key => {
      const u = typeof key === 'string' ? key : key.url, rel = Object.keys(split).find(k => u.endsWith(k)); if (!rel) return undefined;
      const parts = []; for (let i = 0; i < split[rel]; i++){ const r = await fetch(base + 'models/' + rel + '.part' + i); if (!r.ok) throw new Error('model part ' + i); parts.push(await r.blob()); if (onProgress) onProgress((i + 1) / split[rel] * 0.95); }
      const blob = new Blob(parts); return new Response(blob, { headers:{ 'content-length':String(blob.size) } }); } };
    // best model this phone can run: SAM ViT-B on the GPU (fast, accurate) → SAM ViT-B on the CPU → SlimSAM (small, fallback)
    let gpu = false; try{ const ad = navigator.gpu && await navigator.gpu.requestAdapter(); gpu = !!(ad && ad.features && ad.features.has('shader-f16')); }catch(e){}
    const prog = p => { if (onProgress && p.status === 'progress' && p.total) onProgress(p.loaded / p.total); };
    // on the CPU the big model only when it can use several threads (else ~30 s per photo) — the small one is ~3× faster
    const many = self.crossOriginIsolated && (navigator.hardwareConcurrency || 2) >= 6;
    let pref = 'auto'; try{ pref = localStorage.getItem('ihabAIModel') || 'auto'; }catch(e){}   // 🧠 user's choice: auto / strong / fast
    const tries = self.__aiModel ? [self.__aiModel] : pref === 'fast' ? ['slim'] : pref === 'strong' ? (gpu ? ['gpu', 'base', 'slim'] : ['base', 'slim'])
      : gpu ? ['gpu', 'base', 'slim'] : many ? ['base', 'slim'] : ['slim'];
    for (const t of tries){
      const id = t === 'slim' ? 'Xenova/slimsam-77-uniform' : 'Xenova/sam-vit-base', opt = t === 'gpu' ? { device:'webgpu', dtype:'fp16', progress_callback:prog } : { dtype:'q8', progress_callback:prog };
      try{ model = await T.SamModel.from_pretrained(id, opt); processor = await T.AutoProcessor.from_pretrained(id); modelName = t; break; }
      catch(e){ console.warn('AI model', t, 'failed', e); model = null; }
    }
    if (!model) throw new Error('no AI model');
    return true;
  })();
  try{ return await loading; }catch(e){ loading = null; throw e; }
}

async function embed(cv){
  if (embCache.cv === cv) return embCache;
  const ctx = cv.getContext('2d', { willReadFrequently:true }), d = ctx.getImageData(0, 0, cv.width, cv.height);
  const img = new T.RawImage(d.data, cv.width, cv.height, 4).rgb();
  const inp = await processor(img);
  const emb = await model.get_image_embeddings(inp);
  embCache = { cv, emb, inp };
  return embCache;
}

/* box: {x0,y0,x1,y1} 0..1 (search area), pos: [[x,y]] 0..1 inside the lens, neg: [[x,y]] 0..1 on the frame/background */
export async function segmentLens(cv, box, pos = [], neg = [], prior = null, lim = null){
  await loadAI();
  const { emb, inp } = await embed(cv);
  const [rh, rw] = inp.reshaped_input_sizes[0];
  const P = pos.map(p => [p[0] * rw, p[1] * rh]).concat(neg.map(p => [p[0] * rw, p[1] * rh]));
  const L = pos.map(() => 1).concat(neg.map(() => 0));
  const feeds = { ...emb };
  if (P.length){
    feeds.input_points = new T.Tensor('float32', P.flat(), [1, 1, P.length, 2]);
    feeds.input_labels = new T.Tensor('int64', L.map(BigInt), [1, 1, P.length]);
  }
  // (this distilled model takes points only; the box is used below to choose the right mask)
  const out = await model(feeds);
  const masks = await processor.post_process_masks(out.pred_masks, inp.original_sizes, inp.reshaped_input_sizes);
  const M = masks[0], [, , nm, H, W] = M.dims.length === 5 ? M.dims : [1, ...M.dims], data = M.data, sc = out.iou_scores.data;
  // pick the mask that looks like a lens opening: decent IoU score, fills the box well, doesn't spill out of it
  let best = null;
  for (let k = 0; k < nm; k++){
    let area = 0, inBox = 0, x0 = W, x1 = 0, y0 = H, y1 = 0;
    const bx0 = box ? box.x0 * W : 0, bx1 = box ? box.x1 * W : W, by0 = box ? box.y0 * H : 0, by1 = box ? box.y1 * H : H;
    for (let y = 0; y < H; y += 2) for (let x = 0; x < W; x += 2){
      if (!data[k * W * H + y * W + x]) continue; area++;
      if (x >= bx0 && x <= bx1 && y >= by0 && y <= by1) inBox++;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    if (!area) continue;
    const boxA = (bx1 - bx0) * (bx1 - bx0 > 0 ? (by1 - by0) : 1) / 4, fill = area / boxA, spill = 1 - inBox / area;
    let s = sc[k] - 0.8 * spill - 0.6 * Math.abs(Math.min(1.2, fill) - 0.8);
    if (prior){ let pa = 0; for (let i = 0; i < prior.length; i++){ const a = prior[i], b = prior[(i + 1) % prior.length]; pa += a[0] * b[1] - b[0] * a[1]; }
      pa = Math.abs(pa) / 2 * (W / cv.width) * (H / cv.height); const ra = area * 4 / (pa || 1); s = sc[k] - 1.5 * Math.abs(Math.log(Math.max(0.05, ra))); }
    else if (!box){ const fr = area * 4 / (W * H), edge = x0 <= 2 || y0 <= 2 || x1 >= W - 3 || y1 >= H - 3;   // tap mode: a lens is a closed, mid-size region
      s = sc[k] - (edge ? 0.6 : 0) - (fr > 0.3 ? 0.6 : 0) - (fr < 0.004 ? 0.6 : 0); }
    if (!best || s > best.s) best = { k, s, x0, x1, y0, y1, score:sc[k] };
  }
  if (!best) return null;
  // outline: star sampling from the lens centre, 360 rays; with a prior outline (classic tracer) the rays are kept
  // within 0.82…1.15 of it, so a mask that leaks through the bridge gap or onto the shelf can't run away
  const k = best.k, at = (x, y) => x >= 0 && y >= 0 && x < W && y < H && data[k * W * H + (y | 0) * W + (x | 0)];
  const sx = W / cv.width, sy = H / cv.height;   // mask is at original size = canvas size, normally 1
  let cx = 0, cy = 0, n = 0;
  if (prior){ prior.forEach(p => { cx += p[0]; cy += p[1]; }); cx /= prior.length; cy /= prior.length; }
  else { for (let y = best.y0; y <= best.y1; y += 2) for (let x = best.x0; x <= best.x1; x += 2) if (at(x, y)){ cx += x; cy += y; n++; } cx /= n; cy /= n; }
  const pr = t => { let bd = 9, br = 0; for (const p of prior){ const a = Math.atan2(cy - p[1], p[0] - cx), d = Math.abs(Math.atan2(Math.sin(a - t), Math.cos(a - t))); if (d < bd){ bd = d; br = Math.hypot(p[0] - cx, p[1] - cy); } } return br; };
  const Rall = Math.hypot(best.x1 - best.x0, best.y1 - best.y0) * 0.6, rr = new Float32Array(360), used = new Uint8Array(360);
  for (let a = 0; a < 360; a++){
    const t = a / 360 * Math.PI * 2, ux = Math.cos(t), uy = -Math.sin(t), p0 = prior ? pr(t) : 0, R = prior ? p0 * (lim ? lim.hi : 1.12) : Rall;
    let last = 0; for (let s = 0; s < R; s += 0.5) if (at((cx + ux * s) * sx, (cy + uy * s) * sy)) last = s;
    if (prior && last < p0 * (lim ? lim.lo : 0.8)){ rr[a] = p0; } else { rr[a] = last; used[a] = 1; }
  }
  const med = Array.from(rr, (_, a) => { const w = []; for (let q = -6; q <= 6; q++) w.push(rr[(a + q + 360) % 360]); w.sort((p, q) => p - q); return w[6]; });
  const pts = med.map((r, a) => { const t = a / 360 * Math.PI * 2; return [cx + Math.cos(t) * r, cy - Math.sin(t) * r]; });
  const agree = used.reduce((s2, v) => s2 + v, 0) / 360;
  return { pts, score:best.score, agree, cx, cy };
}

/* 🔎 find the lenses by itself on a photo straightened with the card (sPx px per mm): a grid of single-point prompts
   on one image embedding; keep masks that are lens-sized in mm, closed, fairly full, not the card; pair two alike side by side */
export async function autoLenses(cv, sPx, avoid, onProg){
  await loadAI();
  const { emb, inp } = await embed(cv);
  const [rh, rw] = inp.reshaped_input_sizes[0], W0 = cv.width, H0 = cv.height, step = 16 * sPx, cands = [];
  autoLenses.stat = { pts:0, masks:0, sized:0, model:modelName };
  const pts = []; for (let y = step / 2; y < H0; y += step) for (let x = step / 2; x < W0; x += step){
    if (avoid && x > avoid.x0 - 5 * sPx && x < avoid.x1 + 5 * sPx && y > avoid.y0 - 5 * sPx && y < avoid.y1 + 5 * sPx) continue; pts.push([x, y]); }
  pts.sort((a, b) => Math.hypot(a[0] - W0 / 2, a[1] - H0 / 2) - Math.hypot(b[0] - W0 / 2, b[1] - H0 / 2));
  const inside = (c, x, y) => c.m && x >= c.bx0 && x <= c.bx1 && y >= c.by0 && y <= c.by1 && c.m[Math.floor((y - c.by0) / 2) * c.mw + Math.floor((x - c.bx0) / 2)];
  for (let i = 0; i < pts.length; i++){
    if (onProg) onProg(i / pts.length); autoLenses.stat.pts++;
    if (self.__aiStop && Date.now() > self.__aiStop){ autoLenses.stat.timeout = true; break; }
    const [px, py] = pts[i]; if (cands.some(c => inside(c, px, py))) continue;   // already inside a found lens
    const feeds = { ...emb, input_points:new T.Tensor('float32', [px / W0 * rw, py / H0 * rh], [1, 1, 1, 2]), input_labels:new T.Tensor('int64', [1n], [1, 1, 1]) };
    const out = await model(feeds);
    const masks = await processor.post_process_masks(out.pred_masks, inp.original_sizes, inp.reshaped_input_sizes);
    const M = masks[0], dims = M.dims, H = dims[dims.length - 2], W = dims[dims.length - 1], nm = dims[dims.length - 3], data = M.data, sc = out.iou_scores.data;
    for (let k = 0; k < nm; k++){
      let area = 0, x0 = W, x1 = 0, y0 = H, y1 = 0;
      for (let y = 0; y < H; y += 2) for (let x = 0; x < W; x += 2) if (data[k * W * H + y * W + x]){ area++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
      if (!area) continue; area *= 4;
      const wm = (x1 - x0) / sPx, hm = (y1 - y0) / sPx, fill = area / Math.max(1, (x1 - x0) * (y1 - y0)), edge = x0 < 3 || y0 < 3 || x1 > W - 4 || y1 > H - 4;
      if (self.__dbgL) self.__dbgL.push([Math.round(px), Math.round(py), k, +wm.toFixed(0), +hm.toFixed(0), +fill.toFixed(2), edge, +sc[k].toFixed(2)]);
      autoLenses.stat.masks++; if (wm >= 25 && wm <= 90) autoLenses.stat.sized++;
      if (wm < 32 || wm > 72 || hm < 20 || hm > 62 || wm / hm < 0.85 || wm / hm > 2.4 || fill < 0.6 || fill > 0.96 || edge || sc[k] < 0.6) continue;
      if (avoid){ const ix = Math.max(0, Math.min(x1, avoid.x1) - Math.max(x0, avoid.x0)), iy = Math.max(0, Math.min(y1, avoid.y1) - Math.max(y0, avoid.y0)); if (ix * iy > 0.1 * (x1 - x0) * (y1 - y0)) continue; }
      // keep a low-res copy of the mask (to skip grid points inside it) and the outline (360 rays from its centre)
      const mw = Math.ceil((x1 - x0 + 1) / 2), mh = Math.ceil((y1 - y0 + 1) / 2), m = new Uint8Array(mw * mh); let cx = 0, cy = 0, n = 0;
      for (let y = y0; y <= y1; y += 2) for (let x = x0; x <= x1; x += 2) if (data[k * W * H + y * W + x]){ m[((y - y0) / 2 | 0) * mw + ((x - x0) / 2 | 0)] = 1; cx += x; cy += y; n++; }
      cx /= n; cy /= n;
      const at = (x, y) => x >= 0 && y >= 0 && x < W && y < H && data[k * W * H + (y | 0) * W + (x | 0)], R = Math.hypot(x1 - x0, y1 - y0) * 0.6, rr = new Float32Array(360);
      for (let a = 0; a < 360; a++){ const t = a / 180 * Math.PI, ux = Math.cos(t), uy = -Math.sin(t); let last = 0; for (let s = 0; s < R; s += 0.5) if (at(cx + ux * s, cy + uy * s)) last = s; rr[a] = last; }
      const med = Array.from(rr, (_, a) => { const w = []; for (let q = -4; q <= 4; q++) w.push(rr[(a + q + 360) % 360]); w.sort((p, q) => p - q); return w[4]; });
      const P = med.map((r, a) => { const t = a / 180 * Math.PI; return [cx + Math.cos(t) * r, cy - Math.sin(t) * r]; });
      const sx = W0 / W, sy = H0 / H;
      cands.push({ s:sc[k] - Math.abs(fill - 0.8) * 0.5, cx:cx * sx, cy:cy * sy, bx0:x0 * sx, bx1:x1 * sx, by0:y0 * sy, by1:y1 * sy, mw, m, pts:P.map(([x, y]) => [x * sx, y * sy]) });
    }
    // de-duplicate: overlapping finds → keep the better one
    for (let a = 0; a < cands.length; a++) for (let b = cands.length - 1; b > a; b--){ const A = cands[a], B = cands[b];
      const ix = Math.max(0, Math.min(A.bx1, B.bx1) - Math.max(A.bx0, B.bx0)), iy = Math.max(0, Math.min(A.by1, B.by1) - Math.max(A.by0, B.by0)), u = (A.bx1 - A.bx0) * (A.by1 - A.by0) + (B.bx1 - B.bx0) * (B.by1 - B.by0) - ix * iy;
      if (ix * iy / u > 0.5){ if (B.s > A.s) cands[a] = B; cands.splice(b, 1); } }
  }
  if (onProg) onProg(1);
  // best pair: same height line, alike in size, a bridge gap of 8–28 mm between them
  let best = null;
  for (const A of cands) for (const B of cands){ if (A === B || A.cx > B.cx) continue;
    const wa = A.bx1 - A.bx0, wb = B.bx1 - B.bx0, ha = A.by1 - A.by0, hb = B.by1 - B.by0, gap = (B.bx0 - A.bx1) / sPx;
    if (Math.abs(A.cy - B.cy) / sPx > 10 || wa / wb < 0.82 || wa / wb > 1.22 || ha / hb < 0.82 || ha / hb > 1.22 || gap < 8 || gap > 28) continue;
    const s = A.s + B.s - Math.abs(Math.log(wa / wb)) - Math.abs(Math.log(ha / hb)); if (!best || s > best.s) best = { s, A, B, gap:B.bx0 - A.bx1 }; }
  if (best) return { lenses:[best.A, best.B], gap:best.gap };
  cands.sort((a, b) => b.s - a.s);
  return { lenses:cands.slice(0, 1), gap:0 };
}
