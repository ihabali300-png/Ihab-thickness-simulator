/* On-device lens segmentation (SlimSAM, Segment Anything distilled) — runs offline in the browser.
   Everything (runtime + model, ~35 MB) is served from this site and cached after the first load.
   segmentLens(canvas, box, points) → closed outline of the lens opening in image pixels, or null. */
let T = null, model = null, processor = null, loading = null, embCache = { cv:null, emb:null, inp:null };
const base = new URL('..', import.meta.url).href;

export function aiReady(){ return !!model; }
export async function loadAI(onProgress){
  if (model) return true;
  if (!loading) loading = (async () => {
    T = await import('./transformers.min.js');
    T.env.allowRemoteModels = false; T.env.allowLocalModels = true;
    T.env.localModelPath = base + 'models/';
    T.env.backends.onnx.wasm.wasmPaths = base + 'ai/';
    T.env.backends.onnx.wasm.numThreads = (self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1);
    const opt = { dtype:'q8', progress_callback:p => { if (onProgress && p.status === 'progress' && p.total) onProgress(p.loaded / p.total); } };
    model = await T.SamModel.from_pretrained('Xenova/slimsam-77-uniform', opt);
    processor = await T.AutoProcessor.from_pretrained('Xenova/slimsam-77-uniform');
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
