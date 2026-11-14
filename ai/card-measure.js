/* 📏 Card-scale measuring (OpenCV.js, offline after the first load)
   A bank-size card (ISO/IEC 7810 ID-1: 85.60 × 53.98 mm) lies on the same paper as the frame.
   1) find the card's 4 corners  2) warp the photo so the paper is seen straight from above, at a known px/mm
   → the lens shape is true (no tilt distortion) and A / B / DBL come out in millimetres. */
const CW = 85.60, CH = 53.98;
let cvP = null;
export function loadCV(){
  if (cvP) return cvP;
  cvP = new Promise((res, rej) => {
    const s = document.createElement('script'); s.src = new URL('./opencv.js', import.meta.url).href; s.async = true;
    s.onerror = () => { cvP = null; rej(new Error('opencv load failed')); };
    s.onload = async () => {
      try{
        let cv = window.cv; if (cv instanceof Promise) cv = await new Promise(r => cv.then(m => r({ m }))).then(o => o.m);
        if (!cv.Mat) await new Promise((r, j) => { cv.onRuntimeInitialized = r; const t0 = Date.now(), iv = setInterval(() => { if (cv.Mat){ clearInterval(iv); r(); } else if (Date.now() - t0 > 60000){ clearInterval(iv); j(new Error('opencv init timeout')); } }, 100); });   // init may finish before onload
        if (typeof cv.then === 'function') delete cv.then;   // emscripten's thenable Module would make the promise resolve forever
        window.cv = cv; res(cv);
      }catch(e){ cvP = null; rej(e); }
    };
    document.head.appendChild(s);
  });
  return cvP;
}

const order = P => {   // TL, TR, BR, BL
  const c = P.reduce((a, p) => [a[0] + p[0] / 4, a[1] + p[1] / 4], [0, 0]);
  const Q = P.slice().sort((a, b) => Math.atan2(a[1] - c[1], a[0] - c[0]) - Math.atan2(b[1] - c[1], b[0] - c[0]));   // clockwise on screen (y down)
  let i0 = 0, best = 1e18; Q.forEach((p, i) => { const v = p[0] + p[1]; if (v < best){ best = v; i0 = i; } });
  return [0, 1, 2, 3].map(k => Q[(i0 + k) % 4]);
};
const fitLine = pts => {   // PCA line through points → [px, py, dx, dy]
  let mx = 0, my = 0; pts.forEach(p => { mx += p[0]; my += p[1]; }); mx /= pts.length; my /= pts.length;
  let a = 0, b = 0, c = 0; pts.forEach(p => { const x = p[0] - mx, y = p[1] - my; a += x * x; b += x * y; c += y * y; });
  const t = 0.5 * Math.atan2(2 * b, a - c); return [mx, my, Math.cos(t), Math.sin(t)];
};
const cross = (L1, L2) => { const d = L1[2] * L2[3] - L1[3] * L2[2]; if (Math.abs(d) < 1e-9) return null;
  const t = ((L2[0] - L1[0]) * L2[3] - (L2[1] - L1[1]) * L2[2]) / d; return [L1[0] + L1[2] * t, L1[1] + L1[3] * t]; };

/* → [[x,y]×4] (TL,TR,BR,BL) in the canvas' pixels, or null */
export function findCard(cv, canvas){
  const k = Math.min(1, 1000 / Math.max(canvas.width, canvas.height));
  const sm = document.createElement('canvas'); sm.width = Math.round(canvas.width * k); sm.height = Math.round(canvas.height * k);
  const sg = sm.getContext('2d', { willReadFrequently:true }); sg.drawImage(canvas, 0, 0, sm.width, sm.height);
  const D = sg.getImageData(0, 0, sm.width, sm.height).data, W = sm.width, H = sm.height;
  const gray = (x, y) => { const i = (y * W + x) * 4; return 0.3 * D[i] + 0.59 * D[i + 1] + 0.11 * D[i + 2]; };
  const sat = (x, y) => { const i = (y * W + x) * 4, mx = Math.max(D[i], D[i + 1], D[i + 2]); return mx ? (mx - Math.min(D[i], D[i + 1], D[i + 2])) / mx * 255 : 0; };
  const src = cv.imread(sm), g = new cv.Mat(), e = new cv.Mat(), cs = new cv.MatVector(), hi = new cv.Mat(), ker = cv.Mat.ones(3, 3, cv.CV_8U);
  const cands = [];
  try{
    cv.cvtColor(src, g, cv.COLOR_RGBA2GRAY); cv.GaussianBlur(g, g, new cv.Size(5, 5), 0);
    for (const [lo, hiT] of [[30, 90], [60, 160], [15, 45]]){
      cv.Canny(g, e, lo, hiT); cv.dilate(e, e, ker);
      cv.findContours(e, cs, hi, cv.RETR_LIST, cv.CHAIN_APPROX_NONE);
      for (let i = 0; i < cs.size(); i++){
        const c = cs.get(i), area = cv.contourArea(c);
        if (area < 0.004 * W * H || area > 0.45 * W * H){ c.delete(); continue; }
        const peri = cv.arcLength(c, true), ap = new cv.Mat(); cv.approxPolyDP(c, ap, 0.03 * peri, true);
        if (ap.rows === 4 && cv.isContourConvex(ap)){
          const P = order([0, 1, 2, 3].map(j => [ap.data32S[j * 2], ap.data32S[j * 2 + 1]]));
          const sd = [0, 1, 2, 3].map(j => Math.hypot(P[(j + 1) % 4][0] - P[j][0], P[(j + 1) % 4][1] - P[j][1]));
          let asp = (sd[0] + sd[2]) / (sd[1] + sd[3]); if (asp < 1) asp = 1 / asp;
          const rr = cv.minAreaRect(c), fill = area / Math.max(1, rr.size.width * rr.size.height);
          const touch = P.some(p => p[0] < 3 || p[1] < 3 || p[0] > W - 4 || p[1] > H - 4);
          // straight sides: almost every contour point lies on one of the 4 sides (lens openings are curved)
          let onLine = 0; const tol = Math.max(2, 0.012 * peri);
          for (let j = 0; j < c.rows; j += 2){ const x = c.data32S[j * 2], y = c.data32S[j * 2 + 1]; let dm = 1e9;
            for (let q = 0; q < 4; q++){ const A = P[q], B = P[(q + 1) % 4], dx = B[0] - A[0], dy = B[1] - A[1], L = Math.hypot(dx, dy); dm = Math.min(dm, Math.abs((x - A[0]) * dy - (y - A[1]) * dx) / L); }
            if (dm < tol) onLine++; }
          const straight = onLine / Math.ceil(c.rows / 2);
          if (typeof window !== 'undefined' && window.__dbg) window.__dbg.push([+asp.toFixed(2), +fill.toFixed(2), +straight.toFixed(2), +(area / (W * H)).toFixed(3), touch]);
          if (asp > 1.38 && asp < 1.82 && fill > 0.9 && straight > 0.93 && area < 0.25 * W * H && !touch){
            // the card is coloured / darker than the white paper around it; lens openings and the paper are bright
            let x0 = W, x1 = 0, y0 = H, y1 = 0; P.forEach(p => { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); });
            const inside = (x, y) => { let r = false; for (let a = 0, b = 3; a < 4; b = a++){ const A = P[a], B = P[b]; if ((A[1] > y) !== (B[1] > y) && x < (B[0] - A[0]) * (y - A[1]) / (B[1] - A[1]) + A[0]) r = !r; } return r; };
            let gi = 0, si = 0, ni = 0, go = 0, no = 0; const st = Math.max(1, Math.round((x1 - x0) / 40)), mx = (x1 - x0) * 0.25, my = (y1 - y0) * 0.25;
            for (let y = Math.max(0, Math.round(y0 - my)); y < Math.min(H, y1 + my); y += st) for (let x = Math.max(0, Math.round(x0 - mx)); x < Math.min(W, x1 + mx); x += st){
              if (inside(x, y)){ gi += gray(x, y); si += sat(x, y); ni++; } else { go += gray(x, y); no++; } }
            if (ni && no){ gi /= ni; si /= ni; go /= no;
              const contrast = Math.abs(go - gi) / 255 + si / 255 * 0.8;
              const pts = []; for (let j = 0; j < c.rows; j++) pts.push([c.data32S[j * 2], c.data32S[j * 2 + 1]]);
              cands.push({ P, pts, area, score:contrast - Math.abs(Math.log(asp / (CW / CH))) * 1.2 + Math.min(0.15, area / (W * H)) }); } }
        }
        ap.delete(); c.delete();
      }
    }
  } finally { [src, g, e, cs, hi, ker].forEach(m => m.delete()); }
  cands.sort((a, b) => b.score - a.score);
  const best = cands[0]; if (!best || best.score < 0.18) return null;   // a lens opening (paper seen through it) has almost no contrast/colour → never taken for the card
  // refine: a straight line through each side's contour points (skipping the rounded corners) → corners = line crossings
  const P = best.P, L = [];
  for (let j = 0; j < 4; j++){
    const A = P[j], B = P[(j + 1) % 4], dx = B[0] - A[0], dy = B[1] - A[1], len = Math.hypot(dx, dy), sel = [];
    for (const p of best.pts){ const t = ((p[0] - A[0]) * dx + (p[1] - A[1]) * dy) / (len * len), d = Math.abs((p[0] - A[0]) * dy - (p[1] - A[1]) * dx) / len;
      if (t > 0.15 && t < 0.85 && d < Math.max(3, len * 0.04)) sel.push(p); }
    L.push(sel.length > 8 ? fitLine(sel) : [A[0], A[1], dx / len, dy / len]);
  }
  const R = [0, 1, 2, 3].map(j => cross(L[(j + 3) % 4], L[j]) || P[j]);
  // a bank card has rounded corners (r = 3.18 mm ≈ 3.7 % of its width): the outline cuts each corner by ~1.5 % of the width.
  // posters, screens, boxes have sharp corners → not a card
  const wC = Math.hypot(R[1][0] - R[0][0], R[1][1] - R[0][1]) + Math.hypot(R[2][0] - R[1][0], R[2][1] - R[1][1]);
  const gaps = R.map(q => { let d = 1e9; for (const p of best.pts) d = Math.min(d, Math.hypot(p[0] - q[0], p[1] - q[1])); return d / (wC / 2.586 * 1.586 / 1); });
  const rounded = gaps.filter(gp => gp > 0.0055 && gp < 0.05).length;
  if (typeof window !== 'undefined') window.__cardGaps = gaps;
  if (rounded < 3 && !(typeof window !== "undefined" && window.__noRound)) return null;
  if (R.some((p, j) => Math.hypot(p[0] - P[j][0], p[1] - P[j][1]) > 0.15 * Math.sqrt(best.area))) return P.map(p => [p[0] / k, p[1] / k]);
  return R.map(p => [p[0] / k, p[1] / k]);
}

/* Warp so the paper is seen from straight above. The card (corners TL,TR,BR,BL in canvas px) becomes an exact
   85.6 × 53.98 mm rectangle at `s` px/mm. Returns { canvas, s }. The picture is mirrored (frame lies face-down). */
export function rectify(cv, canvas, C, opts = {}){
  const side = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);
  const horiz = side(C[0], C[1]) + side(C[3], C[2]) >= side(C[0], C[3]) + side(C[1], C[2]);
  const w = horiz ? CW : CH, h = horiz ? CH : CW;
  const H0 = (s, ox, oy) => { const sm = cv.matFromArray(4, 1, cv.CV_32FC2, C.flat()), dm = cv.matFromArray(4, 1, cv.CV_32FC2, [ox, oy, ox + w * s, oy, ox + w * s, oy + h * s, ox, oy + h * s]);
    const M = cv.getPerspectiveTransform(sm, dm); sm.delete(); dm.delete(); return M; };
  // working window: the card ± ~230 × 170 mm (a whole A4 sheet around it), cut to what the photo actually shows
  let M = H0(1, 0, 0); const m = M.data64F, map = (x, y) => { const z = m[6] * x + m[7] * y + m[8]; return z > 1e-9 ? [(m[0] * x + m[1] * y + m[2]) / z, (m[3] * x + m[4] * y + m[5]) / z] : null; };
  M.delete();
  const cx = w / 2, cy = h / 2; let bx0 = cx - 230, bx1 = cx + 230, by0 = cy - 175, by1 = cy + 175;
  const IC = [[0, 0], [canvas.width, 0], [canvas.width, canvas.height], [0, canvas.height]].map(p => map(p[0], p[1]));
  if (IC.every(Boolean)){ bx0 = Math.max(bx0, Math.min(...IC.map(p => p[0]))); bx1 = Math.min(bx1, Math.max(...IC.map(p => p[0])));
    by0 = Math.max(by0, Math.min(...IC.map(p => p[1]))); by1 = Math.min(by1, Math.max(...IC.map(p => p[1]))); }
  const s = Math.min(opts.maxPxMm || 9, (opts.maxSide || 2200) / Math.max(bx1 - bx0, by1 - by0));
  const ow = Math.round((bx1 - bx0) * s), oh = Math.round((by1 - by0) * s);
  M = H0(s, -bx0 * s, -by0 * s);
  const src = cv.imread(canvas), dst = new cv.Mat();
  cv.warpPerspective(src, dst, M, new cv.Size(ow, oh), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(128, 128, 128, 255));
  if (opts.mirror !== false) cv.flip(dst, dst, 1);
  const out = document.createElement('canvas'); cv.imshow(out, dst);
  src.delete(); dst.delete(); M.delete();
  // where the card ended up (so the lens finder can ignore it)
  const card = { x0:-bx0 * s, x1:(w - bx0) * s, y0:-by0 * s, y1:(h - by0) * s };
  if (opts.mirror !== false){ const a = ow - card.x1, b = ow - card.x0; card.x0 = a; card.x1 = b; }
  return { canvas:out, s, card };
}

/* 📄 Measuring mat (ai/mat-spec.json): 4 ArUco 4x4 markers (ids 0–3, 22 mm) at known mm positions on an A4 sheet.
   OpenCV.js here has no aruco module → own detector: dark convex quads → sample the 6×6 grid → black border +
   inner 4×4 bits = one of the 4 codes (in any rotation) → every found corner gives an image↔mm pair → homography. */
const MAT = { page:[297, 210], ms:22, m:9, codes:{
  0:[[1,0,1,1],[0,1,0,1],[0,0,1,1],[0,0,1,0]], 1:[[0,0,0,0],[1,1,1,1],[1,0,0,1],[1,0,1,0]],
  2:[[0,0,1,1],[0,0,1,1],[0,0,1,0],[1,1,0,1]], 3:[[1,0,0,1],[1,0,0,1],[0,1,0,0],[0,1,1,0]],
  // mat v3 (bigger zones, ruler on the edges only): ids 4–7, 18 mm, 6 mm from the page edge
  4:[[0,1,0,1],[0,1,0,0],[1,0,0,1],[1,1,1,0]], 5:[[0,1,1,1],[1,0,0,1],[1,1,0,0],[1,1,0,1]],
  6:[[1,0,0,1],[1,1,1,0],[0,0,1,0],[1,1,1,0]], 7:[[1,1,0,0],[0,1,0,0],[1,1,1,1],[0,0,1,0]] } };
MAT.ver = id => id >= 4 ? 3 : 2;
MAT.size = id => id >= 4 ? 18 : 22;
{ const P = MAT.page, cornerPos = (j, ms, m) => [[m, m], [P[0] - m - ms, m], [P[0] - m - ms, P[1] - m - ms], [m, P[1] - m - ms]][j];
  MAT.pos = {}; for (let id = 0; id < 8; id++) MAT.pos[id] = id >= 4 ? cornerPos(id - 4, 18, 6) : cornerPos(id, 22, 9); }
const rotCW = g => g[0].map((_, c) => g.map(r => r[c]).reverse());
const homog = (src, dst) => {   // 4-point homography (src→dst) as a 3×3 array
  const A = [], b = [];
  for (let i = 0; i < 4; i++){ const [x, y] = src[i], [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u); A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v); }
  for (let i = 0; i < 8; i++){ let p = i; for (let r = i + 1; r < 8; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
    [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
    for (let r = 0; r < 8; r++) if (r !== i){ const f = A[r][i] / A[i][i]; for (let c = i; c < 8; c++) A[r][c] -= f * A[i][c]; b[r] -= f * b[i]; } }
  const h = b.map((v, i) => v / A[i][i]); return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1]; };
const apply = (H, x, y) => { const z = H[6] * x + H[7] * y + H[8]; return [(H[0] * x + H[1] * y + H[2]) / z, (H[3] * x + H[4] * y + H[5]) / z]; };

/* → { pts:[[imgX,imgY,mmX,mmY]…], ids:[…] } in the canvas' pixels, or null (needs ≥ 2 markers) */
export function findMat(cv, canvas){
  const k = Math.min(1, 1600 / Math.max(canvas.width, canvas.height));
  const sm = document.createElement('canvas'); sm.width = Math.round(canvas.width * k); sm.height = Math.round(canvas.height * k);
  const sg = sm.getContext('2d', { willReadFrequently:true }); sg.drawImage(canvas, 0, 0, sm.width, sm.height);
  const W = sm.width, H = sm.height, D = sg.getImageData(0, 0, W, H).data, G = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) G[i] = 0.3 * D[i * 4] + 0.59 * D[i * 4 + 1] + 0.11 * D[i * 4 + 2];
  const at = (x, y) => { const xi = Math.max(0, Math.min(W - 1, Math.round(x))), yi = Math.max(0, Math.min(H - 1, Math.round(y))); return G[yi * W + xi]; };
  const src = cv.imread(sm), g = new cv.Mat(), bin = new cv.Mat(), cs = new cv.MatVector(), hi = new cv.Mat(), found = {};
  try{
    cv.cvtColor(src, g, cv.COLOR_RGBA2GRAY);
    for (const bs of [Math.round(W / 40) | 1, Math.round(W / 15) | 1]){
      cv.adaptiveThreshold(g, bin, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, Math.max(11, bs), 7);
      cv.findContours(bin, cs, hi, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
      for (let i = 0; i < cs.size(); i++){
        const c = cs.get(i), area = cv.contourArea(c);
        if (area < 250 || area > 0.06 * W * H){ c.delete(); continue; }
        const ap = new cv.Mat(); cv.approxPolyDP(c, ap, 0.05 * cv.arcLength(c, true), true);
        if (ap.rows === 4 && cv.isContourConvex(ap)){
          const Q = order([0, 1, 2, 3].map(j => [ap.data32S[j * 2], ap.data32S[j * 2 + 1]]));
          // read the 6×6 grid: canonical (0..6) → image
          const Hc = homog([[0, 0], [6, 0], [6, 6], [0, 6]], Q), cell = [];
          for (let r = 0; r < 6; r++){ const row = []; for (let q = 0; q < 6; q++){ let s = 0;
            for (const dy of [0.3, 0.5, 0.7]) for (const dx of [0.3, 0.5, 0.7]){ const [x, y] = apply(Hc, q + dx, r + dy); s += at(x, y); } row.push(s / 9); } cell.push(row); }
          const flat = cell.flat(), lo = Math.min(...flat), hiV = Math.max(...flat), thr = (lo + hiV) / 2;
          if (hiV - lo > 40){
            const border = [0, 1, 2, 3, 4, 5].every(t => cell[0][t] < thr && cell[5][t] < thr && cell[t][0] < thr && cell[t][5] < thr);
            if (border){
              const bits = [1, 2, 3, 4].map(r => [1, 2, 3, 4].map(q => cell[r][q] > thr ? 1 : 0));
              for (let id = 0; id < 8; id++){ let code = MAT.codes[id];
                for (let rot = 0; rot < 4; rot++){
                  if (code.every((row, r) => row.every((v, q) => v === bits[r][q]))){
                    // the marker's own TL corner sits at image corner `rot`
                    const mm = MAT.pos[id], z = MAT.size(id), mc = [[mm[0], mm[1]], [mm[0] + z, mm[1]], [mm[0] + z, mm[1] + z], [mm[0], mm[1] + z]];
                    const pts = [0, 1, 2, 3].map(j => { const p = Q[(rot + j) % 4]; return [p[0] / k, p[1] / k, mc[j][0], mc[j][1]]; });
                    if (!found[id] || area > found[id].area) found[id] = { area, pts };
                  }
                  code = rotCW(code);
                } }
            } }
        }
        ap.delete(); c.delete();
      }
    }
  } finally { [src, g, bin, cs, hi].forEach(m => m.delete()); }
  // old (v2) or new (v3) sheet — whichever has more markers in the photo
  const all = Object.keys(found).map(Number), n3 = all.filter(i => i >= 4).length, ver = n3 > all.length - n3 ? 3 : 2;
  const ids = all.filter(i => MAT.ver(i) === ver); if (ids.length < 2) return null;
  return { ids, ver, pts:ids.flatMap(id => found[id].pts) };
}

/* straighten the whole sheet to s px/mm (page coordinates, mirrored when the frame lies face-down) */
export function rectifyMat(cv, canvas, M, opts = {}){
  const s = Math.min(opts.maxPxMm || 8, (opts.maxSide || 2200) / MAT.page[0]), ow = Math.round(MAT.page[0] * s), oh = Math.round(MAT.page[1] * s);
  const sp = cv.matFromArray(M.pts.length, 1, cv.CV_32FC2, M.pts.flatMap(p => [p[0], p[1]]));
  const dp = cv.matFromArray(M.pts.length, 1, cv.CV_32FC2, M.pts.flatMap(p => [p[2] * s, p[3] * s]));
  const Hm = cv.findHomography(sp, dp, 0);
  // check: how far each corner lands from where it should (px) → quality
  const h = Hm.data64F; let err = 0; M.pts.forEach(p => { const z = h[6] * p[0] + h[7] * p[1] + h[8], u = (h[0] * p[0] + h[1] * p[1] + h[2]) / z, v = (h[3] * p[0] + h[4] * p[1] + h[5]) / z; err = Math.max(err, Math.hypot(u - p[2] * s, v - p[3] * s)); });
  const src = cv.imread(canvas), dst = new cv.Mat();
  cv.warpPerspective(src, dst, Hm, new cv.Size(ow, oh), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(128, 128, 128, 255));
  if (opts.mirror !== false) cv.flip(dst, dst, 1);
  const out = document.createElement('canvas'); cv.imshow(out, dst);
  [sp, dp, Hm, src, dst].forEach(m => m.delete());
  return { canvas:out, s, card:{ x0:-1e6, x1:-1e6 + 1, y0:-1e6, y1:-1e6 + 1 }, mat:true, ver:M.ver || 2, mirror:opts.mirror !== false, err:err / s };   // err in mm
}
