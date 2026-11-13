#!/bin/sh
# Usage: ./bump.sh "ملاحظة 1" "ملاحظة 2" ...  → bumps the version everywhere and saves the "what's new" notes
# + the update's download size (files changed since the last bump, gzip like GitHub Pages sends them).
set -e
cd "$(dirname "$0")"
v=$(python3 -c "import json;print(json.load(open('version.json'))['v']+1)")
sed -i "s/const APP_V = [0-9]*;/const APP_V = $v;/" index.html
sed -i "s/ihab-thk-v[0-9]*/ihab-thk-v$v/" sw.js
python3 - "$v" "$@" <<'PY'
import json, sys, subprocess, os, gzip, re
base = subprocess.run(['git','log','-1','--format=%H','--','version.json'], capture_output=True, text=True).stdout.strip()
ch = set(subprocess.run(['git','diff','--name-only', base], capture_output=True, text=True).stdout.split()) if base else set()
ch |= set(subprocess.run(['git','ls-files','--others','--exclude-standard'], capture_output=True, text=True).stdout.split())
core = lazy = 0
for f in ch:
    if not os.path.isfile(f) or f.startswith(('_t', '_tr', 'scripts/', '.github/')) or f in ('version.json', 'bump.sh'): continue
    n = os.path.getsize(f)
    # AI runtime / OpenCV / models download only when that tool is first used
    if f.startswith('models/') or re.search(r'^ai/(.*\.(wasm|mjs)|transformers\.min\.js|opencv\.js)$', f): lazy += n
    else: core += len(gzip.compress(open(f, 'rb').read())) if re.search(r'\.(html|js|json|css|svg|txt)$', f) else n
kb = max(1, round(core / 1024)); tot = (core + lazy) / 1048576   # level by the whole size, AI files included
lvl = 'صغير' if tot < 10 else 'متوسط' if tot <= 100 else 'كبير'
json.dump({"v": int(sys.argv[1]), "notes": sys.argv[2:], "kb": kb, "size": lvl, "lazyMB": round(lazy / 1048576, 1), "totalKB": round((core + lazy) / 1024)}, open("version.json", "w"), ensure_ascii=False)
print('size', kb, 'KB', lvl, '+ lazy', round(lazy / 1048576), 'MB')
PY
echo "version $v"
