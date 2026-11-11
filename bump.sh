#!/bin/sh
# Usage: ./bump.sh "ملاحظة 1" "ملاحظة 2" ...  → bumps the version everywhere and saves the "what's new" notes.
set -e
cd "$(dirname "$0")"
v=$(python3 -c "import json;print(json.load(open('version.json'))['v']+1)")
sed -i "s/const APP_V = [0-9]*;/const APP_V = $v;/" index.html
sed -i "s/ihab-thk-v[0-9]*/ihab-thk-v$v/" sw.js
python3 - "$v" "$@" <<'PY'
import json, sys
json.dump({"v": int(sys.argv[1]), "notes": sys.argv[2:]}, open("version.json", "w"), ensure_ascii=False)
PY
echo "version $v"
