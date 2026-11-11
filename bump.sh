#!/bin/sh
# Bump the app version everywhere (offline cache name, in-page APP_V, version.json) before each push.
set -e
cd "$(dirname "$0")"
v=$(python3 -c "import json;print(json.load(open('version.json'))['v']+1)")
sed -i "s/const APP_V = [0-9]*;/const APP_V = $v;/" index.html
sed -i "s/ihab-thk-v[0-9]*/ihab-thk-v$v/" sw.js
printf '{"v": %s}\n' "$v" > version.json
echo "version $v"
