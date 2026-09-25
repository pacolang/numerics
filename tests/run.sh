#!/usr/bin/env bash
# Runs every tests/*.paco with the paco compiler and diffs its stdout
# against the matching tests/<name>.expected. No framework: PACO_BIN
# defaults to `paco` on PATH, override it to point at a specific build.
set -u

PACO_BIN="${PACO_BIN:-paco}"
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fail=0

for src in "$dir"/*.paco; do
    name="$(basename "$src" .paco)"
    expected="$dir/$name.expected"
    if [[ ! -f "$expected" ]]; then
        echo "FAIL $name: no $expected"
        fail=1
        continue
    fi
    actual="$("$PACO_BIN" run "$src" 2>&1)"
    if [[ "$actual" != "$(cat "$expected")" ]]; then
        echo "FAIL $name"
        diff <(echo "$actual") "$expected"
        fail=1
    else
        echo "PASS $name"
    fi
done

exit $fail
