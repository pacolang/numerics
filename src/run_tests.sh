#!/usr/bin/env bash
# Runs every src/*.paco that has a matching src/<name>.expected (this
# repository's unit tests, colocated with the source they test) and diffs
# its stdout against that file. No framework: PACO_BIN defaults to `paco`
# on PATH, override it to point at a specific build.
set -u

PACO_BIN="${PACO_BIN:-paco}"
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fail=0

for expected in "$dir"/*.expected; do
    name="$(basename "$expected" .expected)"
    src="$dir/$name.paco"
    if [[ ! -f "$src" ]]; then
        echo "FAIL $name: no $src"
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
