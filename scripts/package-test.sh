#!/usr/bin/env bash
# Proves scripts/package.sh on the cargo at hand. A throwaway workspace of
# three crates, with its own CARGO_HOME and target dir and no network:
# pkgtest-c calls `pkgtest_b::f`, which pkgtest-b re-exports from pkgtest-a.
# Each step changes the tree at one version and checks the verdict on
# pkgtest-c. Plain `cargo package --workspace` gets steps 2, 3 and 5 wrong,
# because it verifies against the siblings as they were at the first run
# (issue #3).
set -euo pipefail
package=$(cd "$(dirname "$0")" && pwd)/package.sh
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export CARGO_HOME=$tmp/home CARGO_TARGET_DIR=$tmp/target
export CARGO_NET_OFFLINE=true CARGO_TERM_COLOR=never
ws=$tmp/ws

mkdir -p "$ws"/{a,b,c}/src
cat >"$ws/Cargo.toml" <<'EOF'
[workspace]
resolver = "3"
members = ["a", "b", "c"]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "MIT"
description = "The fixture of scripts/package-test.sh"
EOF
for crate in a b c; do
    cat >"$ws/$crate/Cargo.toml" <<EOF
[package]
name = "pkgtest-$crate"
version.workspace = true
edition.workspace = true
license.workspace = true
description.workspace = true
EOF
done
printf '\n[dependencies]\npkgtest-a = { path = "../a", version = "0.1.0" }\n' >>"$ws/b/Cargo.toml"
printf '\n[dependencies]\npkgtest-b = { path = "../b", version = "0.1.0" }\n' >>"$ws/c/Cargo.toml"

lib() { printf '%s\n' "$2" >"$ws/$1/src/lib.rs"; }
lib a 'pub fn f() {}'
lib b 'pub use pkgtest_a::*;'
lib c 'pub fn g() { pkgtest_b::f() }'
(cd "$ws" && cargo generate-lockfile --quiet)

# check green|red LABEL [CHANGED]: package the workspace, and fail unless
# pkgtest-c was verified with the verdict wanted, a red one for want of
# `pkgtest_b::f`, and package.sh found exactly CHANGED changed. Outside a
# git checkout --allow-dirty changes nothing; it keeps a TMPDIR inside one
# from failing cargo's dirty check.
check() {
    local want=$1 label=$2 changed=${3-} got
    if (cd "$ws" && "$package" --allow-dirty) >"$tmp/log" 2>&1; then
        got=green
    elif grep -q 'error\[E0425\]' "$tmp/log"; then
        got=red
    else
        got='red, but not for want of pkgtest_b::f'
    fi
    grep -q 'Verifying pkgtest-c' "$tmp/log" || got="$got, without verifying pkgtest-c"
    if [ -n "$changed" ] && ! grep -q "changed since the last green verify: $changed\$" "$tmp/log"; then
        got="$got, without exactly $changed changed"
    fi
    if [ "$got" != "$want" ]; then
        cat "$tmp/log" >&2
        echo "package-test: $label: $got, wanted $want" >&2
        exit 1
    fi
    echo "package-test: $label: $got"
}

check green 'cold'
lib a ''
check red 'a drops f, which c still calls through b' pkgtest-a-0.1.0
check red 'the same tree again' pkgtest-a-0.1.0
lib a 'pub fn f() {}'
check green 'f back as it was at the last green run' pkgtest-a-0.1.0
lib a 'pub fn f(_: u8) {}'
lib c 'pub fn g() { pkgtest_b::f(1) }'
check green 'f takes an argument, and c passes one' 'pkgtest-a-0.1.0 pkgtest-c-0.1.0'
