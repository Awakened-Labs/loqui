#!/usr/bin/env bash
# Packages every crate of the workspace in the current directory as crates.io
# would receive it, keeps each under crates.io's 10 MiB limit, and verifies
# each against its siblings as they are now. Arguments go to `cargo package`.
#
# `cargo package --workspace` alone verifies each crate against its siblings
# as they were the first time it packaged this version (issue #3). It serves
# them from a temporary registry and takes a registry package to be immutable
# by name and version, so it never refreshes
#   - the siblings it extracted to $CARGO_HOME/registry/src/-<hash>/, where
#     <hash> is a hash of the temporary registry's path, so never changes; or
#   - their builds, whose fingerprints leave out a registry package's sources.
# So package without verifying first. Then delete the extracted siblings,
# which cost little to extract again, and `cargo clean -p` every crate whose
# files changed since the last green verify. Cargo itself rebuilds whatever
# depends on a crate it rebuilt. scripts/package-test.sh proves all of this on
# the cargo at hand.
set -euo pipefail
shopt -s nullglob

meta=$(cargo metadata --format-version 1 --no-deps)
target=$(jq -r .target_directory <<<"$meta")
build=$(jq -r '.build_directory // .target_directory' <<<"$meta")
# "<crate>-<version> <hash>" for each crate, as of the last green verify.
stamp=$build/package-verified

# Start from nothing, so that target/package holds only this version's crates
# and the temporary registry no copy of an older one. Before 1.95, cargo
# overwrote a copy without truncating it (rust-lang/cargo#16683). The stale
# tail fails its checksum, and cargo then quietly asks crates.io instead,
# which may well have this version already.
rm -rf "$target/package" "$build/package"
cargo package --workspace --locked --no-verify "$@"
crates=("$target"/package/*.crate)
if [ ${#crates[@]} -eq 0 ]; then
    echo "package.sh: cargo packaged nothing into $target/package" >&2
    exit 1
fi
limit=$((10 * 1024 * 1024))
for crate in "${crates[@]}"; do
    size=$(stat -c %s "$crate")
    echo "$(basename "$crate"): $size bytes"
    if [ "$size" -ge "$limit" ]; then
        echo "package.sh: $crate is over crates.io's 10 MiB limit" >&2
        exit 1
    fi
done
packaged=$(sha256sum "${crates[@]}")

home=${CARGO_HOME:-$HOME/.cargo}
now=() kept=() changed=() clean=()
for crate in "${crates[@]}"; do
    id=$(basename "$crate" .crate)
    # <crate>-<version>, and a crate name has no dots.
    if ! [[ $id =~ ^([A-Za-z0-9_-]+)-[0-9]+\. ]]; then
        echo "package.sh: no crate name and version in $crate" >&2
        exit 1
    fi
    name=${BASH_REMATCH[1]}
    # The name and contents of every file but two, which change with every
    # commit and never change what compiles: .cargo_vcs_info.json names HEAD,
    # and Cargo.lock pins the checksums of the siblings' tarballs.
    sum=$(tar -xzf "$crate" --exclude="$id/.cargo_vcs_info.json" --exclude="$id/Cargo.lock" \
        --to-command='printf "%s " "$TAR_FILENAME"; sha256sum' | sha256sum | cut -c1-64)
    now+=("$id $sum")
    if grep -qxF "$id $sum" "$stamp" 2>/dev/null; then
        kept+=("$id $sum")
    else
        changed+=("$id")
        clean+=(-p "$name")
    fi
    rm -rf "$home"/registry/src/-*/"$id"
done

if [ ${#changed[@]} -gt 0 ]; then
    echo "package.sh: changed since the last green verify: ${changed[*]}"
    # Forget them before rebuilding anything, so that a red verify leaves
    # them to be rebuilt again next time rather than trusted.
    printf '%s\n' "${kept[@]}" >"$stamp"
    cargo clean --locked "${clean[@]}"
fi

cargo package --workspace --locked "$@"
if [ "$(sha256sum "${crates[@]}")" != "$packaged" ]; then
    echo "package.sh: the tree changed while it was being packaged; run again" >&2
    exit 1
fi
printf '%s\n' "${now[@]}" >"$stamp"
