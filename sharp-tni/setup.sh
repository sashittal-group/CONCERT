#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
deps_dir="$script_dir/.deps"
build_dir="$script_dir/.build"
lemon_source="$deps_dir/lemon-1.3.1"
sharptni_source="$deps_dir/SharpTNI"
lemon_prefix="$script_dir/.local/lemon"

sharptni_revision=8b0a1554bb1f3fe38a72e19d792157e35c492059
lemon_revision=032f077a63e493ab5a8b616cafcd4f81cfaa5356

for tool in git cmake patch c++; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required tool not found: $tool" >&2
        exit 1
    fi
done

if command -v brew >/dev/null 2>&1; then
    missing_packages=()
    for package in cmake boost graphviz; do
        if ! brew list --versions "$package" >/dev/null 2>&1; then
            missing_packages+=("$package")
        fi
    done
    if ((${#missing_packages[@]})); then
        brew install "${missing_packages[@]}"
    fi
fi

clone_revision() {
    local url=$1
    local destination=$2
    local revision=$3

    if [[ -d "$destination/.git" ]]; then
        git -C "$destination" fetch --depth 1 origin "$revision"
    elif [[ -e "$destination" ]]; then
        echo "Refusing to use a non-git dependency directory: $destination" >&2
        exit 1
    else
        mkdir -p "$(dirname "$destination")"
        git clone --no-checkout "$url" "$destination"
        git -C "$destination" fetch --depth 1 origin "$revision"
    fi
    git -C "$destination" checkout --detach "$revision"
}

mkdir -p "$deps_dir" "$build_dir" "$(dirname "$lemon_prefix")"
clone_revision \
    https://github.com/Skantz/lemon-1.3.1.git \
    "$lemon_source" \
    "$lemon_revision"
clone_revision \
    https://github.com/elkebir-group/SharpTNI.git \
    "$sharptni_source" \
    "$sharptni_revision"

lemon_cmake="$lemon_source/CMakeLists.txt"
if ! grep -Fxq 'CMAKE_MINIMUM_REQUIRED(VERSION 3.5)' "$lemon_cmake"; then
    patch -d "$lemon_source" -p0 <<'PATCH'
--- CMakeLists.txt
+++ CMakeLists.txt
@@ -1,5 +1,3 @@
-CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
-
-CMAKE_POLICY(SET CMP0048 OLD)
+CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PATCH
fi

cmake_prefix_args=()
if command -v brew >/dev/null 2>&1; then
    cmake_prefix_args+=("-DCMAKE_PREFIX_PATH=$(brew --prefix)")
fi

cmake -S "$lemon_source" -B "$build_dir/lemon" \
    -DCMAKE_INSTALL_PREFIX="$lemon_prefix" \
    -DCMAKE_CXX_STANDARD=11 \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DLEMON_ENABLE_GLPK=NO \
    -DLEMON_ENABLE_COIN=NO \
    -DLEMON_ENABLE_ILOG=NO
cmake --build "$build_dir/lemon" --parallel
cmake --install "$build_dir/lemon"

cmake -S "$sharptni_source" -B "$build_dir/SharpTNI" \
    "${cmake_prefix_args[@]}" \
    -DLIBLEMON_ROOT="$lemon_prefix" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$build_dir/SharpTNI" --parallel

printf '\nSharpTNI is ready:\n'
printf '  %s\n' "$build_dir/SharpTNI/sankoff"
printf '  %s\n' "$build_dir/SharpTNI/gamma"
printf '  %s\n' "$build_dir/SharpTNI/sample_sankoff"
