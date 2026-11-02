#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

python_executable="${PYTHON:-$repo_root/.venv/bin/python}"
if ! command -v "$python_executable" >/dev/null 2>&1; then
    echo "Python executable not found: $python_executable" >&2
    echo "Set PYTHON to the project's Python interpreter and retry." >&2
    exit 1
fi

output_root=data/KPTracer-Data/processed_data
transition_edges=src/transition_edges_kptracer.txt
initial_times="$output_root/transition-times.txt"

for input_file in \
    "$output_root/nodes_0.csv" \
    "$output_root/edges_0.csv" \
    "$initial_times" \
    "$transition_edges"; do
    if [[ ! -f "$input_file" ]]; then
        echo "Required input not found: $input_file" >&2
        echo "Run the KPTracer preprocessing pipeline first." >&2
        exit 1
    fi
done

cache_root=$(mktemp -d)
trap 'rm -rf "$cache_root"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
export TMPDIR="$cache_root/tmp"
export DYLD_FALLBACK_LIBRARY_PATH=/usr/lib
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"

initial_result=""
for grid_size in 100 200 500 1000; do
    stage="inference_progressive_fixed_sampling_0.05_20_grid_${grid_size}"
    checkpoint_directory="$output_root/$stage"

    args=(
        -m src.progressive_joint_transition_times
        --tree-directory "$output_root" \
        --num-states 5 \
        --num-trees 1 \
        --transition-edges-file "$transition_edges" \
        --num-transition-times 20 \
        --fixed-sampling-probability 0.05 \
        --initial-transition-times-file "$initial_times" \
        --initial-transition-count 10 \
        --checkpoint-directory "$checkpoint_directory" \
        --max-time 1 \
        --grid-size "$grid_size" \
        --output "$output_root/$stage.out" \
        --transition-times-output "$output_root/$stage.txt" \
        --log "$output_root/$stage.log" \
        --device cpu \
        --optimizer adam \
        --normalize-root-likelihood
    )
    if [[ -n "$initial_result" ]]; then
        args+=(--initial-result-file "$initial_result")
    fi

    "$python_executable" "${args[@]}" \
        2>&1 | tee "$output_root/$stage.stdout.log"

    initial_times="$checkpoint_directory/transition_count_10_times.txt"
    initial_result="$checkpoint_directory/transition_count_10.out"
done
