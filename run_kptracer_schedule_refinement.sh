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
source_directory="$output_root/inference_progressive_fixed_sampling_0.05_5_20_regenerated_grid_100"
refined_output="$output_root/inference_progressive_fixed_sampling_0.05_5_20_regenerated_refined_grid_100"
initial_times="$output_root/transition-times.txt"

for input_file in \
    "$output_root/nodes_0.csv" \
    "$output_root/edges_0.csv" \
    "$initial_times" \
    src/transition_edges_kptracer.txt; do
    if [[ ! -f "$input_file" ]]; then
        echo "Required input not found: $input_file" >&2
        exit 1
    fi
done

if [[ -e "$source_directory" || -e "$refined_output" ]]; then
    echo "Refusing to overwrite an existing output directory." >&2
    echo "Move or rename these directories before retrying:" >&2
    echo "  $source_directory" >&2
    echo "  $refined_output" >&2
    exit 1
fi

cache_root=/tmp/kptracer-schedule-refinement-regenerated-grid-100
export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
export TMPDIR="$cache_root/tmp"
export DYLD_FALLBACK_LIBRARY_PATH=/usr/lib
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR" "$source_directory" "$refined_output"

run_python_module() {
    local module_name=$1
    shift
    "$python_executable" -c '
import runpy
import sys
import torch._inductor.config as inductor_config

inductor_config.cpp_cache_precompile_headers = False
module_name = sys.argv[1]
sys.argv = [module_name, *sys.argv[2:]]
runpy.run_module(module_name, run_name="__main__")
' "$module_name" "$@"
}

run_python_module src.progressive_joint_transition_times \
    --tree-directory "$output_root" \
    --num-states 5 \
    --num-trees 1 \
    --transition-edges-file src/transition_edges_kptracer.txt \
    --num-transition-times 20 \
    --fixed-sampling-probability 0.05 \
    --initial-transition-times-file "$initial_times" \
    --initial-transition-count 5 \
    --max-time 1 \
    --grid-size 100 \
    --checkpoint-directory "$source_directory" \
    --output "$source_directory/final.out" \
    --transition-times-output "$source_directory/final.txt" \
    --log "$source_directory/final.log" \
    --device cpu \
    --optimizer adam \
    --normalize-root-likelihood \
    2>&1 | tee "$source_directory/stdout.log"

run_python_module src.refine_transition_time_schedules \
    --tree-directory "$output_root" \
    --num-states 5 \
    --num-trees 1 \
    --transition-edges-file src/transition_edges_kptracer.txt \
    --small-source-directory "$source_directory" \
    --nine-source-directory "$source_directory" \
    --large-source-directory "$source_directory" \
    --output-directory "$refined_output" \
    --fixed-sampling-probability 0.05 \
    --max-time 1 \
    --grid-step 0.0001 \
    --max-rounds 5 \
    --device cpu \
    --normalize-root-likelihood \
    2>&1 | tee "$refined_output/stdout.log"
