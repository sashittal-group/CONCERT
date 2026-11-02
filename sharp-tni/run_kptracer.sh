#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
processed_dir="$repo_root/data/KPTracer-Data/processed_data"
sharp_tni_build="$script_dir/.build/SharpTNI"
run_stamp=$(date '+%Y%m%d_%H%M%S')
output_dir="${SHARPTNI_OUTPUT_DIR:-$processed_dir/sharptni/run_$run_stamp}"

for executable in sankoff gamma; do
    if [[ ! -x "$sharp_tni_build/$executable" ]]; then
        echo "SharpTNI executable not found: $sharp_tni_build/$executable" >&2
        echo "Run $script_dir/setup.sh first." >&2
        exit 1
    fi
done
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to prepare the KPTracer inputs." >&2
    exit 1
fi
if ! command -v dot >/dev/null 2>&1; then
    echo "Graphviz 'dot' is required; run $script_dir/setup.sh first." >&2
    exit 1
fi
if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite an existing output directory: $output_dir" >&2
    echo "Set SHARPTNI_OUTPUT_DIR to a new path and retry." >&2
    exit 1
fi

python3 "$script_dir/prepare_kptracer.py" \
    --processed-dir "$processed_dir" \
    --output-dir "$output_dir"

"$sharp_tni_build/sankoff" \
    "$output_dir/hosts.txt" \
    "$output_dir/ptree.txt" \
    "$output_dir/solution.ptree" \
    > "$output_dir/run.log" 2>&1
cat "$output_dir/run.log"

"$sharp_tni_build/gamma" \
    "$output_dir/hosts.txt" \
    "$output_dir/solution.ptree" \
    > "$output_dir/gamma.log" \
    2> "$output_dir/transmission_network.dot"
cat "$output_dir/gamma.log"

dot -Tpdf "$output_dir/transmission_network.dot" \
    -o "$output_dir/transmission_network.pdf"
printf '\nResults written to %s\n' "$output_dir"
