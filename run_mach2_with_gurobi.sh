#!/bin/sh

set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
license_file="$HOME/gurobi.lic"
project_license="$project_root/gurobi.lic"
python="$project_root/.venv/bin/python"
mach2="$project_root/.venv/bin/mach2"

if [ ! -f "$license_file" ]; then
    if [ -f "$project_license" ]; then
        chmod 600 "$project_license"
        mv "$project_license" "$license_file"
    else
        printf '%s\n' "No Gurobi license found at $license_file or $project_license." >&2
        printf '%s\n' "Install your WLS credentials at $license_file and rerun." >&2
        exit 1
    fi
else
    if [ -f "$project_license" ]; then
        printf '%s\n' "A second license file exists in the project root: $project_license" >&2
        printf '%s\n' "Remove or securely relocate it before continuing." >&2
        exit 1
    fi
    chmod 600 "$license_file"
fi

if [ ! -x "$python" ] || [ ! -x "$mach2" ]; then
    printf '%s\n' "The workspace .venv is missing Python or mach2." >&2
    exit 1
fi

printf '%s\n' "Checking Gurobi license..."
"$python" -c 'import gurobipy as gp; model = gp.Model(); variable = model.addVar(vtype=gp.GRB.BINARY); model.setObjective(variable); model.optimize(); status = model.Status; model.dispose(); assert status == gp.GRB.OPTIMAL, f"Gurobi test solve failed with status {status}"; print("Gurobi license check passed.")'

printf '%s\n' "Running mach2..."
cd "$project_root"
exec "$mach2" \
    data/KPTracer-Data/processed_data/1461.tree \
    data/KPTracer-Data/processed_data/1461.labeling \
    -p T1 \
    -o data/KPTracer-Data/processed_data