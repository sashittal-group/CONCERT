# SharpTNI on KPTracer

From the repository root, build SharpTNI and its LEMON dependency:

```bash
./sharp-tni/setup.sh
```

Then run the KPTracer analysis:

```bash
./sharp-tni/run_kptracer.sh
```

The runner expects the normalized KPTracer inputs under
`data/KPTracer-Data/processed_data/`. If they are missing, first run the
preprocessing pipeline described in
[`src/process_data/kptracer/README.md`](../src/process_data/kptracer/README.md).

Each run writes a new timestamped directory under
`data/KPTracer-Data/processed_data/sharptni/`. Set `SHARPTNI_OUTPUT_DIR` to
choose a different output path. The directory contains the generated host and
tree inputs, optimal labeling, logs, and a DOT/PDF transmission network.

The run uses T1, L1, L2, L3, and S1 as the five host states and assigns each
the full normalized interval `[0,1]`. KPTracer does not provide per-location
entry/removal intervals in these inputs, so interpret the result as exploratory
state-transition analysis, not a validated host-to-host infection estimate.

`setup.sh` installs missing Homebrew packages on macOS and builds pinned
SharpTNI and LEMON source revisions locally under this directory. Generated
dependencies and build files are git-ignored.

To group the state-changing edge intervals in enumerated Sankoff labelings
across different transition directions, run:

```bash
python sharp-tni/analyze_sankoff_solutions.py \
  --solutions sharp-tni/results/kptracer_sharptni_20261101_1118/enumerated_solutions/sankoff_solutions_sorted_by_cotransmission.ptree.gz \
  --output-dir sharp-tni/results/kptracer_sharptni_20261101_1118/enumerated_solutions \
  --gamma 21 22 \
  --overwrite
```

The script uses the midpoint of each group's positive-width common overlap
interval and writes both SharpTNI and CARTA time coordinates. It expects
enumerated solution blocks sorted by nondecreasing gamma; the `--node-count`
option overrides the default KPTracer value of 2,921.
