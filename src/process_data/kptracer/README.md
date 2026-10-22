# KPTracer preprocessing

Run the modular preprocessing pipeline from the repository root:

```bash
python -m src.process_data.kptracer.pipeline
```

The defaults use `data/laml/problin_1461.nwk` and
`data/laml/KPTracer_meta.csv`, together with the refined tree and inferred
location labels in `data/KPTracer-Data/processed_data/`. Override these paths
and the minimum-edge and tip-depth settings with the CLI options shown by
`--help`.

The pipeline writes `1461.tree` and `1461.labeling`, then normalizes and exports
`nodes_0.csv` and `edges_0.csv`. It does not add a synthetic stem node or edge;
the original Newick root is the tree root. If the legacy location-label file
contains a node-0 stem label, that entry is ignored. The pipeline also builds
`transitions_0.csv`, `transitions_0_adjusted.csv`, and `transition-times.txt`
from the refined edge intervals. `transition-times.txt` contains sorted,
unique event times; the adjusted table still retains a row for every
transition edge. Tree normalization keeps the notebook's one-based tree-node
IDs, unit minimum tip depth, and 0.001 minimum edge length.

To run the preprocessing through Snakemake:

```bash
snakemake --cores 1 data/KPTracer-Data/processed_data/transitions_0_adjusted.csv
```
