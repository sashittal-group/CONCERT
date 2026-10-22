# SGOOTR tree processing

Run the modular pipeline from the repository root:

```bash
python -m src.process_data.sgootr.pipeline
```

The input defaults to `data/sgootr/CRC11.nwk`; LSD2 and MACH2 are resolved from
`PATH` or the project `.venv`. Override paths and processing parameters with
the CLI options shown by `--help`.

The pipeline writes four stages under `data/sgootr/processed/`: `input`,
`ultrametric`, `birth_shifted_min_branch`, and `transition_anchored`. Each
stage contains `tree.nwk`, `nodes.csv`, `edges.csv`, `transition_times.csv`,
`transition_times.txt`, and `tree.png`. The input stage has an empty transition
file because transitions are inferred after dating.

Transition times exported in `transition_times.csv` and `transition_times.txt`
use the same age direction as the node `T` values: `root_age - time + 1e-6 -
1e-4`. Transition intervals in the CSV use `root_age - time + 1e-6` without
the event-time decrement. Internal processing continues to use root-to-tip
depth coordinates.

No synthetic stem node or stem edge is added. The node and edge CSVs contain
only nodes and branches present in the stage's Newick tree.

To infer parameters from either processed tree, run the corresponding target:

```bash
snakemake --cores 1 \
  data/sgootr/processed/transition_anchored/inference_result.txt
snakemake --cores 1 \
  data/sgootr/processed/birth_shifted_min_branch/inference_result.txt
```

This uses the stem-free crown likelihood with the known root state fixed to
state 0, conditioning on both daughter lineages surviving.
