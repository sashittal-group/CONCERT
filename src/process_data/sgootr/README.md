# SGOOTR tree processing

Run the modular pipeline from the repository root:

```bash
python -m src.process_data.sgootr.pipeline
```

The input defaults to `data/sgootr/CRC11.nwk`; LSD2 and MACH2 are resolved from
`PATH` or the project `.venv`. Override paths and processing parameters with
the CLI options shown by `--help`.

To construct an ultrametric tree by UPGMA from the already exported input tree,
run the dedicated preprocessor:

```bash
snakemake --cores 1 data/sgootr/processed/upgma/tree.nwk
```

It calculates the pairwise patristic distances between input tips, writes
`distance_matrix.csv`, and exports the UPGMA tree with the standard nodes,
edges, transition-time files, and plot under `data/sgootr/processed/upgma/`.
This is an additional stage; the existing pipeline's LSD2 dating stage remains
unchanged.

For an additional UPGMA result after subtracting the smallest off-diagonal
distance from every pair (while keeping the diagonal at zero), run:

```bash
snakemake --cores 1 \
  data/sgootr/processed/upgma_min_subtracted/tree.nwk
```

The adjusted matrix and the same tree, node, edge, transition-time, and plot
outputs are written under `data/sgootr/processed/upgma_min_subtracted/`.
To run inference on that adjusted UPGMA tree, target the inference output:

```bash
snakemake --cores 1 \
  data/sgootr/processed/upgma_min_subtracted/inference_result.txt
```

To scale the adjusted UPGMA tree and all corresponding times to total height
1, while keeping the unscaled result intact, run:

```bash
snakemake --cores 1 \
  data/sgootr/processed/upgma_min_subtracted_height_1/inference_result.txt
```

The scaled tree, distance matrix, node ages, transition intervals, and
transition times are written under
`data/sgootr/processed/upgma_min_subtracted_height_1/` before inference.
To also keep the tree above the latest transition fixed and rebuild each
descendant subtree independently with UPGMA, target:

```bash
snakemake --cores 1 \
  data/sgootr/processed/upgma_min_subtracted_height_1_transition_cut_upgma/inference_result.txt
```

The cutoff is the most recent transition (the smallest `T` value). For each
branch crossing that time, the pairwise tip distances below the fixed region
are recorded, their minimum is subtracted, and UPGMA is run separately for
that descendant subtree. Nonzero-height UPGMA trees are scaled to the cutoff
so all leaves retain the same overall height. Degenerate zero-height
subtrees receive equal terminal extensions to preserve ultrametricity.
The output directory includes the transformed tree, inference inputs, and
`subtree_pairwise_distances.csv` / `subtree_summary.csv` diagnostics.

To partition this tree by its complete per-edge transition history and fit
state-specific birth, death, and sampling parameters, run:

```bash
snakemake --cores 1 \
  data/sgootr/processed/upgma_min_subtracted_height_1_transition_cut_upgma/single_state_inference.json
```

The output JSON records state-specific birth counts, leaf counts, lineage
exposures, fitted rates, and boundary warnings. The accompanying
`single_state_segments.csv` reports the state and duration of each branch
segment. Sampling probabilities are estimated separately by state. Rate
optimization is bounded at birth 100 and death 50; check `boundary_warnings`
before interpreting estimates, since a boundary hit indicates that the fit is
bound-sensitive. A zero-length synthetic stem is added to this crown tree
solely for the root normalization used by the single-state likelihood.

To include this inference in the SGOOTR parameter-profile plots alongside the
existing datasets, preprocess its profile data and render the figures:

```bash
snakemake --cores 1 \
  src/results/sgootr/data/upgma_min_subtracted_height_1/parameter_value_ranges.csv \
  src/results/sgootr/data/upgma_min_subtracted_height_1_transition_cut_upgma/parameter_value_ranges.csv
.venv/bin/python -m src.results.sgootr.plot \
  --config src/results/sgootr/config.yaml
```

This writes profile tables for both normalized trees and regenerates the
composite and per-parameter figures in both configured formats.

To compare against inference with every event time shifted earlier by
`1e-4`, target:

```bash
snakemake --cores 1 \
  data/sgootr/processed/upgma_min_subtracted_eps_1e-4/inference_result.txt
```

This keeps the original run intact and writes shifted transition-time inputs,
the inference result, and logs in the `upgma_min_subtracted_eps_1e-4/` folder.

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
