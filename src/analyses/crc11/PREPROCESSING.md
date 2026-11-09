# SGOOTR tree processing

> **Workflow status:** The standard UPGMA target
> `data/sgootr/processed/upgma/tree.nwk` remains in the root `Snakefile`.
> Snakemake commands below for alternative preprocessing and inference targets
> refer to auxiliary rules that have been removed; their Python entry points
> remain in the repository for those separate analyses.

Run the modular pipeline from the repository root:

```bash
python -m src.analyses.crc11.preprocessing.pipeline
```

The input defaults to `data/sgootr/CRC11.nwk`; LSD2 and MACH2 are resolved from
`PATH` or the project `.venv`. Override paths and processing parameters with
the CLI options shown by `--help`.

Terminal labels map to states as follows: `gray=NC`, `blue=PT`, `orange=LN`,
`red=ML`, `purple=MO`, and `green=MP`. The original CRC11 colors retain their
existing state IDs; the additional states support the trees in
`data/sgootr/all_trees/`.

To run the four tree-processing stages for every available SGOOTR all-tree
input, run from the repository root:

```bash
for tree in data/sgootr/all_trees/S*/CRC*/tree.nwk; do
  relative_tree="${tree#data/sgootr/all_trees/}"
  output_root="data/sgootr/all_trees/processed/${relative_tree%/tree.nwk}"
  .venv/bin/python -m src.analyses.crc11.preprocessing.pipeline \
    --input-tree "$tree" --output-root "$output_root" \
    --birth-minimum-branch-length 0.01 \
    --post-transition-terminal-floor 0.01
done
```

Each tree is processed independently under its `S10` or `S15` patient folder.
The four stages are `input`, `ultrametric`, `birth_shifted_min_branch`, and
`transition_anchored`. The batch uses 0.01 for both branch-length floors
because the CRC11 default of 0.025 cannot preserve tip dates for every tree.

To run the minimum-distance-subtracted UPGMA stage for every processed input
tree:

```bash
for tree in data/sgootr/all_trees/S*/CRC*/tree.nwk; do
  relative_tree="${tree#data/sgootr/all_trees/}"
  dataset_root="data/sgootr/all_trees/processed/${relative_tree%/tree.nwk}"
  .venv/bin/python -m src.concert --analysis crc11-upgma \
    --input-tree "$dataset_root/input/tree.nwk" \
    --output-dir "$dataset_root/upgma_min_subtracted" \
    --subtract-minimum-distance
done
```

This writes the UPGMA tree, adjusted distance matrix, node/edge tables,
transition-time files, and plot under each tree's
`upgma_min_subtracted/` folder.

To plot the combined fixed-sampling inference with TLS-style parameter panels
and a transition-probability heatmap:

```bash
.venv/bin/python -m src.analyses.crc11.plot_combined_inference \
  data/sgootr/all_trees/combined_inference/S10_upgma_min_subtracted/inference_fixed_sampling_0.1.txt
```

This writes composite and individual parameter figures in PDF and SVG under
the inference result's `figures/` folder.

To calculate fixed-sampling profile likelihoods for the combined inference
and render TLS-style interval plots:

```bash
.venv/bin/python -m src.analyses.crc11.profile_combined_inference \
  data/sgootr/all_trees/combined_inference/S10_upgma_min_subtracted/inference_fixed_sampling_0.1.txt
```

The profile grid and likelihood thresholds follow `config.yaml`. Sampling
remains fixed at 0.1; its plot shows a point rather than a profiled interval.
Profile tables are written to `profiles/data/` and figures to `profiles/figures/`.

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
  outputs/sgootr/profiles/upgma_min_subtracted_height_1/parameter_value_ranges.csv \
  outputs/sgootr/profiles/upgma_min_subtracted_height_1_transition_cut_upgma/parameter_value_ranges.csv
.venv/bin/python -m src.analyses.crc11.plot \
  --config src/analyses/crc11/config.yaml
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
