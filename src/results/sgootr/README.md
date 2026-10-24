# SGOOTR parameter profiles

The SGOOTR profile workflow reuses the shared profile grid, likelihood-threshold
interval calculation, and TLS figure renderer. It profiles one parameter at a
time around each fit and uses the stem-free crown likelihood with root state 0
known, conditioning on both daughter lineages surviving.

Run either profile target, or use the default workflow to generate both:

```bash
snakemake --cores 1 src/results/sgootr/figures/SGOOTR_composite_sensitivity_compact.pdf
snakemake --cores 1
```

The workflow writes five matching figure types in PDF and SVG under
`src/results/sgootr/figures/` and profile tables under
`src/results/sgootr/data/`. The middle panel of the composite figure shows the
individual transition probabilities `NC->PT` and `PT->LN`.
Composite profile figures use a tighter layout with narrower panels and wider
boxes; state-specific profile boxes are colored consistently by cell type (NC,
PT, and LN).

To rerun inference with the sampling probability fixed to 0.1 for all four
configured tree variants and regenerate their profiles and figures, run:

```bash
snakemake --cores 1 sgootr_fixed_sampling_01
```

These results are isolated under
`src/results/sgootr/fixed_sampling_0.1/`; the existing shared-sampling
inference results and figures are left unchanged.

## Combined S10/S15 profiles

For a combined cohort fit, pass its inference result to the plotting and
profile commands. The cohort label and sampling mode are included in profile
figure names. To run the six-tree combined S10 and S15 fits with state-specific
sampling probabilities inferred, use a separate output folder for each cohort:

```bash
for cohort in S10 S15; do
  result="data/sgootr/all_trees/combined_inference/${cohort}_upgma_min_subtracted_inferred_sampling/inference_inferred_sampling.txt"
  .venv/bin/python -m src.main2_given_transitions \
    "data/sgootr/all_trees/combined_inference/${cohort}_upgma_min_subtracted_inferred_sampling" \
    --num-states 3 --num-trees 6 \
    --transition-edges-file "data/sgootr/all_trees/combined_inference/${cohort}_upgma_min_subtracted_inferred_sampling/transition_edges.txt" \
    --transition-times-file "data/sgootr/all_trees/combined_inference/${cohort}_upgma_min_subtracted_inferred_sampling/transition_times.txt" \
    --per-edge-transition-times-pattern "data/sgootr/all_trees/combined_inference/${cohort}_upgma_min_subtracted_inferred_sampling/transition_times_{index}.csv" \
    --output "$result" \
    --log "data/sgootr/all_trees/combined_inference/${cohort}_upgma_min_subtracted_inferred_sampling/optimizer_inferred_sampling.log" \
    --device cpu --optimizer lbfgs --no-stem
  .venv/bin/python -m src.results.sgootr.plot_combined_inference "$result"
  .venv/bin/python -m src.results.sgootr.profile_combined_inference "$result"
done
```

Sampling is inferred independently for NC, PT, and LN because neither a fixed
sampling probability nor a shared-sampling option is supplied. The agreed
transition graph (NC->PT, PT->LN, LN->PT, NC->LN), per-edge histories, common
epochs, and six CRCs match the fixed-sampling fits.

To infer one sampling probability independently for each tree, shared across
NC, PT, and LN, add `--per-tree-sampling-rates` to the runner command. This flag
is incompatible with `--shared-sampling-rate` and
`--fixed-shared-sampling-probability`. Result files retain the `sampling:`
statewise mean for compatibility and include one value per tree under
`sampling_by_tree:`; the point-estimate plotter adds a per-tree heatmap with a
single shared-state column. The combined profile command creates one sampling
profile per tree, changing that tree's probability across all states together.

To infer separate NC, PT, and LN sampling probabilities within every tree,
use `--per-tree-state-sampling-rates` instead. It cannot be combined with
`--per-tree-sampling-rates`, `--shared-sampling-rate`, or
`--fixed-shared-sampling-probability`. These runs write a `sampling_by_tree:`
matrix and should use their own output directory; the plotter displays a
tree-by-cell-type heatmap and the profile workflow profiles each tree/state
pair independently.

To use each tree's own observed transition-event times as fixed epochs and
learn a separate transition matrix for each tree, add
`--per-tree-transition-rates` and keep the per-edge history pattern enabled.
Supply `--per-tree-transition-edges-file` to use the CRC-specific graphs in
[per_tree_transition_edges_S10.txt](./per_tree_transition_edges_S10.txt) and
[per_tree_transition_edges_S15.txt](./per_tree_transition_edges_S15.txt); each tree's
row-normalized probabilities are constant across that tree's epochs. The union
of those graphs is the shared `--transition-edges-file`; the per-tree graph
file determines which edges are available to each tree. The configured graphs
are CRC04 (NC->PT, PT->LN), CRC10 (NC->PT, PT->LN, LN->PT), CRC11 (NC->PT,
PT->LN), CRC12 (NC->PT, PT->LN), CRC13 (NC->LN, PT->LN, LN->PT), and CRC14
(NC->PT, PT->LN, LN->PT). This mode can be combined with
`--per-tree-state-sampling-rates` and requires a separate output folder:

```bash
for cohort in S10 S15; do
  input="data/sgootr/all_trees/combined_inference/${cohort}_upgma_min_subtracted_per_tree_sampling"
  output="data/sgootr/all_trees/combined_inference/${cohort}_upgma_min_subtracted_per_tree_transition_rates"
  mkdir -p "$output"
  cp "$input/trees.txt" "$output/trees.txt"
  .venv/bin/python -m src.main2_given_transitions "$input" \
    --num-states 3 --num-trees 6 \
    --transition-edges-file "$input/transition_edges.txt" \
    --transition-times-file "$input/transition_times.txt" \
    --per-edge-transition-times-pattern "$input/transition_times_{index}.csv" \
    --per-tree-transition-edges-file "src/results/sgootr/per_tree_transition_edges_${cohort}.txt" \
    --per-tree-state-sampling-rates --per-tree-transition-rates \
    --output "$output/inference.txt" --log "$output/optimizer.log" \
    --device cpu --optimizer lbfgs --no-stem
  .venv/bin/python -m src.results.sgootr.plot_combined_inference \
    "$output/inference.txt"
  .venv/bin/python -m src.results.sgootr.profile_combined_inference \
    "$output/inference.txt" --input-dir "$input"
done
```
