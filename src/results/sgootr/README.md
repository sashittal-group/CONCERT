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

## CRC11 manuscript figure

Generate the four-panel S10 CRC11 manuscript figure with:

```bash
.venv/bin/python -m src.results.sgootr.plot_state_specific_transitions \
  data/sgootr/all_trees/combined_inference/S10_CRC11_state_specific_3transition_shared_source_search/schedule_3857/inference.txt \
  --input-dir data/sgootr/all_trees/combined_inference/S10_CRC11_transition_chain_input
.venv/bin/python -m src.results.sgootr.figure_generator
```

By default, the figure uses the shared-source two-PT->LN fit
`S10_CRC11_state_specific_3transition_shared_source_search/schedule_3857`,
its parameter profile tables and colored tree, plus the PT->LN event-count
likelihood analysis. The generator writes panels A-D and a composite in PNG,
PDF, and SVG under `data/sgootr/figures/figure_generator/`. Panel labels use
bold lowercase parenthesized letters, and the plots use reduced heights and
tighter spacing; panel (b) is shorter still. The composite has three columns:
panels A and B are stacked in the left column, panel C occupies the middle
column, and panel D occupies the right column.

Panel A shows the transition graph without probability values. Panel B plots
the log-likelihood change for one through five PT->LN transition times relative
to the selected two-event fit, with a thicker, fully opaque curve and the
selected fit marked with a star. Panel C contains the
CRC11 state-colored tree, fitted event-time guides, and a compact NC/PT/LN
color map; its time axis is labeled `Time` with five major ticks. Panel D shows
the net-growth and relative-birth profile intervals side by side, with equally
wide filled boxes. Net growth is labeled by cell type; relative-birth ticks
show `β_PT / β_NC` and `β_LN / β_NC`. Plot text is 16 pt, with 18-pt x ticks
and 18-pt panel labels. The boxes show the likelihood fraction `exp(-0.25)`
and whiskers show `exp(-1)`. The older A-C panels and composite are preserved as
`panel_A_old`, `panel_B_old`, `panel_C_old`, and `figure_composite_old` in all
three formats. Profile intervals are also written to
`panel_D_profile_intervals.csv`.

The point estimates shown by the generator come from
`S10_CRC11_state_specific_3transition_shared_source_search/schedule_3857/inference.txt`.
Its `birth:` and `death:` vectors give net growth as birth minus death, while
the fitted birth rates give relative birth rates against NC. The profile tables
in the same schedule directory contain the likelihood profiles derived from
that fit. These are the project's SGOOTR inference results for patient CRC11,
S10 region; they are not parameter estimates copied from the SGOOTR publication.

## CRC11 genetic transition-schedule search

The runner searches schedules of three distinct transition times on the
0.001-spaced grid `i/1000`, for `i=0..999`. Times at or beyond the CRC11 root
cannot affect this tree's likelihood, so only grid values strictly before the
root are sampled (444 values for the current CRC11 tree). It starts with 500
random schedules, then carries the top 50 into each next cohort and generates
450 new schedules through tournament parent selection, per-position crossover,
and 15% per-time mutation. Each schedule gets the existing LBFGS fit, which
uses up to 20 iterations. The search stops after 10 cohorts at most, or after
three consecutive cohorts without a global log-likelihood improvement greater
than `1e-6`.

Run the search with:

```bash
.venv/bin/python -m src.results.sgootr.schedule_search \
  --input-dir data/sgootr/all_trees/combined_inference/S10_CRC11_transition_chain_input \
  --transition-edges-file data/sgootr/all_trees/combined_inference/S10_CRC11_transition_chain_input/transition_edges.txt \
  --output-dir data/sgootr/all_trees/combined_inference/S10_CRC11_genetic_schedule_search \
  --workers 10 --seed 42
```

To rule out a transition exactly at the sampling time, add
`--exclude-zero-time`; the smallest eligible transition time is then 0.001.
Use a separate output directory for this constrained run.

Every new schedule rebuilds the CRC11 tree cuts and optimizes the other
parameters in a separate `main2.py` process; fits within a cohort run in
parallel. The worker count is capped at the available CPU count. The
`schedule_search_summary.csv` ranks every unique evaluated schedule and records
failed fits; `cohort_summary.csv` records generation-level progress, and
`best_schedule.txt` contains the best successful schedule. Runs are
reproducible with `--seed`. The options `--cohorts`, `--cohort-size`, and
`--elite-count` can adjust the corresponding limits. Subsequent cohorts retain
the best schedules, use tournament-selected crossover, add random schedules
for 20% of new slots, and mutate individual transition times at a 15% rate.
Most mutations stay within 0.05 time units of the inherited time; one quarter
of mutations make a global jump across the eligible grid.

To generate parameter-profile figures for the best genetic-search fit, run:

```bash
.venv/bin/python -m src.results.sgootr.profile_combined_inference \
  data/sgootr/all_trees/combined_inference/S10_CRC11_genetic_schedule_search/schedule_4545/inference.txt \
  --input-dir data/sgootr/all_trees/combined_inference/S10_CRC11_transition_chain_input \
  --output-dir data/sgootr/all_trees/combined_inference/S10_CRC11_genetic_schedule_search/schedule_4545/tight_drop_analysis/profiles \
  --transition-times-only --likelihood-drops 0.1 0.25 0.5 1.0
```

The profile baseline is checked against the fitted likelihood. Profile tables
are written to `schedule_4545/tight_drop_analysis/profiles/data/`, and
parameter/transition and ratio figures to
`schedule_4545/tight_drop_analysis/profiles/figures/`. The numbered composite
growth/rate and relative-sampling interval figures are written to
`schedule_4545/tight_drop_analysis/figures/`. `--likelihood-drops` overrides
the configured profile thresholds for this run only; the ratio figures mark
the reference ratio of 1.

For inference run with `src.main2` using only global transition-time epochs,
add `--transition-times-only` to the profile command. This skips per-edge
transition assignments and uses the same unconditioned likelihood as `main2`.

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
