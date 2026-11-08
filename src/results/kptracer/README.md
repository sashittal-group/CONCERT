# KPTracer parameter profiles

> **Workflow status:** KPTracer is outside the manuscript reproduction
> workflows. Its Snakemake rules have been removed from the root `Snakefile`;
> the Snakemake commands below are historical and no longer run as written.
> The analysis code remains available for separate use.

The workflow profiles one parameter at a time around the stem-free,
crown-conditioned KPTracer fit. It uses the known root state 0 and conditions
on both crown daughter lineages surviving. Rate-profile domains are bounded
relative to each state's fitted net growth rate; adjust
`rate_profile_growth_span` in `config.yaml` to widen or narrow those domains.
Birth and death figures use logarithmic y-axes to accommodate the fitted rates.

Run the profile preprocessing and plotting rules with Snakemake:

```bash
snakemake --cores 1 src/results/kptracer/figures/KPTracer_composite_sensitivity_compact.pdf
```

The workflow writes composite and individual birth, death, transition, and
sampling figures in PDF and SVG under `src/results/kptracer/figures/`, with
profile tables under `src/results/kptracer/data/KPTracer/`.
Transition probabilities are shown as a log-scaled matrix heatmap, and the
composite scaled-sampling panel omits L1. KPTracer plots label states directly
rather than repeating the dataset name.

## Source-specific transition times

The source-specific fit assigns one transition time to each KPTracer source state
with outgoing edges (T1, L2, L3, and S1). At each time, only that source state's
transition-probability row is active. The fit is seeded from the unconstrained
CONCERT N=11 result, fixes sampling probabilities at 0.05, and jointly refines
the source times, birth/death rates, and allowed transition probabilities. It
uses four cyclic assignments of seed times to compare starts, and does not
impose a minimum self-transition probability.

Run the inference with Snakemake:

```bash
snakemake --cores all --rerun-incomplete \
  data/KPTracer-Data/processed_data/inference_source_specific_transition_times/transition_count_4.out
```

The result and source-to-time mapping are written under
`data/KPTracer-Data/processed_data/inference_source_specific_transition_times/`.
