# KPTracer parameter profiles

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
