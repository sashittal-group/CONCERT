# KPTracer analysis

The KPTracer workflow calculates crown-conditioned parameter profiles and
generates the sensitivity figures from the processed trees and fitted
parameters. Run both commands from the repository root:

```bash
python -m src.analyses.kptracer.preprocess \
  --config src/analyses/kptracer/config.yaml

python -m src.analyses.kptracer.plot \
  --config src/analyses/kptracer/config.yaml
```

The configured input root is `data/KPTracer-Data/processed_data`. It must
contain the processed tree files and the `inference_per_edge.out` fit consumed
by the profile workflow. Profile tables are written under
`outputs/kptracer/profiles/KPTracer/`; figures are written under
`outputs/kptracer/figures/`.

The profile calculation fixes the known root state to 0 and conditions on both
crown daughter lineages surviving. Rate-profile domains are bounded relative
to each state's fitted net growth rate; `rate_profile_growth_span` in the
configuration controls their width. Birth and death figures use logarithmic
y-axes. The composite transition panel is a log-scaled heatmap, and its
scaled-sampling panel omits L1.

## Source-specific transition times

The optional source-specific inference runner is
`src.analyses.kptracer.infer_source_specific_transition_times`. It requires an
input tree directory, a compatible seed fit, the transition-edge list, and
separate output paths for the fit and source-time mapping. Full fits and logs
belong under `data/`; compact summaries and figures belong under `outputs/`.
See the command-line options with:

```bash
python -m src.analyses.kptracer.infer_source_specific_transition_times --help
```
