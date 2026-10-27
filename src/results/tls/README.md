# TLS parameter profiles for 100 transition times

To calculate likelihood profiles and regenerate boxplot figures for the
fixed-sampling TLS fit with 100 transition times, run from the repository root:

```bash
python -m src.results.tls.profile_transition_run \
  data/troupe/TLS/fixed_sampling_0.1/inference_transition_times_100.tmp.out
```

The workflow evaluates each parameter profile against the 273 retained TLS
subtrees in `data/troupe/TLS/subtrees_kept_states`, uses the same normalized
root likelihood as the inference run, and checks the fitted likelihood against
the checkpoint. It uses the configured likelihood fractions and profile grid
from `config_fixed_sampling_0.1.yaml`. It writes profile tables to
`data/troupe/TLS/fixed_sampling_0.1/profiles/data/TLS_transition_times_100/`
and PDF/SVG figures to the corresponding `figures/` directory. Sampling remains
fixed at 0.1.

The same command accepts the TLSC completed inference output:

```bash
python -m src.results.tls.profile_transition_run \
  data/troupe/TLSC/fixed_sampling_0.1/inference_transition_times_100.out
```

After both profile runs finish, render TLS and TLSC as adjacent boxes in the
same panels:

```bash
python -m src.results.tls.compare_transition_profiles \
  --tls-profiles data/troupe/TLS/fixed_sampling_0.1/profiles \
  --tlsc-profiles data/troupe/TLSC/fixed_sampling_0.1/profiles
```

The comparison PNGs are written to
`data/troupe/TLS/fixed_sampling_0.1/profiles/figures/`.

## TLS/TLSC figure panels

Generate the current transition diagram (Panel B), separate TLS and TLSC
condition subpanels, and the composite figure with the other panel slots left
blank:

```bash
python -m src.results.tls.figure_generator
```

The pipeline reads each condition's 100-transition fit, uses the median
spacing of `data/troupe/transition-times-100.txt` to calculate the effective
transition rates `p/Δt`, and generates PNG, PDF, and SVG versions of each
condition subpanel, Panel B, and the composite. Outputs are written to
`data/troupe/figures/figure_generator/`. The TLS result is marked as a
checkpoint if its final `.out` file is absent. The composite reserves a blank
Panel A slot for a future subpanel. The panels use Arial, abbreviate the
terminal states as NT and S, label nodes with `β_s − δ_s` and `β_s ρ`, and show
the fitted values for both node quantities. They show `p_{NMP,target} / Δt` on
differentiation edges plus `p_{NMP,NMP} / Δt` on the progenitor self-loop.
Node area increases with net growth (with a minimum size retained for text
legibility), and edge width and arrowhead size increase with the displayed
rate on a logarithmic scale so that both the self-loop and differentiation
edges remain visible. Parameter values are printed inline with their formulas.

## State likelihoods for cut edge segments

To compute and store the likelihood and posterior probability of each state for
every edge segment in the TLS and TLSC subtrees, run:

```bash
python -m src.results.tls.process_state_coloring --dataset TLS
python -m src.results.tls.process_state_coloring --dataset TLSC
```

The processor uses the corresponding 100-transition-time fit, fixes the root
state to state 0 as in inference, and applies the same root-likelihood
normalization. It calculates marginal state probabilities by message passing
through the likelihood hierarchy, then records the most probable state for
each edge segment. The long-format CSV contains one row per segment/state,
including its log likelihood, posterior probability, and most-likely-state
label. Results are stored under each dataset's
`fixed_sampling_0.1/state_coloring/` directory with a `metadata.json` file.

To render the stored most-likely segment states as colored subtree plots, run:

```bash
python -m src.results.tls.plot_state_coloring --dataset TLS
python -m src.results.tls.plot_state_coloring --dataset TLSC
```

The PNGs are grouped by source tree under each dataset's
`fixed_sampling_0.1/state_coloring/plots/` directory. They use the same
rectangular, root-at-top ultrametric layout as `draw_ultrametric_tree`; each
edge segment is colored by its most likely state (NMPs gray, Somite pink,
NeuralTube green). The `plot_index.csv` file records the number of plotted
segments and states for each subtree, along with the mean posterior probability
of the selected state.

To plot the number of most-likely-state edge segments by time across all TLS
subtrees, and the corresponding TLSC plot, run:

```bash
python -m src.results.tls.plot_state_segment_counts
```

Each plot uses 101 intervals bounded by that dataset's 100 fitted transition
times plus the time-range endpoints. The intervals are not all equal width:
their boundaries follow the inference model. Subtree times remain on the
absolute fitted scale: leaf endpoints are anchored at zero (floating-point
residuals within `1e-12` are clamped to zero), while each root time is
preserved. A segment is counted once in every interval it overlaps. The PNGs
and their count tables are written as
`state_segment_counts_by_time_100.png` and
`state_segment_counts_by_time_100.csv` in each dataset's
`fixed_sampling_0.1/state_coloring/plots/` directory.

The same command also writes 100%-stacked state-proportion plots,
`state_segment_proportions_by_time_100.png`, and their normalized tables,
`state_segment_proportions_by_time_100.csv`. Each interval's three state
proportions sum to one.
