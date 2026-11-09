# TLS parameter profiles for 100 transition times

To calculate likelihood profiles and regenerate boxplot figures for the
fixed-sampling TLS fit with 100 transition times, run from the repository root:

```bash
python -m src.analyses.tls.profile_transition_run \
  data/troupe/TLS/fixed_sampling_0.1/inference_transition_times_100.tmp.out
```

The workflow evaluates each parameter profile against the 273 retained TLS
subtrees in `data/troupe/TLS/subtrees_kept_states`, uses the same normalized
root likelihood as the inference run, and checks the fitted likelihood against
the checkpoint. It uses the configured likelihood fractions and profile grid
from `config_fixed_sampling_0.1.yaml`. It writes profile tables and PDF/SVG
figures under
`outputs/tls/fixed_sampling_0.1/transition_profiles/TLS/`. Sampling remains
fixed at 0.1.

The same command accepts the TLSC completed inference output:

```bash
python -m src.analyses.tls.profile_transition_run \
  data/troupe/TLSC/fixed_sampling_0.1/inference_transition_times_100.out
```

After both profile runs finish, render TLS and TLSC as adjacent boxes in the
same panels:

```bash
python -m src.analyses.tls.compare_transition_profiles \
  --tls-profiles outputs/tls/fixed_sampling_0.1/transition_profiles/TLS/data/TLS_transition_times_100 \
  --tlsc-profiles outputs/tls/fixed_sampling_0.1/transition_profiles/TLSC/data/TLSC_transition_times_100
```

The comparison PNGs are written to
`outputs/tls/fixed_sampling_0.1/profile_comparison/`.

## TLS/TLSC figure panels

Generate the supplied PDF (Panel A), convergence plots (Panel B), the combined
boxplot panel (Panel C), the separate TLS and TLSC condition subpanels, and the
composite:

```bash
python -m src.analyses.tls.figure_generator
```

The pipeline plots parameter convergence for TLS and TLSC using inference runs
from 5 through 100 transitions, and reads the saved profile tables for the
100-transition TLS and TLSC fits. It generates PNG, PDF, and SVG versions of
each condition subpanel, Panels A-C, and the composite.
Outputs are written to
`outputs/tls/figure_generator/`. The TLS result is marked as a
checkpoint if its final `.out` file is absent. Panel A uses
`figure_generator/tls_panelA.pdf`; its standalone PDF output is copied directly
from that source, and a high-resolution render is used in the composite. Panel B
is drawn in `figure_generator/panel_b.py` as three convergence plots (net
growth, scaled birth, and transition probability / Δt); color denotes state or
transition destination, and line style distinguishes TLS from TLSC. Panel C
combines the net growth, transition-probability ratio, and scaled-birth
boxplots previously shown as separate panels D-F. Each panel can be generated independently:

```bash
python -m src.analyses.tls.figure_generator.panel_a
python -m src.analyses.tls.figure_generator.panel_b
python -m src.analyses.tls.figure_generator.panel_c
```

Each panel script has a `FIGURE_SIZE` setting near the top for easy adjustment.
The composite places A, B, and the three-part Panel C in one row. Panel C is
exported as a compact standalone figure. Its plots are ordered as net growth,
scaled birth, then Somite/NeuralTube transition ratio. The first two are
grouped by cell type, with TLS and TLS$^{\mathrm{CL}}$ boxes using filled and
outlined styles, respectively. The transition-ratio plot uses
`p_{NMP,S} / p_{NMP,NT}` on the y-axis, and the scaled-birth plot uses
`β_s ρ`. Panel B uses `β_s ρ` and `p_{s,d} / Δt` as the second and third
y-axis labels. In the composite, Panel B net-growth and scaled-birth axes
share their heights and y-axis scales with the first and second Panel C plots,
respectively. The composite uses Arial throughout.

The panels use Arial, abbreviate the terminal states as NT and S, label nodes
with `β_s − δ_s` and `β_s ρ`, and show the fitted values for both node
quantities. Panel B shows `p_{NMP,target} / Δt` on differentiation edges plus
`p_{NMP,NMP} / Δt` on the progenitor self-loop. Node area increases with net
growth (with a minimum size retained for text legibility), and edge width and
arrowhead size increase with the displayed rate on a logarithmic scale so that
both the self-loop and differentiation edges remain visible. Parameter values
are printed inline with their formulas.

## State likelihoods for cut edge segments

To compute and store the likelihood and posterior probability of each state for
every edge segment in the TLS and TLSC subtrees, run:

```bash
python -m src.analyses.tls.process_state_coloring --dataset TLS
python -m src.analyses.tls.process_state_coloring --dataset TLSC
```

The processor uses the corresponding 100-transition-time fit, fixes the root
state to state 0 as in inference, and applies the same root-likelihood
normalization. It calculates marginal state probabilities by message passing
through the likelihood hierarchy, then records the most probable state for
each edge segment. The long-format CSV contains one row per segment/state,
including its log likelihood, posterior probability, and most-likely-state
label. Results are stored under
`outputs/tls/state_coloring/<dataset>/` with a `metadata.json` file.

To render the stored most-likely segment states as colored subtree plots, run:

```bash
python -m src.analyses.tls.plot_state_coloring --dataset TLS
python -m src.analyses.tls.plot_state_coloring --dataset TLSC
```

The PNGs are grouped by source tree under
`outputs/tls/state_coloring/<dataset>/plots/`. They use the same
rectangular, root-at-top ultrametric layout as `draw_ultrametric_tree`; each
edge segment is colored by its most likely state (NMPs gray, Somite pink,
NeuralTube green). The `plot_index.csv` file records the number of plotted
segments and states for each subtree, along with the mean posterior probability
of the selected state.

To plot the number of most-likely-state edge segments by time across all TLS
subtrees, and the corresponding TLSC plot, run:

```bash
python -m src.analyses.tls.plot_state_segment_counts
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
`outputs/tls/state_coloring/<dataset>/plots/` directory.

The same command also writes 100%-stacked state-proportion plots,
`state_segment_proportions_by_time_100.png`, and their normalized tables,
`state_segment_proportions_by_time_100.csv`. Each interval's three state
proportions sum to one.
