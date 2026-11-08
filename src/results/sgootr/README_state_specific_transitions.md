# State-specific CRC11 transitions

The separate `src.main2_state_specific_transitions` runner accepts named
NC→PT and PT→LN event times, sorts the epochs while keeping each time attached
to its event, and permits only the assigned state change at each epoch. It fits
one transition probability per source state. If multiple epochs use the same
source state, they share that state's transition row and probability. Every
other state's transition row is fixed to an identity row at that epoch.

The two times are provided explicitly and must be distinct. They remain
associated with their named edge even though the likelihood internally sorts
the epoch boundaries. For example:

```bash
.venv/bin/python -m src.main2_state_specific_transitions \
  data/sgootr/all_trees/combined_inference/S10_CRC11_transition_chain_input \
  --nc-to-pt-time 0.354 \
  --pt-to-ln-time 0.131 \
  --output data/sgootr/all_trees/combined_inference/S10_CRC11_state_specific_fit/inference.txt \
  --log data/sgootr/all_trees/combined_inference/S10_CRC11_state_specific_fit/optimizer.log \
  --device cpu
```

The output uses a distinct `transition_matrices_by_epoch` result section.
Existing inference and plotting code continues to use its original shared
transition matrix behavior and does not parse this new result format.

To search S15 schedules with one NC→PT epoch and two PT→LN epochs, run:

```bash
.venv/bin/python -m src.results.sgootr.state_specific_schedule_search \
  --input-dir data/sgootr/all_trees/combined_inference/S15_CRC11_transition_chain_input \
  --output-dir data/sgootr/all_trees/combined_inference/S15_CRC11_state_specific_3transition_shared_source_search \
  --summary-dir outputs/sgootr/state_specific_schedule_search/S15_CRC11_state_specific_3transition_shared_source_search \
  --workers 4
```

The schedule search randomly initializes candidate schedules, then applies the
existing genetic search on the 0.001 time grid. At each schedule, state rates
and one transition probability per source state are fit by LBFGS. For the
one-NC→PT/two-PT→LN model, the two PT→LN epochs share the same PT transition
row and probability. Event times are ordered from recent to old as PT→LN,
PT→LN, NC→PT; only state changes matching the current epoch are allowed.
This is a grid-based schedule search, not continuous gradient optimization of
event times. Use the corresponding S10 input and output directory names to run
the same search on S10.

To compare likelihood across one through five PT→LN times when no existing
shared-source two-time result is available, run:

```bash
.venv/bin/python -m src.results.sgootr.transition_count_analysis \
  --input-dir data/sgootr/all_trees/combined_inference/S10_CRC04_transition_chain_input \
  --output-dir data/sgootr/all_trees/combined_inference/S10_CRC04_ptln_transition_count_analysis \
  --summary-dir outputs/sgootr/transition_count_analysis/S10_CRC04_ptln_transition_count_analysis
```

The analysis keeps one NC→PT event and searches its time along with the PT→LN
event times. It searches counts 1 through 5, fitting all model parameters for
each candidate schedule. If `--existing-two-count-inference` is supplied, the
two-time result is reused instead of searching count 2. It writes likelihood
and schedule summaries plus PNG/PDF likelihood plots under the summary
directory; each newly searched count saves its full best inference result
under `data/`. The input directory name sets the cohort/CRC label in the plot.

Full schedule fits remain under `data/`; ranked schedules and cohort summaries
are written under `outputs/sgootr/state_specific_schedule_search/`. The same
command supports other single-tree S10 or S15 CRC input directories.

To profile the likelihood by moving each event time individually while holding
the other event times and every fitted rate, sampling probability, and transition
probability fixed, run:

```bash
.venv/bin/python -m src.results.sgootr.transition_time_profile \
  --input-dir data/sgootr/all_trees/combined_inference/S10_CRC11_transition_chain_input \
  --inference data/sgootr/all_trees/combined_inference/S10_CRC11_state_specific_3transition_shared_source_search/schedule_3857/inference.txt \
  --output-dir outputs/sgootr/CRC11/fixed_parameter_time_profile
```

The script checks that the fixed-parameter likelihood at the saved event times
reproduces the inference result before writing the CSV and PNG/PDF profiles.
Each event is swept across the 0.001 time grid subject to preserving the order
of all event times and keeping them before the tree root time. A zero-probability
likelihood is retained as negative infinity in the profile.

Generate the matching CRC11 plots for either state-specific fit with:

```bash
.venv/bin/python -m src.results.sgootr.plot_state_specific_transitions \
  data/sgootr/all_trees/combined_inference/S10_CRC11_state_specific_transition_test/inference.txt \
  --input-dir data/sgootr/all_trees/combined_inference/S10_CRC11_transition_chain_input \
  --output-dir outputs/sgootr/CRC11/state_specific_transition_test
```

Use the corresponding `S15_...` paths for S15. The script verifies that its
likelihood calculation matches the saved fit before writing any plots. It
creates `outputs/sgootr/CRC11/state_specific_transition_test/figures/CRC11_state_coloring.{pdf,png}`,
`figures/plot_1_net_growth_scaled_birth_relative_birth_profiles.{pdf,svg,png}`,
`figures/plot_3_relative_sampling_probability_profile_intervals_by_tree.{pdf,svg,png}`,
and `tree_with_inferred_transition_times.png` under the selected output
directory. Profile tables are written to its `profiles/data/` folder. The
transition times remain fixed at the values used by the fit; this command does
not search for new times.
