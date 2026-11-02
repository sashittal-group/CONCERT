# KPTracer preprocessing

Run the modular preprocessing pipeline from the repository root:

```bash
python -m src.process_data.kptracer.pipeline
```

The defaults use `data/laml/problin_1461.nwk` and
`data/laml/KPTracer_meta.csv`, together with the refined tree and inferred
location labels in `data/KPTracer-Data/processed_data/`. Override these paths
and the minimum-edge and tip-depth settings with the CLI options shown by
`--help`.

The pipeline writes `1461.tree` and `1461.labeling`, then normalizes and exports
`nodes_0.csv` and `edges_0.csv`. It does not add a synthetic stem node or edge;
the original Newick root is the tree root. If the legacy location-label file
contains a node-0 stem label, that entry is ignored. The pipeline also builds
`transitions_0.csv`, `transitions_0_adjusted.csv`, and `transition-times.txt`
from the refined edge intervals. `transition-times.txt` contains sorted,
unique event times; the adjusted table still retains a row for every
transition edge. Tree normalization keeps the notebook's one-based tree-node
IDs, unit minimum tip depth, and 0.001 minimum edge length.

To run the preprocessing through Snakemake:

```bash
snakemake --cores 1 data/KPTracer-Data/processed_data/transitions_0_adjusted.csv
```

To progressively infer 20 transition times on the normalized KPTracer tree
while fixing the sampling probability at 0.05, run the shell script from the
repository root:

```bash
./run_kptracer_progressive.sh
```

The script runs grids 100, 200, 500, and 1000 in sequence. It starts with an
evenly spaced 10-time subset of the existing grouped schedule, then seeds each
finer grid from the previous grid's fitted 10-transition result and schedule.
It prints progress to the terminal and saves a `.stdout.log` for each stage
under `data/KPTracer-Data/processed_data/`. To follow the first stage from
another terminal:

```bash
tail -f data/KPTracer-Data/processed_data/inference_progressive_fixed_sampling_0.05_20_grid_100.stdout.log
```

The transition-edge file allows the eight directed state changes observed in
the current KPTracer annotations. Each grid stage progressively fits counts
10 through 20, refining the transition times and refitting model parameters at
every count. Each stage directory contains `transition_count_10.out` through
`transition_count_20.out` with fitted parameters and matching
`transition_count_*_times.txt` schedules. Optimizer logs for counts 10 through
19 are in that directory; the count-20 optimizer log is the stage's `.log`
file. Stages' combined terminal output and stderr are saved in their
`.stdout.log` files. This seeded initialization is needed because one-time
schedules have non-finite likelihoods for the KPTracer tree.

The same workflow is also available as the Snakemake target
`infer_kptracer_progressive`.

To fit CARTA using the 23 co-transmission times derived from a SharpTNI run,
first run `./sharp-tni/run_kptracer.sh`, then set
`kptracer_sharptni_cotransmission_run_directory` in
`src/results/kptracer/config.yaml` to that run's output directory. Run:

```bash
snakemake --cores 1 infer_kptracer_sharptni_cotransmission
```

The rule reads `cotransmission_times_from_sharptni.txt` and initializes from
`carta_sharptni_fixed_times.out` in that directory. It stops after
`kptracer_sharptni_cotransmission_patience` consecutive Adam steps without a
new best likelihood, with `kptracer_sharptni_cotransmission_max_steps` as a
safety cap. Both default to 500 and 50,000 respectively and can be overridden,
for example:

```bash
snakemake --cores 1 infer_kptracer_sharptni_cotransmission \
  --config kptracer_sharptni_cotransmission_patience=1000
```

The fitted parameters are written to
`carta_sharptni_cotransmission_times.out` in the SharpTNI run directory.

To fit CARTA on every distinct mixed-direction overlap schedule from the
enumerated gamma=21 and gamma=22 Sankoff solutions, initialized each time from the
CONCERT N=10 parameters, run:

```bash
snakemake --cores 4 infer_kptracer_grouped_transition_schedules
```

The workflow deduplicates identical schedules across solutions and runs one
fit per distinct schedule (96 for the current inputs). Each fit is an
independent one-thread job, so `--cores` controls the parallelism. Sampling
is fixed at 0.05; max steps and patience default to 50,000 and 500 and can be
overridden with `kptracer_grouped_transition_max_steps` and
`kptracer_grouped_transition_patience`. Individual fits and logs are written
under `sharp-tni/results/kptracer_sharptni_20261101_1118/concert_initialized/grouped_schedule_fits/`;
the aggregate likelihood table is `fit_summary.tsv` in that directory.

To add the completed 11-time SharpTNI grouped-schedule likelihoods to the
regenerated likelihood-versus-transition-count plot, run:

```bash
python sharp-tni/plot_grouped_schedule_likelihoods.py
```

The script updates
`data/KPTracer-Data/processed_data/likelihood_vs_transition_count_grid_100_regenerated_precision_repaired.png`
and writes the plotted likelihoods with their SharpTNI solution indices to
`completed_11_time_schedule_likelihoods.tsv` in the grouped-schedule fit
directory. It includes completed fits only.

To compare transition-type counts across the CONCERT N=7/N=11 fits, the latest
CONCERT-initialized MACH2 schedules, and the best completed SharpTNI grouped
11-time fit, run:

```bash
python -m src.results.kptracer.plot_transition_type_counts
```

The stacked bars count changes between adjacent posterior-MAP colored edge
segments at each fit's exact transition times, on a shared root-to-tip time
axis from 0 to 1. Bars remain at their exact elapsed times unless close events
need a small display-only shift to leave a visible gap. Zero-count times are
marked with a dot; the table records both exact times and any display shift.
The plot and complete transition-count table are written under
`data/KPTracer-Data/processed_data/transition_type_counts/`.

To regenerate and refine transition counts 5–20, run:

```bash
./run_kptracer_schedule_refinement.sh
```

The script starts from the original grouped schedule, regenerates grid-100
progressive fits for N=5–20, then performs up to five coordinate rounds for
each count. Each round scans legal neighbor intervals at a 0.0001 time step
and refits model parameters after each coordinate update. Existing checkpoints
remain unchanged. Regenerated checkpoints are written under
`data/KPTracer-Data/processed_data/inference_progressive_fixed_sampling_0.05_5_20_regenerated_grid_100/`;
refined parameters and schedules, a per-count CSV summary, and the combined log
are written under
`data/KPTracer-Data/processed_data/inference_progressive_fixed_sampling_0.05_5_20_regenerated_refined_grid_100/`.
