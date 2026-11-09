# CONCERT

This repository contains research code for **“Phylodynamic Inference with
Coordinated Cell State Transitions.”** It implements the Coordinated State
Transition Model (CSTM) and CONCERT, an inference framework for birth, death,
sampling, and coordinated cell-state transitions on lineage trees.

The commands below are organized by the paper’s computational results. Run them
from the repository root unless a command says otherwise.

## Results and workflows

| Paper result | Workflow | Main outputs |
|---|---|---|
| Simulation comparisons | [Simulation studies](#simulation-studies) | Inference results and parameter-accuracy plots for star and linear graphs |
| TLS/TLSC analysis | [TLS and TLSC](#tls-and-tlsc) | Fixed-sampling fits, likelihood profiles, and figure panels |
| CRC11 analysis | [CRC11](#crc11) | State-specific fit, transition-count comparison, and manuscript figure |
| KPTracer analysis | [KPTracer](#kptracer) | Crown-conditioned parameter profiles and sensitivity figures |
| Numerical convergence (Supplementary B.1) | [CSTM–BDT-CTMC convergence](#cstm--bdt-ctmc-convergence) | CSTM likelihoods and convergence plot |

## Setup, data, and external software

Create an environment before running the analyses:

```bash
git clone https://github.com/sashittal-group/phylodynamic-CARTA.git
cd phylodynamic-CARTA
python -m venv .venv
source .venv/bin/activate
```

This repository snapshot does not include a pinned Python dependency manifest,
so there is no authoritative `pip install -r ...` command. The workflows use Python scientific packages including PyTorch, NumPy, pandas,
SciPy, Matplotlib, NetworkX, and PyYAML. Snakemake is needed for the workflow
targets below.
Install versions compatible with your environment before running the analyses.

The input trees and generated intermediate results are not a self-contained
dataset in this checkout; many paths under `data/` are local or ignored files.
Obtain the study inputs and place them at the paths specified below and in the
workflow notes before running a result. Depending on the result, external
software is also required: LSD2 and MACH2 for SGOOTR preprocessing, R with
diversitree, BEAM, MACH2, and the companion `phylodynamic-carta-c` executable
for BDT-CTMC.

Generated files follow a two-root convention:

- `data/` contains input data, generated simulation trees, and full per-tree or
  per-schedule inference outputs.
- `outputs/` contains figures, aggregate tables, profile summaries, and other
  compact analysis results.

Both directories are ignored by Git. Keep these roles separate when
overriding output paths; figures and aggregate summaries belong in `outputs/`.
The compact, generated linear A → D example is stored separately under
`examples/linear_A_to_D/` so its inputs and fit are available in a clean clone.

## Repository layout

- `src/concert/` contains the shared CONCERT tree, likelihood, optimization, and
  inference implementation.
- `src/analyses/common/` contains reusable profile, plotting, and tree-data
  utilities.
- `src/analyses/simulation/`, `tls/`, `crc11/`, `kptracer/`, and `bdt_ctmc/`
  contain code and configuration specific to each result group.
- `examples/linear_A_to_D/` contains an eight-tree input dataset and its
  CONCERT inference output.
- `data/` stores study inputs, simulated trees, and full inference fits;
  `outputs/` stores figures and compact analysis summaries.

The root `Snakefile` discovers some simulation inputs while it is being
parsed. In a clean checkout without those generated inputs, Snakemake may
fail before it can select even a specific target. Its default `all` target
only selects the configured TLS fixed-sampling plots; it does not reproduce
all paper results. Use the explicit targets below after preparing the
required inputs.

## Input and output formats

The main inference CLI is `python -m src.concert`. With no selector it jointly
optimizes transition times and model parameters. Use
`--given-transition-times` to fit with fixed transition times. The inference
application accepts one or more paired tree files from an input directory.
Each tree is represented by:

| File | Required columns | Meaning |
|---|---|---|
| `nodes.csv` | `node_id,T,is_leaf,state_id` | One row per node; `node_id` is unique, `T` is its normalized time, `is_leaf` is `0` or `1`, and sampled-tip `state_id` is a zero-based integer. Internal or unobserved states may be `-1`. |
| `edges.csv` | `parent,child,branch_length` | One directed parent-child edge per row, with a nonnegative branch length. |

Use matching filenames such as `nodes.csv`/`edges.csv` or
`nodes_1.csv`/`edges_1.csv`. The command searches the input directory
recursively and pairs files by replacing `edges` with `nodes` in the filename.
Use `--num-trees` to limit how many complete pairs are loaded.

The transition inputs are separate from the tree pair:

- The file passed to `--transition-edges-file` has no header; each line is a
  whitespace-separated `source_state target_state` pair using zero-based
  state IDs. For example, [`src/analyses/simulation/config/transition_edges_star.txt`](src/analyses/simulation/config/transition_edges_star.txt)
  defines the three non-self transitions from state `0` in a four-state star.
- The file passed to `--transition-times-file` has one normalized transition
  time per line. For simulator settings, use
  `src.concert --analysis extract-star-transition-times` to convert the scheduled times to the
  normalized time scale used by the tree files. A simulator-generated
  `transition_times.csv` records realized events (and starts with an event
  count); it is not the scheduled-time input expected by
  `python -m src.concert --given-transition-times`.

The `--output` file is a plain-text fit summary containing the log likelihood,
birth/death/sampling vectors, nonzero transition probabilities, and transition
times. `--log` writes the optimizer log. Keep full fits and logs in `data/`;
figures and compact summaries belong in `outputs/`.

## Quick start: one simulated tree

This example creates one four-state, three-transition star setting, simulates
a single tree, extracts its scheduled transition times, and fits the model to
that tree. It does not require any of the paper's external comparison tools:

```bash
# Create one star_3 setting.
python -m src.concert --analysis generate-star-transition-settings \
  --output-root data/example --num-settings 1

# Simulate one tree from the setting.
python -m src.concert --analysis simulate-coordinate-tree \
  data/example/star_3/setting_1/setting.txt \
  0 data/example/star_3/setting_1/tree_0

# Convert setting times to the normalized scale used by the tree.
python -m src.concert --analysis extract-star-transition-times \
  data/example/star_3/setting_1/setting.txt \
  data/example/star_3/setting_1/transition_times.txt

# Fit one tree; the complete result and optimizer log stay under data/.
python -m src.concert --given-transition-times data/example/star_3/setting_1 \
  --num-states 4 --num-trees 1 \
  --transition-edges-file src/analyses/simulation/config/transition_edges_star.txt \
  --transition-times-file data/example/star_3/setting_1/transition_times.txt \
  --output data/example/star_3/setting_1/inference.out \
  --log data/example/star_3/setting_1/inference.log \
  --device cpu
```

The simulator writes `nodes.csv`, `edges.csv`, tree images, and realized-event
CSV files under `tree_0/`. Its `transition_times.csv` starts with the event
count followed by one time per row; `transition_times_with_edges.csv` starts
with the count followed by rows of
`time,start_node,end_node,source_state_id,target_state_id` (no header). These
files describe realized events, whereas the separate `transition_times.txt`
contains the scheduled times used by this inference example. The inference
command estimates model parameters while holding those scheduled times fixed.
The generated setting and tree are reproducible inputs for trying other
options such as `--num-trees` and `--device`.

## Example: eight linear A → D trees

The repository includes a small example with one generated A → B → C → D
setting, eight simulated trees under `examples/linear_A_to_D/input/`, and a
CONCERT fit under `examples/linear_A_to_D/output/`. To regenerate the example
inputs from the repository root:

```bash
python -m src.concert --analysis generate-linear-transition-settings \
  --output-dir examples/linear_A_to_D/setting --num-settings 1 --seed 3 \
  --self-transition-probabilities 0.75 0.6666666666666666 0.5

for tree in 0 1 2 3 4 5 6 7; do
  seed=$((10000 + tree * 100))
  python -m src.concert --analysis simulate-coordinate-tree \
    examples/linear_A_to_D/setting/setting_1/setting.txt \
    "$seed" "examples/linear_A_to_D/input/tree_$tree"
done
```

The simulator retries consecutive seeds when a trial produces no sampled tips.
The eight inputs are node/edge CSV pairs in `input/tree_0/` through
`input/tree_7/`. To jointly estimate the transition times and model parameters
with the same state-specific coordinate-search application used by the
Snakemake linear-inference rule:

```bash
mkdir -p examples/linear_A_to_D/output
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m src.concert --analysis joint-state-specific-transition-times \
  --tree-directory examples/linear_A_to_D/input \
  --num-states 4 --num-trees 8 \
  --transition-edges-file src/analyses/simulation/config/transition_edges_linear.txt \
  --max-time 1 --grid-size 1000 --random-seed 1000 \
  --lbfgs-max-iter 20 --search-rounds 6 \
  --output examples/linear_A_to_D/output/inference.out \
  --log examples/linear_A_to_D/output/inference.log \
  --device cpu \
  > examples/linear_A_to_D/output/stdout.log \
  2> examples/linear_A_to_D/output/stderr.log
```

The fit reports the estimated birth, death, sampling, transition probabilities,
and one transition time per linear edge. Each `epoch_N` transition matrix
corresponds to one directed edge in the linear graph; only that edge's source
state can transition during that epoch. `transition_times_by_edge` maps each
edge to its fitted normalized time. The edges are listed in reverse state order
(`2->3`, `1->2`, `0->1`): the simulation setting's times for A, B, and C map
to `0->1`, `1->2`, and `2->3`, respectively. This uses the same grid and
search rounds as the full linear Snakemake inference rule. For the full
simulation workflow,
`generate_linear_transition_settings` and `simulate_linear_transition_3`
produce the configured data, while
`infer_linear_transition_3_joint_coordinate_first8` fits the first eight trees
in a setting/batch. The aggregate target
`infer_linear_transition_3_joint_coordinate_all` expands this fit over the
configured settings and batches; it is substantially larger than this example.
The example sets self-transition probabilities to 3/4, 2/3, and 1/2 for A, B,
and C respectively, with the remainder assigned to the next state. This keeps
more lineages in ancestor states while allowing lineages to reach D. The
example-only generator option leaves the simulation workflow's default
randomized probabilities unchanged.

## CONCERT command-line interface

Use one entry point for inference and the Python analysis tasks invoked by
Snakemake. With no selector, the CLI runs joint transition-time and parameter
inference.

| Selector | Application |
|---|---|
| *(none)* | Jointly optimize transition times and model parameters. |
| `--given-transition-times` | Fit parameters with fixed transition times. |
| `--given-edge-specific-transition-times` | Fit using observed edge-specific transitions. |
| `--continue` | Continue a fit from saved parameter estimates. |
| `--progressive-schedule` | Estimate transition times by progressive grid sweeps. |
| `--state-specific-transitions` | Fit the CRC11 state-specific transition model. |
| `--analysis TASK` | Run a named Python analysis task used by Snakemake. |

Run `python -m src.concert --help` to list selectors and analysis tasks. For
example, `python -m src.concert --given-transition-times --help` shows the
fixed-time fit options, while `python -m src.concert --analysis profile-plot
--help` shows the profile plotting options.

## Simulation studies

The comparison workflow evaluates CONCERT against diversitree, BEAM, and
MACH2 on four-state star and linear transition graphs. The inference-completed
comparison covers the 20 configured star and 20 configured linear
setting/batch combinations, using the first eight trees in each batch.
For the linear simulation workflow, the Snakemake rules are
`generate_linear_transition_settings`, `simulate_linear_transition_3`, and
`infer_linear_transition_3_joint_coordinate_first8`; the latter fits the
state-specific A → B → C → D transition times together with the model
parameters. The smaller checked-in example above uses the same inference
application without requiring the full configured simulation set.

1. Generate the star settings. The linear settings are generated by the
   Snakemake workflow when needed:

   ```bash
   python -m src.concert --analysis generate-star-transition-settings \
     --output-root data/simulation/coord_diff \
     --num-settings 100
   ```

2. Run the curated comparison:

   ```bash
   snakemake --cores 1 coord_diff_inference_completed_batches_methods
   ```

   This target requires the comparison software and its configuration to be
   available. It writes simulated trees and per-tree method results under
   `data/simulation/coord_diff/`.

3. Generate plots from completed inference outputs:

   ```bash
   python -m src.analyses.simulation.plot_star_inference_accuracy \
     --data-dir data/simulation/coord_diff/star_3 \
     --progressive-transition-times 3

   python -m src.concert --analysis plot-linear-transition-inference-accuracy \
     --data-dir data/simulation/coord_diff/linear_A_to_D

   python -m src.concert --analysis plot-linear-transition-inference-accuracy \
     --data-dir data/simulation/coord_diff/linear_A_to_D \
     --transition-time-errors
   ```

The linear plot rule in the Snakefile is also available as
`snakemake --cores 1 plot_linear_transition_3_identifiable_parameters`; it
requires the full configured set of linear simulations and diversitree fits,
not only the listed inference-completed batches.

## TLS and TLSC

The root workflow target for the configured fixed-sampling TLS/TLSC analysis
is:

```bash
snakemake --cores 1 tls_fixed_sampling_01
```

It uses the subtree exports under `data/troupe/`, writes fits under each
dataset's `data/troupe/<dataset>/fixed_sampling_0.1/` directory, and writes
profile tables and figures under `outputs/tls/fixed_sampling_0.1/`.

For the manuscript’s 100-transition-time profile analysis, use the completed
inference files/checkpoints and regenerate profiles and comparison plots:

```bash
python -m src.analyses.tls.profile_transition_run \
  data/troupe/TLS/fixed_sampling_0.1/inference_transition_times_100.tmp.out

python -m src.analyses.tls.profile_transition_run \
  data/troupe/TLSC/fixed_sampling_0.1/inference_transition_times_100.out

python -m src.analyses.tls.compare_transition_profiles \
  --tls-profiles outputs/tls/fixed_sampling_0.1/transition_profiles/TLS/data/TLS_transition_times_100 \
  --tlsc-profiles outputs/tls/fixed_sampling_0.1/transition_profiles/TLSC/data/TLSC_transition_times_100

python -m src.analyses.tls.figure_generator
```

The TLS profile command uses a saved `.tmp.out` checkpoint; it is not produced
by the profile command itself. The figure generator also expects the relevant
inference runs and profile tables to be present. To reproduce the state-colored
subtree plots after the 100-transition fits are available, run:

```bash
python -m src.analyses.tls.process_state_coloring --dataset TLS
python -m src.analyses.tls.process_state_coloring --dataset TLSC
python -m src.analyses.tls.plot_state_coloring --dataset TLS
python -m src.analyses.tls.plot_state_coloring --dataset TLSC
```

See [the TLS workflow notes](src/analyses/tls/README.md) for individual panel
commands and output locations. TLS figures, profiles, and state-coloring
summaries are written under `outputs/tls/`.

## CRC11

The CRC11 result fits a state-specific NC → PT → LN model, compares candidate
numbers of PT → LN transition times, and generates the manuscript figure.
Its prepared transition-chain input is expected at
`data/sgootr/all_trees/combined_inference/S10_CRC11_transition_chain_input/`.
The input directory must contain the tree and transition files consumed by
the inference scripts.

If the raw CRC11 tree and preprocessing tools are available, the general
SGOOTR processing pipeline can be run with:

```bash
python -m src.analyses.crc11.preprocessing.pipeline
```

This pipeline expects `data/sgootr/CRC11.nwk` by default and LSD2/MACH2 on
`PATH` or in the project environment. It does not by itself create the
prepared `S10_CRC11_transition_chain_input` directory; see
[the SGOOTR processing notes](src/analyses/crc11/PREPROCESSING.md) for its
preprocessing stages.

The CRC11 inference and downstream results are available as named Snakemake
targets. They require the prepared S10 input directory above. Run the full
workflow with:

```bash
snakemake --cores 4 sgootr_crc11_results
```

This aggregate includes the full schedule search (up to 10 cohorts of 500
schedules), the fixed-time fit for the manuscript's selected schedule 3857,
the selected-fit profiles and tree visualization, the one-through-five
PT → LN transition-count analysis, and the manuscript figure. The search and
transition-count analysis are computationally expensive. To run or rerun only
one stage, use its individual target:

```bash
snakemake --cores 4 sgootr_crc11_schedule_search
snakemake --cores 1 sgootr_crc11_fit_schedule_3857
snakemake --cores 1 sgootr_crc11_selected_fit_plots
snakemake --cores 4 sgootr_crc11_transition_count_analysis
snakemake --cores 1 sgootr_crc11_manuscript_figure
```

The fixed fit uses NC → PT at 0.356 and PT → LN at 0.131 and 0.282. Each
target writes to a separate `_snakemake` output location so it will not
overwrite the existing manuscript results. The search results and full fits
are written under `data/sgootr/all_trees/combined_inference/`; summaries,
profiles, plots, and the figure are written under `outputs/sgootr/`.

The figure is assembled from the fixed schedule-3857 fit. A fresh schedule
search may select another schedule; the fixed-fit target intentionally
recreates the fit associated with the current manuscript figure. Additional
state-specific fitting, profiling, and plotting commands are in
[the CRC11 workflow notes](src/analyses/crc11/README.md) and
[the state-specific transition notes](src/analyses/crc11/README_state_specific_transitions.md).

## KPTracer

The KPTracer sensitivity analysis requires the processed data and fitted
parameters under `data/KPTracer-Data/processed_data`. Generate its profile
tables and figures with:

```bash
python -m src.analyses.kptracer.preprocess \
  --config src/analyses/kptracer/config.yaml

python -m src.analyses.kptracer.plot \
  --config src/analyses/kptracer/config.yaml
```

Profile tables are written under `outputs/kptracer/profiles/KPTracer/`, and
the composite and individual parameter figures are written under
`outputs/kptracer/figures/`. The input format and optional source-specific
inference runner are documented in
[the KPTracer workflow notes](src/analyses/kptracer/README.md).

## CSTM–BDT-CTMC convergence

Supplementary Sec. B.1 compares CSTM likelihoods across increasing numbers of
transition times on 100 simulated trees. The CSTM computation is:

```bash
python -m src.analyses.bdt_ctmc.compare_cstm_bdt_ctmc
```

It reads trees from
`data/simulation/bdt_ctmc/emperical_proof_data/tree_001/` through
`tree_100/` and writes the compact likelihood table to
`outputs/supplementary/bdt_ctmc/tree_likelihoods_cstm.txt`.
The retained `infer` rule in the `Snakefile` runs the BDT-CTMC companion
executable for a prepared setting, but that executable and the setting inputs
are not included in this repository. For a setting named `SETTING_NAME`, its
target is:

```bash
snakemake --cores 1 data/simulation/bdt_ctmc/SETTING_NAME/inference.out
```

The aggregate BDT-CTMC likelihood table belongs alongside the CSTM table at
`outputs/supplementary/bdt_ctmc/tree_likelihoods.txt`.
The convergence plot is in
[`notebooks/cstm_vs_ctmc.ipynb`](notebooks/cstm_vs_ctmc.ipynb); open and run it
after producing both CSTM and BDT-CTMC likelihood files:

```bash
jupyter lab notebooks/cstm_vs_ctmc.ipynb
```

The notebook saves its convergence figures under
`outputs/supplementary/bdt_ctmc/`.

## Retained analysis notebooks

The repository retains notebooks used for paper results and supplementary
analysis:

- [`notebooks/TLS_100_transition_methods_and_results.ipynb`](notebooks/TLS_100_transition_methods_and_results.ipynb)
- [`notebooks/tls_transition_resolution_stability.ipynb`](notebooks/tls_transition_resolution_stability.ipynb)
- [`notebooks/SGOOTR_CRC11_methods_and_results.ipynb`](notebooks/SGOOTR_CRC11_methods_and_results.ipynb)
- [`notebooks/cstm_vs_ctmc.ipynb`](notebooks/cstm_vs_ctmc.ipynb)

Open a notebook from the repository root with `jupyter lab <notebook-path>`.
Notebooks may depend on the input files and intermediate outputs described in
the result sections above.
