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

The root `Snakefile` discovers some simulation inputs while it is being
parsed. In a clean checkout without those generated inputs, Snakemake may
fail before it can select even a specific target. Its default `all` target
only selects the configured TLS fixed-sampling plots; it does not reproduce
all paper results. Use the explicit targets below after preparing the
required inputs.

## Input and output formats

The main inference command, `python -m src.main2`, accepts one or more paired
tree files from an input directory. Each tree is represented by:

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
  state IDs. For example, [`src/transition_edges_star.txt`](src/transition_edges_star.txt)
  defines the three non-self transitions from state `0` in a four-state star.
- The file passed to `--transition-times-file` has one normalized transition
  time per line. For simulator settings, use
  `src.extract_star_transition_times` to convert the scheduled times to the
  normalized time scale used by the tree files. A simulator-generated
  `transition_times.csv` records realized events (and starts with an event
  count); it is not the scheduled-time input expected by `src.main2`.

The `--output` file is a plain-text fit summary containing the log likelihood,
birth/death/sampling vectors, nonzero transition probabilities, and transition
times. `--log` writes the optimizer log. Keep full fits and logs in `data/`;
figures and compact summaries belong in `outputs/`.

## Quick start: one simulated tree

This example creates one four-state star setting, simulates a single tree,
extracts its scheduled transition times, and fits the model to that tree. It
does not require any of the paper's external comparison tools:

```bash
# Create one setting for each star_2 and star_3 model.
python -m src.generate_star_transition_settings \
  --output-root data/example --num-settings 1

# Simulate one tree from the first two-transition setting.
python -m src.simulator_coord \
  data/example/star_2/setting_1/setting.txt \
  0 data/example/star_2/setting_1/tree_0

# Convert setting times to the normalized scale used by the tree.
python -m src.extract_star_transition_times \
  data/example/star_2/setting_1/setting.txt \
  data/example/star_2/setting_1/transition_times.txt

# Fit one tree; the complete result and optimizer log stay under data/.
python -m src.main2 data/example/star_2/setting_1 \
  --num-states 4 --num-trees 1 \
  --transition-edges-file src/transition_edges_star.txt \
  --transition-times-file data/example/star_2/setting_1/transition_times.txt \
  --output data/example/star_2/setting_1/inference.out \
  --log data/example/star_2/setting_1/inference.log \
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

## Simulation studies

The comparison workflow evaluates CONCERT against diversitree, BEAM, and
MACH2 on four-state star and linear transition graphs. The curated comparison
target covers the top twenty setting/batch combinations and the first eight
trees in each batch.

1. Generate the star settings. The linear settings are generated by the
   Snakemake workflow when needed:

   ```bash
   python -m src.generate_star_transition_settings \
     --output-root data/simulation/coord_diff \
     --num-settings 100
   ```

2. Run the curated comparison:

   ```bash
   snakemake --cores 1 coord_diff_top_twenty_batches_methods
   ```

   This target requires the comparison software and its configuration to be
   available. It writes simulated trees and per-tree method results under
   `data/simulation/coord_diff/`.

3. Generate plots from completed inference outputs:

   ```bash
   python -m src.plot_star_inference_accuracy \
     --data-dir data/simulation/coord_diff/star_3 \
     --progressive-transition-times 3

   python -m src.plot_linear_transition_inference_accuracy \
     --data-dir data/simulation/coord_diff/linear_A_to_D

   python -m src.plot_linear_transition_inference_accuracy \
     --data-dir data/simulation/coord_diff/linear_A_to_D \
     --transition-time-errors
   ```

The linear plot rule in the Snakefile is also available as
`snakemake --cores 1 plot_linear_transition_3_identifiable_parameters`; it
requires the full configured set of linear simulations and diversitree fits,
not only the curated top-twenty comparison.

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
python -m src.results.tls.profile_transition_run \
  data/troupe/TLS/fixed_sampling_0.1/inference_transition_times_100.tmp.out

python -m src.results.tls.profile_transition_run \
  data/troupe/TLSC/fixed_sampling_0.1/inference_transition_times_100.out

python -m src.results.tls.compare_transition_profiles \
  --tls-profiles outputs/tls/fixed_sampling_0.1/transition_profiles/TLS/data/TLS_transition_times_100 \
  --tlsc-profiles outputs/tls/fixed_sampling_0.1/transition_profiles/TLSC/data/TLSC_transition_times_100

python -m src.results.tls.figure_generator
```

The TLS profile command uses a saved `.tmp.out` checkpoint; it is not produced
by the profile command itself. The figure generator also expects the relevant
inference runs and profile tables to be present. To reproduce the state-colored
subtree plots after the 100-transition fits are available, run:

```bash
python -m src.results.tls.process_state_coloring --dataset TLS
python -m src.results.tls.process_state_coloring --dataset TLSC
python -m src.results.tls.plot_state_coloring --dataset TLS
python -m src.results.tls.plot_state_coloring --dataset TLSC
```

See [the TLS workflow notes](src/results/tls/README.md) for individual panel
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
python -m src.process_data.sgootr.pipeline
```

This pipeline expects `data/sgootr/CRC11.nwk` by default and LSD2/MACH2 on
`PATH` or in the project environment. It does not by itself create the
prepared `S10_CRC11_transition_chain_input` directory; see
[the SGOOTR processing notes](src/process_data/sgootr/README.md) for its
preprocessing stages.

To rerun the state-specific schedule search on the prepared S10 input:

```bash
python -m src.results.sgootr.state_specific_schedule_search \
  --input-dir data/sgootr/all_trees/combined_inference/S10_CRC11_transition_chain_input \
  --output-dir data/sgootr/all_trees/combined_inference/S10_CRC11_state_specific_3transition_shared_source_search \
  --summary-dir outputs/sgootr/state_specific_schedule_search/S10_CRC11_state_specific_3transition_shared_source_search \
  --workers 4 --seed 42
```

Full schedule fits remain under `data/`; ranked schedule and cohort summaries
are written under `outputs/sgootr/state_specific_schedule_search/`. To
regenerate the selected-fit plots and profile tables, compare one through
five PT → LN transition times using the saved two-time fit, and assemble the
manuscript figure:

```bash
python -m src.results.sgootr.plot_state_specific_transitions \
  data/sgootr/all_trees/combined_inference/S10_CRC11_state_specific_3transition_shared_source_search/schedule_3857/inference.txt \
  --input-dir data/sgootr/all_trees/combined_inference/S10_CRC11_transition_chain_input \
  --output-dir outputs/sgootr/CRC11/selected_fit

python -m src.results.sgootr.transition_count_analysis \
  --input-dir data/sgootr/all_trees/combined_inference/S10_CRC11_transition_chain_input \
  --output-dir data/sgootr/all_trees/combined_inference/S10_CRC11_ptln_transition_count_analysis \
  --summary-dir outputs/sgootr/transition_count_analysis/S10_CRC11_ptln_transition_count_analysis \
  --existing-two-count-inference data/sgootr/all_trees/combined_inference/S10_CRC11_state_specific_3transition_shared_source_search/schedule_3857/inference.txt

python -m src.results.sgootr.figure_generator
```

The figure generator’s defaults refer to the saved manuscript fit at
`schedule_3857` in `data/`, and its associated profile, tree-coloring, and
transition-count summaries in `outputs/sgootr/`. A fresh schedule search may
select a different schedule; update the figure-generator input paths if using
that new result. Additional state
specific fitting, profiling, and plotting commands are in
[the CRC11 workflow notes](src/results/sgootr/README.md) and
[the state-specific transition notes](src/results/sgootr/README_state_specific_transitions.md).

## CSTM–BDT-CTMC convergence

Supplementary Sec. B.1 compares CSTM likelihoods across increasing numbers of
transition times on 100 simulated trees. The CSTM computation is:

```bash
python -m src.cstm_vs_ctmc
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
