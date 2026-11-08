# Supplementary Methods: CSTM–MTBD likelihood convergence analysis

## Tree data and likelihood calculations

We compared likelihoods for the same 100 simulated trees (`tree_001` through `tree_100`) under an MTBD reference and the CSTM calculation. Each tree was loaded from its `nodes.csv` and `edges.csv` files in `data/simulation/bdt_ctmc/emperical_proof_data/`.

For the CSTM calculation, the two-state birth, death, and sampling rates were fixed at \(\lambda=(7.0, 6.0)\), \(\mu=(1.5, 0.5)\), and \(\psi=(0.5, 0.4)\), respectively. We evaluated each tree at eight temporal resolutions, \(N \in \{5, 10, 25, 50, 100, 500, 1000, 10000\}\), using \(\Delta t=1/N\). At each resolution, the transition matrix was

\[
P(\Delta t)=
\begin{pmatrix}
1-\Delta t & \Delta t \\
0 & 1
\end{pmatrix},
\]

and the transition-time grid was constructed as `[0, 1e-15]` followed by \(k\Delta t\) for \(k=1,\ldots,N-1\). The CSTM likelihoods were calculated with the closed-form routines in [`src/cstm_vs_ctmc.py`](../src/cstm_vs_ctmc.py).

The notebook loads one reference log-likelihood per tree from `outputs/supplementary/bdt_ctmc/tree_likelihoods.txt` and the eight CSTM log-likelihoods per tree from `outputs/supplementary/bdt_ctmc/tree_likelihoods_cstm.txt`. In the notebook, the reference values are held in a variable named `ctmc_likelihoods`; the final figure labels the reference as MTBD.

## Error summary and figure

For tree \(i\) at time step \(\Delta t_j\), we calculated the paired absolute log-likelihood difference

\[
e_{ij}=\left|\mathcal{L}_{\mathrm{MTBD},i}
-\mathcal{L}_{\mathrm{CSTM},ij}\right|.
\]

The light-gray curves show these errors for individual trees. At each time step, the black curve shows their arithmetic mean across the 100 trees. Absolute differences are taken before averaging so that signed errors cannot cancel one another.

To summarize how the mean error changes with time-step size, we fit an ordinary least-squares line to \(\log_{10}(\overline{e})\) as a function of \(\log_{10}(\Delta t)\). The fitted slope (the power-law exponent) and \(R^2\) are reported in the figure legend. The axes are logarithmic; decreasing error as \(\Delta t\) decreases indicates convergence of the CSTM likelihoods toward the MTBD reference as the discretization is refined.

The plotted analysis is in [`cstm_vs_ctmc.ipynb`](./cstm_vs_ctmc.ipynb), which writes `likelihood_error_scaling.pdf` and `likelihood_error_scaling.svg`.
