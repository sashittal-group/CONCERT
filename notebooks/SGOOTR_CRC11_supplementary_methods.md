# Supplementary Methods: CRC11 S10 lineage-tree inference

## Dataset and ultrametric tree

We analyzed the CRC11 S10 single-cell DNA-methylation dataset originally
reported by Bian et al.~\cite{bian2018single}. Pairwise methylation-based
distances estimated using Sgootr~\cite{liu2023single} were used to reconstruct
the lineage tree by unweighted pair group method with arithmetic mean
(UPGMA). UPGMA produces an ultrametric tree, in which all tips have equal
root-to-tip distance. The resulting branch lengths define the relative time
coordinate for inference; they are not absolute calendar dates.

The source description reports 228 cells, whereas the tree used for the
selected fit contains 227 labeled tips: 25 non-cancer (NC), 134 primary-tumor
(PT), and 68 lymph-node (LN) cells. This one-cell discrepancy is unresolved.
Accordingly, these state counts describe the analyzed tree only and should not
be interpreted as a complete partition of all 228 cells in the source
description.

For inference, the tip time was set to zero and time increased toward the root.
The UPGMA tree's root age, and hence \(t_{\mathrm{MRCA}}\), was
0.443103042 tree-time units. The transition-time search was restricted to this
coordinate and to times earlier than the root. When displaying event times on
the reversed tree coordinate, we used
\(t_{\mathrm{display}}=t_{\mathrm{MRCA}}-t_{\mathrm{inference}}\).

## State-specific model and inference

We considered a sequential state progression \(NC\rightarrow PT\rightarrow
LN\). Within each state \(s\), the model has a birth rate \(\lambda_s\), a
death rate \(\mu_s\), and a sampling probability \(\pi_s\), the probability
that a lineage in state \(s\) is sampled. At each
transition epoch, the transition matrix permits only the designated
source-to-destination change; all other states remain unchanged. The model
allows one \(NC\rightarrow PT\) epoch and \(K\) distinct \(PT\rightarrow LN\)
epochs. Repeated \(PT\rightarrow LN\) epochs share a single transition
probability for the PT source state, as do repeated epochs with any common
source state.

For a fixed set of event times, the tree likelihood was maximized over the
three state-specific birth rates, three death rates, three sampling
probabilities, and one transition probability for each source state used by
the schedule. The transition times were searched discretely rather than
optimized continuously: candidate times lie on a 0.001-spaced grid, are
distinct, and must precede the root. Candidate schedules were evaluated by
fitting the continuous parameters with the L-BFGS optimizer (up to 20
iterations per schedule). A seeded genetic schedule search initialized
candidate schedules at random and used high-likelihood schedules to generate
subsequent candidates; the saved best fit is therefore the best schedule
found by this search, not a guarantee of the global maximum over all possible
event times.

The selected S10 fit has one \(NC\rightarrow PT\) event and two
\(PT\rightarrow LN\) events. In the inference coordinate, their times are
0.356, 0.282, and 0.131, respectively. The saved fit has reported
root-normalized log likelihood 327.294240. These are the project's fitted
estimates for CRC11 S10, rather than estimates copied from the Sgootr
publication.

The fitted state-specific sampling probabilities were
\(\hat{\pi}_{NC}=0.051098\), \(\hat{\pi}_{PT}=0.025462\), and
\(\hat{\pi}_{LN}=0.032316\). These are model sampling probabilities, not the
observed proportions of tips in the three states.

## Selection of the number of \(PT\rightarrow LN\) epochs

We compared models with \(K=1,\ldots,5\) \(PT\rightarrow LN\) epochs while
retaining one \(NC\rightarrow PT\) epoch. For each value of \(K\), the
transition schedule and the model parameters were fit in a separate schedule
search. The nominal number of free parameters is
\[
p_K = 3_{\lambda}+3_{\mu}+3_{\pi}+2_{q}+1_{NC\rightarrow PT}+K_{PT\rightarrow LN}
    = 12+K,
\]
where \(q\) denotes the transition probability associated with a source state.
We ranked the saved best fits using
\[
\mathrm{AIC}_K=-2\ell_K+2p_K,\qquad
\Delta\mathrm{AIC}_K=\mathrm{AIC}_K-\min_j\mathrm{AIC}_j.
\]

| \(PT\rightarrow LN\) epochs (\(K\)) | \(p_K\) | Best log likelihood | AIC | \(\Delta\)AIC |
|---:|---:|---:|---:|---:|
| 1 | 13 | 290.555649 | -555.111299 | 71.477181 |
| 2 | 14 | 327.294240 | -626.588480 | 0.000000 |
| 3 | 15 | 327.108987 | -624.217975 | 2.370505 |
| 4 | 16 | 328.125035 | -624.250071 | 2.338409 |
| 5 | 17 | 328.345254 | -622.690509 | 3.897971 |

Among these candidates, \(K=2\) has the minimum AIC. The comparison is
exploratory rather than a significance test: AIC's usual assumptions may be
imperfect for discrete change-point schedules, and the likelihoods are the
best fits found in separate searches rather than guaranteed global maxima.

## Likelihood profiles

The parameter profiles used for the CRC11 rate summaries were calculated at
the selected event schedule. For each profile, one parameter was assigned a
sequence of feasible candidate values (a 121-point grid plus the fitted value
if it was not already on the grid), and the tree likelihood was recomputed at
each value. Event times, transition probabilities, and all other rate and
sampling parameters were held at their fitted values; the remaining
parameters were not re-optimized.
Thus, these curves are one-parameter, fixed-nuisance likelihood slices, rather
than profiles in the strict sense of maximizing over nuisance parameters.

For state-specific net growth \(g_s=\lambda_s-\mu_s\), a candidate value was
evaluated by setting \(\lambda_s=\mu_s+g_s\) and leaving the other parameters
unchanged. For the relative-birth parameter
\(r_s=\lambda_s/\lambda_{NC}\), the two birth rates were varied to attain the
candidate ratio while holding their sum fixed. The resulting likelihood drop
was compared with the fitted likelihood. The plotted interval boundaries
were linearly interpolated at likelihood fractions
\(\exp(-0.25)\) and \(\exp(-1)\), corresponding to log-likelihood drops of
0.25 and 1.0. These plotted ranges are descriptive likelihood-drop intervals,
not confidence intervals with a claimed coverage level.

In the separate transition-time sensitivity calculation, one event time at a
time was swept across the 0.001 grid, subject to the root-time and event-order
constraints, while all other event times and all fitted model parameters were
held fixed. The likelihood was recalculated at each candidate time; zero
likelihoods were retained as \(-\infty\). This is likewise a fixed-parameter
sensitivity profile and does not refit nuisance parameters.
