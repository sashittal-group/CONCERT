# Supplementary methods: parameter inference and optimization

## PyTorch implementation of the likelihood and parameter fitting

### Model parameters and inference target

For a set of rooted, time-scaled trees, the model assigns each state \(s\) a
birth rate \(\lambda_s\), death rate \(\mu_s\), and sampling probability
\(\psi_s\). State changes at specified transition epochs are represented by a
row-stochastic matrix \(Q\). The diagonal \(Q_{ss}\) represents remaining in
state \(s\); the user-specified set of allowed off-diagonal edges determines
which changes \(s\to r\) can have nonzero probability. In the fixed-epoch
inference workflow (`src.main2`), transition times are supplied as input and
are held fixed while rates and transition probabilities are estimated.

The implementation loads each node/edge pair as a tree, partitions the tree at
the supplied transition times, and builds level-wise sufficient data for
likelihood evaluation. Transition times later than a tree's root are omitted
for that tree. Per-level data include birth counts, sampled-tip counts by
state, edge start and end times, and incidence matrices linking components
across adjacent levels. These tree-derived quantities are held fixed during
optimization; differentiable tensors represent the model parameters and
likelihood calculations.

### Differentiable parameterization

Optimization is performed over unconstrained `torch.nn.Parameter` tensors.
Reported rates and probabilities are differentiable transformations of these
raw tensors, so the optimizer does not need to enforce constraints by clipping
the fitted values.

For each state, the death rate is parameterized as

\[
\mu_s = \mu_{\max}\,\sigma(d_s),
\]

where \(d_s\) is unconstrained and \(\sigma\) is the logistic sigmoid. The
birth rate is then

\[
\lambda_s = \mu_s + (\lambda_{\max}-\mu_s)\,\sigma(b_s),
\]

which enforces \(\mu_s \leq \lambda_s \leq \lambda_{\max}\). The defaults in
the fixed-time fitter are \(\mu_{\max}=500\) and \(\lambda_{\max}=1000\).
When death rates are fixed by the caller, they are registered as buffers
rather than optimized parameters.

For state-specific sampling, \(\psi_s=\sigma(p_s)\). With the
`--shared-sampling-rate` option, one raw scalar is transformed and expanded
across states. Alternatively, sampling probabilities can be supplied as fixed
values. The default initial rates are \(\lambda_s=5\), \(\mu_s=1\), and
\(\psi_s=0.5\), unless initial values are read from a previous inference
result. Raw initialization values are obtained with the inverse sigmoid after
clamping probabilities away from exactly zero and one.

Transition probabilities are parameterized row by row. For source state \(s\),
the logits include the diagonal entry and only the permitted outgoing edges;
a softmax normalizes these logits to form that row of \(Q\). All other
off-diagonal entries are exactly zero. In the absence of user-supplied initial
values, each permitted destination, including the diagonal, receives the same
initial logit and therefore the same initial probability within that row.

The implementation uses 64-bit floating-point tensors. It selects CUDA when
available, otherwise Apple MPS when available, and otherwise CPU; the device
can also be set explicitly.

### Likelihood calculation

The likelihood is evaluated in log space. First, the code computes the
state-specific probability \(U\) of leaving no sampled descendants, together
with an auxiliary quantity \(C\), from the sampling probabilities, birth and
death rates, and transition matrix. At the present,
\(U_{0,s}=1-\psi_s\), and \(C_{0,s}\) is initialized as

\[
C_{0,s} = \frac{U_{0,s}-\mu_s/\lambda_s}{U_{0,s}-1}.
\]

Between epochs, the closed-form birth--death flow is evaluated using
\(r_s=\lambda_s-\mu_s\). For elapsed time \(\Delta t\), the implementation
forms

\[
U^{\mathrm{flow}}_s
= 1-\frac{r_s}{\lambda_s(1-C_s e^{-r_s\Delta t})},
\]

applies the transition matrix to that vector at the epoch, and recomputes
\(C_s=(U_s-\mu_s/\lambda_s)/(U_s-1)\). These expressions are implemented with
PyTorch tensor operations, so derivatives propagate through the birth--death
quantities and \(Q\).

For each tree level and candidate state, the log likelihood adds birth-event
contributions \(n_s^{\mathrm{birth}}\log\lambda_s\) and
state-specific sampling contributions
\(\sum_s n_s^{\mathrm{sample}}\log\psi_s\). State combinations incompatible
with the observed sampled-tip states are assigned negative infinity. Each
edge contributes the birth--death survival term and the change in the
closed-form \(C\)-dependent term between its parent and child times. At a
transition boundary, state messages from the adjacent level are combined
with the permitted transition probabilities using `torch.logsumexp`. This
log-space message passing sums over latent ancestral states while avoiding
direct products of many small probabilities.

The tree likelihoods are summed to obtain the total log likelihood. The
implementation uses the root likelihood for state 0. With
`--normalize-root-likelihood`, it additionally subtracts
\(\log[1-U_0(t_{\mathrm{root}})]\) for each tree. This option changes the
root-likelihood normalization; it does not change the rate or transition
parameterization.

### Gradient optimization and stopping

Autograd computes derivatives of the total log likelihood with respect to the
trainable raw tensors. The optimizer minimizes its negative. The command-line
entry point defaults to PyTorch L-BFGS; Adam is also available. The fitting
function's own default is Adam, but the CLI explicitly selects L-BFGS unless
overridden.

For Adam, the CLI defaults are a learning rate of 0.01, at most 50,000 steps,
and a patience of 500 steps without improvement. Trees are grouped in chunks
of ten and the corresponding likelihood modules are passed through
`torch.compile`; chunk likelihoods are summed for the objective. Each step
zeros gradients, evaluates the negative log likelihood, backpropagates, and
updates the raw parameters. A non-finite loss raises an error. The best
pre-update parameter state is retained, and optimization stops early when the
patience threshold is reached. The best state is restored after fitting.

For L-BFGS, PyTorch's optimizer uses a strong-Wolfe line search, a history size
of 100, and up to 20 internal iterations per optimizer step. Its closure
recomputes the full objective, clears gradients, backpropagates, and tracks
the best finite likelihood encountered during line-search evaluations. The
same best-state restoration is applied at the end. In either mode, the final
likelihood is recomputed with gradients disabled, and the fitted likelihood,
rates, sampling probabilities, transition matrix, and fixed transition times
are written to the inference output.

The objective in this workflow is the sum of tree log likelihoods, with the
optional root normalization described above; the optimizer does not add a
parameter-prior penalty to this objective.

### Estimating transition times

The differentiable fixed-time fit does not estimate transition times. Although
the times are represented as tensors in the likelihood code, the tree is
partitioned at those times before fitting, and the times are not trainable
parameters in `src.mle_closed_form2`. Transition-time estimation is handled
by separate outer search routines. Both evaluate a schedule by rebuilding the
tree cuts and likelihood data structures for those times, then evaluating the
tree likelihood. During a time-search step, the fitted model parameters are
held fixed; after the schedule is changed, the continuous model parameters can
be refitted at that schedule. This separation avoids differentiating through
the discrete changes to the tree partition.

The joint-search routine, `src.main2_joint_transition_times`, first seeks a
finite-likelihood initial schedule. It tries an evenly spaced schedule and
then randomly generated schedules projected onto a 1,000-point grid over
\((0,t_{\max})\), stopping at the first finite candidate (up to the configured
number of initialization attempts). It then alternates two operations:

1. Fit birth rates, death rates, sampling probabilities, and allowed
   transition probabilities at the current schedule using the selected
   PyTorch optimizer.
2. With those fitted parameters held fixed, visit each transition time in
   order and compare the current value with two candidates: the current value
   minus or plus the current step size. Candidates are constrained to remain
   inside \((0,t_{\max})\) and to preserve strict ordering. A candidate is
   accepted only if it has a finite likelihood greater than the best value
   already found for that coordinate.

The initial coordinate step is
\(t_{\max}/[2(K+1)]\), where \(K\) is the number of transition times; it is
halved after each search round. The default is three rounds. Once these rounds
finish, the model parameters are refitted at the final schedule, initialized
from the previous fit. This is a local coordinate search, not a grid-wide
maximization or a gradient update of the times.

The progressive search routine,
`src.progressive_joint_transition_times`, instead searches transition times
on an ordered grid. It begins with one transition time, fits the continuous
parameters, and sweeps that time across the admissible grid values while
holding the current parameter estimates fixed. It then refits the parameters
at the selected time. For each additional transition, it evaluates candidate
insertion times on the grid (excluding times already in the schedule), chooses
the finite-likelihood insertion with the highest likelihood under the current
parameters, and refits the model. It subsequently sweeps each transition
coordinate over grid values strictly between its neighboring times, followed
by another parameter refit. This repeats until the requested number of
transitions has been added. The grid size, maximum time, and number of
transitions are inputs to this procedure.

Thus, gradients are used for rates, sampling probabilities, and allowed
transition probabilities, whereas transition times are selected by explicit
likelihood comparisons. The joint and progressive routines use different
time-search strategies; neither backpropagates through the tree-cutting
operation.
