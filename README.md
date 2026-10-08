# process-spine — Bayesian experimental design on measured process tables

A surrogate model and its acquisition functions, benchmarked on a table of
**4599 real experiments** — no simulator anywhere in the loop.

The question this answers: when each evaluation costs a plate of real reactions
instead of a forward pass, how much does a Bayesian campaign actually save, and
what does it save it *relative to*?

## The finding, in one paragraph

On the Doyle-group amination table (3 bases x 4 ligands x 16 aryl halides x 24
additives, 4599 of 4608 cells run), give every method the same 64 experiments in
batches of 8, over 5 seeds. Expected improvement reaches **0.976 +- 0.033** of
the table's own best, one-factor-at-a-time reaches **0.978 +- 0.026**, and plain
random reaches **0.939 +- 0.061**. So the Bayesian gain over random is worth
about **+0.037 of the ceiling**, and it is *not* a gain over the oldest
no-model heuristic in process engineering: coordinate ascent ties it. Two arms
lose to random outright — greedy space-filling (0.902) and, notably, log-scaled
EI (0.923), which is my own code failing for a reason I can name.

The reason matters more than the ranking. Held-out rank correlation of the
surrogate rises from **0.205 (16 training cells) to 0.794 (256)**, but its
top-10 recovery stays at **0.00** until 256 cells and reaches only **0.10**
there. The surrogate orders cells coarsely and cannot yet point at the best ten.
That is exactly the regime where EI degrades to *follow the posterior mean*, and
where "follow your best observation so far, one factor at a time" is already
near-optimal. The Bayesian arm is not losing to a straw man; it is tying a real
heuristic in the small-sample regime this domain lives in.

## What is measured, and what is asserted

    claim                                  how it is established
    the campaign only costs experiments     every queried cell is a row of the
                                           published table; a cell the lab never
                                           ran raises instead of returning a value
    EI/UCB/PI reach ~0.98 of the ceiling    bench/results_full.json, 5 seeds each
    the surrogate is the bottleneck         held-out Spearman and top-10 recovery
                                           as a function of training-set size
    logEI underperforms plain EI            same budget, same seeds, same table
    the pipeline is deterministic           same seed -> same trace, asserted in
                                           the test suite
    the data is the data                    SHA-256 verified on load; a mismatch
                                           raises rather than proceeding

## Run it

    python -m pip install -e .
    python -m pspine.bench          # or: python bench/run_campaign.py

    arm     best_frac  std    worst   q90_hit  q90_mean  s/run
    random      0.939  0.061   83.29     0.80      24.0    0.0
    ofat        0.978  0.026   92.96     1.00      24.0    0.0
    fill        0.902  0.049   86.76     0.40       8.0    1.9
    ei          0.976  0.033   91.39     1.00      25.6    0.8
    logei       0.923  0.097   76.11     0.60      13.3    0.4
    ucb         0.976  0.033   91.39     1.00      24.0    0.5
    pi          0.976  0.033   91.39     1.00      27.2    0.5

`best_frac` is best-found over the table's maximum. `q90_hit` is the fraction of
seeds that ever found 90% of the ceiling within the budget; `q90_mean` is the
experiments that took.

## Why this exists, and what it is not

In machine learning you can call the objective a million times, so "brute force"
is a strategy. In a process it is not: one evaluation is a day, a batch of
material and a destroyed coupon. This package exists to hold that constraint
literally -- the unit of cost is an experiment, and the only way to spend one is
to name a cell that somebody already ran.

What it is **not**: a synthetic benchmark. Any Bayesian optimisation library
benchmarks on functions someone wrote down, which makes both the surrogate and
the acquisition unaccountable -- if the campaign wins, you cannot tell whether
the landscape was easy or the method was good. Here the landscape is a table of
real yields, so the oracle line is real and losing to it is a real loss.

The honest weakness, stated plainly: this is **one** table, in chemistry, with
four categorical factors and a smooth response. A surrogate that ties
one-factor-at-a-time here is not evidence that it will tie it on a 30-factor
process with hard constraints. The GP's low top-10 recovery is measured, not
hypothesised, and it is the specific thing to fix next.

## The surrogate

    k(a,b) = signal * exp( -sum_f d2_f(a_f, b_f) / (2 l_f^2) )

Additive over factors, one length scale each, fitted by marginal likelihood.
For a categorical factor the squared distance is 0 on a match and 2 otherwise,
so each factor contributes 1 or exp(-1/(l_f^2)) -- a factor whose length scale
collapses to the floor is a factor the data says does not matter. This is the
same statement contact-sid makes about a robot joint that cannot excite gravity.

Kernels are built in **row chunks** (fit at most 300 cells, predict 192 cells at
a time). That is not a micro-optimisation: the process that produced these
numbers runs under a hard heap ceiling of a couple of MB, so a full 4599x4599
matrix is not merely slow, it is impossible. Peak allocation is a few hundred KB.

## Fuel

    pspine/_datasets/buchwald/data_table.csv   4599 rows, 1.2 MB, SHA-256 pinned
    source: doylelab/rxnpredict, MIT, (c) 2017 Ahneman, Estrada, Lin, Dreher, Doyle

Columns: `base`, `ligand`, `aryl_halide`, `additive` -> `yield` (%). 564 rows are
an exact zero, and a zero here is "no product detected" -- a censored reading,
not a measurement of 0.0. It is kept as a row because it cost a real experiment,
and that censoring is the most likely reason log-scaled EI misbehaves.

## Related, by the same author

- [autoforge](https://github.com/ZhangYangyi03/autoforge) -- a tool's fitness, until an oracle outside the tool agrees
- [agentic-eda](https://github.com/ZhangYangyi03/agentic-eda) -- a circuit's area, until equivalence to the reference netlist is proven
- [contact-sid](https://github.com/ZhangYangyi03/contact-sid) -- a contact model, until the recorded excitation says which parameters are identifiable
- [craftagent](https://github.com/ZhangYangyi03/craftagent) -- a machine's drift, until it is recovered against a planted ground truth
- [debt-verify](https://github.com/ZhangYangyi03/debt-verify) -- a debt clause decision, until it survives the published revision record
