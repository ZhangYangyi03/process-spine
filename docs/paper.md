# What a Bayesian campaign is worth when every evaluation is an experiment

*Working note. All numbers are reproducible with `python bench/run_campaign.py`
and `python bench/gp_fidelity.py`; the tables they print are the ones below.*

## 1. The claim under test

Bayesian optimisation is usually sold with a number from a benchmark function:
"finds the optimum in 40 evaluations where random needs 400." That comparison is
made on an objective someone wrote down, which makes the result
uninterpretable in a specific way -- when the campaign wins, nothing in the
setup says whether the landscape was easy or the method was good. Every arm is
also allowed to query a point the *function* can produce, including points no
experimenter would ever run.

This note makes the same claim where it can be falsified. The objective is a
table of 4599 recorded experiments (Ahneman et al., *Science* 2018, the
Buchwald-Hartwig amination set distributed with `doylelab/rxnpredict`). Four
categorical factors -- base (3), ligand (4), aryl halide (16), additive (24) --
give 4608 cells, of which 4599 were actually run. A campaign may query a cell
only if the table contains it, and the cost of a query is one experiment.

The consequence is that the oracle is real. A campaign cannot beat looking at
the finished table, so "how close to the best row" is a meaningful score, and
"lost to one-factor-at-a-time" is a real loss.

## 2. Setup

    budget          64 experiments, in batches of 8, 8 used as the initial design
    seeds           5
    arms            random, greedy space-filling, one-factor-at-a-time,
                    EI, log-EI, UCB (kappa=2), PI
    surrogate       additive categorical GP, per-factor length scale, fitted by
                    marginal likelihood after every batch
    reported        best-so-far as a fraction of the table maximum (100.0),
                    plus whether each seed ever reached 90% of it and when

## 3. Result

| arm | best/ceiling | sd | worst | reach 90% | exp. to 90% |
|-----|--------------|----|-------|-----------|-------------|
| random  | 0.939 | 0.061 | 83.29 | 0.8 | 24.0 |
| fill    | 0.902 | 0.049 | 86.76 | 0.4 | 8.0 |
| ofat    | 0.978 | 0.026 | 92.96 | 1.0 | 24.0 |
| ei      | 0.976 | 0.033 | 91.39 | 1.0 | 25.6 |
| ucb     | 0.976 | 0.033 | 91.39 | 1.0 | 24.0 |
| pi      | 0.976 | 0.033 | 91.39 | 1.0 | 27.2 |
| logei   | 0.923 | 0.097 | 76.11 | 0.6 | 13.3 |

Three things in that table are worth more than the ranking.

**The Bayesian gain over random is small and consistent: +0.037 of the ceiling.**
EI, UCB and PI land on 0.976 +- 0.033 against random's 0.939 +- 0.061. This is a
real gain -- every Bayesian arm's worst seed beats random's worst seed -- but it
is a gain of a few percent, not an order of magnitude, and on 64 experiments.

**The Bayesian arms do not beat one-factor-at-a-time.** Coordinate ascent with
no model at all reaches 0.978 +- 0.026, which is the best mean and the best worst
case in the table. It got there by doing the thing process engineers have done
for a century: from the best run so far, vary one factor, keep the improvement.
Any claim that a surrogate is *necessary* on this landscape is false as stated.

**Two arms lose to random.** Greedy space-filling (0.902) is a real strategy
that is simply wrong for a 4608-cell space at this budget: covering the space
spends the whole budget on coverage and finds nothing special. Log-scaled EI
(0.923) is more interesting, because it is this package's own idea failing.
The reasoning behind it was sound: 562 of the 4599 cells are an exact zero, and
an exact zero is a *censored* reading ("no product detected"), not a measurement
of 0.0, so EI on the raw scale should keep proposing corroborating zeros. It
does not help; it hurts, and the mechanism is visible in the trace -- log-scaling
compresses the objective's upside as well as its floor, so the arm becomes
conservative and stalls.

## 4. Why: the surrogate, not the acquisition

A campaign result is a joint statement about the surrogate and the acquisition.
Separating them is the point of this section. Train on a random subset of cells,
predict the rest, and measure *rank* correlation (the acquisition only orders
candidates; a model with the right order and the wrong scale is fine).

| train cells | Spearman | Pearson | top-10 recovery |
|-------------|----------|---------|-----------------|
| 16  | 0.205 | 0.126 | 0.000 |
| 32  | 0.463 | 0.392 | 0.000 |
| 64  | 0.578 | 0.495 | 0.000 |
| 128 | 0.672 | 0.623 | 0.000 |
| 256 | 0.794 | 0.741 | 0.100 |

At the budget this note uses -- 64 experiments, of which 8 are the initial design
-- the surrogate sits at roughly 0.5 Spearman and recovers **none** of the true
top ten. That is the regime in which EI degenerates: expected improvement is
driven by the posterior mean wherever the posterior spread is broad and similar
across candidates, and the posterior spread is broad and similar across
*everything* at 56 training cells. "Follow the posterior mean with a bit of
optimism" and "follow your best observation, one factor at a time" are then the
same policy, which is why the fourth row of the first table ties the third.

The interesting quantity in the second table is not the Spearman column, which
looks respectable, but the top-10 column, which is zero. A surrogate can order
the bulk of the space correctly and still be unable to point at the ten best
cells, and it is the second ability the last batch of a campaign needs. This is
measurable, and it was measured, rather than argued.

## 5. What this does not show

One table, one domain, four factors, a smooth response, no constraints, no
categorical levels that are physically infeasible. A surrogate that ties
coordinate ascent here is not evidence that it will tie it on a 30-factor
process with hard constraints -- it is evidence that it ties it *here*, which is
the only claim the numbers support. The most likely reason the Bayesian arm does
not win is stated in section 4 and is fixable: an acquisition that accounts for
the *batch* exactly (the implementation here penalises duplication greedily)
and a surrogate with a stronger prior over which levels are similar would both
change the picture, and neither is in this note.

## 6. Availability

    repo      https://github.com/ZhangYangyi03/process-spine
    data      doylelab/rxnpredict, MIT, SHA-256 pinned and verified on load
    tests     python -m pytest tests -q   (21 tests; 2 of them re-check the
              published numbers and fail if they drift)
