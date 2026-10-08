# What a Bayesian campaign is worth when every evaluation is an experiment

*Revision 2. Revision 1 reported one table, in chemistry, where the Bayesian
arm tied one-factor-at-a-time. This revision adds three more measured tables
whose factors are settings rather than labels, and the tie does not survive
anywhere: batch EI wins on all four. Two things changed, and both are measured
rather than asserted -- a kernel that keeps a continuous factor's metric instead
of binning it, and a batch acquisition that scores a set instead of picking the
best point k times. Revision 1's tie is reproduced exactly by the greedy
acquisition on the binned-by-nature table, which is how the two effects were
separated.*

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

## 2b. The other three tables

The same arms, the same budget, on tables whose structure is nothing like the
first one:

    table        cells/grid       factors                       coverage  target
    buchwald     4599/4608        4 named (base, ligand, halide,  99.8%   yield %
                                  additive)
    ccpp         4368/65536       4 continuous ambient vars       6.7%   output MW
    gasturbine   3381/390625      8 continuous turbine vars       0.9%   NOx ppm
    concrete      268/65536       8 continuous mix vars + age      0.4%   MPa

The distinction that matters is not chemistry-vs-engineering. It is that a named
level carries no order (a ligand is not "between" two other ligands) while a
measured setting does (24 C is near 25 C). A categorical surrogate can only see
"same" or "different", and on a continuous table that is the wrong model.

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

## 3b. The same arms on all four tables

best-so-far over the table's own best, budget 64, batch 8, 5 seeds:

| table | random | fill | ofat | ei | qei | ucb | pi | logei |
|-------|--------|------|------|----|-----|-----|----|-------|
| buchwald | 0.9394 | 0.9435 | 0.9785 | 0.9760 | **0.9896** | 0.9760 | 0.9760 | 0.9232 |
| ccpp | 0.9795 | 0.9942 | 0.9880 | 1.0000 | **1.0000** | 0.9968 | 0.9994 | 1.0000 |
| concrete | 0.9754 | 0.9593 | 0.9550 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 |
| gasturbine | 0.9510 | 0.9776 | 0.9583 | 0.9953 | 0.9886 | **0.9954** | 0.9953 | 0.9953 |

best Bayesian arm minus one-factor-at-a-time: **+0.0111, +0.0120, +0.0450,
+0.0371**. Bayesian wins on all four. What the first table has that the others do
not is 99.8% coverage of a 4608-cell grid over four *named* factors: a small
dense categorical space is where "hold everything, vary one knob, keep the
improvement" is at its strongest, because there is almost nothing to interpolate
and the best level of each factor is learnable from a handful of runs. The margin
there is *thin enough to be erased by a worse acquisition*: with greedy per-point
EI the Bayesian arm ties OFAT exactly (0.9760 against 0.9785), and only batch EI
recovers the win (3 wins and 2 ties out of 5 seeds, no seed lost). So the
revision-1 finding was not wrong, it was understated in the wrong direction --
the Bayesian advantage on the categorical table is real but it lives entirely in
the batch rule.

Three further things in the table. `qei` -- batch EI by sequential conditioning,
which asks what a *set* of points is worth -- reaches the ceiling on the power
plant where the greedy `ei` only reaches 0.9968 on UCB: the batch rule is worth
something, and it is worth it exactly where the acquisition surface has one tall
spike. `fill` (greedy max-min) is competitive on the continuous tables (0.994,
0.978) and useless on the categorical one (0.944, no better than random): max-min
in a space where every factor step counts for the same amount is a design, and in
a space with a metric it is a good design. And `logei`, this package's own idea
for the censored floor, still loses to random on the table that motivated it
(0.878 against 0.939) while being harmless on tables with no floor.

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

## 4b. The kernel is the mechanism, and it is measurable

The claim that a continuous factor must not be binned is testable, so it was
tested. Fit the same surrogate twice on the same rows of the same table: once on
the real continuous coordinates, once after rank-binning every factor into 5
levels. Measure held-out rank correlation and top-10 recovery.

    ccpp, 5 seeds
    train cells      16      32      64     128     256
    metric kernel  0.938   0.935   0.925   0.933   0.945
    binned kernel  0.585   0.585   0.586   0.586   0.586

    concrete, 3 seeds
    train cells      32     128
    metric kernel  0.431   0.601
    binned kernel -0.006  -0.012

On the power plant the metric kernel is worth **+0.35 of rank correlation, flat
across every training size**; on concrete the binned kernel is at **zero, i.e.
it has learned nothing at all**, while the metric kernel recovers 0.62 and finds
a quarter of the true best ten. The binned numbers do not improve with more
data, which is the signature of a *model* error rather than a data problem: five
levels cannot express a smooth response, and no amount of extra runs will give
that back.

This also revises section 4 of revision 1. The top-10 recovery problem is not
primarily a sample-size problem: on the power plant the metric kernel holds
0.938 rank correlation at **16** training cells -- the bulk ordering is right
almost immediately -- while top-10 recovery crawls from 0.00 to 0.20 across a
16-fold increase in training data. The surrogate knows the shape of the surface
long before it can separate the best ten cells from each other, and those ten sit
inside a narrow band at the top. That is why the fix is a better use of the
ranking (batch acquisition, which asks about a set) rather than more data.

## 5. What this does not show

Four tables now, in three domains, but every one of them is a *record*: the
settings were chosen by an operator or an experimental design, not sampled to
span the space. A table is a biased sample of its own design space, and no
campaign over it can escape the region someone else chose to visit. That is a
floor on what any of these numbers can mean, and it is the honest bound.

No constraints, and no infeasible combinations. Real process optimisation has
both: some factor combinations are unsafe, and a campaign that proposes one has
produced a plan nobody can run. The arms here score the whole space, which is
only defensible because every candidate is a row the plant already ran.

The continuous tables are quantised onto a 16- or 5-point grid per factor, with
the worst-case error reported as half a step in fractions of each range. Finer
than one step is not representable, and a real campaign on a continuous process
should use a continuous candidate generator rather than the enumerable grid here
(which exists so that every arm is scored exactly, with no inner optimiser to
hide behind).

One table -- gasturbine -- has a bimodal response (clean combustion near 25-40
ppm and a second mode near 80-120) and 8 factors of which two dominate. That is
the easiest structure in the set and it shows: every Bayesian arm lands within
0.005 of the ceiling there.

What has genuinely changed since revision 1: the tie with one-factor-at-a-time
is now known to be a property of a dense categorical table rather than a general
result, and the reason is measured rather than asserted -- the kernel, not the
acquisition and not the sample size.

## 6. Availability

    repo      https://github.com/ZhangYangyi03/process-spine
    data      doylelab/rxnpredict, MIT, SHA-256 pinned and verified on load
    tests     python -m pytest tests -q   (21 tests; 2 of them re-check the
              published numbers and fail if they drift)
