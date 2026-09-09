# What a drift threshold measures, and why nothing here pages

## Status

Accepted. The distribution monitor ships and reports. It is not wired to a pager, and the
measurement below is the reason.

## Context

Three different things in this repo get called skew, and only one of them moves.

The Spark job and the reference implementation are compared cell for cell on every run.
That has never once differed. `tests/run_with_spark.py` reports 21,000 cells and 0
differing, and the corpus scale version in `scripts/oracle_check.py` is the same answer at
a larger size. A band around a signal that has never moved is degenerate: it either refuses
to judge or it fires on everything. So implementation skew is held as a constant by
`drift.constant_agreement`, and any disagreement at all is the alert.

The stored vector against what the offline pipeline says at request time does move, and it
moves a lot. That is the population every number below is measured on.

The third is a feature's own distribution moving between one window of traffic and the
next, which is what a drift monitor usually means.

The statistic is the population stability index, over bins fitted on a reference window.

    PSI = sum over bins of (c - r) * ln(c / r)

It is here because it is what everybody uses. The two thresholds everybody uses with it
are 0.1 and 0.25, and those turned out to be the thing worth measuring rather than the
thing to copy.

## The algebra was wrong and the measurement is what said so

`predicted_null_mean` first returned `(k - 1) / (2n)`. That is the standard result: twice a
likelihood ratio statistic is asymptotically chi square on `k - 1` degrees of freedom, so a
one directional divergence has that mean.

This index is not one directional. Expanding it,

    sum (c - r) ln(c / r)  =  sum c ln(c / r)  +  sum r ln(r / c)

which is the divergence in both directions added together. The answer is `(k - 1) / n` and
the first version was exactly half of it.

Nothing about reading the code says which of those two it is. What said so was the measured
null coming back at almost exactly twice the prediction on all seven features at once. A
multinomial with a known bin count then agreed with `(k - 1) / n` at five bin counts and two
window sizes. A factor of two is not cosmetic here. A threshold set from the wrong form sits
at half the level a clean window produces on its own.

Measured with `scripts/drift_probe.py`, at a window of 200 rows:

| feature | bins | distinct | null mean | predicted | null q95 |
|---|---|---|---|---|---|
| txn_count_1h | 2 | 17 | 0.0055 | 0.0050 | 0.0193 |
| txn_count_24h | 7 | 36 | 0.0292 | 0.0300 | 0.0621 |
| amount_sum_1h | 2 | 12,880 | 0.0053 | 0.0050 | 0.0222 |
| amount_max_24h | 9 | 14,836 | 0.0424 | 0.0400 | 0.0885 |
| amount_mean_7d | 10 | 251,382 | 0.0452 | 0.0450 | 0.0828 |
| distinct_merchants_1h | 2 | 16 | 0.0047 | 0.0050 | 0.0161 |
| distinct_merchants_7d | 7 | 30 | 0.0298 | 0.0300 | 0.0579 |

The upper quantile is measured rather than taken from a chi square table, because a
critical value written from memory is how a fabricated constant reaches a fixture.

## A threshold with no window size attached is a statement about traffic volume

The same null at a range of window sizes. Nothing has drifted in any of these.

| feature | 50 | 100 | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|
| txn_count_1h | 0.0889 | 0.0345 | 0.0193 | 0.0090 | 0.0031 | 0.0018 |
| txn_count_24h | 0.3064 | 0.1335 | 0.0704 | 0.0242 | 0.0121 | 0.0066 |
| amount_sum_1h | 0.0589 | 0.0374 | 0.0222 | 0.0089 | 0.0033 | 0.0018 |
| amount_max_24h | 0.3127 | 0.1628 | 0.0778 | 0.0309 | 0.0157 | 0.0079 |
| amount_mean_7d | 0.3739 | 0.1937 | 0.0858 | 0.0339 | 0.0171 | 0.0085 |
| distinct_merchants_1h | 0.0838 | 0.0345 | 0.0193 | 0.0069 | 0.0044 | 0.0021 |
| distinct_merchants_7d | 0.2839 | 0.1391 | 0.0658 | 0.0254 | 0.0131 | 0.0059 |

Five of seven features have a 95th percentile above 0.10 at a window of 50 rows with
nothing wrong. At a window of 2,000 every one of them is under 0.01, so the same threshold
is somewhere between deafening and deaf depending only on how much traffic arrived.

The bin count is the other half of it, and it is not the bin count anybody asked for. Ten
bins were requested and `txn_count_1h` got two, because it is a small integer where most
rows hold zero and quantile edges collapse onto each other. Since the null scales with `k`,
a per project threshold and a per feature threshold are different numbers.

## The threshold is correctly calibrated and the assumption under it is wrong

A threshold at the measured 95th percentile of the null should fire on about 5 percent of
clean windows. Held out consecutive windows of real traffic the fit never saw, 979 of them:

| feature | threshold | in time order | shuffled | ratio |
|---|---|---|---|---|
| txn_count_1h | 0.0227 | 0.2727 | 0.0398 | 6.8x |
| txn_count_24h | 0.0664 | 0.0582 | 0.0470 | 1.2x |
| amount_sum_1h | 0.0190 | 0.2084 | 0.0572 | 3.6x |
| amount_max_24h | 0.0783 | 0.0409 | 0.0531 | 0.8x |
| amount_mean_7d | 0.0924 | 0.1399 | 0.0368 | 3.8x |
| distinct_merchants_1h | 0.0227 | 0.2962 | 0.0398 | 7.4x |
| distinct_merchants_7d | 0.0693 | 0.5832 | 0.0255 | 22.8x |

The shuffled column is the control and it is the only column where a fire is certainly a
false one. Shuffling the whole table before splitting makes the reference and every window
the same distribution by construction, so the only difference between the two columns is
time order. On the shuffle the threshold fires at 0.0255 to 0.0572, which is the 5 percent
it was built for. In time order the same threshold on the same data fires at up to 0.5832.

That matters for how the in time order column is described. Those windows are later
traffic, and nothing says later traffic is free of real drift, so that column is not a
false positive rate. It is what a monitor would page for. Which is the number that decides
whether to build the pager.

## Decision

Nothing here pages.

The monitor reports, per feature, with the threshold and the source of the threshold
attached. `Monitor.source` exists so that a threshold copied from a rule of thumb and a
threshold measured off a null cannot be confused in a report.

Wiring a pager to the only threshold that has a measured false positive rate would page on
between a twentieth and a half of all windows, and a guardrail that fires that often gets
switched off, at which point it protects nothing. Raising it to 0.25 makes it quiet, at
0.0000 to 0.0194 held out, and quiet by being far above anything this corpus produces
rather than by separating anything.

The fix is named rather than built: the null has to be resampled in a way that keeps time
structure, which is a moving block bootstrap over consecutive runs of rows rather than
independent draws. That would raise the threshold to whatever real windows do on their own
and the fire rate would mean something. It is a day of work and it is the first thing this
monitor needs.

## The consequence nobody would guess

The distribution monitor is blind to the failure this project actually has.

Served against offline, over the same 195,989 requests, so the request weighting is present
on both sides and cancels:

| feature | index | over 0.10 | over 0.25 | cells wrong | share |
|---|---|---|---|---|---|
| txn_count_1h | 0.0012 | no | no | 54,526 | 0.2782 |
| txn_count_24h | 0.0030 | no | no | 157,299 | 0.8026 |
| amount_sum_1h | 0.0002 | no | no | 60,900 | 0.3107 |
| amount_max_24h | 0.0022 | no | no | 185,106 | 0.9445 |
| amount_mean_7d | 0.0034 | no | no | 193,926 | 0.9895 |
| distinct_merchants_1h | 0.0012 | no | no | 53,882 | 0.2749 |
| distinct_merchants_7d | 0.0005 | no | no | 63,850 | 0.3258 |

769,489 of 1,371,923 cells are wrong. That is 0.5609. And 16,357 of 195,989 requests get
the opposite decision from the served vector to the one the offline answer calls for, a
rate of 0.0835. The largest index any feature reaches is 0.0034, against a rule of thumb for
moderate drift of 0.10.

`amount_mean_7d` is the clearest case. It is wrong on 98.95 percent of requests and its
distribution index is 0.0034. Checked directly rather than inferred, the mean of the served
column is 32.8343 against 32.8599 for the offline one, and the standard deviations are
18.8082 and 19.2903.

The reason is that staleness is a per row error and not a distributional one. A stale
seven day mean is still a perfectly ordinary seven day mean. It is attached to the wrong
request. The marginal distribution barely moves and every individual row is wrong.

So a distribution monitor cannot be the skew detector for a feature store. The per row
comparison in `scripts/staleness_probe.py` is the one with the signal in it, and the
distribution monitor's real job is catching an upstream change to the feed. Both ship, and
the README says which is which, because a report showing seven green distribution rows over
a store wrong on 56 percent of its cells is worse than no report.

## The reweighting artefact, which is loud and is not an error

Measured at the load moment, where the store agrees with the reference on all 28,000 cells
and staleness is exactly zero:

| feature | index | over 0.10 | over 0.25 |
|---|---|---|---|
| txn_count_1h | 0.0490 | no | no |
| txn_count_24h | 0.3096 | yes | yes |
| amount_sum_1h | 0.0458 | no | no |
| amount_max_24h | 0.1466 | yes | no |
| amount_mean_7d | 0.0092 | no | no |
| distinct_merchants_1h | 0.0490 | no | no |
| distinct_merchants_7d | 0.1837 | yes | no |

The store holds one row per entity and the traffic reads a busy account far more often than
a quiet one, so the same store has two distributions depending on whether entities or
requests are counted. `txn_count_24h` reads 0.3096, above the rule of thumb for significant
drift, on a store that is exactly correct.

A monitor fitted on the stored population and pointed at request traffic therefore fires
forever, on the first day, with nothing wrong. Profile on the population the monitor will
judge.

## The epsilon nobody thinks about

A bin the current window emptied needs `ln(0)`. The usual fix substitutes a small share.
One real held out window, priced four ways:

| feature | excluded | 1e-3 | 1e-4 | 1e-6 |
|---|---|---|---|---|
| amount_sum_1h | 0.0105 | 0.4666 | 0.7008 | 1.1621 |

A factor of 111 between the smallest and the largest, and the data has no say in which one
is the answer. So an emptied bin is counted and reported as its own verdict rather than
priced. `BIN_EMPTIED` fires regardless of the index, because a category disappearing is a
different event from a distance growing and a distance cannot express it without a constant
somebody picked.

## What this costs

The verdict is coarser. A monitor that returns `moved` or `bin_emptied` cannot say how far
something moved in a way that survives a comparison across windows of different sizes, and
that is the honest position given the table above rather than a limitation of the code.

`MIN_WINDOW` is 30 and it is a floor rather than a fitted number. It was chosen before any
threshold was measured, on the grounds that the null at ten bins and twenty rows is 0.45,
which is above any level anyone would alert at. A window under it is `too_few`, which is not
a drift verdict and does not count in a fire rate.
