# Design notes

Moved out of the README so the front page stays short. Back to the [README](../README.md).

## Measured

The corpus is generated and gitignored, so every figure below names the command that
rebuilds its input. Run `python scripts/generate_corpus.py` first.

```
python scripts/generate_corpus.py
  4000 accounts, 489972 transactions over 60 days, sha256 062cfda8bc58
  402 fraud transactions, rate 0.00082
  60 accounts compromised against a target of 80, 20 too quiet to carry a burst
  median 89 transactions per account, largest account 1253
  60 of 402 fraud rows open their burst, share 0.149
```

That last line is a floor on what any trailing window can catch. The first transaction of a
burst has ordinary history behind it, so no count or sum over prior events separates it
from a normal purchase.

### The two readings of one window

```
python scripts/window_probe.py
  txn_count_24h         calendar_day/left    windows 0.773   values 0.647
  amount_max_24h        calendar_day/left    windows 0.773   values 0.570
  amount_max_24h        event_time/right     windows 1.000   values 0.360
  distinct_merchants_7d calendar_day/left    windows 0.762   values 0.120
```

Shares of 4,000 scored rows, against the `event_time` plus `left` reading the spec
declares. A nightly batch job and an online store reading the same declared 24 hour window
hand the model a different count on 64.7 percent of rows.

Two columns because one of them is a trap. A count over a strictly larger window always
increments, so a 1.000 there is a restatement of addition rather than a measurement. The
probe marks the cells where the two columns agree as forced by the aggregate.

### A snapshot join does not inflate the metric here. It starves it.

This one moved to the front page, in the [README](../README.md).

Both arms use the same spec and the same corpus. Same time based split and the same model
too. The only difference is when the features were computed. The snapshot arm computes one vector per account over the whole
history and joins it onto every labelled row for that account, which is what a feature
table joined on entity alone produces.

The usual warning about this mistake is that it flatters your offline number. It did the
opposite by 33 points.

The third arm is the reason that means anything. The oracle scores every row with its own
account's fraud rate in the test set, which is the best any account constant score can
reach. It reaches 0.9982, so the snapshot arm is not capped by its shape. It had that
ceiling available and missed it by 41 points, because a snapshot taken at one moment
carries nothing about a burst that happened somewhere else in time.

The version of this mistake that does inflate a metric is a snapshot taken close to the
label rather than at the end of the corpus. That arm is not built yet and there is a TODO
in the probe naming it.

Both AUCs are about a synthetic corpus. The comparison between them is about the join.

### The batch job against the reference

The Spark job is generated from the spec. It reads the frame offsets back out of
`compute.window_bounds` rather than stating them, so the two sides are not two readings of
one English sentence. `docs/adr-0003` is why.

```
python scripts/materialize.py --corpus data/transactions.csv --out /tmp/training
  wrote 489972 rows to /tmp/training in 8.08s
  fingerprints on the table: ['534217a2fc6a']
```

```
python scripts/oracle_check.py --accounts 0
  489972 rows over 4000 accounts
  18 rows share an entity and a timestamp with a sibling, largest tie 2
  4703 rows had an empty window, so the zero fallback is exercised
  reference built in 10.11s, spark arm in 10.75s

  shipped                  3429804 cells  exact 3429804  within tol 0  differing       0
  ignore_closed            3429804 cells  exact  757199  within tol 5  differing 2672600
  stretch_lower_by_a_day   3429804 cells  exact 1086612  within tol 2  differing 2343190
```

Zero differences is not a result on its own. It reads the same whether the job is right or
the two sides are being handed different rows. So the same builder produces a damaged arm
for each entry in `spark_job.DAMAGES`, one on each end of the frame, and the check fails if
any of them agrees with the reference. `ignore_closed` is the real mistake, a window join
that admits the row being scored. It moves 77.9 percent of the cells.

The tolerance column exists because a sliding window mean in Spark could add and subtract
terms where the reference sums a list from the left. On this corpus it never needed to. The
two features that carry arithmetic are the sum and the mean, that is 979,944 cells, and
every one of them is bit identical. The only cells that have ever used the tolerance are 7,
all inside the damaged arms.

### How the columns are added changes the plan

Seven features, one entity, one ordering. The obvious way to build that is a `withColumn`
per feature, and it gives a plan with seven Window operators and seven Sorts of the same
489,972 rows.

```
python scripts/plan_probe.py --passes 5
  489972 rows, rows differing between the arms 0
  one_select         window  1  sort  1  exchange  1   median 2.05s  best 1.97s
  withcolumn_chain   window  7  sort  7  exchange  1   median 2.81s  best 2.74s
  chaining costs 1.37x on the median
```

Spark only groups window expressions inside a single projection, and a chain of
`withColumn` calls makes one projection per feature. Putting all seven in one `select`
collapses it to one Window and one Sort. I expected three, one per distinct frame, and the
grouping turned out to be by partition and ordering with the frames handled inside a single
operator.

Both arms come out of the same builder with one flag, so the losing side is not a rewrite
from memory. The arms are alternated after a discarded warmup, and the check that makes the
timing mean anything is the row difference, which is a set difference over whole rows in
both directions. A faster plan that computes something else is not a faster plan.

The ratio needs the passes. Three passes gave 1.32x and five gave 1.37x in the same
session, over the same rows, with only the pass count changing. The operator counts are
exact and do not move at all, which is the half of this worth quoting.

### Ties, which the corpus almost does not have

A frame boundary is exactly where a tied timestamp decides the answer, and the generated
corpus has 18 rows in a tie out of 489,972. That is 0.004 percent, so the agreement above
says almost nothing about that case.

```
python scripts/oracle_check.py --ties
  3000 rows over 20 accounts
  1640 rows share an entity and a timestamp with a sibling, largest tie 7
  28 rows had an empty window, so the zero fallback is exercised
  shipped                    21000 cells  exact 21000  within tol 0  differing     0
  ignore_closed              21000 cells  exact  5425  within tol 0  differing 15575
  stretch_lower_by_a_day     21000 cells  exact  7896  within tol 0  differing 13104
```

### The online store, through a real Redis

One vector per entity, computed at a single moment, under a key that carries the spec
fingerprint. The server is a `redislite` 6.2.14 started inside the process that talks to
it, which is a real Redis and not an in memory imitation.

```
python scripts/store_check.py --accounts 0
  489972 rows over 4000 entities, spec txn_fraud at 534217a2fc6a
  wrote 4000 entities in 0.24s, 16743 entities per second
  the store grew from 988128 to 2111432 bytes, 281 bytes per entity

  shipped                    4000 entities    28000 cells  exact   28000  differing       0
  load_a_day_early           4000 entities    28000 cells  exact   14846  differing   13154
  six_significant_digits     4000 entities    28000 cells  exact   24981  differing    3019

  lookup over 12000 calls  p50 0.162 ms  p95 0.243 ms  p99 0.286 ms  max 0.815 ms
```

Same shape of argument as the batch job. Zero differences is not a result on its own, so
the same loader builds a damaged arm for every entry in `online.DAMAGES` and the check
fails if any of them round trips cleanly. `six_significant_digits` is the interesting one.
It is what a formatted encoder costs, because six digits is what somebody reaches for when
they want the stored value to look tidy in `redis-cli`, and it silently moves 3,019 of
28,000 cells.

The memory figure reproduced to the byte on four consecutive runs, so 281 bytes for seven
numbers is a real number and most of it is Redis rather than the data. That matters more
than it looks. At four thousand accounts the whole store is two megabytes and at forty
million it is eleven gigabytes, and the feature set is small. Nothing here does anything
about that.

The latency is a floor and not a serving number. There is no network, no model and no
service in it. It is one `HGETALL` and a decode, measured on loopback with the server in
the same process, and the whole point of having it now is that the p99 budget for the day
this becomes an HTTP endpoint is 150 ms. Four runs of 12,000 calls put p50 between 0.157
and 0.175 ms and p99 between 0.279 and 0.332 ms, so the third decimal is noise and the
figure to carry forward is a fifth of a millisecond.

### A spec change, and what the key buys

```
python scripts/store_check.py --accounts 0
  fingerprint in the key: yes    answered     0  missed  4000  cells wrong       0 of       0  share no cells
  fingerprint in the key: no     answered  4000  missed     0  cells wrong    2325 of   28000  share    0.083
```

The store is loaded under the shipped spec and then asked for one whose windows moved by an
hour. With the fingerprint in the key every request misses, which is the answer a service
can act on. Without it every request is answered from a definition nobody is serving.

The share is why this is a key problem rather than a validation problem. 8.3 percent of the
cells are wrong, so 91.7 percent of the answers happen to be right, and the payload
validation cannot see any of it because a window length is not a field name. A change that
broke every cell would be found in an afternoon.

### The store is wrong the moment it finishes loading

```
python scripts/staleness_probe.py --accounts 0
  489972 rows over 4000 entities, store loaded at 1754664876 which is 90% through the corpus
  48998 transactions arrive after that, spanning 144.2 hours
  control at the load moment: 28000 cells, 0 differing, 0 misses

  age           requests      cells  differing    share
  under 1h           459       3213        446    0.139
  1h to 6h          2736      19152       6855    0.358
  6h to 24h         5047      35329      17069    0.483
  over 24h         40756     285292     159359    0.559
```

Every transaction after the load moment is a request the service would have answered, and
the training table holds a row for it computed at its own timestamp. So each one is scored
against the vector the store would have handed it. This is the training and serving skew
the project is about, with no model in the way.

Within the first hour 13.9 percent of cells already disagree. The control is what makes
that a number about age: scored at the load moment itself the store is exact on all 28,000
cells, so the table is not measuring a broken comparison.

```
feature                  under 1h   1h to 6h  6h to 24h   over 24h
txn_count_1h                0.176      0.301      0.262      0.272
txn_count_24h               0.166      0.535      0.754      0.807
amount_sum_1h               0.187      0.336      0.288      0.302
amount_max_24h              0.054      0.325      0.770      0.960
amount_mean_7d              0.196      0.653      0.915      0.995
distinct_merchants_1h       0.172      0.300      0.260      0.271
distinct_merchants_7d       0.020      0.056      0.133      0.302
```

The long windows drift hardest and that is the opposite of what I expected. `amount_mean_7d`
is wrong on 99.5 percent of requests more than a day old and `txn_count_1h` on 27.2 percent.
A mean over seven days moves when any transaction arrives, and a count over the last hour is
zero for a quiet account both before and after, so the short window features agree by both
being empty rather than by being fresh. Agreement at zero is not accuracy.

### The loader cannot write an entity it has never seen

An entity with no event strictly before the load moment gets no row, and a lookup for it
raises rather than returning zeros. `compute.aggregate` returns 0 for an empty window, so a
written cold start is indistinguishable from an account that has stopped spending.

```
python scripts/staleness_probe.py --accounts 0 --at 0.10
  489972 rows over 3940 entities, store loaded at 1750519103 which is 10% through the corpus
  1371 requests were for an entity the loader never wrote, 1240 of them had real history by then
```

Loaded a tenth of the way through, the store holds 3,940 of the 4,000 accounts. 1,371 later
requests are for one of the other 60, and 1,240 of those arrive with real history behind
them. Without the filter every one is answered with a vector of zeros and nothing raises.

The boundary is the part worth stating. Every window is closed on the left, so an event at
exactly the load moment is inside none of them, and an entity whose first event lands there
would still be written as zeros. The comparison was `>` when it was first written and a
mutation pass is what asked, because no fixture had put an entity's first event on the
boundary.

### The service, and the model it serves

The scorer is a logistic regression held as a coefficient per feature and an intercept.
`scripts/train_model.py` fits it with scikit-learn and writes the JSON. Nothing on the
serving path imports scikit-learn, so a latency number is a statement about the feature
fetch rather than about somebody's inference runtime. `docs/adr-0001` argues that choice.

```
python scripts/train_model.py --accounts 0 --out models/txn_fraud.json
  489972 rows over 4000 accounts, spec txn_fraud at 534217a2fc6a
  split at 1753627753, 342980 training rows and 146992 test rows
  273 fraud in training, 129 in test
  train AUC 0.9216
  test AUC  0.9226
  largest disagreement with scikit-learn over 489972 rows: 1.998e-15
```

That last line is the reason the script exists. scikit-learn scores through numpy and the
service scores through `math.exp`, so they are two implementations of one function, and
this repo has been caught more than once by two implementations that were each correct and
disagreed. The fit is not written out until every training row has gone through both.

The 0.9226 is not the 0.9226 in the leakage probe above, and that is worth saying because
they round the same. The probe standardises its inputs and this does not, so the fitted
coefficients are entirely different. The two test AUCs are 0.922574 and 0.922591, which is
339 reordered pairs out of 18,945,327. Not standardising is deliberate. A scaler is a
second artefact between training and serving, and the point of a linear model here is that
a feature arriving with the wrong scale moves the log odds by the coefficient times the
difference and nothing else.

The service is `featurestore/service.py` and it imports no web framework.
`featurestore/api.py` is the FastAPI edge and makes no decisions. Validation lives in the
first file and so does what a miss means. So does what an age means and so does the shape of
the log record. That is why every one of them is reachable from a suite that installs one
requirements file.

```
python scripts/serve_probe.py --accounts 0
  3000 requests through the app, statuses {200: 3000}
                                  p50      p95      p99      max
  through the app, ms           1.698    2.168    2.486    3.564
  inside the service, ms        0.250    0.381    0.486
  the edge is 5.1x the service on the p99
```

None of those is worth a third decimal, and the range around them is wider than one
session can tell you. Four runs in one session put the p99 through the app between 2.362
and 3.368 ms, the service half between 0.467 and 0.623, and the ratio at 4.9 to 5.5. Four
more runs on a different day gave 2.377 to 2.455 through the app and 0.374 to 0.387 inside
the service, for a ratio of 6.2 to 6.4. Same machine and same corpus.

The app half landed inside its earlier range and the service half did not, so the ratio
did not either. Read that as the shape rather than as a number. The edge costs several
times what the decision logic does, and any tighter claim is a claim about a session.

The `max` column is the least stable thing in the block. It is one request out of 3000 and
it has read 3.564 ms on one day and about 53 ms on another.

The budget for this project is 150 ms. The store alone was measured at a p99 of 0.286 ms
with no HTTP in it, so the framework and the socket are most of what a request costs and
the whole thing is still well inside budget. This is a single process on loopback with no
network, so it is a floor and not a serving number.

### Running the service, and the four things it reads from the environment

`featurestore/api.py` exposes a factory rather than a module level app, so importing the
module does not try to reach Redis.

```
uvicorn --factory featurestore.api:app
```

`from_environment` assembles the service the way a container would. It raises rather than
degrading. A process with no model, or with a model fitted on a different spec, would
answer requests with numbers nobody can account for.

| variable | default | what it decides |
|---|---|---|
| `FS_SPEC` | `specs/txn_fraud.yml` | the feature set and therefore the online keyspace |
| `FS_MODEL` | `models/txn_fraud.json` | the coefficients |
| `FS_REDIS_URL` | `redis://127.0.0.1:6379/0` | where the online store is |
| `FS_MAX_AGE_S` | unset | the staleness refusal in `service.Policy` |

`FS_MAX_AGE_S` is the one to read twice. Unset is the shipped default and it means a vector
of any age is served with its age attached. That is a decision rather than an omission and
the section below is the argument for it.

The Redis default is port 6379 because that is what a real Redis listens on. Every
`redislite` server started inside this repo uses a different port, so a local bring-up needs
`FS_REDIS_URL` pointed at whichever one is running. `scripts/store_check.py` is 6399 and
`tests/run_with_api.py` is 6403. `scripts/serve_probe.py` defaults to 6404 and
`scripts/load_probe.py` defaults to 6405.

### An age threshold refuses far more than it saves

A decision flip is a request where the stored vector and its own point in time vector land
on opposite sides of the model threshold. It is the only error measure here with a
consequence attached.

```
python scripts/serve_probe.py --accounts 0
  max age        refused   share   served   flips    rate       95% interval refused rate
  no threshold         0   0.000    48998    3848  0.0785   0.0762 to 0.0810          n/a
  1h               48539   0.991      459      17  0.0370   0.0217 to 0.0586       0.0789
  6h               45803   0.935     3195     258  0.0808   0.0715 to 0.0907       0.0784
  1d               40756   0.832     8242     605  0.0734   0.0679 to 0.0792       0.0796
```

The last column is the flip rate among the requests each threshold threw away. At six hours
the service refuses 93.5 percent of its traffic and keeps requests failing at 0.0808 against
0.0784 for the ones it discarded. At a day it refuses 83.2 percent and the interval on what
it keeps covers the number it started from.

A flip is defined by the model threshold, which ships at 0.5 as a default nobody chose, so
the same sweep runs at four others.

```
  threshold           all   1h served   6h served   1d served
  0.10             0.2362      0.0763      0.1468      0.1924
  0.25             0.2673      0.0959      0.2081      0.2338
  0.50             0.0785      0.0370      0.0808      0.0734
  0.75             0.0238      0.0131      0.0232      0.0228
  0.90             0.0092      0.0065      0.0075      0.0091
```

The level moves by a factor of 29, so 0.0785 is a fact about staleness at a threshold of
0.5 rather than a fact about staleness. What holds across the range is the exchange rate.
At 0.25, the friendliest row a threshold gets, six hours refuses 93.5 percent of traffic to
remove 22 percent of the flips.

So `max_age_s` ships as None. The service attaches the age to every scored response and
does not judge it. `docs/adr-0005` is the argument and the intervals are Clopper Pearson,
because 17 of 459 printed as 0.0370 next to 0.0785 invites a reader to treat a gap as
settled that the count does not license.

### A feature goes stale while nothing happens

Age is standing in for whether anything arrived. Splitting the same requests on that
directly beats every threshold on both axes at once.

```
  requests split on new arrivals      requests   flips    rate       95% interval
  nothing arrived since the load          3933     112  0.0285   0.0235 to 0.0342
  something arrived since the load       45065    3736  0.0829   0.0804 to 0.0855
```

Those intervals do not overlap, and the quiet set is 8.6 times the size of what the one
hour threshold serves at a lower flip rate. Then the quiet side turns out not to be clean
either.

```
  of the 3933 quiet requests the stored vector had already moved on 2685 (0.6827)
  feature                       moved   moved upward
  txn_count_1h                    364              0
  txn_count_24h                  1749              0
  amount_sum_1h                   364              0
  amount_max_24h                 1309              0
  amount_mean_7d                 1969           1103
  distinct_merchants_1h           363              0
  distinct_merchants_7d           631              0
```

More than two thirds of the vectors for entities where nothing happened were already wrong.
Every window is trailing, so it sheds rows off its back edge as the clock moves. A count of
the last hour falls as the hour it counted recedes, with nobody transacting.

Six of the seven can only fall, because removing rows from a count or a sum or a max or a
distinct count cannot raise it. The mean is the exception and it moved upward 1,103 times,
since dropping a below average transaction pushes a mean up. A streaming writer fed by
arrivals fixes the larger half of this and not all of it.

### The worker thread costs a third of the throughput and is still the right default

Every latency figure above this line came off `fastapi.testclient`, which sends one request
at a time and never opens a socket. `scripts/load_probe.py` drives a real uvicorn over real
connections instead, and the first thing it can see is the one decision the HTTP edge makes.
`featurestore/api.py` runs the blocking work on a worker thread. The comment saying why had
never been measured, because a serial harness cannot tell the two arrangements apart.

Six paired rounds per arm per level. Arms rotated so none is always first. A warmup pass
discarded and the driver's own control checked on every pass. All 96 passes kept their load
up. The comparison is a sign test over the paired rounds, and six is the smallest number of
rounds that can reach a p below 0.05, so the probe refuses five.

```
python scripts/load_probe.py --accounts 0 --workers 1,8 --requests 800 --rounds 6
python scripts/load_probe.py --accounts 0 --workers 32,64 --requests 800 --rounds 6

  medians over six rounds

  clients   offload rps   inline rps   offload p99   inline p99   throughput   p99      p
        1           827         1055          2.02         1.66       0.78x  1.21x  0.0312
        8           792         1094         16.42         9.58       0.72x  1.71x  0.0312
       32           858         1168         98.51        30.83       0.73x  3.20x  0.0312
       64           961         1332        126.87        53.24       0.72x  2.38x  0.0312
```

The worker thread lost every round at every level. So the first reading was that the change
should be reverted, and that reading is wrong.

It is a result about the store rather than about the arrangement. Redis here is redislite
over loopback, so there is almost nothing for a worker thread to overlap and the dispatch is
pure cost. Putting a wait in front of the store finds where it flips.

```
python scripts/load_probe.py --accounts 0 --workers 32 --requests 320 --rounds 6 \
    --warmup 60 --store-delay-ms 0,0.5,1,2

  32 clients, six rounds each, a sleep in front of the store

  store delay   offload rps   inline rps   ratio   rounds won        p
       0.0 ms           964         1383   0.70x       0 of 6   0.0312
       0.1 ms           921         1290   0.71x       0 of 6   0.0312
       0.2 ms          1036         1129   0.92x       0 of 6   0.0312
       0.3 ms           966         1009   0.96x       3 of 6   1.0000
       0.4 ms          1045          911   1.15x       4 of 6   0.6875
       0.5 ms          1041          847   1.23x       6 of 6   0.0312
       1.0 ms          1065          578   1.84x       6 of 6   0.0312
       2.0 ms           902          354   2.55x       6 of 6   0.0312
```

The crossover sits between 0.2 and 0.5 ms, and this store answers in about 0.12 ms with a
lookup p99 near 0.29 ms. So the deployment being measured sits on the crossover rather than
safely to one side of it.

The two costs are not symmetric. Below the crossover the thread costs about 28 percent of
throughput and stops there. Above it the loop arrangement loses without bound, because every
request in flight waits behind the one holding the loop. Half a millisecond of round trip is
any Redis that is not on this machine. `offload` therefore stays on by default, and
`docs/adr-0006` is the argument.

### What a perfect cache would be worth, which is why there is not one

The day's line says caching where it earns its place. What that needs first is a ceiling.

The cached arms replace Redis with a dict holding every key, read out of the real store so
both sides decode identical bytes. That is not a proposal. A perfect cache is a dict with
everything already in it. No eviction and no round trip and no invalidation. Nothing real
can beat it.

```
  from the same two runs as the arrangement table above

  throughput, cache against no cache inside the same arrangement

  clients   on inline   on offload
        1       1.32x        1.25x
        8       1.26x        1.35x
       32       1.24x        1.27x
       64       1.21x        1.24x
```

Twenty one to thirty two percent, as a best case. No cache is built, and the reason that
decides it is not the size of the number. The store is loaded once and never refreshed, and
the section above measures what that already costs. A cache in front of it adds a second
unbounded age with no invalidation path, on a service whose age policy ships as `None`
precisely because the first layer's staleness could not be bounded usefully. The p99 budget
is 150 ms and the shipped arrangement is at 126.87 ms at sixty four clients, which is the
only number here under any pressure, and the cache is worth 1.24x of throughput there while
the arrangement question is worth 2.38x of p99. The smaller lever, on the axis that is not
binding, bought with an error already measured.

### A drift threshold is a statement about how much traffic arrived

Rebuild every figure in this section with:

```
python scripts/drift_probe.py --json /tmp/drift.json
```

The statistic is the population stability index over bins fitted on a reference window. It
is here because it is what everybody uses. The two thresholds everybody uses with it are
0.1 and 0.25, and those turned out to be the thing worth measuring rather than the thing to
copy.

Two samples from one distribution do not give zero. The expected value under no drift at
all is `(k - 1) / n`, so the index a clean window returns depends on the bin count and on
how many rows arrived. Measured at a window of 200 rows, against that closed form:

```
feature                 bins   distinct  null mean  predicted   null q95
txn_count_1h               2         17     0.0055     0.0050     0.0193
txn_count_24h              7         36     0.0292     0.0300     0.0621
amount_sum_1h              2      12880     0.0053     0.0050     0.0222
amount_max_24h             9      14836     0.0424     0.0400     0.0885
amount_mean_7d            10     251382     0.0452     0.0450     0.0828
distinct_merchants_1h      2         16     0.0047     0.0050     0.0161
distinct_merchants_7d      7         30     0.0298     0.0300     0.0579
```

Ten bins were asked for and `txn_count_1h` got two, because it is a small integer where
most rows hold zero and quantile edges collapse onto each other. Since the null scales with
the bin count, a threshold set once for a project and a threshold set per feature are
different numbers.

The same null at a range of window sizes. Nothing has drifted in any of these:

```
feature                      50      100      200      500     1000     2000
txn_count_1h             0.0889   0.0345   0.0193   0.0090   0.0031   0.0018
txn_count_24h            0.3064   0.1335   0.0704   0.0242   0.0121   0.0066
amount_sum_1h            0.0589   0.0374   0.0222   0.0089   0.0033   0.0018
amount_max_24h           0.3127   0.1628   0.0778   0.0309   0.0157   0.0079
amount_mean_7d           0.3739   0.1937   0.0858   0.0339   0.0171   0.0085
distinct_merchants_1h    0.0838   0.0345   0.0193   0.0069   0.0044   0.0021
distinct_merchants_7d    0.2839   0.1391   0.0658   0.0254   0.0131   0.0059
```

Five of seven sit above 0.10 at fifty rows with nothing wrong, and every one is under 0.01
at two thousand. The same threshold is deafening or deaf depending only on traffic.

The closed form was wrong when it was first written, at `(k - 1) / (2n)`. That is the
standard result for a one directional divergence and this index is the divergence in both
directions added together, so the answer is twice it. Nothing in the code says which one it
is. The measured column coming back at almost exactly double the predicted one on all seven
features is what said so, and a multinomial with a known bin count then agreed with
`(k - 1) / n` at five values of `k` and two of `n`.

### The threshold is calibrated correctly and the assumption under it is not

A threshold at the 95th percentile of that null should fire on about five percent of clean
windows. Over 979 held out consecutive windows the fit never saw, and over the same rows
shuffled before splitting:

```
feature                 threshold     in order     shuffled      ratio
txn_count_1h               0.0227       0.2727       0.0398       6.8x
txn_count_24h              0.0664       0.0582       0.0470       1.2x
amount_sum_1h              0.0190       0.2084       0.0572       3.6x
amount_max_24h             0.0783       0.0409       0.0531       0.8x
amount_mean_7d             0.0924       0.1399       0.0368       3.8x
distinct_merchants_1h      0.0227       0.2962       0.0398       7.4x
distinct_merchants_7d      0.0693       0.5832       0.0255      22.8x
```

Shuffling the whole table before splitting makes the reference and every window the same
distribution by construction, so the only difference between those two columns is time
order. On the shuffle the threshold fires at 0.0255 to 0.0572, which is the five percent it
was built for. In time order, on the same rows, it reaches 0.5832.

The in order column is not a false positive rate and it is not described as one. Those
windows are later traffic and nothing establishes that later traffic is free of real drift.
It is the rate at which a monitor would page somebody, which is the number that decides
whether to build the pager.

**Nothing here pages.** A guardrail firing on between a twentieth and a half of all windows
gets switched off, and then it protects nothing. The named fix is a null that keeps time
structure, which is a moving block resample over consecutive runs rather than independent
draws. `docs/adr-0007` has the argument.

### The distribution monitor is blind to the failure this project has

Served against offline over the same 195,989 requests, so the request weighting is on both
sides and cancels:

```
feature                     index over 0.10 over 0.25  cells wrong      share
txn_count_1h               0.0012        no        no        54526     0.2782
txn_count_24h              0.0030        no        no       157299     0.8026
amount_sum_1h              0.0002        no        no        60900     0.3107
amount_max_24h             0.0022        no        no       185106     0.9445
amount_mean_7d             0.0034        no        no       193926     0.9895
distinct_merchants_1h      0.0012        no        no        53882     0.2749
distinct_merchants_7d      0.0005        no        no        63850     0.3258
```

769,489 of 1,371,923 cells are wrong. That is 0.5609. And 16,357 of 195,989 requests get
the opposite decision from the served vector to the one the offline answer calls for, a
rate of 0.0835. The largest index any feature reaches is 0.0034 against a rule of thumb of 0.10.

`amount_mean_7d` is the clearest case, wrong on 98.95 percent of requests at an index of
0.0034. Checked directly rather than inferred, the served column has a mean of 32.8343
against 32.8599 offline, with standard deviations of 18.8082 and 19.2903.

Staleness is a per row error and not a distributional one. A stale seven day mean is a
perfectly ordinary seven day mean attached to the wrong request. So a distribution monitor
cannot be the skew detector for a feature store. `scripts/staleness_probe.py` is the
comparison with the signal in it, and this one is for catching a change in the feed.

### A store that is exactly right reads as significant drift

Measured at the load moment, where the store agrees with the reference on all 28,000 cells:

```
feature                     index     null q95 over 0.10 over 0.25
txn_count_1h               0.0490     0.000030        no        no
txn_count_24h              0.3096     0.000043       yes       yes
amount_sum_1h              0.0458     0.000022        no        no
amount_max_24h             0.1466     0.000112       yes        no
amount_mean_7d             0.0092     0.000081        no        no
distinct_merchants_1h      0.0490     0.000028        no        no
distinct_merchants_7d      0.1837     0.000096       yes        no
```

The store holds one row per entity and traffic reads a busy account far more often than a
quiet one, so the same store has two distributions depending on whether entities or
requests are counted. `txn_count_24h` reads 0.3096, above the rule of thumb for significant
drift, at a moment when staleness is exactly zero. Profile on the population the monitor is
going to judge.

### An emptied bin is a verdict rather than a number

A bin the window emptied needs the logarithm of zero. The usual fix substitutes a small
share. One real held out window, priced four ways:

```
feature                excluded       0.001      0.0001       1e-06
amount_sum_1h            0.0105      0.4666      0.7008      1.1621
```

A factor of 111 across a constant the data has no opinion about. So an emptied bin is
counted and reported as `bin_emptied`, which fires whatever the index is, because a
category disappearing is a different event from a distance growing.

## Verification

```
python tests/run_all.py                          248 passed, 0 failed
python tests/run_with_spark.py                   248 passed, 0 failed, spark arm agrees
python tests/run_with_redis.py                   248 passed, 0 failed, store arm agrees
python tests/run_with_api.py                     248 passed, 0 failed, api arm agrees
python scripts/drift_probe.py                    exit 0, load moment control clean
```

Four runners because three of them reach something the first cannot. `tests/run_all.py`
runs on a clone that installed only `requirements.txt`, which is what keeps the base
install small. The cost is that no check under `tests/` executes a line inside a pyspark
call, opens a socket or starts a web framework. `tests/run_with_spark.py` runs the
same checks and then the real job against the reference. `tests/run_with_redis.py` runs
them and then the loader and the lookup against a real server. `tests/run_with_api.py` runs
them and then the real FastAPI app over a real Redis, in both of the arrangements
`scripts/load_probe.py` compares, asserting they return the same body before anything times
them. All three fail rather than skip when their dependency is missing.

No runner starts a socket. `tests/run_with_api.py` drives the app through `TestClient`,
which is in process. The only thing here that binds a port and runs an event loop is
`scripts/load_probe.py`, and it is a script rather than a fifth runner because what it
produces is a measurement rather than a pass or a fail. What is checkable in it lives in
`featurestore/load.py` and is graded by the first runner.

Every one of those counts was produced by running the command. A runner's number is a
property of the runner and cannot be reasoned about from the module the last change
touched, which is a mistake this repo has already made once.

That second runner exists because a mutation score was being quoted for
`featurestore/spark_job.py` that nothing in the repo could produce. Any harness that
takes a repo and reads the exit code of `tests/run_all.py` reproduces the numbers below,
and for the Spark module it is pointed at a copy whose `run_all.py` is the Spark runner.

Mutation over the library, every pass against a copy outside the repo, control clean at
both ends. Every row but the last was re-run against the suite as it stands. The
`run_with_spark.py` row is carried, because neither that module nor the comparison it is
graded against has changed and a pass there costs a Spark session per mutant.

The runner is part of the figure, so it is in the table. The same module scores differently
under the two of them and a score with no runner beside it cannot be reproduced.

```
                           killed   runner
featurestore/spec.py       29 of 29 run_all.py
featurestore/compute.py    20 of 20 run_all.py
featurestore/corpus.py      2 of 2  run_all.py
featurestore/online.py     31 of 31 run_all.py
featurestore/service.py    27 of 28 run_all.py
featurestore/model.py      18 of 20 run_all.py
featurestore/generate.py   60 of 63 run_all.py
featurestore/frames.py     17 of 30 run_all.py
featurestore/load.py       53 of 67 run_all.py
featurestore/drift.py      79 of 79 run_all.py
featurestore/stats.py      86 of 92 run_all.py
featurestore/spark_job.py  11 of 29 run_all.py
featurestore/spark_job.py  24 of 29 run_with_spark.py
```

Four of the seventy nine on `drift.py` were killed by not terminating rather than by a
failing assertion, and all four are in `bin_index`, which is a hand written binary search.
Widening its loop condition, negating the midpoint, or stepping the lower bound backwards
each produce a loop with no exit. The harness reports nothing at all in that case, because
its own subprocess timeout is longer than the call it runs inside, so the evidence a pass
happened is a mutant left in the copy. They are counted as killed. A mutant that hangs the
suite forever is one the suite noticed.

The pass over `drift.py` started at fourteen survivors across the file and reached zero.
None of them was an equivalent mutant and none needed the module changing. Every one was a
fixture that agreed with the mutation by accident, and three are worth reading.

Replacing `log(c / r)` with `log(c * r)`, which is the centre of the statistic, survived
everything. The two forms differ by `sum((c - r) * 2 * log(r))`, and when every reference
share is the same number that pulls out of the sum and leaves it multiplied by
`sum(c - r)`, which is zero because both sides are shares. Quantile binning makes the
reference shares equal by construction, so every fixture built from continuous data made
that mutant exactly equivalent. It dies on ninety zeros and ten ones, where the shares are
0.9 and 0.1 and the index is 0.8788898310 forwards and the negative of it backwards.

Two more were tie handling, and both fixtures had put their ties on the smallest value,
where the final filter removes the duplicates anyway. A repeated value in the middle is
where accepting an equal edge really produces eight bins nothing can fall into, and a
unique smallest value is where comparing against the second smallest turns a feature
holding two values into a constant.

The floor on a judged window was written as `MIN_WINDOW - 1`, so it moved with the constant
it was supposed to pin, and a mutant shifting that constant survived. Literal sizes on both
sides of the boundary now.

`stats.py` moved out of `scripts/serve_probe.py` this pass, because the drift thresholds
need the same exact interval and two Clopper Pearson implementations in one repo is the
shape where each half is right on its own. Its one real survivor was the `n == 0` guard on
`binomial_interval`, and it survived because the only case anything asserted was zero
successes in zero trials, which returns the same pair whether the guard fires or not. One
trial separates them.

The remaining six are floating point. Two are iteration caps. One is 300 on the continued
fraction, where the most any tested input uses is 29. The other is 200 on a bisection over
the unit interval, which is already past what a double can represent after about 52. Three
are comparisons against a tolerance that differ only when a value lands exactly on it.

The last one is the bisection's own `<` against `<=` and it is not equivalent. It moves the
answer by one unit in the last place on two of the eight published inputs and by nothing on
the other six. That unit is 2.22e-16. Pinning it would mean asserting a bit that carries no
information.

The pass over `load.py` started at 47 of 67 and moved to 53 by fixing what it named. Two of
those were the module rather than the checks. A clamp on the quantile index could never
bind, because for any q below one the index is already inside the list, so a mutant widening
it changed nothing. A guard on a zero wall clock in `summarise` was unreachable for the same
kind of reason, since the function that raises on it runs one line earlier. Both were
deleted rather than tested.

Three were fixtures that agreed with a mutant by coincidence. A wall clock of exactly one
second makes dividing by it and multiplying by it the same operation. Nothing sat on the
inclusive edge of the driver floor. Nothing carried a single request, and a guard against an
arm of no requests had been written as `n <= 0` where `n <= 1` passed everything.

One was a real defect and it is the one worth reading. `all_ok` is the control saying an arm
served nothing but scores. An arm of zero requests has a status map of `{200: 0}`, which
matched the expected set and matched its own count, so a pass that served nothing at all
cleared the control that exists to say whether it served the right thing. That is the same
shape as a checker reporting clean because it was pointed at no files.

The fourteen survivors are four unreachable guards and ten equivalent mutants. The guards
are the zero denominator checks in `compare`. They cannot be reached, because an arm with a
throughput of zero fails the status control first, and a check asserts that rather than
pretending a fixture reaches the branch. The equivalents are four things. Guards whose
result is already clamped by a `min(1.0, ...)`. A binomial term that is zero by definition.
Comparisons on ties that were filtered out one line above. And boundaries at 0.05 which the
p values cannot land on, because they are all of the form `2k / 2**n`.

The pass over the serving path found four holes and they were one mistake in four places.
Every check compared a status against `service.OK` rather than against 200, so renumbering
the constant moved both sides of the comparison and the whole suite stayed green on a
service answering 201. A status code is a fact about HTTP and not a value this repo gets to
choose, so the literals are asserted once. The same shape sat under `max_age_s <= 0` and
`event_ts <= 0`, where the fixtures had the refusing side and nothing said a threshold of
one second or an epoch second of 1 is legal.

The fifth was the decision boundary. No fixture put a score exactly on the model threshold,
so `>=` and `>` were the same code. The threshold in that check is built from the score
rather than chosen, because a logistic will not land on a round number.

The three survivors are a rounding place in a log field and two ways of writing the same
JSON. The scorer's overflow branch was a fourth until it was measured. Moving its pivot
from 0 to 1 looks equivalent, both forms being algebraically identical, and it changes the
served score in the last bit on 4,738 of 489,972 corpus rows. The check that had been
passing used a tolerance of 1e-15, which is the whole lesson. Making the pivot strict
instead really is equivalent and that one moved 0 rows of 489,972.

`online.py` has one row and not two, because it imports no client and the suite already
reaches every line of it. The third runner is not there to raise a mutation score. It is
there because the stub in `tests/test_online.py` is a dict and cannot say whether a real
server round trips a value.

The three survivors in the generator are float equality boundaries. A transaction landing
exactly on the end of the horizon, a uniform draw landing exactly on the thinning
threshold, and an account rate landing exactly on the skip cutoff. All three have measure
zero on a continuous draw and none is a rule anybody can get wrong.

All thirteen survivors in `frames.py` are inside the probe tuple. Moving one probe by a
second changes no answer, and that is either an equivalent mutant or a hole nobody has
looked in. `check_the_offsets_do_not_depend_on_which_probes_are_used` says which, by
asking an unrelated probe set the same question and requiring the same offsets. If a probe
value ever starts to matter, that fails.

The five in `spark_job.py` are a literal inside a `count` and two coalesces over expressions
that cannot be null. The other two are the default number of local cores and the flag that
chooses between the two assemblies. The last one is the interesting case. It survives because the
two assemblies really do produce the same table. `plan_probe.py` measures that as zero rows
differing rather than leaving it as a claim.

The two coalesces are equivalent by measurement rather than by argument, which is worth the
distinction. The tie input the second runner uses has 28 rows whose windows admit nothing,
so the branch a coalesce would matter on is reached 28 times per run and adding one there
still changes no cell. `count` returns 0 over an empty frame and `collect_set` returns an
empty array, so neither was ever null to begin with.

`spark_job.py` needs a second oracle and that is worth saying plainly. Under
`tests/run_all.py` the score is 11 of 29, because every decision inside a pyspark call
survives untouched. Under `tests/run_with_spark.py` the same 29 sites score 24, and the
13 that flipped are the frame bounds, the aggregate expressions and the cast. Those are
the whole module. A pass that reported 11 was measuring the runner and not the code.

That pass found a hole in the check rather than in the job. A mutant turning the CSV
header off made Spark read the header line as a row, and the comparison scored perfectly,
because it only ever looked at rows both sides had. It now fails on a row Spark produced
and the reference did not.

The pass over the store found a real defect and it was in the line written to stop one.
The loader skips an entity it has not seen, and the comparison was `>` rather than `>=`,
so an entity whose first event landed exactly on the load moment was still written. Every
window is closed on the left, so that vector is zeros. The fixture had no entity on the
boundary, which is the only reason the mutant survived. Two more survivors were the batch
size default and the counter a flush resets, both invisible until a check named neither
and used three flushes.

What the earlier pass over the definition layer found, in the order it found it.

An `__eq__` on the window type that nothing called and nothing needed. Defining it without
a `__hash__` had quietly made windows unhashable, so deleting it was a repair rather than a
tidy up.

Two of the four unit conversions. Every fixture builds its windows from the same table it
is checked against, so an hour could have been 3,601 seconds and the whole suite stayed
green.

A `sort_keys` flag. Turning it off changes nothing today, because both sides of every
comparison are built by the same code, and it would silently move every fingerprint the
first time somebody reordered a dict literal.

The median helper behind two published figures, and the transaction counter beside it.
Both were only ever read through an inequality.

The zero fallbacks for an empty corpus, which nothing had ever handed an empty corpus.

And the default arguments. Every check passes explicit values to stay fast, so the seed and
the horizon that build the published corpus were the one thing in the file nothing touched.
