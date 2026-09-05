# Feature Store and Real Time Inference

One feature definition, meant to be read by the job that builds the training table and by
the service that answers a request. This is the definition layer, the reference
implementation both of those get checked against, and the distributed job that is the
first thing to be checked against it.

There are now two implementations of the same spec and a comparison between them. The
Spark job reproduces the reference on 3,429,804 cells with nothing differing. Redis and
the service are still not built, so the training and serving claim is half evidenced
rather than unevidenced.

    pip install -r requirements.txt
    python tests/run_all.py

The suite needs nothing beyond that file. `requirements-spark.txt` is for the batch job and
`requirements-model.txt` is for the leakage probe, and `featurestore/spark_job.py` imports
pyspark inside the functions that build Spark objects so the frame arithmetic stays
readable without a JVM. A check asserts that arrangement rather than trusting it, because
this machine has pyspark installed and would never notice it breaking.

## What is here

```
  specs/txn_fraud.yml
        |
        v
  featurestore/spec.py  ---> fingerprint  ---> fs:<set>:<fingerprint>:<entity>
        |                                            (the online key, and a column
        |                                             on the training table)
        |
        +--> featurestore/compute.py    the reference implementation
        |         |                     plain python, deliberately slow
        |         |
        |         +--> featurestore/frames.py   reads the window bounds back out
        |         |          |                  as spark RANGE offsets
        |         |          v
        |         |    featurestore/spark_job.py    the training table
        |         |          |
        |         +----------+--> scripts/oracle_check.py   grades one against the other
        |
        +--> payload schema             what a request must carry
```

The online store loader and the service are not built yet.

## The spec

A feature is declared once.

```yaml
- name: txn_count_24h
  dtype: int
  aggregate: count
  window:
    length: 24h
    closed: left
    align: event_time
```

Three required window fields and no defaults. `closed: left` excludes the row being scored,
because at serving time that row does not exist yet. `align` says whether the window hangs
off the scoring timestamp or off midnight. See `docs/adr-0002` for why both are required
and what it costs to get either wrong.

`FeatureSet.fingerprint()` is a hash over the semantic content. It goes in the online store
key, so a loader running an older definition cannot overwrite rows a newer one wrote, and a
service asking for the new fingerprint gets a miss rather than a stale answer. Descriptions
and feature ordering are outside the hash, so rewording a comment does not invalidate a
warehouse.

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

```
python scripts/leakage_probe.py
  point in time features        AUC 0.9226
  one snapshot per account      AUC 0.5898
  best possible account score   AUC 0.9982
  coin flip                     AUC 0.5000
```

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
  reference built in 9.66s, spark arm in 11.53s

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

## Verification

```
python tests/run_all.py                          83 passed, 0 failed
python tests/run_with_spark.py                   83 passed, 0 failed, spark arm agrees
```

Two runners because one of them cannot reach the job. `tests/run_all.py` runs on a clone
that installed only `requirements.txt`, which is what keeps the base install small, and
the cost is that no check under `tests/` executes a line inside a pyspark call.
`tests/run_with_spark.py` runs the same checks and then the real job against the
reference, and it fails rather than skips when pyspark is missing.

That second runner exists because a mutation score was being quoted for
`featurestore/spark_job.py` that nothing in the repo could produce. Any harness that
takes a repo and reads the exit code of `tests/run_all.py` reproduces the numbers below,
and for the Spark module it is pointed at a copy whose `run_all.py` is the Spark runner.

Mutation over the library, every pass against a copy outside the repo, control clean at
both ends.

The runner is part of the figure, so it is in the table. The same module scores differently
under the two of them and a score with no runner beside it cannot be reproduced.

```
                           killed   runner
featurestore/spec.py       29 of 29 run_all.py
featurestore/compute.py    20 of 20 run_all.py
featurestore/corpus.py      2 of 2  run_all.py
featurestore/generate.py   60 of 63 run_all.py
featurestore/frames.py     17 of 30 run_all.py
featurestore/spark_job.py  11 of 29 run_all.py
featurestore/spark_job.py  24 of 29 run_with_spark.py
```

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

## Known limitations

A calendar aligned window has no batch implementation. A Spark RANGE frame is two
constants and that alignment is not a constant offset from the scoring event, so
`frames.frame_offsets` refuses it and names the six frames it produced instead of
returning one of them. The shipped feature set is entirely event aligned so the job runs,
and half the spec's legal alignments have no training path. `docs/adr-0003` names what
building it would take.

`distinct_count` goes through `collect_set` because Spark has no `count(distinct)` over a
window frame. That holds the whole set per row. A seven day window on the busiest account
here is a few hundred merchant ids and it is fine, and on a real card portfolio it would
not be.

The second runner grades against the tie heavy input and not against the corpus, because a
session costs about eighteen seconds and a mutation pass pays that once per mutant. So the
24 of 29 is a score over 3,000 rows and 20 accounts. A mistake that only shows at scale
would survive it. The full corpus comparison is a separate command and it is the one the
cell counts above come from.

The Spark job agreeing with the reference on every cell says the two implementations
agree. It does not say either is right. The reference is what defines correct in this repo
and an error inside it would be reproduced faithfully by the job and reported as a clean
run.

An empty window returns 0 for every aggregate, so a sum of nothing and a max of nothing are
indistinguishable from a real zero. The model has to be handed a number, so the choice is
made in one place in `compute.aggregate` rather than pushed downstream.

`compute.py` is plain Python over a list of dicts and it is slow on purpose. It exists to
be the oracle the distributed job is checked against, so it is written to be obviously
correct. It is also not slower than the distributed job at this size, which is the next
paragraph.

The corpus makes compromise an account level property, which is why the account constant
oracle sits at 0.9982. On a real portfolio, accounts are compromised and then recover, so
account identity would be worth less than it is here. That inflates the value of the point
in time discipline this repo is about.

Spark loses to the reference on this corpus. In the same run over the same 489,972 rows the
reference took 9.66 s and the Spark arm took 11.53 s, and the Spark figure excludes the 18 s
the session takes to come up. Two local cores is not a cluster and 490,000 rows is not a lot
of data, so this is the expected direction and it is worth stating rather than implying the
distributed one is the fast one. The reason to have the Spark job is that the real corpus is
not this one.

Nothing here talks to Redis yet.

## What I would do differently

The spec started with one window field and the probe is what forced the other two out. The
right order was the other way round. Writing the measurement first would have shown that a
window with one field cannot be implemented twice and produce the same number, and the
`calendar_day` with an hour length combination would never have been legal in the first
place.

The same mistake in a second form. The mutation harness was pointed at the suite, and the
suite has no JVM, so most of `spark_job.py` was ungraded and the score read 11 of 29.
Pointing it at the comparison against the reference instead took it to 24 and found a hole
in the comparison itself. The oracle a mutation pass uses is a choice and I made it by
default.

Then I published the 24 before there was anything in the repo that could produce it. The
first pass was run by hand against a runner that existed for an afternoon, so the number
was correct and unreproducible, which is a worse position than being wrong. A figure whose
command lives in somebody's shell history is a figure nobody can check, including me a week
later. `tests/run_with_spark.py` is that afternoon's runner, committed.
