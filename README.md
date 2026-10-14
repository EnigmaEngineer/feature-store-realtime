# Feature Store and Real Time Inference

One feature definition, meant to be read by the job that builds the training table and by
the service that answers a request. This is the definition layer, the reference
implementation both of those get checked against, and the distributed job that is the
first thing to be checked against it.

There are now four readers of the same spec and a comparison for each. The Spark job
reproduces the reference on 3,429,804 cells with nothing differing. The online store round
trips 28,000 cells through a real Redis with nothing differing. The service scores from the
store and is graded against the same reference through a real HTTP stack.

    pip install -r requirements.txt
    python tests/run_all.py

The suite needs nothing beyond that file. There is one optional file per heavy thing. The
batch job takes `requirements-spark.txt` and the store takes `requirements-online.txt`. The
HTTP edge takes `requirements-service.txt` and fitting takes `requirements-model.txt`.
`featurestore/spark_job.py` imports pyspark
inside the functions that build Spark objects and `featurestore/api.py` imports fastapi
inside the one that builds the app, so the frame arithmetic and the serving rules stay
readable without a JVM or a web framework. `featurestore/online.py` imports no client at
all and takes one as an argument. Checks assert that arrangement rather than trusting it,
because this machine has all of them installed and would never notice it breaking.

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
        |         |
        |         +--> featurestore/online.py    the loader and the lookup
        |                    |                   one vector per entity, in redis
        |                    |
        |                    +--> scripts/store_check.py      grades the round trip
        |                    +--> scripts/staleness_probe.py  grades it against the clock
        |                    |
        |                    +--> featurestore/service.py   validate, fetch, score, log
        |                              |                    no web framework in it
        |                              |
        |                              +--> featurestore/api.py     the fastapi edge
        |                              |         |
        |                              |         +--> scripts/load_probe.py  many at once
        |                              |                    |
        |                              |         featurestore/load.py  what a pass is
        |                              |                               allowed to claim
        |                              +--> scripts/serve_probe.py  grades what it serves
        |
        +--> featurestore/drift.py      has a distribution moved, and what a
        |          |                    threshold on that is worth
        |          |
        |          +--> scripts/drift_probe.py   fits, then judges windows the
        |                                        fit never saw
        |
        +--> payload schema             what a request must carry
                  ^
                  |
        featurestore/model.py           coefficients, keyed to the same fingerprint
```

`scripts/train_model.py` fits the model and writes `models/txn_fraud.json`. Nothing on the
serving path imports scikit-learn, because the arithmetic is a dot product and a logistic.

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

## Results

The corpus is generated from one command and every figure here names the script that
rebuilds it. 4,000 accounts and 489,972 transactions over 60 days, with a fraud rate of
0.00082.

**Two honest readings of one 24 hour window disagree on 64.7 percent of rows.** A
nightly batch job and an online store that both honour the spec hand the model a different
count on most requests, unless the spec says which edge is closed and what the window hangs
off. That is why the spec has three required window fields and no defaults.

**A snapshot join does not inflate the metric here. It starves it.**

```
python scripts/leakage_probe.py
  point in time features        AUC 0.9226
  one snapshot per account      AUC 0.5898
  best possible account score   AUC 0.9982
  coin flip                     AUC 0.5000
```

**The online store is wrong the moment it finishes loading, and an age threshold does not
fix that.** Against a model threshold, 7.85 percent of served requests land on the other
side from their own point in time vector. Refusing anything older than an hour refuses 99.1
percent of traffic and still flips 3.70 percent of what it serves.

**A drift threshold is a statement about how much traffic arrived.** The same PSI threshold
on the same rows fires at 0.0255 to 0.0572 when the rows are shuffled and at up to 0.5832
when they keep their time order. The threshold is calibrated correctly for a model of
traffic this traffic does not follow, so nothing is wired to a pager.

The Spark job loses to the plain Python reference at this size. Two runs gave ratios of
1.19 and 1.06, not counting the 18 seconds the session takes to start. The counted cells
reproduced to the digit across both runs and only the clock moved.

## Known limitations

- The drift monitor assumes a window is an independent draw from the reference, and real
  traffic is not.
- There is no injection harness for drift, so the monitor's noise is measured and its
  sensitivity is not.
- A calendar aligned window has no batch implementation. Half the spec's legal alignments
  have no training path.
- The Spark job agreeing with the reference on every cell says they agree. It does not say
  either is right.
- The store is loaded once and never refreshed. Nothing deletes an old keyspace after a spec
  change.
- Load figures come from one process on one machine with one uvicorn worker and no network.
- The model threshold ships at 0.5, which is a default and not a decision.
- There is no authentication, rate limit or request size cap on the service.

The full list is in [docs/limitations.md](docs/limitations.md), with what I would do
differently. The short version of that is that I wrote the spec before the measurement that
would have shaped it.

## More detail

**[Design notes](docs/design-notes.md)**

- [Measured](docs/design-notes.md#measured)
  - [The two readings of one window](docs/design-notes.md#the-two-readings-of-one-window)
  - [A snapshot join does not inflate the metric here. It starves it.](docs/design-notes.md#a-snapshot-join-does-not-inflate-the-metric-here-it-starves-it)
  - [The batch job against the reference](docs/design-notes.md#the-batch-job-against-the-reference)
  - [How the columns are added changes the plan](docs/design-notes.md#how-the-columns-are-added-changes-the-plan)
  - [Ties, which the corpus almost does not have](docs/design-notes.md#ties-which-the-corpus-almost-does-not-have)
  - [The online store, through a real Redis](docs/design-notes.md#the-online-store-through-a-real-redis)
  - [A spec change, and what the key buys](docs/design-notes.md#a-spec-change-and-what-the-key-buys)
  - [The store is wrong the moment it finishes loading](docs/design-notes.md#the-store-is-wrong-the-moment-it-finishes-loading)
  - [The loader cannot write an entity it has never seen](docs/design-notes.md#the-loader-cannot-write-an-entity-it-has-never-seen)
  - [The service, and the model it serves](docs/design-notes.md#the-service-and-the-model-it-serves)
  - [Running the service, and the four things it reads from the environment](docs/design-notes.md#running-the-service-and-the-four-things-it-reads-from-the-environment)
  - [An age threshold refuses far more than it saves](docs/design-notes.md#an-age-threshold-refuses-far-more-than-it-saves)
  - [A feature goes stale while nothing happens](docs/design-notes.md#a-feature-goes-stale-while-nothing-happens)
  - [The worker thread costs a third of the throughput and is still the right default](docs/design-notes.md#the-worker-thread-costs-a-third-of-the-throughput-and-is-still-the-right-default)
  - [What a perfect cache would be worth, which is why there is not one](docs/design-notes.md#what-a-perfect-cache-would-be-worth-which-is-why-there-is-not-one)
  - [A drift threshold is a statement about how much traffic arrived](docs/design-notes.md#a-drift-threshold-is-a-statement-about-how-much-traffic-arrived)
  - [The threshold is calibrated correctly and the assumption under it is not](docs/design-notes.md#the-threshold-is-calibrated-correctly-and-the-assumption-under-it-is-not)
  - [The distribution monitor is blind to the failure this project has](docs/design-notes.md#the-distribution-monitor-is-blind-to-the-failure-this-project-has)
  - [A store that is exactly right reads as significant drift](docs/design-notes.md#a-store-that-is-exactly-right-reads-as-significant-drift)
  - [An emptied bin is a verdict rather than a number](docs/design-notes.md#an-emptied-bin-is-a-verdict-rather-than-a-number)
- [Verification](docs/design-notes.md#verification)

**Other docs**

- [adr-0001-dataset-and-model](docs/adr-0001-dataset-and-model.md)
- [adr-0002-window-semantics](docs/adr-0002-window-semantics.md)
- [adr-0003-spark-frames](docs/adr-0003-spark-frames.md)
- [adr-0004-online-keys-and-misses](docs/adr-0004-online-keys-and-misses.md)
- [adr-0005-what-a-vector-age-is-worth](docs/adr-0005-what-a-vector-age-is-worth.md)
- [adr-0006-a-worker-thread-and-a-cache](docs/adr-0006-a-worker-thread-and-a-cache.md)
- [adr-0007-what-a-drift-threshold-measures](docs/adr-0007-what-a-drift-threshold-measures.md)
