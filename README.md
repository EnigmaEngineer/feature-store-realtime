# Feature Store and Real Time Inference

One feature definition, meant to be read by the job that builds the training table and by
the service that answers a request. This is the definition layer and the reference
implementation both of those get checked against.

The claim it is honest to make today is narrow. A spec pins what a feature means and the
fingerprint stops two definitions being confused for each other. It does not yet stop two
implementations of the same definition returning different numbers, because only one
implementation exists. That check is the point of the project and it is not built.

    pip install -r requirements.txt
    python tests/run_all.py

## What is here

```
  specs/txn_fraud.yml
        |
        v
  featurestore/spec.py  ---> fingerprint  ---> fs:<set>:<fingerprint>:<entity>
        |                                            (the online key)
        |
        +--> featurestore/compute.py    the reference implementation
        |         |
        |         +--> point in time features for a training table
        |         +--> the same numbers a lookup has to return
        |
        +--> payload schema             what a request must carry
```

The Spark job, the online store loader and the service are not built yet. What is built is
the thing all three have to agree with.

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

## Verification

```
python tests/run_all.py                          55 passed, 0 failed
```

Mutation over the library, every pass against a copy outside the repo, control clean at
both ends.

```
featurestore/spec.py      29 of 29 killed
featurestore/compute.py   20 of 20 killed
featurestore/corpus.py     2 of 2 killed
featurestore/generate.py  60 of 63 killed
```

The three survivors in the generator are float equality boundaries. A transaction landing
exactly on the end of the horizon, a uniform draw landing exactly on the thinning
threshold, and an account rate landing exactly on the skip cutoff. All three have measure
zero on a continuous draw and none is a rule anybody can get wrong.

What the pass found, in the order it found it.

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

An empty window returns 0 for every aggregate, so a sum of nothing and a max of nothing are
indistinguishable from a real zero. The model has to be handed a number, so the choice is
made in one place in `compute.aggregate` rather than pushed downstream.

`compute.py` is plain Python over a list of dicts and it is slow on purpose. Building point
in time features for 489,972 rows takes about 10 seconds. It exists to be the oracle the
distributed job is checked against, so it is written to be obviously correct.

The corpus makes compromise an account level property, which is why the account constant
oracle sits at 0.9982. On a real portfolio, accounts are compromised and then recover, so
account identity would be worth less than it is here. That inflates the value of the point
in time discipline this repo is about.

Nothing here talks to Redis or Spark yet.

## What I would do differently

The spec started with one window field and the probe is what forced the other two out. The
right order was the other way round. Writing the measurement first would have shown that a
window with one field cannot be implemented twice and produce the same number, and the
`calendar_day` with an hour length combination would never have been legal in the first
place.
