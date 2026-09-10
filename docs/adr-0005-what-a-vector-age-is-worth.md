# What the service does about a stored vector's age

## Status

Accepted.

## The question

`online.lookup` returns the feature vector and the moment it was computed. Something has to
act on that second value or the field should not exist. Three readings were available.

Refuse a vector older than some threshold. Attach the age to the response and let the
caller decide. Delete the field.

## What was measured

The store is loaded once, ninety percent of the way through the corpus. Every transaction
after that moment is a request the service would have answered, and the training table
holds the row the model was taught to expect for it. So each request can be scored twice.
Once from the stored vector, which is what the service really returns. Once from its own
point in time vector, which is the answer it should have had.

A **decision flip** is a request where those two land on opposite sides of the model
threshold. That is the only error measure here with a consequence attached to it. A score
that moves without crossing the line changes nothing anybody acts on.

    python scripts/serve_probe.py --accounts 0

```
max age        refused   share   served   flips    rate       95% interval refused rate
no threshold         0   0.000    48998    3848  0.0785   0.0762 to 0.0810          n/a
1h               48539   0.991      459      17  0.0370   0.0217 to 0.0586       0.0789
6h               45803   0.935     3195     258  0.0808   0.0715 to 0.0907       0.0784
1d               40756   0.832     8242     605  0.0734   0.0679 to 0.0792       0.0796
```

Read the last two columns together. The refused column is the flip rate among the requests
that threshold threw away.

At six hours the service refuses 93.5 percent of its traffic, and the requests it keeps
flip at 0.0808 against 0.0784 for the ones it discarded. At a day it refuses 83.2 percent
to move the rate from 0.0785 to 0.0734, and the interval on that covers the number it
started from. Neither threshold separates anything. They are pure loss.

Only the one hour threshold does anything at all, and it costs 99.1 percent of the traffic
to take the flip rate from 0.0785 to 0.0370.

## What that table depends on

A flip is defined by the model threshold, which ships at 0.5 because that is the default
rather than because anybody measured it. So the headline is a function of a number nobody
chose, and the same sweep at other thresholds says how much of the finding is about
staleness.

```
the same table at other model thresholds, because a flip is defined by one
threshold           all   1h served   6h served   1d served
0.10             0.2362      0.0763      0.1468      0.1924
0.25             0.2673      0.0959      0.2081      0.2338
0.50             0.0785      0.0370      0.0808      0.0734
0.75             0.0238      0.0131      0.0232      0.0228
0.90             0.0092      0.0065      0.0075      0.0091
cuts that beat no cut at all, out of 5 thresholds: 1h 5, 6h 4, 1d 5
```

The level moves by a factor of 29 and it is a property of the threshold rather than of the
store. So "0.0785 of requests get the wrong decision" is not a fact about staleness, it is a
fact about staleness at a threshold of 0.5.

The claim this table does support is the exchange rate, and it is bad everywhere. At 0.25,
the friendliest row for a threshold, six hours refuses 93.5 percent of traffic to move the
rate from 0.2673 to 0.2081. That is 22 percent of the flips for 93.5 percent of the
service. At 0.50 the six hour cut is worse than no cut at all.

The first draft of this document said neither the six hour nor the day threshold separates
anything. That is true at 0.5 and it is stronger than the data supports across the range.

## Why the clock is a bad proxy

Age is standing in for something else. A vector is wrong because the world moved after it
was computed. Splitting the same requests on whether anything actually arrived for that
entity says how good a stand in the clock is.

```
requests split on new arrivals      requests   flips    rate       95% interval
nothing arrived since the load          3933     112  0.0285   0.0235 to 0.0342
something arrived since the load       45065    3736  0.0829   0.0804 to 0.0855
```

Those intervals do not overlap. And the arrival signal beats every threshold on both axes
at once. It identifies 3,933 requests against the one hour threshold's 459, which is 8.6
times the coverage, at a lower flip rate. Anything the clock can do here, knowing whether
an event landed does better.

## The part that stops this being simple

The quiet side is not zero, and the reason is not noise.

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
Every window here is trailing, so it sheds rows off its back edge as the clock moves. A
count of the last hour falls as the hour it counted recedes, with nobody transacting.

Six of the seven can only fall, because removing rows from a count or a sum or a max or a
distinct count cannot raise it. `amount_mean_7d` is the exception and it moved upward 1,103
times, since dropping a below average transaction pushes a mean up.

So a feature goes stale while nothing happens, and a streaming writer fed by arrivals fixes
the larger half of this and not all of it.

## The decision

The service attaches the age to every scored response and does not judge it by default.
`max_age_s` exists, it is honoured, and it ships as None.

An operator sets it with `FS_MAX_AGE_S`, in seconds, which `featurestore/api.py` reads in
`from_environment` and passes to `Policy`. A value of zero or below is refused at start
rather than treated as off. Leaving the variable unset is the shipped position and the
table above is why.

That is a decision and not a refusal to make one. A threshold here is a guardrail whose
subject is a store with no refresh loop, so any setting short enough to protect the seven
day mean refuses essentially all traffic, and a guardrail that refuses everything correct
gets switched off by whoever is on call. The measurement says the two settings a person
would reach for first separate nothing at all.

Deleting the field was the other honest option and it is worse, because the age is real and
a caller scoring a marginal transaction may well want it. Attaching it costs three keys on
a response.

What the response carries:

```json
{
  "score": 0.0271,
  "decision": "allow",
  "features_as_of": 1754664876,
  "features_age_s": 3521,
  "freshness": "unbounded",
  "fingerprint": "534217a2fc6a"
}
```

`unbounded` is a distinct verdict from `fresh`, deliberately. It says no threshold was
configured, so nothing judged the age. That is a fact about how the process was started
rather than a property of the row, and a dashboard counting fresh responses must not be
able to count these among them.

## What this does not settle

The flip rate is measured against a model trained on this corpus, so 0.0785 is a number
about this generator. What survives the corpus being synthetic is the shape. A threshold on
the clock refuses far more than it saves, and the direct signal dominates it.

The refresh loop that all of this argues for is not built. This is the measurement it would
have to be justified against.
