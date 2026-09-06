# The fingerprint goes in the key, and a miss is an error

## Status

Accepted.

## The problem

The online store holds one feature vector per entity and the service reads it back to
build a request payload. Two decisions sit in that sentence and both are the kind that get
made by accident.

The first is what the key is. The obvious answer is the entity id, and it works until
somebody changes a window length. Then a loader running the new definition and a service
running the old one share a keyspace, and every row either overwrites or is overwritten.
Nothing raises, because the field names did not move. A window is a number inside the
definition and the payload looks exactly the same either way.

The second is what a lookup returns when there is no row. A vector of zeros is convenient,
because the model has to be handed numbers and `compute.aggregate` already returns 0 for an
empty window. It is also the same vector a real account with a quiet week gets. So the
convenient answer makes a cold start indistinguishable from a customer who has stopped
spending.

## The decision

The key is `fs:<feature set>:<fingerprint>:<entity>`, built by
`spec.FeatureSet.online_key`. The fingerprint is a hash over the semantic content of the
spec, so a changed definition writes to a different keyspace and a service running the new
definition reads an empty one.

A miss raises `online.Miss`. It is a `LookupError` and deliberately not a `KeyError`, so a
caller wrapping its own dict access does not swallow it. What to do about a miss is the
service's decision and it is made where somebody can see it.

The loader writes an entity only if the log holds an event strictly before the load moment.
Every window is closed on the left, so an event landing exactly on that moment cannot be
inside any of them, and the vector would be zeros. In production the log would not contain
that entity at all. Handed a corpus that runs past the load moment, which is what every
probe here does, the filter is what stops a cold start being written as history.

## What it is worth, measured

`python scripts/store_check.py --accounts 0` over 4,000 entities, with the store loaded and
then asked for a definition whose windows moved by an hour.

| fingerprint in the key | answered | missed | cells wrong |
|---|---|---|---|
| yes | 0 | 4,000 | 0 |
| no | 4,000 | 0 | 2,325 |

The second row is the one to read carefully. 2,325 wrong cells out of 28,000 is 8.3 percent,
which means 91.7 percent of the answers under the old definition happened to be right. That
is what makes it dangerous rather than harmless. A change that broke every cell would be
found in an afternoon.

The cold start filter is measured by `python scripts/staleness_probe.py --at 0.10`, which
loads the store a tenth of the way through the corpus. It holds 3,940 of the 4,000
entities, and 1,371 later requests are for an entity it never wrote. 1,240 of those have
real history by the time they arrive. Without the filter every one of them is answered with
a vector of zeros.

## What it costs

Two keyspaces exist during a definition change and nothing here deletes the old one. A
rollout is a load under the new fingerprint, then a deploy, then a decision about the old
keys that this repo does not make. The alternative is a store that answers wrongly during
the same window, which is worse and quieter.

A miss that raises means the service has to handle it. That is the point and it is still a
cost, because there is no correct default and the caller now owns a decision it might get
wrong in its own way.
