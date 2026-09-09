# ADR 0006: whether the blocking work leaves the event loop, and whether anything is cached

Status: accepted.

## The two questions

`featurestore/api.py` runs `service.handle` on a worker thread through
`starlette.concurrency.run_in_threadpool`. That was written on the argument that redis-py
is synchronous, so calling it from an async endpoint holds the event loop for the whole
round trip. The argument was never measured, because every latency figure this repo had
came off `fastapi.testclient`, which sends one request at a time and never opens a socket.
A serial harness cannot tell the two arrangements apart.

The second question is whether a cache earns its place here. What
that needs is a ceiling on what a cache could buy, not a cache.

`scripts/load_probe.py` answers both. Four arms, all built by `api.build_app`, driven by a
closed loop of real HTTP connections against a real uvicorn.

## What was measured

Six paired rounds per arm per level. Arms rotated so no arm is always first. A warmup pass
discarded and the driver's own control checked on every pass. All 96 passes kept their
load up. The comparison is a sign test over the paired rounds. Six rounds is the smallest
number that can reach a p below 0.05, which is why the probe defaults to six and refuses
five.

Medians over six rounds, requests per second and p99 in milliseconds.

| concurrency | offload rps | inline rps | offload p99 | inline p99 | throughput ratio | p |
|---|---|---|---|---|---|---|
| 1  |  827 | 1055 |   2.02 |   1.66 | 0.78x | 0.0312 |
| 8  |  792 | 1094 |  16.42 |   9.58 | 0.72x | 0.0312 |
| 32 |  858 | 1168 |  98.51 |  30.83 | 0.73x | 0.0312 |
| 64 |  961 | 1332 | 126.87 |  53.24 | 0.72x | 0.0312 |

The worker thread lost every round at every level. Throughput 0.72x to 0.78x, and p99 from
1.21x worse at one client to 3.20x worse at thirty two.

So the comment in `api.py` was wrong about this machine, and the first draft of this
document said to revert the change.

## Why that draft was wrong

The result is about the store, not about the arrangement. Redis here is redislite over
loopback and its round trip is a fraction of a millisecond, so there is almost nothing for
a worker thread to overlap and the dispatch is pure cost. `service.handle` is otherwise
Python arithmetic under the GIL, which a thread cannot help with.

Putting a wait in front of the store finds where it flips. `time.sleep` releases the GIL,
which is the property a network wait has.

| store delay | offload rps | inline rps | ratio | rounds won | p |
|---|---|---|---|---|---|
| 0.0 ms |  964 | 1383 | 0.70x | 0 of 6 | 0.0312 |
| 0.1 ms |  921 | 1290 | 0.71x | 0 of 6 | 0.0312 |
| 0.2 ms | 1036 | 1129 | 0.92x | 0 of 6 | 0.0312 |
| 0.3 ms |  966 | 1009 | 0.96x | 3 of 6 | 1.0000 |
| 0.4 ms | 1045 |  911 | 1.15x | 4 of 6 | 0.6875 |
| 0.5 ms | 1041 |  847 | 1.23x | 6 of 6 | 0.0312 |
| 1.0 ms | 1065 |  578 | 1.84x | 6 of 6 | 0.0312 |
| 2.0 ms |  902 |  354 | 2.55x | 6 of 6 | 0.0312 |

At 32 clients. The crossover sits between 0.2 and 0.5 ms, with 0.3 and 0.4 undecided at
six rounds. The store round trip check measures this store answering `HGETALL` in about
0.12 ms and the lookup
p99 at about 0.29 ms, so the deployment being measured sits on the crossover rather than
comfortably to one side of it.

## Decision

**The worker thread stays, and `offload` defaults to True.**

The two costs are not symmetric. Below the crossover the thread costs about 28 percent of
throughput and that is where it stops. Above it the loop arrangement loses without bound,
2.55x at a two millisecond store and worse beyond that, because every request in flight
waits behind the one holding the loop. Half a millisecond of store round trip is not an
unusual store. It is any Redis that is not on this machine.

A measurement taken on the fastest store this service will ever see should not set the
default for the stores it will actually meet.

`offload=False` is kept as an argument rather than deleted. It is how the losing arm gets
built by the same builder, and a comparison whose losing side is hand written is one the
author can rig without meaning to.

## The cache

The cached arms replace Redis with a dict holding every key, read out of the real store so
both sides decode identical bytes. That is not a proposal. It is the ceiling. A perfect
cache is a dict with everything already in it. No eviction and no round trip and no
invalidation. Nothing real can beat it.

Medians over six rounds, cache against no cache within the same arrangement.

| concurrency | cache on inline | cache on offload |
|---|---|---|
| 1  | 1.32x | 1.25x |
| 8  | 1.26x | 1.35x |
| 32 | 1.24x | 1.27x |
| 64 | 1.21x | 1.24x |

**No cache is built.** Three reasons, in the order they matter.

There is no throughput problem to solve. The budget is a p99 of 150 ms and the shipped
arrangement is at 126.87 ms at sixty four concurrent clients on this machine, which is the
one number here that is close to anything. The cache does not fix that. It is worth 1.24x
of throughput there and the arrangement question is worth 2.38x of p99, so the cache is the
smaller lever on the metric that binds.

The ceiling is small enough to be spent elsewhere. Twenty one to thirty two percent is the
best case of a component that can only be worse in practice.

And the staleness is not free, which is the part that decides it. The store is loaded once
and never refreshed, and `docs/adr-0005` measured what that already costs. A cache in front
of it adds a second unbounded age with no invalidation path, on a service whose age policy
ships as `None` precisely because the first layer's staleness could not be bounded
usefully. Buying a fifth of the throughput on a service that does not need throughput, by
adding a second layer of an error already measured at 7.85 percent of decisions, is the
wrong trade and the numbers are why rather than the instinct.

## What would change this

A store that is not on loopback. Everything above is one process against a Redis on the
same machine. The delay sweep is a model of a slower store and not a measurement of one,
and the honest reading of it is a crossover with a bracket rather than a number.

A second uvicorn worker. One process is one GIL, and most of what `service.handle` does is
Python. The first thing to try for throughput is another process, not a cache, and nothing
here has measured that.

A p99 budget that binds. At 150 ms nothing in this table is under pressure except the
offload arm at sixty four clients.
