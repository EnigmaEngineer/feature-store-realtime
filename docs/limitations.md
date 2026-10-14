# Limitations

Moved out of the README so the front page stays short. Back to the [README](../README.md).

## Known limitations

The drift monitor's null assumes a window is an independent draw from the reference, and a
window of real traffic is not one. The measured cost of that assumption is in the shuffled
control above, where the same threshold on the same rows fires at 0.0255 to 0.0572 without
time order and at up to 0.5832 with it. So the threshold is right for a model of traffic
this traffic does not follow, which is why nothing is wired to a pager. The fix is a
resample that keeps time structure and it is not built.

There is no injection harness for drift. Nothing in this repo takes a clean window, damages
it in a named way, and checks that the monitor notices. Every rate reported above is a rate
on data nobody deliberately broke, so the monitor's sensitivity is unmeasured and only its
noise is measured. Half a control is worth less than it looks, and this is the half that is
missing.

The reference profile is fitted once and never refitted, exactly as the online store is
loaded once and never refreshed. A monitor whose reference is three months stale reports
the age of its own profile.

Nothing stores a fitted profile, which hides a second gap behind the first. Profiles are
fitted at the top of a run and read back inside the same run, off the same list of feature
names, so a feature can never arrive without one. The day they are persisted that stops
being true. A spec that gains a feature after the profiles were written has no reference to
judge it against, and the honest answer then is a verdict saying so rather than a monitor
that silently judges whichever features it happens to hold. There is no such verdict here,
deliberately, because a constant nothing can assign reads like a handled path.

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

Spark loses to the reference on this corpus, and the margin is a range rather than a number.
Over the same 489,972 rows in the same process, two runs on two days gave the reference
9.66 s and 10.11 s against a Spark arm of 11.53 s and 10.75 s, so the ratio ran 1.19 and
1.06. The Spark figure excludes the 18 s the session takes to come up. The counted cells
reproduced to the digit across both runs and only the clock moved, which is why the
direction is worth stating and the ratio is not worth quoting to two places. Two local cores
is not a cluster and 490,000 rows is not a lot of data. The reason to have the Spark job is
that the real corpus is not this one.

The store is loaded in one pass and never updated. There is no streaming writer and no
refresh loop, so the staleness table above is a description of the design rather than a
problem the code solves. It is the argument a refresh interval would have to be made
against and the refresh itself is not built.

Nothing deletes the old keyspace after a spec change. The fingerprint in the key means two
definitions can coexist, which is the point, and it also means an unbounded number of them
can. A real deployment needs an expiry or a sweep and this has neither.

The controls in `scripts/store_check.py` decide whether the day's headline holds and they
live in a script, so nothing in `tests/` imports them and no mutant can reach them. Same
position as `scripts/oracle_check.py` and it is a gap in both.

The lookup is one `HGETALL` per entity. A request scoring several entities would pay a
round trip each, and nothing here batches reads or pipelines them.

The latency figures in the sections above the load one are a single process on loopback
driving the app through a test client, with no concurrency in them. They are a floor. The
load sections replace the concurrency half of that and not the rest. It is still one
process on one machine with one uvicorn worker and no network. One process is one GIL on a
workload that is mostly Python. The first thing to try for throughput here is a second
worker process and nothing has measured that.

The delay sweep that puts the crossover between 0.2 and 0.5 ms is a model of a slower store
rather than a measurement of one. It is `time.sleep` in front of `hgetall`, which releases
the GIL the way a socket wait does and is otherwise nothing like a network. The bracket is
worth more than the midpoint of it.

The model artefact is committed and the corpus it was fitted on is gitignored. The corpus
regenerates byte identically from one command and the fit is one more, so it is
reproducible rather than merely asserted, and a check refuses to pass if the committed
artefact and the shipped spec disagree about the fingerprint. What no check covers is
somebody refitting on a different corpus and committing that.

`class_weight="balanced"` is doing a lot of work at a fraud rate of 0.00082. It was chosen
because an unweighted fit predicts the majority class everywhere, and nothing here compares
it against the alternatives or against a threshold tuned on a holdout. The model threshold
ships at 0.5, which is a default and not a decision.

The flip rate is measured against a model trained on this corpus, so 0.0785 is a number
about this generator. What survives the corpus being synthetic is the shape of the finding
rather than the level.

There is no authentication, no rate limit and no request size cap on the service. It is a
scoring path and not a deployment.

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
