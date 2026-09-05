# The Spark frame is read out of the reference, not written beside it

## Status

Accepted.

## The problem

The batch job has to compute the same numbers as `featurestore/compute.py`. The obvious
way to write it is to read the spec, understand what `closed: left` means, and type the
offsets into a `rangeBetween` call.

That gives two independent readings of one English sentence. When the two agree, all you
have learned is that you made the same interpretation twice. That is not a check, and it
is the shape of mistake this whole repo exists to catch.

## The decision

`featurestore/frames.py` never states an offset. It calls `compute.window_bounds` at a
set of probe timestamps and expresses each answer as a pair of offsets from that probe.
If every probe agrees, the pair is what Spark gets.

```python
lo, hi, hi_inclusive = compute.window_bounds(as_of, window)
upper = hi - as_of if hi_inclusive else hi - as_of - 1
seen[(lo - as_of, upper)] = ...
```

There is exactly one subtraction of my own in that, the one that turns an exclusive upper
bound into an inclusive one. It is only sound on whole seconds, so
`spark_job.check_event_time` refuses a float or a string event time column rather than
quietly being off by a fraction of a second.

## What it bought immediately

A Spark RANGE frame is two constants. A window in this spec is not always that, and the
probes say so without anybody reasoning about it.

`align: event_time` gives one frame at every probe. `align: calendar_day` gives six
different frames across six probes, from `(-172799, -86400)` to `(-86400, -1)`, because
the anchor is midnight and the distance from the scoring event to midnight is whatever
time of day it is. So `frame_offsets` refuses that window and says why.

Half the spec's legal alignments cannot be a window frame at all. Nothing about reading
the Spark documentation would have told me that. The probe did.

## The trap inside the probe set

Probes have to disagree on the time of day. Probes exactly a day apart look varied and
are not, because a calendar aligned window resolves to the same frame under both, and the
refusal then silently stops working.

`probes_differ_in_time_of_day` is a guard against a later edit trimming the set down to
something convenient, and `tests/test_frames.py` shows the bad probe set really would have
hidden the problem rather than asserting that it would.

## What is not solved

A calendar aligned feature has no batch implementation here. It needs a join to a table
of day boundaries rather than a frame, which is a different piece of work. The shipped
feature set is entirely event aligned, so the job runs, and a spec that used the other
alignment would be refused at plan time rather than computed wrongly.

Refusing is the right answer today and it is not a complete one. The point of the spec is
that both alignments are legal, and one of them has a training path and the other does
not.
