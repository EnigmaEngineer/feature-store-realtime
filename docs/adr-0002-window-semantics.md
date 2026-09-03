# A window needs three fields, not one

## Status

Accepted.

## The problem

`txn_count_24h` names a window and does not define one. Two more decisions sit behind the
number 24 and neither has an obvious default.

Does the transaction being scored count towards its own feature? At serving time that row
does not exist yet, so the online answer is no. At training time the row is sitting right
there in the table, so a join written without care says yes.

Does the window hang off the scoring timestamp or off midnight? An online store answers
from the moment the request arrives. A nightly batch job runs at midnight over whole days,
because that is what a partitioned table makes easy. Nobody chooses the second one. It
arrives with the scheduler.

Those are the training path and the serving path of this repo, and they disagree.

## The decision

A window is three required fields.

    window:
      length: 24h
      closed: left        # left excludes the row being scored
      align: event_time   # event_time or calendar_day

No defaults. `featurestore/spec.py` refuses a window missing any of them. A default would
be a decision made once by whoever wrote the parser and then inherited silently by every
feature after it.

## What it costs to get wrong

Measured over 4,000 scored rows sampled across the corpus, with
`python scripts/window_probe.py`. Each figure is the share of rows where that reading
returns a different value from `event_time` plus `left`, which is what the shipped spec
declares.

| feature | reading | windows differ | values differ |
|---|---|---|---|
| txn_count_24h | calendar_day/left | 0.773 | 0.647 |
| amount_max_24h | calendar_day/left | 0.773 | 0.570 |
| amount_max_24h | event_time/right | 1.000 | 0.360 |
| distinct_merchants_1h | event_time/right | 1.000 | 0.965 |
| distinct_merchants_7d | event_time/right | 1.000 | 0.132 |
| distinct_merchants_7d | calendar_day/left | 0.762 | 0.120 |

Two columns because one of them is a trap. Windows that admit different events can still
agree on a value, and windows that admit the same events cannot disagree at all. When the
two columns are equal the disagreement is arithmetic rather than a fact about the data. A
count over a strictly larger window always increments, so the 1.000 it reports under
`event_time/right` is a restatement of addition. The probe marks those rows rather than
publishing them as findings.

The row that matters is the first one. The nightly reading and the online reading of the
same declared 24 hour window hand the model a different count on 64.7 percent of rows.

## The part the format also has to refuse

`align: calendar_day` with `length: 1h` resolves to the hour before midnight on the
previous day, for every event in the day being scored. It is a legal combination of two
legal values and it means nothing.

The first version of the parser accepted it. It was found by running the probe and getting
a disagreement share for a window nobody would ever write. `calendar_day` now requires a
length that is a whole number of days, and `tests/test_spec.py` holds both halves.

## Consequences

The `closed` and `align` fields are inside the spec fingerprint, so changing either one
writes to a different online keyspace and a model trained under the old reading cannot be
served under the new one.

A feature that genuinely wants both readings has to be declared twice under two names.
That is the intended outcome. Two readings of one window are two features.
