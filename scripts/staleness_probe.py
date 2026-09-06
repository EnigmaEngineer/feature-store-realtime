"""How wrong the online store is, as a function of how long ago it was loaded.

    python scripts/staleness_probe.py --accounts 0

The training table holds one row per transaction, with features computed at that
transaction's own timestamp. The online store holds one row per entity, computed once at
load time. Both come from the same spec and the same reference implementation, so every
difference between them is the gap between when the value was computed and when it is
read.

That gap is not a bug to be fixed. It is what a batch loaded online store is. The number
worth having is how fast it opens, because that is what a refresh interval has to be
argued against.

The comparison is against real requests rather than against a grid of offsets. Every
transaction after the load moment is a request the service would have answered, and the
row the training table holds for it is what the model was taught to expect. So each one
is scored against the vector the store would have handed it, and the results are grouped
by how old that vector was.

No Redis here. The store's contents at load time are `online.online_rows`, and putting a
server in the middle would only add the round trip `scripts/store_check.py` already
grades.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import compute
from featurestore import corpus as corpus_io
from featurestore import online, spec

HOUR = 3600
DAY = compute.DAY

# Upper edges in seconds. The last bucket is everything past the previous edge, which on
# this corpus is the tail nobody would ever leave a store unrefreshed for.
BUCKETS = ((HOUR, "under 1h"), (6 * HOUR, "1h to 6h"), (DAY, "6h to 24h"), (None, "over 24h"))


def bucket_for(age_s):
    for edge, label in BUCKETS:
        if edge is None or age_s < edge:
            return label
    return BUCKETS[-1][1]


def pick_rows(rows, n_accounts):
    if n_accounts is None:
        return rows
    keep = set(sorted({r["account_id"] for r in rows})[:n_accounts])
    return [r for r in rows if r["account_id"] in keep]


def load_moment(rows, time_field, share):
    """The timestamp the store is loaded at, as a share of the way through the corpus.

    Taken from the event times rather than from the span, so the tail after it holds a
    predictable number of requests whatever the arrival rate is doing.
    """
    times = sorted(r[time_field] for r in rows)
    return times[int(len(times) * share)]


def blank_report():
    return {"rows": 0, "cells": 0, "differing": 0, "by_feature": {}}


def score(log, feature_set, requests, as_of, stored):
    """Every request after the load moment, against the vector the store would give it.

    `stored` is what the loader wrote. A request for an entity the loader never saw is
    counted separately, because a miss is a different failure from a wrong number and
    folding them together loses the one the service can actually detect.
    """
    entity = feature_set.entity
    time_field = feature_set.event_time
    names = feature_set.feature_names()

    buckets = {label: blank_report() for _, label in BUCKETS}
    misses = 0
    # A miss whose true vector is not all zeros is one a loader writing every entity in
    # the log would have answered wrongly rather than harmlessly. That is the cost of
    # letting the loader see past its own load moment, as a count.
    misses_that_are_not_zero = 0
    for row in requests:
        entity_id = row[entity]
        if entity_id not in stored:
            misses += 1
            truth = compute.features_as_of(log, feature_set, entity_id, row[time_field])
            if any(truth[name] != 0 for name in names):
                misses_that_are_not_zero += 1
            continue
        truth = compute.features_as_of(log, feature_set, entity_id, row[time_field])
        served = stored[entity_id]
        report = buckets[bucket_for(row[time_field] - as_of)]
        report["rows"] += 1
        for name in names:
            report["cells"] += 1
            if truth[name] != served[name]:
                report["differing"] += 1
                report["by_feature"][name] = report["by_feature"].get(name, 0) + 1
    return buckets, misses, misses_that_are_not_zero


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="specs/txn_fraud.yml")
    ap.add_argument("--corpus", default="data/transactions.csv")
    ap.add_argument("--accounts", type=int, default=400, help="0 means every account")
    ap.add_argument(
        "--at",
        type=float,
        default=0.9,
        help="load the store this far through the corpus, by event count",
    )
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args(argv)

    feature_set = spec.load(args.spec)
    rows = pick_rows(list(corpus_io.read_csv(args.corpus)), args.accounts or None)
    log = compute.EventLog(rows, feature_set.entity, feature_set.event_time)
    time_field = feature_set.event_time

    as_of = load_moment(rows, time_field, args.at)
    stored = {eid: vals for eid, vals in online.online_rows(log, feature_set, as_of)}
    requests = sorted(
        (r for r in rows if r[time_field] >= as_of), key=lambda r: r[time_field]
    )

    print(
        "{} rows over {} entities, store loaded at {} which is {:.0%} through the "
        "corpus".format(len(rows), len(stored), as_of, args.at)
    )
    print(
        "{} transactions arrive after that, spanning {:.1f} hours".format(
            len(requests), (requests[-1][time_field] - as_of) / HOUR
        )
    )

    # The control. Scored at the load moment itself the store is exactly right, so a
    # table of nonzero rates below is about age and not about the comparison being wrong.
    control_rows = [
        {
            feature_set.entity: eid,
            time_field: as_of,
            feature_set.label: 0,
        }
        for eid in sorted(stored)
    ]
    control, control_misses, _ = score(log, feature_set, control_rows, as_of, stored)
    control_differing = sum(b["differing"] for b in control.values())
    control_cells = sum(b["cells"] for b in control.values())
    print(
        "control at the load moment: {} cells, {} differing, {} misses".format(
            control_cells, control_differing, control_misses
        )
    )

    buckets, misses, live_misses = score(log, feature_set, requests, as_of, stored)
    print(
        "{} requests were for an entity the loader never wrote, {} of them had real "
        "history by then".format(misses, live_misses)
    )
    print()
    print("{:<12} {:>9} {:>10} {:>10} {:>8}".format("age", "requests", "cells", "differing", "share"))
    for _, label in BUCKETS:
        report = buckets[label]
        share = report["differing"] / report["cells"] if report["cells"] else 0.0
        print(
            "{:<12} {:>9} {:>10} {:>10} {:>8.3f}".format(
                label, report["rows"], report["cells"], report["differing"], share
            )
        )

    # The per feature table rather than the worst one. The worst is the same feature in
    # every bucket and printing only that hides the reason, which is that the features
    # sitting still are the ones agreeing at zero on both sides.
    print()
    header = "{:<22}".format("feature") + "".join(
        "{:>11}".format(label) for _, label in BUCKETS
    )
    print(header)
    for name in feature_set.feature_names():
        row = "{:<22}".format(name)
        for _, label in BUCKETS:
            report = buckets[label]
            share = report["by_feature"].get(name, 0) / report["rows"] if report["rows"] else 0.0
            row += "{:>11.3f}".format(share)
        print(row)

    if control_differing:
        print()
        print(
            "CONTROL FAILED. The store disagrees with the reference at the moment it was "
            "loaded, so the table above is not measuring age."
        )
        return 2

    report = {
        "as_of": as_of,
        "at": args.at,
        "rows": len(rows),
        "entities": len(stored),
        "requests": len(requests),
        "misses": misses,
        "misses_with_real_history": live_misses,
        "control_cells": control_cells,
        "control_differing": control_differing,
        "buckets": buckets,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
