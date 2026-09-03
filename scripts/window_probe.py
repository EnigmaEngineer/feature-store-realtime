"""How far apart the readings of one window string are.

    python scripts/window_probe.py

`length: 7d` on its own does not say what a window is. Two more fields decide it and both
are required in the spec because of what this prints. `closed` says whether the row being
scored sits inside its own window, and `align` says whether the window hangs off the
scoring timestamp or off midnight.

The pairing that matters is event_time/left against calendar_day/left. The first is what
an online store computes when a request arrives. The second is what a nightly batch job
computes without anybody deciding it, because the job runs at midnight over whole days.
Those two are the training path and the serving path of this project, and the column
below says how often they would hand the model a different number.

Two shares are printed per cell and the second one is the honest one. Windows that admit
different events can still agree on a value. Windows that admit the same events cannot
disagree at all. When the two shares are equal the disagreement is arithmetic rather than
a fact about the data, which is what a count over a strictly larger window does.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import compute, corpus, spec

READINGS = [
    ("event_time", "left"),
    ("event_time", "right"),
    ("calendar_day", "left"),
    ("calendar_day", "right"),
]
BASELINE = ("event_time", "left")


def variants(feature):
    """Every reading the spec would accept for this feature, keyed by reading.

    calendar_day is refused for a window that is not a whole number of days, so a 1h
    feature has two readings and a 7d feature has four. The refusal is reported rather
    than skipped, because a blank cell and a meaningless cell look the same in a table.
    """
    out = {}
    refused = {}
    for align, closed in READINGS:
        try:
            w = spec.parse_window(
                {"length": _length_text(feature.window.length_s), "closed": closed, "align": align},
                "probe",
            )
        except spec.SpecError as exc:
            refused[(align, closed)] = str(exc)
            continue
        out[(align, closed)] = spec.Feature(
            feature.name, feature.dtype, feature.aggregate, feature.column, w
        )
    return out, refused


def _length_text(seconds):
    for suffix, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds % size == 0:
            return "{}{}".format(seconds // size, suffix)
    return "{}s".format(seconds)


def probe_feature(log, feature_set, scoring_rows, feature):
    entity = feature_set.entity
    time_field = feature_set.event_time
    vs, refused = variants(feature)

    values = {key: [] for key in vs}
    members = {key: [] for key in vs}
    for r in scoring_rows:
        for key, variant in vs.items():
            lo, hi, inc = compute.window_bounds(r[time_field], variant.window)
            rows = log.between(r[entity], lo, hi, inc)
            members[key].append(tuple(row["txn_id"] for row in rows))
            values[key].append(compute.aggregate(rows, variant))

    base_v = values[BASELINE]
    base_m = members[BASELINE]
    cells = {}
    for key in vs:
        if key == BASELINE:
            continue
        v_diff = sum(1 for a, b in zip(base_v, values[key]) if a != b)
        m_diff = sum(1 for a, b in zip(base_m, members[key]) if a != b)
        n = len(base_v)
        cells[key] = {
            "windows_differ": m_diff / n if n else 0.0,
            "values_differ": v_diff / n if n else 0.0,
            "forced": m_diff == v_diff,
        }
    return cells, refused


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="specs/txn_fraud.yml")
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--sample", type=int, default=4000)
    args = ap.parse_args(argv)

    fs = spec.load(args.spec)
    events = list(corpus.read_csv(args.data))
    log = compute.EventLog(events, fs.entity, fs.event_time)

    # Scoring every row would take too long for a probe. A stride keeps the sample spread
    # over the whole horizon instead of clustering it at one end.
    stride = max(1, len(events) // args.sample)
    scoring = events[::stride][: args.sample]

    print("spec {} fingerprint {}".format(fs.name, fs.fingerprint()))
    print(
        "{} events, scoring 1 in {} of them for a sample of {}".format(
            len(events), stride, len(scoring)
        )
    )
    print("")
    row_fmt = "{:<24} {:<20} {:>9} {:>9}  {}"
    print(row_fmt.format("feature", "reading", "windows", "values", "note"))
    any_refused = {}
    for feature in fs.features:
        cells, refused = probe_feature(log, fs, scoring, feature)
        any_refused.update({(feature.name, k): v for k, v in refused.items()})
        for key in READINGS:
            if key == BASELINE:
                continue
            label = "{}/{}".format(*key)
            if key in refused:
                print(row_fmt.format(feature.name, label, "", "", "refused by the spec"))
                continue
            cell = cells[key]
            note = "forced by the aggregate" if cell["forced"] else ""
            print(
                row_fmt.format(
                    feature.name,
                    label,
                    "{:.3f}".format(cell["windows_differ"]),
                    "{:.3f}".format(cell["values_differ"]),
                    note,
                )
            )
    print("")
    print("shares are of scored rows, against event_time/left which is what the spec declares")
    if any_refused:
        print("")
        print(sorted(any_refused.values())[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
