"""What the serving path costs, and what an age threshold buys.

    python scripts/serve_probe.py --accounts 0

Two questions, both about the day the store gets an HTTP endpoint in front of it.

The first is arithmetic. `scripts/store_check.py` measured one HGETALL and a decode at a
p99 of about 0.29 ms. This adds the model, the framework and the socket and reports the
same quantile, so the gap is what the edge costs rather than a fresh number with no
reference behind it.

The second is the one that decides a design. `featurestore/service.Policy` can refuse a
vector older than a threshold. Whether it should is a measurement, not a preference, and
the measurement is a sweep. For each candidate threshold, how many real requests does it
refuse, and how wrong are the ones it still serves. A guardrail is judged first on what it
refuses that it should not.

The requests are real. Every transaction after the load moment is one the service would
have answered, and the training table holds the row the model was taught to expect for it.
So a served response can be scored against the score the same model gives the point in
time vector, which is the answer the request should have had.

That comparison is the thing the age threshold is a proxy for. The whole question is how
good a proxy it is.
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import api, compute, load, service, stats
from featurestore import corpus as corpus_io
from featurestore import model as model_mod
from featurestore import online, spec

HOUR = 3600
DAY = compute.DAY

# Candidate thresholds, plus None which is what ships. Chosen to span the buckets the
# staleness table already uses rather than fitted to anything, because a threshold picked
# off the numbers it is about to be judged against is not a measurement.
CANDIDATES = (None, HOUR, 6 * HOUR, DAY, 7 * DAY)


def pick_rows(rows, n_accounts):
    if n_accounts is None:
        return rows
    keep = set(sorted({r["account_id"] for r in rows})[:n_accounts])
    return [r for r in rows if r["account_id"] in keep]


def label_for(max_age_s):
    if max_age_s is None:
        return "no threshold"
    if max_age_s % DAY == 0:
        return "{}d".format(max_age_s // DAY)
    return "{}h".format(max_age_s // HOUR)


# One definition of a quantile, in the package. It lived here until `scripts/load_probe.py`
# also needed a p99, and two modules reporting one off two definitions is a disagreement
# neither half looks wrong on its own.
quantiles = load.quantiles


def truth_scores(log, feature_set, model, requests):
    """The score each request should have had, from its own point in time vector.

    This is the training table's row for that transaction, put through the same model.
    Nothing about the store is in it.
    """
    entity = feature_set.entity
    time_field = feature_set.event_time
    out = []
    for row in requests:
        values = compute.features_as_of(log, feature_set, row[entity], row[time_field])
        out.append(model.score(values))
    return out


def sweep(served, wanted, ages, threshold, flip_at):
    """One row of the table. What it refuses, what it serves, and how wrong each half is.

    `flip_at` is the model threshold, so a decision flip is a request the service answered
    with the opposite action to the one its own point in time vector called for. That is
    the only error measure here with a consequence attached, and it is much smaller than
    the share of scores that moved at all.

    Both halves get a flip rate. A threshold that refuses a set flipping at the same rate
    as the set it keeps has separated nothing, and reporting only the served side hides
    that. This is the column the first version of this function did not have.
    """
    kept_gaps = []
    flips = 0
    refused = 0
    refused_flips = 0
    for got, want, age in zip(served, wanted, ages):
        flipped = (got >= flip_at) != (want >= flip_at)
        if threshold is not None and age > threshold:
            refused += 1
            refused_flips += 1 if flipped else 0
            continue
        kept_gaps.append(abs(got - want))
        flips += 1 if flipped else 0
    total = len(served)
    kept = total - refused
    low, high = stats.binomial_interval(flips, kept)
    return {
        "threshold": threshold,
        "label": label_for(threshold),
        # A threshold longer than the span of requests cannot refuse anything, so its row
        # is a restatement of the no threshold row rather than an observation about it.
        "binds": threshold is not None and refused > 0,
        "requests": total,
        "refused": refused,
        "refused_share": refused / total if total else 0.0,
        "refused_flips": refused_flips,
        "refused_flip_share": refused_flips / refused if refused else 0.0,
        "served": kept,
        "flips": flips,
        "flip_share": flips / kept if kept else 0.0,
        "flip_share_low": low,
        "flip_share_high": high,
        "mean_abs_gap": statistics.fmean(kept_gaps) if kept_gaps else 0.0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="specs/txn_fraud.yml")
    ap.add_argument("--model", default="models/txn_fraud.json")
    ap.add_argument("--corpus", default="data/transactions.csv")
    ap.add_argument("--accounts", type=int, default=200, help="0 means every account")
    ap.add_argument("--at", type=float, default=0.9, help="load the store this far through the corpus")
    ap.add_argument("--latency-calls", type=int, default=3000)
    ap.add_argument("--rdb", default=os.path.join("/tmp", "fsr_serve_probe"))
    ap.add_argument("--port", default="6404")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args(argv)

    import redis
    import redislite
    from fastapi.testclient import TestClient

    feature_set = spec.load(args.spec)
    model = model_mod.load(args.model).bind(feature_set)
    time_field = feature_set.event_time
    entity = feature_set.entity

    rows = pick_rows(list(corpus_io.read_csv(args.corpus)), args.accounts or None)
    log = compute.EventLog(rows, feature_set.entity, time_field)
    times = sorted(r[time_field] for r in rows)
    as_of = times[int(len(times) * args.at)]

    os.makedirs(args.rdb, exist_ok=True)
    server = redislite.Redis(os.path.join(args.rdb, "probe.rdb"), serverconfig={"port": args.port})
    try:
        client = redis.Redis(host="127.0.0.1", port=int(args.port))
        client.flushall()
        written = online.load(client, log, feature_set, as_of)

        svc = service.Service(client, feature_set, model, service.Policy())
        sink = []
        http = TestClient(api.build_app(svc, sink=sink.append))

        requests = sorted((r for r in rows if r[time_field] >= as_of), key=lambda r: r[time_field])
        print(
            "{} rows over {} entities, spec {} at {}, model {}".format(
                len(rows), written, feature_set.name, feature_set.fingerprint(), model.fingerprint
            )
        )
        print(
            "store loaded at {} which is {:.0%} through the corpus, {} requests arrive "
            "after it".format(as_of, args.at, len(requests))
        )

        # The control. A request timed at the load moment itself must get the score its
        # own point in time vector gives, or the table below is measuring a broken
        # comparison rather than age.
        stored = {eid: vals for eid, vals in online.online_rows(log, feature_set, as_of)}
        control_gap = 0.0
        for eid in sorted(stored):
            got = model.score(stored[eid])
            want = model.score(compute.features_as_of(log, feature_set, eid, as_of))
            control_gap = max(control_gap, abs(got - want))
        print("control at the load moment: {} entities, largest score gap {:.3e}".format(
            len(stored), control_gap))
        if control_gap > 1e-12:
            print("CONTROL FAILED. The store and the reference disagree at the load moment.")
            return 2

        # Latency through the whole edge, against the store floor measured without one.
        # A warmup pass is discarded because the first request through this stack pays for
        # the route table and the encoder, and charging that to whichever arm ran first is
        # a mistake this program has made three times.
        warm = requests[0]
        for _ in range(50):
            http.post("/score", json={"entity_id": warm[entity], "event_ts": warm[time_field]})
        sink.clear()

        sample = requests[: args.latency_calls]
        http_ms = []
        statuses = {}
        for row in sample:
            started = time.perf_counter()
            resp = http.post("/score", json={"entity_id": row[entity], "event_ts": row[time_field]})
            http_ms.append((time.perf_counter() - started) * 1000.0)
            statuses[resp.status_code] = statuses.get(resp.status_code, 0) + 1
        service_ms = [r["service_ms"] for r in sink if "service_ms" in r]

        p50, p95, p99, worst = quantiles(http_ms)
        s50, s95, s99, _ = quantiles(service_ms)
        print()
        print("{} requests through the app, statuses {}".format(len(sample), dict(sorted(statuses.items()))))
        print("{:<26} {:>8} {:>8} {:>8} {:>8}".format("", "p50", "p95", "p99", "max"))
        print("{:<26} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f}".format("through the app, ms", p50, p95, p99, worst))
        print("{:<26} {:>8.3f} {:>8.3f} {:>8.3f}".format("inside the service, ms", s50, s95, s99))
        print("the edge is {:.1f}x the service on the p99".format(p99 / s99 if s99 else 0.0))

        # Every request answered, and what the point in time answer would have been.
        answered = []
        ages = []
        for row in requests:
            try:
                values, row_as_of = online.lookup(client, feature_set, row[entity])
            except online.Miss:
                continue
            answered.append((row, model.score(values)))
            ages.append(row[time_field] - row_as_of)
        served = [s for _, s in answered]
        wanted = truth_scores(log, feature_set, model, [r for r, _ in answered])

        print()
        print(
            "{:<13} {:>8} {:>7} {:>8} {:>7} {:>7} {:>18} {:>12}".format(
                "max age", "refused", "share", "served", "flips", "rate", "95% interval",
                "refused rate",
            )
        )
        table = []
        for candidate in CANDIDATES:
            row = sweep(served, wanted, ages, candidate, model.threshold)
            table.append(row)
            interval = "{:.4f} to {:.4f}".format(row["flip_share_low"], row["flip_share_high"])
            refused_rate = "{:.4f}".format(row["refused_flip_share"]) if row["refused"] else "n/a"
            print(
                "{:<13} {:>8} {:>7.3f} {:>8} {:>7} {:>7.4f} {:>18} {:>12}".format(
                    row["label"], row["refused"], row["refused_share"], row["served"],
                    row["flips"], row["flip_share"], interval, refused_rate,
                )
            )
        dead = [r["label"] for r in table if r["threshold"] is not None and not r["binds"]]
        if dead:
            print(
                "{} cannot refuse anything on this corpus, because the requests span "
                "less than that. Those rows restate the first one.".format(", ".join(dead))
            )

        # What the clock is standing in for. A vector is wrong because events arrived
        # after it was computed, and age is only a proxy for that. Splitting the same
        # requests on whether anything arrived says how good a proxy, and it is the
        # measurement that decides whether the answer is a shorter threshold or a
        # different signal entirely.
        # Half open, and the open end is the whole point. `between` with an inclusive top
        # counts the request's own transaction, which every request has by definition, so
        # the first version of this reported zero quiet requests out of 48,998 and the
        # split said nothing. Every window in this spec is closed on the left, so a
        # vector computed at the request's own timestamp would not see that row either.
        # A sibling at exactly the same second is excluded with it, and there are 18 tied
        # rows in 489,972.
        quiet_total = quiet_flips = busy_total = busy_flips = 0
        for (row, got), want in zip(answered, wanted):
            arrived = len(log.between(row[entity], as_of, row[time_field], False))
            flipped = (got >= model.threshold) != (want >= model.threshold)
            if arrived:
                busy_total += 1
                busy_flips += 1 if flipped else 0
            else:
                quiet_total += 1
                quiet_flips += 1 if flipped else 0
        print()
        print("{:<34} {:>9} {:>7} {:>7} {:>18}".format(
            "requests split on new arrivals", "requests", "flips", "rate", "95% interval"))
        for name, tot, flip in (
            ("nothing arrived since the load", quiet_total, quiet_flips),
            ("something arrived since the load", busy_total, busy_flips),
        ):
            low, high = stats.binomial_interval(flip, tot)
            print("{:<34} {:>9} {:>7} {:>7.4f} {:>18}".format(
                name, tot, flip, flip / tot if tot else 0.0, "{:.4f} to {:.4f}".format(low, high)))
        # Why the quiet side is not zero. A trailing window sheds rows off its back edge
        # whether or not anything arrives, so a vector rots with the clock alone. Six of
        # these can only fall, because removing rows from a count or a sum or a max or a
        # distinct count cannot raise it. A mean is the exception and it is the one worth
        # knowing, since dropping a below average transaction pushes it up.
        moved = {name: 0 for name in feature_set.feature_names()}
        upward = {name: 0 for name in feature_set.feature_names()}
        quiet_moved = 0
        for row, _ in answered:
            if log.between(row[entity], as_of, row[time_field], False):
                continue
            truth = compute.features_as_of(log, feature_set, row[entity], row[time_field])
            held = stored[row[entity]]
            differing = [n for n in moved if truth[n] != held[n]]
            if differing:
                quiet_moved += 1
            for name in differing:
                moved[name] += 1
                if truth[name] > held[name]:
                    upward[name] += 1
        print()
        print(
            "of the {} quiet requests the stored vector had already moved on {} "
            "({:.4f}), with nothing arriving".format(
                quiet_total, quiet_moved, quiet_moved / quiet_total if quiet_total else 0.0
            )
        )
        print("{:<24} {:>10} {:>14}".format("feature", "moved", "moved upward"))
        for name in feature_set.feature_names():
            print("{:<24} {:>10} {:>14}".format(name, moved[name], upward[name]))

        # A flip is defined by the model threshold, which ships at 0.5 because that is the
        # default and not because anybody measured it. So the headline is a function of a
        # number nobody chose, and the sweep below is what says whether the conclusion is
        # about staleness or about 0.5.
        print()
        print("the same table at other model thresholds, because a flip is defined by one")
        print("{:<12} {:>10} {:>11} {:>11} {:>11}".format(
            "threshold", "all", "1h served", "6h served", "1d served"))
        sensitivity = []
        for candidate_threshold in (0.10, 0.25, 0.50, 0.75, 0.90):
            row = {"threshold": candidate_threshold}
            cells = []
            for cut in (None, HOUR, 6 * HOUR, DAY):
                pairs = [
                    (got, want)
                    for got, want, age in zip(served, wanted, ages)
                    if cut is None or age <= cut
                ]
                flipped = sum(
                    1 for got, want in pairs
                    if (got >= candidate_threshold) != (want >= candidate_threshold)
                )
                rate = flipped / len(pairs) if pairs else 0.0
                row["all" if cut is None else label_for(cut)] = rate
                cells.append(rate)
            sensitivity.append(row)
            print("{:<12.2f} {:>10.4f} {:>11.4f} {:>11.4f} {:>11.4f}".format(
                candidate_threshold, *cells))
        # Derived rather than asserted. The first version of this line said a shorter cut
        # is better at every threshold, and the table printed directly above it says
        # otherwise at 0.50, where the six hour cut is worse than no cut at all.
        beats = {}
        for cut_label in ("1h", "6h", "1d"):
            beats[cut_label] = sum(
                1 for row in sensitivity if row[cut_label] < row["all"]
            )
        print(
            "the level is a property of the threshold, not of staleness. It moves by a "
            "factor of {:.0f} across these five.".format(
                max(r["all"] for r in sensitivity) / min(r["all"] for r in sensitivity)
            )
        )
        print(
            "cuts that beat no cut at all, out of {} thresholds: {}".format(
                len(sensitivity),
                ", ".join("{} {}".format(k, v) for k, v in beats.items()),
            )
        )

        report_arrivals = {
            "quiet_requests": quiet_total,
            "quiet_flips": quiet_flips,
            "busy_requests": busy_total,
            "busy_flips": busy_flips,
            "quiet_vectors_moved": quiet_moved,
            "moved_by_feature": moved,
            "moved_upward_by_feature": upward,
        }

        report = {
            "as_of": as_of,
            "entities": written,
            "requests": len(requests),
            "answered": len(answered),
            "control_gap": control_gap,
            "http_ms": {"p50": p50, "p95": p95, "p99": p99, "max": worst},
            "service_ms": {"p50": s50, "p95": s95, "p99": s99},
            "sweep": table,
            "arrivals": report_arrivals,
            "threshold_sensitivity": sensitivity,
        }
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, sort_keys=True)
        return 0
    finally:
        server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
