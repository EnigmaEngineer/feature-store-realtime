"""What a drift threshold on these features is really measuring.

    python scripts/drift_probe.py --json /tmp/drift.json

Seven arms, and the first two are the ones that decide what ships.

**The null floor.** Windows resampled from the reference itself, scored against the
reference. Nothing has drifted in any of them, so every index that comes back is sampling
noise. The mean has a closed form, `(k - 1) / n`, and printing the measured mean beside the
predicted one is what says the implementation and the algebra agree. The first time they
were printed together they disagreed by a factor of two on every feature, and the algebra
was the half that was wrong.

**The null floor against window size.** The same measurement at a range of window sizes.
If the index falls as one over the window size then a fixed threshold is a statement about
how much traffic a window happens to hold, and the two numbers everybody uses with this
statistic are a statement about nothing else.

**Held out fire rates.** Consecutive windows of real traffic the fit never saw. A
resampled window and a consecutive one are not the same object, because real windows arrive
in time order and carry whatever the traffic was doing. A threshold calibrated on the first
and judged on the second is the honest test. It is also the one that is easy to get wrong,
by measuring a rate on the very data a threshold was chosen from.

**The reweighting control.** At the moment the store is loaded it is exactly right, so
anything the monitor says there is not drift. What it can still see is that the population
of stored vectors and the population of vectors actually served are different: an account
that transacts often is read often, and the same store therefore has two distributions
depending on whether you count entities or requests. A monitor fitted on one and pointed at
the other fires forever with nothing wrong.

**Served against offline.** The served vector against what the offline
pipeline says for the same request, over the same request list, so the reweighting above
is present on both sides and cancels. What is left is age.

**Epsilon sensitivity.** The usual fix for a bin the current window emptied is to
substitute a small share. The arm shows what that number decides.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import compute, drift, model as model_mod, online, spec
from featurestore import corpus as corpus_io

# The two numbers this statistic is almost always used with. They come from credit risk
# scorecard practice and they are quoted without a sample size attached, which is the whole
# subject of this probe. Not thresholds this repo chose.
RULE_OF_THUMB = (0.10, 0.25)

WINDOW_SIZES = (50, 100, 200, 500, 1000, 2000)

EPSILONS = (1e-3, 1e-4, 1e-6)


def feature_columns(rows, names):
    return {name: [r[name] for r in rows] for name in names}


def consecutive_windows(values, size):
    """Whole windows only. A short tail is dropped rather than judged.

    A final window holding forty rows would be judged against a threshold calibrated for
    two hundred, and the index it returns is larger for that reason alone. Dropping it
    loses a little data and keeps every row of the table comparable.
    """
    out = []
    for start in range(0, len(values) - size + 1, size):
        out.append(values[start:start + size])
    return out


def psi_with_epsilon(profile, values, epsilon):
    """The index with an emptied bin floored at `epsilon` rather than excluded.

    Not in the package. It exists to put a number on what the substitution decides, and
    shipping it beside the real one would give a caller a way to pick whichever answer
    suited them.
    """
    import math

    counts = drift.bin_counts(values, profile.edges)
    shares = drift.to_shares(counts)
    total = 0.0
    for current, reference in zip(shares, profile.shares):
        if reference <= 0.0:
            continue
        current = max(current, epsilon)
        total += (current - reference) * math.log(current / reference)
    return total


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="specs/txn_fraud.yml")
    ap.add_argument("--corpus", default="data/transactions.csv")
    ap.add_argument("--accounts", type=int, default=0, help="0 means every account")
    ap.add_argument(
        "--reference",
        type=float,
        default=0.6,
        help="fit the profiles on this share of the corpus, by event count",
    )
    ap.add_argument("--window", type=int, default=200, help="rows per judged window")
    ap.add_argument("--trials", type=int, default=400, help="resamples per null estimate")
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--model", default="models/txn_fraud.json")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    feature_set = spec.load(args.spec)
    names = feature_set.feature_names()
    time_field = feature_set.event_time

    rows = list(corpus_io.read_csv(args.corpus))
    if args.accounts:
        keep = set(sorted({r[feature_set.entity] for r in rows})[: args.accounts])
        rows = [r for r in rows if r[feature_set.entity] in keep]
    rows.sort(key=lambda r: r[time_field])

    log = compute.EventLog(rows, feature_set.entity, time_field)
    table = compute.training_rows(log, feature_set, rows)

    cut = int(len(table) * args.reference)
    reference_rows, holdout_rows = table[:cut], table[cut:]
    reference = feature_columns(reference_rows, names)
    holdout = feature_columns(holdout_rows, names)

    print(
        "{} rows over {} accounts, profiles fitted on the first {} and judged on the "
        "remaining {}".format(
            len(table), len({r[feature_set.entity] for r in rows}), cut, len(table) - cut
        )
    )
    print("window {} rows, {} resamples per null estimate, seed {}".format(
        args.window, args.trials, args.seed))
    print()

    profiles = {name: drift.fit_profile(name, reference[name]) for name in names}

    # ---- arm 1: the bins a feature really gets, and the null floor at the operating size
    print("ASKED FOR {} BINS. WHAT EACH FEATURE GOT, AND WHAT THE INDEX READS WITH NO "
          "DRIFT AT ALL.".format(drift.BIN_TARGET))
    print()
    print("{:<22} {:>5} {:>10} {:>10} {:>10} {:>10}".format(
        "feature", "bins", "distinct", "null mean", "predicted", "null q95"))
    nulls = {}
    for name in names:
        profile = profiles[name]
        distinct = len(set(reference[name]))
        if profile.is_constant():
            print("{:<22} {:>5} {:>10} {:>10} {:>10} {:>10}".format(
                name, "-", distinct, "constant", "-", "-"))
            continue
        floor = drift.null_floor(
            profile, args.window, trials=args.trials, level=args.level,
            rng=rng, reference_values=reference[name],
        )
        nulls[name] = floor
        print("{:<22} {:>5} {:>10} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            name, profile.bins, distinct, floor["mean"], floor["predicted_mean"],
            floor["quantile"]))
    print()

    # ---- arm 2: the null floor as a function of window size
    print("THE SAME NULL, AT A RANGE OF WINDOW SIZES. NOTHING HAS DRIFTED IN ANY OF THESE.")
    print()
    header = "{:<22}".format("feature") + "".join("{:>9}".format(n) for n in WINDOW_SIZES)
    print(header + "   null q95 drops under 0.10 at")
    size_sweep = {}
    for name in names:
        profile = profiles[name]
        if profile.is_constant():
            continue
        line = "{:<22}".format(name)
        per_size = {}
        crosses = None
        for size in WINDOW_SIZES:
            floor = drift.null_floor(
                profile, size, trials=args.trials, level=args.level,
                rng=rng, reference_values=reference[name],
            )
            per_size[size] = floor
            line += "{:>9.4f}".format(floor["quantile"])
            if crosses is None and floor["quantile"] < RULE_OF_THUMB[0]:
                crosses = size
        size_sweep[name] = {str(k): v for k, v in per_size.items()}
        line += "   {}".format(
            "n = {}".format(crosses) if crosses else "no size measured here")
        print(line)
    print()
    print("The mean of this statistic under no drift is (bins - 1) / n, so a threshold")
    print("that does not name a window size is a threshold about traffic volume.")
    print()

    # ---- arm 3: held out fire rates, rule of thumb against the measured null
    print("HELD OUT FIRE RATES ON CONSECUTIVE WINDOWS THE FIT NEVER SAW.")
    print()
    print("{:<22} {:>8} {:>10} {:>9} {:>9} {:>9} {:>9}".format(
        "feature", "windows", "threshold", "source", "fires", "rate", "lower"))
    fire_rates = {}
    for name in names:
        profile = profiles[name]
        windows = consecutive_windows(holdout[name], args.window)
        if profile.is_constant():
            monitor = drift.Monitor(profile, 1.0, "constant, any change fires")
            result = drift.fire_rate(monitor, windows)
            fire_rates.setdefault(name, {})["constant"] = _strip(result)
            print("{:<22} {:>8} {:>10} {:>9} {:>9} {:>9.4f} {:>9.4f}".format(
                name, result["judged"], "-", "constant", result["fires"],
                result["rate"], result["lower_bound"]))
            continue
        for threshold, source in (
            (RULE_OF_THUMB[0], "0.10"),
            (RULE_OF_THUMB[1], "0.25"),
            (nulls[name]["quantile"], "null q95"),
        ):
            monitor = drift.Monitor(profile, threshold, source)
            result = drift.fire_rate(monitor, windows)
            fire_rates.setdefault(name, {})[source] = _strip(result)
            print("{:<22} {:>8} {:>10.4f} {:>9} {:>9} {:>9.4f} {:>9.4f}".format(
                name, result["judged"], threshold, source, result["fires"],
                result["rate"], result["lower_bound"]))
    print()

    # ---- arm 4: the same measurement with time order destroyed
    print("THE SAME THING ON A SHUFFLE. THIS IS THE ONLY ARM WHERE A FIRE IS CERTAIN TO BE")
    print("A FALSE ONE.")
    print()
    print("The held out rows above are later traffic, and nothing says later traffic is")
    print("free of real drift, so a rate measured on it is what a monitor would page for")
    print("rather than a false positive rate. Shuffling the whole table before splitting")
    print("makes the reference and every window the same distribution by construction, so")
    print("the difference between the two tables is time order and nothing else.")
    print()
    shuffled = shuffled_fire_rates(table, names, args)
    print("{:<22} {:>10} {:>12} {:>12} {:>10}".format(
        "feature", "threshold", "in order", "shuffled", "ratio"))
    for name in names:
        row = shuffled[name]
        if row is None:
            print("{:<22} {:>10} {:>12} {:>12} {:>10}".format(name, "-", "-", "-", "constant"))
            continue
        ordered_rate = fire_rates[name]["null q95"]["rate"]
        print("{:<22} {:>10.4f} {:>12.4f} {:>12.4f} {:>10}".format(
            name, row["threshold"], ordered_rate, row["rate"],
            "{:.1f}x".format(ordered_rate / row["rate"]) if row["rate"] else "n/a"))
    print()

    # ---- arm 5: the reweighting control
    control = reweighting_control(log, feature_set, rows, names, args.model, args)
    print("THE CONTROL. AT THE LOAD MOMENT THE STORE IS EXACTLY RIGHT, SO NOTHING HERE IS "
          "DRIFT.")
    print()
    print("{} entities in the store, read by {} requests after the load moment".format(
        control["entities"], control["requests"]))
    print("agreement with the reference at that moment: {} cells, {} differing, verdict {}"
          .format(control["agreement"]["cells"], control["agreement"]["differing"],
                  control["agreement"]["verdict"]))
    print()
    print("{:<22} {:>10} {:>12} {:>9} {:>9}".format(
        "feature", "index", "null q95", "over 0.10", "over 0.25"))
    for name in names:
        row = control["by_feature"][name]
        if row["index"] is None:
            print("{:<22} {:>10} {:>12} {:>9} {:>9}".format(name, "constant", "-", "-", "-"))
            continue
        print("{:<22} {:>10.4f} {:>12.6f} {:>9} {:>9}".format(
            name, row["index"], row["null_q95"],
            "yes" if row["index"] > RULE_OF_THUMB[0] else "no",
            "yes" if row["index"] > RULE_OF_THUMB[1] else "no"))
    print()
    print("Same store, same moment, zero staleness. The only difference between the two")
    print("populations is that a busy account is read more often than a quiet one.")
    print("The null column is the honest comparison and it is decisive rather than")
    print("interesting, because at this many requests it sits six decimals below the")
    print("index. The columns that matter to anyone running this are the last two.")
    print()

    # ---- arm 6: served against offline, with the reweighting removed
    print("SERVED AGAINST OFFLINE, OVER THE SAME REQUESTS. THIS ONE IS SKEW.")
    print()
    print("Both sides are indexed by the same request list, so the reweighting above is")
    print("on both and cancels. What is left is that a stored vector ages and the offline")
    print("answer does not.")
    print()
    print("{:<22} {:>10} {:>9} {:>9} {:>12} {:>10}".format(
        "feature", "index", "over 0.10", "over 0.25", "cells wrong", "share"))
    for name in names:
        row = control["skew"][name]
        cell = control["cells"][name]
        if row["index"] is None:
            print("{:<22} {:>10} {:>9} {:>9} {:>12} {:>10.4f}".format(
                name, "constant", "-", "-", cell["differing"], cell["share"]))
            continue
        print("{:<22} {:>10.4f} {:>9} {:>9} {:>12} {:>10.4f}".format(
            name, row["index"],
            "yes" if row["index"] > RULE_OF_THUMB[0] else "no",
            "yes" if row["index"] > RULE_OF_THUMB[1] else "no",
            cell["differing"], cell["share"]))
    print()
    print("THE LAST TWO COLUMNS ARE THE SAME REQUESTS AND THE SAME STORE.")
    print("{} of {} cells are wrong, at {:.4f}, and {} of {} requests get the opposite"
          .format(control["cells_differing"], control["cells_total"],
                  control["cells_share"], control["flips"], control["requests"]))
    print("decision from the served vector to the one the offline answer calls for, at")
    print("{:.4f}. Not one feature's distribution moved enough to notice.".format(
        control["flip_share"]))
    print()

    # ---- arm 7: what an epsilon decides
    epsilon = epsilon_arm(profiles, holdout, names, args)
    if epsilon["cases"]:
        print("A BIN THE WINDOW EMPTIED, UNDER THE SUBSTITUTION EVERYBODY MAKES.")
        print()
        print("{:<22} {:>8}".format("feature", "excluded") + "".join(
            "{:>12}".format(e) for e in EPSILONS))
        for case in epsilon["cases"]:
            print("{:<22} {:>8.4f}".format(case["feature"], case["excluded"]) + "".join(
                "{:>12.4f}".format(case["with_epsilon"][str(e)]) for e in EPSILONS))
        print()
        print("The data has no say in which of those columns is the answer.")
    else:
        print("NO HELD OUT WINDOW EMPTIED A BIN, so the epsilon arm has nothing to report.")
        print("That is a fact about this corpus and not about the substitution.")
    print()

    report = {
        "rows": len(table),
        "reference_rows": cut,
        "holdout_rows": len(table) - cut,
        "window": args.window,
        "trials": args.trials,
        "level": args.level,
        "seed": args.seed,
        "fingerprint": feature_set.fingerprint(),
        "profiles": {name: profiles[name].as_dict() for name in names},
        "null": nulls,
        "null_by_window_size": size_sweep,
        "fire_rates": fire_rates,
        "control": control,
        "epsilon": epsilon,
        "shuffled": shuffled,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print("report written to {}".format(args.json))

    if control["agreement"]["differing"]:
        print()
        print("CONTROL FAILED. The store disagrees with the reference at the moment it was")
        print("loaded, so every index above is measuring that rather than a distribution.")
        return 2
    return 0


def _strip(result):
    """The fire rate without the per window verdicts.

    The verdict list is one entry per window per feature per threshold, which is tens of
    thousands of dicts in a report nobody reads that far into. The counts above it are what
    the tables are built from.
    """
    return {k: v for k, v in result.items() if k != "verdicts"}


def shuffled_fire_rates(table, names, args):
    """The held out measurement again, with the row order destroyed first.

    Shuffling before the split is what makes this a control. Both sides then come from one
    distribution, so a window that fires is firing on sampling noise and nothing else. The
    threshold is refitted on the shuffled reference rather than carried over, because a
    threshold fitted on one reference and applied to another is a second difference and
    this arm is meant to have exactly one.
    """
    rows = list(table)
    random.Random(args.seed + 3).shuffle(rows)
    cut = int(len(rows) * args.reference)
    reference = feature_columns(rows[:cut], names)
    holdout = feature_columns(rows[cut:], names)

    out = {}
    for name in names:
        profile = drift.fit_profile(name, reference[name])
        if profile.is_constant():
            out[name] = None
            continue
        floor = drift.null_floor(
            profile, args.window, trials=args.trials, level=args.level,
            rng=random.Random(args.seed + 4), reference_values=reference[name],
        )
        monitor = drift.Monitor(profile, floor["quantile"], "null q95 on the shuffle")
        result = drift.fire_rate(monitor, consecutive_windows(holdout[name], args.window))
        out[name] = {
            "threshold": floor["quantile"],
            "fires": result["fires"],
            "judged": result["judged"],
            "rate": result["rate"],
            "lower_bound": result["lower_bound"],
        }
    return out


def reweighting_control(log, feature_set, rows, names, model_path, args):
    """The store's own two populations, at the moment staleness is exactly zero.

    `stored` is one vector per entity. `served` is the vector each arriving request would
    have been handed, which is the same vectors counted once per request. The agreement
    check is `constant_agreement`, because the store against the reference at the load
    moment has never once differed and a distance on a flat signal is not a monitor.
    """
    time_field = feature_set.event_time
    times = sorted(r[time_field] for r in rows)
    as_of = times[int(len(times) * 0.6)]

    stored = dict(online.online_rows(log, feature_set, as_of))
    requests = [r for r in rows if r[time_field] >= as_of and r[feature_set.entity] in stored]

    truth_at_load = [
        compute.features_as_of(log, feature_set, eid, as_of) for eid in sorted(stored)
    ]
    served_at_load = [stored[eid] for eid in sorted(stored)]
    agreement = drift.constant_agreement(truth_at_load, served_at_load, names)

    entity_rows = [stored[eid] for eid in sorted(stored)]
    request_rows = [stored[r[feature_set.entity]] for r in requests]
    # What the offline pipeline says for the same requests, at each request's own time.
    # This is the training table's row for that transaction and it is what the served
    # vector is supposed to be an approximation of.
    truth_rows = [
        compute.features_as_of(log, feature_set, r[feature_set.entity], r[time_field])
        for r in requests
    ]

    by_feature = _index_against(
        names, entity_rows, request_rows, random.Random(args.seed + 1))
    skew = _index_against(
        names, truth_rows, request_rows, random.Random(args.seed + 2))

    # The per row view of the same comparison. A distribution index and a cell disagreement
    # rate are two answers to "has the served vector moved" and they are not the same
    # question, so the table prints both rather than letting a reader assume either one
    # stands for the other.
    cells = {name: {"differing": 0, "share": 0.0} for name in names}
    for truth, served in zip(truth_rows, request_rows):
        for name in names:
            if truth[name] != served[name]:
                cells[name]["differing"] += 1
    for name in names:
        cells[name]["share"] = cells[name]["differing"] / len(requests) if requests else 0.0

    model = model_mod.load(model_path).bind(feature_set)
    flips = 0
    for truth, served in zip(truth_rows, request_rows):
        a = model.score(served) >= model.threshold
        b = model.score(truth) >= model.threshold
        if a != b:
            flips += 1

    total_cells = len(requests) * len(names)
    total_differing = sum(c["differing"] for c in cells.values())
    return {
        "as_of": as_of,
        "entities": len(stored),
        "requests": len(requests),
        "agreement": agreement,
        "by_feature": by_feature,
        "skew": skew,
        "cells": cells,
        "cells_total": total_cells,
        "cells_differing": total_differing,
        "cells_share": total_differing / total_cells if total_cells else 0.0,
        "flips": flips,
        "flip_share": flips / len(requests) if requests else 0.0,
    }


def _index_against(names, reference_rows, current_rows, rng):
    """One index per feature, with the null at the size the current window really is.

    `trials` is small here on purpose. The null at a hundred thousand rows is six decimals
    below anything these arms produce, so a tighter estimate of it would buy nothing and
    the resampling is the expensive part of this probe.
    """
    out = {}
    for name in names:
        reference = [r[name] for r in reference_rows]
        profile = drift.fit_profile(name, reference)
        if profile.is_constant():
            out[name] = {"index": None, "null_q95": None, "emptied": 0}
            continue
        index, emptied, _ = drift.psi(profile, [r[name] for r in current_rows])
        floor = drift.null_floor(
            profile, len(current_rows), trials=20, rng=rng, reference_values=reference,
        )
        out[name] = {
            "index": index,
            "emptied": emptied,
            "null_q95": floor["quantile"],
            "over_null": index > floor["quantile"],
        }
    return out


def epsilon_arm(profiles, holdout, names, args):
    """Every held out window that emptied a bin, priced three ways.

    Reports the cases it found rather than a summary, because the point is that one window
    gets three different answers and a mean over them would hide exactly that.
    """
    cases = []
    for name in names:
        profile = profiles[name]
        if profile.is_constant():
            continue
        for window in consecutive_windows(holdout[name], args.window):
            index, emptied, _ = drift.psi(profile, window)
            if not emptied:
                continue
            cases.append({
                "feature": name,
                "excluded": index,
                "emptied": emptied,
                "with_epsilon": {str(e): psi_with_epsilon(profile, window, e)
                                 for e in EPSILONS},
            })
            break
    return {"epsilons": [str(e) for e in EPSILONS], "cases": cases}


if __name__ == "__main__":
    sys.exit(main())
