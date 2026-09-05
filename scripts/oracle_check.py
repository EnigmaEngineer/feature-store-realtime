"""Grade the Spark job against the reference implementation, cell by cell.

    python scripts/oracle_check.py --accounts 400

Both arms read the same rows and the same spec. One is `featurestore/spark_job.py` and
the other is `featurestore/compute.py`, and where they disagree the reference one is
right. Rows are paired on `txn_id`, because an account can transact twice in one second
and then entity plus timestamp names two rows.

The controls are not optional. A comparison that reports zero differences looks the same
whether the job is correct or the two sides are being handed different rows. So the same
builder produces a damaged arm for every entry in `spark_job.DAMAGES`, and each one has
to disagree with the reference. If any of them agrees, this script has proved nothing
and exits nonzero.

The arms are read off `DAMAGES` rather than listed here. A list beside a catalogue is a
list of the entries somebody remembered.

Floats get their own column. A sliding window mean in Spark adds and removes terms as
the frame moves, and the reference adds a list from the left, so the two agree to about
machine precision rather than bit for bit. Counts and distinct counts and maxima carry
no arithmetic and are held to exact equality.
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import compute
from featurestore import corpus as corpus_io
from featurestore import spark_job, spec

REL_TOLERANCE = 1e-9
EXACT_AGGREGATES = ("count", "distinct_count", "max")


def close_enough(a, b):
    if a == b:
        return True
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= REL_TOLERANCE * scale


def pick_rows(rows, n_accounts):
    """The first n accounts by id, and every row belonging to them.

    Sorted rather than sampled at random so a re-run compares the same rows. Taking
    whole accounts matters because a window is per account, and half an account's
    history gives the reference a different answer from the full log.
    """
    if n_accounts is None:
        return rows
    accounts = sorted({r["account_id"] for r in rows})[:n_accounts]
    keep = set(accounts)
    return [r for r in rows if r["account_id"] in keep]


def tie_heavy_rows(accounts=20, per_account=150, seed=7):
    """A corpus built so that many transactions share a timestamp with a sibling.

    The generated corpus has 18 rows in a tie out of 489,972, which is 0.004 percent, so
    the agreement between a RANGE frame and a bisect on tied timestamps rests on almost
    nothing. A frame boundary is exactly where ties decide the answer, so the case gets
    its own input rather than a hope that the main corpus contains enough of it.

    Timestamps are drawn from a coarse grid on purpose. That is what forces collisions,
    and an hourly grid puts them exactly on the edge of the one hour windows. The grid
    spans eight days rather than one, because a shorter input leaves the 24 hour and 7
    day features insensitive to a damaged lower bound and the controls then look weaker
    than they are.
    """
    rng = random.Random(seed)
    grid = 3600
    slots = 8 * 24
    rows = []
    for a in range(accounts):
        account = "tie-{:03d}".format(a)
        base = 1756944000 + a * 37
        for i in range(per_account):
            ts = base + rng.randrange(0, slots) * grid
            rows.append(
                {
                    "txn_id": "t{}-{}".format(a, i),
                    "account_id": account,
                    "event_ts": ts,
                    "amount": round(rng.uniform(1.0, 400.0), 2),
                    "merchant_id": "m{}".format(rng.randrange(0, 12)),
                    "is_fraud": 1 if rng.random() < 0.02 else 0,
                }
            )
    rows.sort(key=lambda r: (r["event_ts"], r["txn_id"]))
    return rows


def tie_summary(rows):
    counts = {}
    for r in rows:
        key = (r["account_id"], r["event_ts"])
        counts[key] = counts.get(key, 0) + 1
    shared = {k: v for k, v in counts.items() if v > 1}
    return {
        "rows": len(rows),
        "pairs_shared": len(shared),
        "rows_in_a_tie": sum(shared.values()),
        "largest_tie": max(counts.values()),
    }


def reference_table(rows, feature_set):
    log = compute.EventLog(rows, feature_set.entity, feature_set.event_time)
    built = compute.training_rows(log, feature_set, rows)
    out = {}
    for source, vals in zip(rows, built):
        out[source["txn_id"]] = vals
    return out


def spark_table(spark, path, feature_set, damage=None):
    df = spark_job.read_log(spark, path)
    table = spark_job.materialize(df, feature_set, carry=("txn_id",), damage=damage)
    return {r["txn_id"]: r.asDict() for r in table.collect()}


def compare(reference, actual, feature_set):
    exact_by = {f.name: f.aggregate in EXACT_AGGREGATES for f in feature_set.features}
    report = {
        name: {"compared": 0, "exact": 0, "within_tolerance": 0, "differing": 0}
        for name in feature_set.feature_names()
    }
    examples = []
    missing = sorted(set(reference) - set(actual))
    extra = sorted(set(actual) - set(reference))

    for key in sorted(set(reference) & set(actual)):
        want = reference[key]
        got = actual[key]
        for name in feature_set.feature_names():
            cell = report[name]
            cell["compared"] += 1
            a, b = want[name], got[name]
            if a == b:
                cell["exact"] += 1
            elif not exact_by[name] and close_enough(float(a), float(b)):
                cell["within_tolerance"] += 1
            else:
                cell["differing"] += 1
                if len(examples) < 5:
                    examples.append(
                        {"txn_id": key, "feature": name, "reference": a, "spark": b}
                    )

    return {
        "rows_paired": len(set(reference) & set(actual)),
        "rows_missing_from_spark": len(missing),
        "rows_only_in_spark": len(extra),
        "by_feature": report,
        "examples": examples,
    }


def empty_window_rows(reference, feature_set):
    """Rows where every window admitted nothing.

    The empty frame is where Spark and the reference are most likely to differ, because
    Spark returns null from sum and avg and max and the reference returns 0. That branch
    is not reachable by any check in `tests/`, since building it needs a JVM. So the
    count is printed. A comparison that never met an empty window has not tested the
    coalesce, and nothing else would say so.
    """
    names = feature_set.feature_names()
    return sum(1 for vals in reference.values() if all(vals[n] == 0 for n in names))


def totals(result):
    out = {"compared": 0, "exact": 0, "within_tolerance": 0, "differing": 0}
    for cell in result["by_feature"].values():
        for k in out:
            out[k] += cell[k]
    return out


def print_arm(label, result):
    t = totals(result)
    print(
        "{:<23} {:>7} rows  {:>8} cells  exact {:>8}  within tol {:>5}  differing "
        "{:>8}".format(
            label,
            result["rows_paired"],
            t["compared"],
            t["exact"],
            t["within_tolerance"],
            t["differing"],
        )
    )
    if result["rows_missing_from_spark"] or result["rows_only_in_spark"]:
        print(
            "  unpaired rows: {} the reference has and spark does not, {} spark has "
            "and the reference does not".format(
                result["rows_missing_from_spark"], result["rows_only_in_spark"]
            )
        )
    for name, cell in sorted(result["by_feature"].items()):
        print(
            "  {:<22} exact {:>8}  within tol {:>5}  differing {:>8}".format(
                name, cell["exact"], cell["within_tolerance"], cell["differing"]
            )
        )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="specs/txn_fraud.yml")
    ap.add_argument("--corpus", default="data/transactions.csv")
    ap.add_argument("--accounts", type=int, default=400, help="0 means every account")
    ap.add_argument(
        "--ties",
        action="store_true",
        help="ignore the corpus and use a tie heavy input instead",
    )
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args(argv)

    feature_set = spec.load(args.spec)
    if args.ties:
        rows = tie_heavy_rows()
    else:
        rows = pick_rows(list(corpus_io.read_csv(args.corpus)), args.accounts or None)
    summary = tie_summary(rows)
    print(
        "{} rows over {} accounts, spec {} at {}".format(
            len(rows),
            len({r["account_id"] for r in rows}),
            feature_set.name,
            feature_set.fingerprint(),
        )
    )
    print(
        "{} rows share an entity and a timestamp with a sibling, largest tie {}".format(
            summary["rows_in_a_tie"], summary["largest_tie"]
        )
    )
    if summary["rows_in_a_tie"] == 0:
        # A frame boundary is where ties decide the answer, so a run over an input with
        # none of them says nothing about that case. Reported rather than refused,
        # because the main corpus is what it is and --ties is the answer.
        print("no ties in this input, so the tied timestamp case is not being checked")

    started = time.time()
    reference = reference_table(rows, feature_set)
    ref_s = time.time() - started
    empties = empty_window_rows(reference, feature_set)
    print("{} rows had an empty window, so the zero fallback is exercised".format(empties))
    if empties == 0:
        print("no empty windows here, so the null to zero path is not being checked")

    scratch = tempfile.mkdtemp(prefix="oracle-check-")
    slice_path = os.path.join(scratch, "slice.csv")
    corpus_io.write_csv(rows, slice_path)

    spark = spark_job.session(app_name="oracle-check")
    spark.sparkContext.setLogLevel("ERROR")
    arms = {}
    try:
        started = time.time()
        arms["shipped"] = spark_table(spark, slice_path, feature_set)
        shipped_s = time.time() - started
        for name in sorted(spark_job.DAMAGES):
            arms[name] = spark_table(spark, slice_path, feature_set, damage=name)
    finally:
        spark.stop()

    results = {name: compare(reference, table, feature_set) for name, table in arms.items()}

    print("reference built in {:.2f}s, spark arm in {:.2f}s".format(ref_s, shipped_s))
    print()
    print_arm("shipped", results["shipped"])
    for name in sorted(spark_job.DAMAGES):
        print()
        print_arm(name, results[name])

    for ex in results["shipped"]["examples"]:
        print("  disagreement {}".format(ex))

    report = {
        "rows": len(rows),
        "accounts": len({r["account_id"] for r in rows}),
        "fingerprint": feature_set.fingerprint(),
        "reference_seconds": round(ref_s, 3),
        "spark_seconds": round(shipped_s, 3),
        "ties": summary,
        "empty_window_rows": empties,
        "arms": results,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)

    blind = [
        name for name in spark_job.DAMAGES if totals(results[name])["differing"] == 0
    ]
    if blind:
        print()
        print(
            "CONTROL FAILED for {}. A damaged job agreed with the reference on every "
            "cell, so this comparison is not sensitive to that mistake.".format(
                ", ".join(sorted(blind))
            )
        )
        return 2

    shipped_totals = totals(results["shipped"])
    # A row Spark produced and the reference did not is a failure, and the first version
    # of this counted it and then never looked at it. Comparing only the intersection
    # means a job emitting extra rows scores perfectly. A mutation that turned the CSV
    # header off is what found it, because the header line then arrives as a row.
    unpaired = (
        results["shipped"]["rows_missing_from_spark"]
        + results["shipped"]["rows_only_in_spark"]
    )
    if shipped_totals["differing"] or unpaired:
        print()
        print("the spark job disagrees with the reference, see above")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
