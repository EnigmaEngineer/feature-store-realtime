"""How the features are assembled changes the plan, so measure it rather than argue.

    python scripts/plan_probe.py

Two arms build the same seven features from the same `spark_job.plan`. One adds them with
a withColumn per feature, which is how everybody writes this. The other puts all seven in
one select. Only the assembly differs, so the losing arm is not a rewrite from memory.

Both arms are run once and discarded before anything is timed, and then alternated, so the
first arm does not get charged for the process warming up.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import spark_job, spec

ARMS = (("one_select", True), ("withcolumn_chain", False))


def operator_counts(table):
    plan = table._jdf.queryExecution().executedPlan().toString()
    return {
        "window": plan.count("Window "),
        "sort": plan.count("Sort ["),
        "exchange": plan.count("Exchange "),
    }


def timed_write(table, out):
    started = time.time()
    table.write.mode("overwrite").parquet(out)
    return time.time() - started


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="specs/txn_fraud.yml")
    ap.add_argument("--corpus", default="data/transactions.csv")
    ap.add_argument("--out", default="/tmp/plan-probe")
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args(argv)

    feature_set = spec.load(args.spec)
    spark = spark_job.session(app_name="plan-probe")
    spark.sparkContext.setLogLevel("ERROR")
    try:
        df = spark_job.read_log(spark, args.corpus)
        tables = {
            name: spark_job.materialize(
                df, feature_set, carry=("txn_id",), one_projection=flag
            )
            for name, flag in ARMS
        }
        counts = {name: operator_counts(t) for name, t in tables.items()}
        rows = {name: t.count() for name, t in tables.items()}
        # A faster plan that computes something else is not a faster plan. Equal row
        # counts would not catch a changed value, so this is a set difference over whole
        # rows in both directions.
        differing = tables["one_select"].exceptAll(tables["withcolumn_chain"]).count()
        differing += tables["withcolumn_chain"].exceptAll(tables["one_select"]).count()

        # Warmup, discarded. The first write in a process pays for the catalog and the
        # planner, and whichever arm runs first would otherwise carry it.
        for name, _ in ARMS:
            timed_write(tables[name], os.path.join(args.out, "warmup"))

        timings = {name: [] for name, _ in ARMS}
        for i in range(args.passes):
            for name, _ in ARMS:
                timings[name].append(
                    timed_write(tables[name], os.path.join(args.out, name))
                )
    finally:
        spark.stop()

    print(
        "{} rows, spec {}, rows differing between the arms {}".format(
            rows["one_select"], feature_set.fingerprint(), differing
        )
    )
    for name, _ in ARMS:
        best = min(timings[name])
        mid = sorted(timings[name])[len(timings[name]) // 2]
        print(
            "{:<18} window {:>2}  sort {:>2}  exchange {:>2}   median {:.2f}s  best "
            "{:.2f}s  passes {}".format(
                name,
                counts[name]["window"],
                counts[name]["sort"],
                counts[name]["exchange"],
                mid,
                best,
                " ".join("{:.2f}".format(t) for t in timings[name]),
            )
        )

    if rows["one_select"] != rows["withcolumn_chain"] or differing:
        print("the two arms did not produce the same table, so this timing is not a comparison")
        return 2
    ratio = (
        sorted(timings["withcolumn_chain"])[args.passes // 2]
        / sorted(timings["one_select"])[args.passes // 2]
    )
    print("chaining costs {:.2f}x on the median".format(ratio))
    return 0


if __name__ == "__main__":
    sys.exit(main())
