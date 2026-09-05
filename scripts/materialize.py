"""Build the training table from the transaction log.

    python scripts/materialize.py --out /tmp/training

Writes parquet partitioned by nothing, because the table is one file's worth here and a
partition scheme chosen before anybody queries it is a guess. Prints the row count, the
spec fingerprint the table carries, and how long the job took.

The correctness argument for this table is not in this script. It is
`scripts/oracle_check.py`, which grades the same job against the reference
implementation.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import spark_job, spec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="specs/txn_fraud.yml")
    ap.add_argument("--corpus", default="data/transactions.csv")
    ap.add_argument("--out", default="/tmp/training")
    ap.add_argument("--cores", type=int, default=2)
    args = ap.parse_args(argv)

    feature_set = spec.load(args.spec)
    spark = spark_job.session(cores=args.cores)
    spark.sparkContext.setLogLevel("ERROR")
    try:
        df = spark_job.read_log(spark, args.corpus)
        table = spark_job.materialize(df, feature_set, carry=("txn_id",))

        started = time.time()
        table.write.mode("overwrite").parquet(args.out)
        elapsed = time.time() - started

        written = spark.read.parquet(args.out)
        rows = written.count()
        fingerprints = [r[0] for r in written.select("spec_fingerprint").distinct().collect()]
    finally:
        spark.stop()

    print("wrote {} rows to {} in {:.2f}s".format(rows, args.out, elapsed))
    print("spec {} at fingerprint {}".format(feature_set.name, feature_set.fingerprint()))
    print("fingerprints on the table: {}".format(sorted(fingerprints)))
    print("columns: {}".format(" ".join(written.columns)))
    if sorted(fingerprints) != [feature_set.fingerprint()]:
        print("the table carries a fingerprint the spec does not")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
