"""Every check in tests/, then the real Spark job against the reference.

    python tests/run_with_spark.py

`tests/run_all.py` runs on a clone that installed only `requirements.txt`, which is the
whole reason `featurestore/spark_job.py` imports pyspark inside its functions. The cost
is that nothing under `tests/` can execute a line inside a pyspark call, so most of that
module is ungraded by the suite that grades everything else.

This is the second oracle. It runs the suite and then `scripts/oracle_check.py --ties`,
which builds the training table with a real session and compares every cell against
`featurestore/compute.py`. Exit code is nonzero if either half fails, so a mutation
harness that takes a runner and reads its exit code can be pointed here instead:

    cp -r <repo> /tmp/mut/fsr
    cp /tmp/mut/fsr/tests/run_with_spark.py /tmp/mut/fsr/tests/run_all.py
    python mutate.py --repo /tmp/mut/fsr --module featurestore/spark_job.py --slice 0:8

The slice is there because one session takes about eighteen seconds to come up and a
whole pass does not fit in one call on the machine this was written on.

A missing pyspark is a failure here and not a skip. A runner whose Spark half quietly
disappears reports the suite's number and reads exactly like a full pass, which is the
same shape as a checker that reports clean having checked nothing.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from runner import run_checks  # noqa: E402  (tests/ is on the path by the line above)


def main():
    suite = run_checks()
    if suite != 0:
        print("the suite failed, so the spark arm was not run")
        return suite

    try:
        import pyspark  # noqa: F401
    except ImportError:
        print("pyspark is not importable, so this runner graded nothing beyond the suite")
        return 2

    proc = subprocess.run(
        [sys.executable, os.path.join("scripts", "oracle_check.py"), "--ties"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    # Only the verdict lines. The full table is what scripts/oracle_check.py is for and
    # printing it once per mutant buries the one line that matters.
    for line in proc.stdout.splitlines():
        if line.startswith(("shipped", "CONTROL FAILED", "the spark job disagrees")):
            print(line.rstrip())
    if proc.returncode != 0:
        print("the spark arm failed with exit {}".format(proc.returncode))
        if proc.stderr.strip():
            print(proc.stderr.strip().splitlines()[-1])
        return proc.returncode
    print("spark arm agrees with the reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
