"""Every check in tests/, then the loader and the lookup against a real Redis.

    python tests/run_with_redis.py

The third runner, and it is here for the same reason as the second. `tests/run_all.py`
runs on a clone that installed only `requirements.txt`, so nothing under `tests/` opens a
socket. The stub in `tests/test_online.py` covers the key, the batching and the refusals,
and it cannot say whether a value survives a real server.

This runs the suite and then `scripts/store_check.py`. That script starts a Redis in this
process and loads the store. It then reads every row back and compares it cell by cell
against the reference. A mutation harness that takes a runner and reads its exit code can
be pointed here instead. Mine is a small script I keep outside this repo, so substitute
your own for `mutate.py` below.

    cp -r <repo> /tmp/mut/fsr
    cp /tmp/mut/fsr/tests/run_with_redis.py /tmp/mut/fsr/tests/run_all.py
    python mutate.py --repo /tmp/mut/fsr --module featurestore/online.py

A missing redis or redislite is a failure here and not a skip. A runner whose store half
quietly disappears reports the suite's number and reads exactly like a full pass.

The account count is small on purpose. A mutation pass pays the load once per mutant, and
the whole corpus comparison is a separate command.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from runner import run_checks  # noqa: E402  (tests/ is on the path by the line above)

ACCOUNTS = "60"
VERDICT_PREFIXES = ("shipped", "CONTROL FAILED", "the round trip lost", "lookup over")


def main():
    suite = run_checks()
    if suite != 0:
        print("the suite failed, so the store arm was not run")
        return suite

    try:
        import redis  # noqa: F401
        import redislite  # noqa: F401
    except ImportError as exc:
        print("{}, so this runner graded nothing beyond the suite".format(exc))
        return 2

    corpus = os.path.join(ROOT, "data", "transactions.csv")
    if not os.path.exists(corpus):
        print("no corpus at data/transactions.csv, run scripts/generate_corpus.py first")
        return 2

    proc = subprocess.run(
        [
            sys.executable,
            os.path.join("scripts", "store_check.py"),
            "--accounts",
            ACCOUNTS,
            "--repeats",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    shown = 0
    for line in proc.stdout.splitlines():
        if line.startswith(VERDICT_PREFIXES):
            print(line.rstrip())
            shown += 1
    if shown == 0:
        # Selecting lines by their opening words is fragile, and the failure it fails
        # into is an empty transcript beside a passing exit code. So finding nothing to
        # print is itself a failure rather than a quiet gap.
        print("the store arm printed nothing this runner recognised, so it reported nothing")
        return 2
    if proc.returncode != 0:
        print("the store arm failed with exit {}".format(proc.returncode))
        if proc.stderr.strip():
            print(proc.stderr.strip().splitlines()[-1])
        return proc.returncode
    print("store arm agrees with the reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
