"""Run every check in this directory.

    python tests/run_all.py

Plain functions named check_* rather than a framework. Nothing here needs a corpus on
disk or a Redis server, so a clone runs the whole suite with only requirements.txt.

That is also the limit of what this runner can grade. Anything inside a pyspark call is
invisible to it, and `tests/run_with_spark.py` is the runner that reaches those.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from runner import run_checks  # noqa: E402  (tests/ is on the path by the line above)


if __name__ == "__main__":
    sys.exit(run_checks())
