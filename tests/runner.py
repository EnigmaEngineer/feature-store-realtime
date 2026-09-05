"""Collecting and running the check_* functions.

Separate from `run_all.py` so that a second entry point can reuse it. If this loop lived
in `run_all.py` then `run_with_spark.py` would have to import that file by name, and a
mutation pass which stands one runner in for the other would import itself.

The name deliberately does not begin with `test_`, so `modules` does not collect it.
"""

import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def modules():
    for name in sorted(os.listdir(HERE)):
        if name.startswith("test_") and name.endswith(".py"):
            yield "tests." + name[:-3]


def run_checks():
    """Returns an exit code. 0 clean, 1 something failed, 2 nothing ran."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    passed = 0
    failed = []
    for modname in modules():
        mod = importlib.import_module(modname)
        for attr in sorted(dir(mod)):
            if not attr.startswith("check_"):
                continue
            try:
                getattr(mod, attr)()
                passed += 1
            except Exception:
                failed.append((modname, attr, traceback.format_exc()))

    for modname, attr, tb in failed:
        print("FAIL {}.{}".format(modname, attr))
        print(tb)

    print("{} passed, {} failed".format(passed, len(failed)))
    if passed == 0:
        # A run that executed nothing is not a pass. This is why the count is printed
        # rather than a bare word.
        print("no checks were collected, that is a failure")
        return 2
    return 1 if failed else 0
