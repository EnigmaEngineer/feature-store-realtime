"""Checks on the sweep the serving probe publishes.

This is the only module here that imports a script. Every other check in this repo grades
the library.

It used to hold the checks on the exact interval too. Those moved into
`tests/test_stats.py` when the interval moved into `featurestore/stats.py`, because the
drift monitor needed the same bound and two Clopper Pearson implementations in one repo is
the shape where each half is right on its own. What is left here is the sweep, which really
does belong to a probe: it decides what a candidate age threshold refuses and how wrong
each half of that split is, and nothing outside this probe has any use for it.

Importing by path rather than adding `scripts/` to the path, because a directory of scripts
on `sys.path` makes any module in it shadow a package with the same name.
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_probe():
    path = os.path.join(ROOT, "scripts", "serve_probe.py")
    spec_obj = importlib.util.spec_from_file_location("serve_probe_under_test", path)
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    return module


PROBE = load_probe()

def check_a_threshold_longer_than_the_data_is_marked_as_not_binding():
    """A row that refuses nothing is a restatement of the no threshold row. Reporting it
    without saying so puts a number in a table that looks like an observation."""
    served = [0.9, 0.1, 0.9]
    wanted = [0.1, 0.1, 0.9]
    ages = [10, 20, 30]
    row = PROBE.sweep(served, wanted, ages, 10 ** 9, 0.5)
    assert row["binds"] is False, row
    assert row["refused"] == 0 and row["served"] == 3
    bites = PROBE.sweep(served, wanted, ages, 15, 0.5)
    assert bites["binds"] is True, bites
    assert bites["refused"] == 2 and bites["served"] == 1


def check_the_sweep_counts_a_flip_and_not_a_moved_score():
    """The fixture has one request whose score moves a long way without crossing the
    threshold and one that crosses it by a hair. A measure that counted movement would
    report two and the one with a consequence attached is one.
    """
    served = [0.95, 0.51]
    wanted = [0.55, 0.49]
    row = PROBE.sweep(served, wanted, [0, 0], None, 0.5)
    assert row["flips"] == 1, row
    assert row["mean_abs_gap"] > 0.2, row


def check_the_refused_side_gets_its_own_rate():
    """A threshold that throws away requests failing at the same rate as the ones it keeps
    has separated nothing, and a table reporting only the served side cannot show it."""
    served = [0.9, 0.9, 0.1, 0.1]
    wanted = [0.1, 0.9, 0.9, 0.1]
    ages = [1, 2, 100, 200]
    row = PROBE.sweep(served, wanted, ages, 50, 0.5)
    assert row["served"] == 2 and row["flips"] == 1 and row["flip_share"] == 0.5
    assert row["refused"] == 2 and row["refused_flips"] == 1
    assert row["refused_flip_share"] == 0.5
