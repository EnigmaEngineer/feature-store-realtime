"""Checks on the confidence interval the serving probe publishes.

This is the first module here that imports a script. Every other check in this repo grades
the library, and the README has said for two days that the controls deciding a headline
live in `scripts/` where no mutant can reach them. That is still true of most of them and
it is no longer true of this one.

The reason this one moved first is that it was wrong. The interval was written as a direct
binomial tail sum starting from `(1 - p) ** n`, which underflows to zero in a double at
n of 48,998, so every evaluation returned zero and the printed interval was `1.0000 to
1.0000` on every row with a large denominator. It printed correct values on the small rows,
which is exactly why it survived being looked at.

The reference values below came from `scipy.stats.beta.ppf` on the same machine. scipy is a
scikit-learn dependency rather than one this repo declares, so it cannot be the oracle at
run time and it can be the source of a fixture. The command that produced them is in the
docstring of `check_the_interval_matches_an_independent_implementation` and rerunning it is
how anybody checks these are not invented.

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

# Each row is k then n then the low and high bounds. Every one came from scipy, and the
# check below carries the command that reproduces them.
# The last three are the denominators the probe really reports on the full corpus, so the
# case that broke is in the fixture rather than represented by a small stand in.
REFERENCE = [
    (0, 10, 0.00000000, 0.30849711),
    (1, 10, 0.00252858, 0.44501612),
    (2, 41, 0.00596312, 0.16533335),
    (5, 20, 0.08657147, 0.49104587),
    (17, 459, 0.02172079, 0.05863892),
    (258, 3195, 0.07153453, 0.09074358),
    (605, 8242, 0.06786503, 0.07924781),
    (3848, 48998, 0.07616671, 0.08095100),
    (10, 10, 0.69150289, 1.00000000),
]


def check_the_interval_matches_an_independent_implementation():
    """Reference values from scipy, which this repo does not declare and cannot import.

    The row for 3848 of 48998 reproduces with two calls, on a machine that has scipy:

        from scipy.stats import beta
        beta.ppf(0.025, 3848, 45151)
        beta.ppf(0.975, 3849, 45150)

    A fixture exists to be the thing that cannot be argued with, so these are computed
    rather than remembered. Two vectors in another repo here were once written from
    memory and were wrong, with correct ones beside them hiding it.
    """
    for k, n, low, high in REFERENCE:
        got_low, got_high = PROBE.binomial_interval(k, n)
        assert abs(got_low - low) < 1e-7, (k, n, got_low, low)
        assert abs(got_high - high) < 1e-7, (k, n, got_high, high)


def check_a_large_denominator_does_not_return_the_whole_unit_interval():
    """The defect, pinned on its own rather than left inside the table above.

    The broken version returned 1.0 for both ends here. A test that only compared against
    the reference list would catch it and would not say what went wrong, and this is the
    line that names the failure for whoever reads it next.
    """
    low, high = PROBE.binomial_interval(3848, 48998)
    assert low < 0.0785 < high, (low, high)
    assert high - low < 0.01, "interval of width {:.4f} on 48998 observations".format(high - low)


def check_the_interval_always_contains_the_point_estimate():
    for k, n, _, _ in REFERENCE:
        low, high = PROBE.binomial_interval(k, n)
        assert low <= k / n <= high, (k, n, low, high)


def check_the_interval_narrows_as_the_count_grows():
    """A rate of one in ten and one in ten thousand are the same number and not the same
    evidence. An implementation that ignored n would pass every check above that reads a
    single row."""
    widths = []
    for n in (10, 100, 1000, 10000):
        low, high = PROBE.binomial_interval(n // 10, n)
        widths.append(high - low)
    assert widths == sorted(widths, reverse=True), widths


def check_zero_and_total_are_one_sided_rather_than_degenerate():
    low, high = PROBE.binomial_interval(0, 100)
    assert low == 0.0 and 0.0 < high < 0.05, (low, high)
    low, high = PROBE.binomial_interval(100, 100)
    assert high == 1.0 and 0.95 < low < 1.0, (low, high)


def check_no_observations_is_the_widest_possible_answer_and_not_a_crash():
    """A threshold that refuses everything serves nothing, and the row for it still
    prints. Returning 0.0 there would read as a rate of zero with certainty."""
    assert PROBE.binomial_interval(0, 0) == (0.0, 1.0)


def check_the_incomplete_beta_is_a_distribution_function():
    """Monotone, and pinned at both ends. The continued fraction has a branch for
    x above the pivot that returns through the mirrored call, so both sides are walked."""
    for a, b in ((0.5, 0.5), (2.0, 3.0), (17.0, 443.0), (3849.0, 45151.0)):
        previous = -1.0
        for i in range(0, 21):
            x = i / 20.0
            value = PROBE.betainc(a, b, x)
            assert 0.0 <= value <= 1.0, (a, b, x, value)
            assert value >= previous - 1e-12, (a, b, x, value, previous)
            previous = value
        assert PROBE.betainc(a, b, 0.0) == 0.0
        assert PROBE.betainc(a, b, 1.0) == 1.0


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
