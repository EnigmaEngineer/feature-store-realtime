"""Checks on the exact interval, now that it is in the package.

These were in `tests/test_serve_probe.py` and they moved with the code they grade. The
serving probe still has its own file, and what is left in it is the sweep, which is the
part that really belongs to a probe.

The interval was wrong once and the shape of the mistake is the reason these checks exist
in this much detail. It was a direct binomial tail sum starting from `(1 - p) ** n`, which
underflows to zero in a double at n of 48,998, so every evaluation returned zero and the
printed interval was `1.0000 to 1.0000` on every row with a large denominator. It printed
correct values on the small rows, which is exactly why it survived being looked at.

The reference values below came from `scipy.stats.beta.ppf` on the same machine. scipy is a
scikit-learn dependency rather than one this repo declares, so it cannot be the oracle at
run time and it can be the source of a fixture. The command that produced them is in the
docstring of `check_the_interval_matches_an_independent_implementation` and rerunning it is
how anybody checks these are not invented.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import stats

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
    rather than remembered. I once wrote two hash vectors into a fixture from memory and
    both were wrong, with three correct ones beside them hiding it.
    """
    for k, n, low, high in REFERENCE:
        got_low, got_high = stats.binomial_interval(k, n)
        assert abs(got_low - low) < 1e-7, (k, n, got_low, low)
        assert abs(got_high - high) < 1e-7, (k, n, got_high, high)


def check_a_large_denominator_does_not_return_the_whole_unit_interval():
    """The defect, pinned on its own rather than left inside the table above.

    The broken version returned 1.0 for both ends here. A test that only compared against
    the reference list would catch it and would not say what went wrong, and this is the
    line that names the failure for whoever reads it next.
    """
    low, high = stats.binomial_interval(3848, 48998)
    assert low < 0.0785 < high, (low, high)
    assert high - low < 0.01, "interval of width {:.4f} on 48998 observations".format(high - low)


def check_the_interval_always_contains_the_point_estimate():
    for k, n, _, _ in REFERENCE:
        low, high = stats.binomial_interval(k, n)
        assert low <= k / n <= high, (k, n, low, high)


def check_the_interval_narrows_as_the_count_grows():
    """A rate of one in ten and one in ten thousand are the same number and not the same
    evidence. An implementation that ignored n would pass every check above that reads a
    single row."""
    widths = []
    for n in (10, 100, 1000, 10000):
        low, high = stats.binomial_interval(n // 10, n)
        widths.append(high - low)
    assert widths == sorted(widths, reverse=True), widths


def check_zero_and_total_are_one_sided_rather_than_degenerate():
    low, high = stats.binomial_interval(0, 100)
    assert low == 0.0 and 0.0 < high < 0.05, (low, high)
    low, high = stats.binomial_interval(100, 100)
    assert high == 1.0 and 0.95 < low < 1.0, (low, high)


def check_no_observations_is_the_widest_possible_answer_and_not_a_crash():
    """A threshold that refuses everything serves nothing, and the row for it still
    prints. Returning 0.0 there would read as a rate of zero with certainty."""
    assert stats.binomial_interval(0, 0) == (0.0, 1.0)
    # And the nonsense call, which is the only input the guard above really protects.
    # With n of 0 the count has to be 0 too, and the k == 0 and k == n branches below
    # already return exactly this pair, so the guard changes nothing on the honest input.
    assert stats.binomial_interval(5, 0) == (0.0, 1.0)


def check_a_single_observation_is_still_a_real_interval():
    """One trial is the smallest thing the guard above must not swallow.

    A mutant moving the `n == 0` guard to `n == 1` survived the whole suite, because the
    only case anything asserted was `(0, 0)` and that returns `(0.0, 1.0)` whether the
    guard fires or not. The fixture agreed with the mutant by coincidence. One row here
    separates them: at one trial and no successes the upper bound is 0.975 and not 1.0.

    From scipy on the same machine. The upper bound at no successes in one trial is the
    first call and the lower bound at one success in one trial is the second.

        beta.ppf(0.975, 1, 1)
        beta.ppf(0.025, 1, 1)
    """
    for k, n, low, high in (
        (0, 1, 0.00000000, 0.97500000),
        (1, 1, 0.02500000, 1.00000000),
        (0, 2, 0.00000000, 0.84188612),
        (1, 2, 0.01257912, 0.98742088),
    ):
        got_low, got_high = stats.binomial_interval(k, n)
        assert abs(got_low - low) < 1e-7, (k, n, got_low, low)
        assert abs(got_high - high) < 1e-7, (k, n, got_high, high)


def check_the_incomplete_beta_is_a_distribution_function():
    """Monotone, and pinned at both ends. The continued fraction has a branch for
    x above the pivot that returns through the mirrored call, so both sides are walked."""
    for a, b in ((0.5, 0.5), (2.0, 3.0), (17.0, 443.0), (3849.0, 45151.0)):
        previous = -1.0
        for i in range(0, 21):
            x = i / 20.0
            value = stats.betainc(a, b, x)
            assert 0.0 <= value <= 1.0, (a, b, x, value)
            assert value >= previous - 1e-12, (a, b, x, value, previous)
            previous = value
        assert stats.betainc(a, b, 0.0) == 0.0
        assert stats.betainc(a, b, 1.0) == 1.0


def check_the_one_sided_bound_sits_above_the_two_sided_one():
    """They answer different questions and reading one as the other is off by the tail.

    The one sided bound at 95 percent is the two sided bound at 90, so it is strictly the
    larger number whenever there is anything to bound. An implementation that returned the
    low end of `binomial_interval` would pass a check that only asked for a number below
    the point estimate.
    """
    for k, n in ((3, 10), (10, 10), (17, 459), (3848, 48998)):
        one_sided = stats.binomial_lower_bound(k, n)
        two_sided_low, _ = stats.binomial_interval(k, n)
        assert one_sided > two_sided_low, (k, n, one_sided, two_sided_low)
        assert one_sided < k / n, (k, n, one_sided)


def check_the_bound_matches_an_independent_implementation():
    """Ten windows is the size a held out fire rate gets measured on here, and the point
    of the bound is that 3 of 10 and 10 of 10 are not the round numbers they print as.

    From scipy on the same machine, one call per row:

        from scipy.stats import beta
        beta.ppf(0.05, k, n - k + 1)

    The first two rows were written into this file from a three decimal figure recorded
    elsewhere and then computed, which is the wrong order and is how a fabricated constant
    reaches a fixture. They happened to agree. All four are scipy output now.
    """
    for k, n, want in (
        (3, 10, 0.08726443),
        (10, 10, 0.74113445),
        (17, 459, 0.02373413),
        (3848, 48998, 0.07654313),
    ):
        got = stats.binomial_lower_bound(k, n)
        assert abs(got - want) < 1e-7, (k, n, got, want)


def check_a_zero_count_bounds_at_zero_rather_than_raising():
    """A monitor that never fired is the normal case and it still has to print a row."""
    assert stats.binomial_lower_bound(0, 100) == 0.0
    assert stats.binomial_lower_bound(0, 0) == 0.0
    assert stats.binomial_lower_bound(5, 0) == 0.0


def check_the_quantile_is_nearest_rank_and_takes_no_average():
    """Ten values, so the level maps onto an index without ambiguity. An interpolating
    implementation returns 5.5 for the median here and this one returns a value that is
    in the list, which is the property the null floor depends on."""
    values = [10, 1, 9, 2, 8, 3, 7, 4, 6, 5]
    assert stats.quantile(values, 0.5) == 6
    assert stats.quantile(values, 0.0) == 1
    assert stats.quantile(values, 0.95) == 10
    for q in (0.0, 0.25, 0.5, 0.9, 0.95):
        assert stats.quantile(values, q) in values


def check_the_quantile_refuses_an_empty_list_and_a_level_of_one():
    """Nothing to take a quantile of is a different answer from zero, and `q` of exactly
    1 would index off the end. Both are refusals rather than clamps, because a clamp here
    silently returns the maximum and calls it a percentile."""
    for bad in ([],):
        try:
            stats.quantile(bad, 0.5)
        except ValueError:
            pass
        else:
            raise AssertionError("took a quantile of {!r}".format(bad))
    for q in (1.0, 1.5, -0.1):
        try:
            stats.quantile([1, 2, 3], q)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted q of {!r}".format(q))
