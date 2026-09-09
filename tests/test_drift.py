"""Checks on the drift monitor.

Two of these exist because the thing they check was wrong when it was written.

`check_the_predicted_null_mean_matches_a_simulation` pins the factor of two. The first
version of `predicted_null_mean` returned `(k - 1) / (2n)`, which is the standard result
for a one directional divergence and is half the answer for this index. Nothing about
reading the code said which one it was, and a simulation over a multinomial with a known
bin count says it in one run.

`check_ties_do_not_produce_a_bin_nothing_can_fall_into` pins the integer case. Every count
feature in this spec is a small integer where most rows hold zero, so quantile edges land
on top of each other, and edges left in place make bins with no reference mass that divide
by zero in the statistic.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import drift


def uniform_psi_mean(k, n, trials, seed):
    """The index under no drift, over a multinomial nobody in this repo built.

    Deliberately not through `fit_profile`. If the null were measured with the same binning
    code it is meant to validate then a binning defect would move both sides together and
    the comparison would pass having compared nothing.
    """
    rng = random.Random(seed)
    reference = [1.0 / k] * k
    total = 0.0
    for _ in range(trials):
        counts = [0] * k
        for _ in range(n):
            counts[rng.randrange(k)] += 1
        current = [c / n for c in counts]
        index = 0.0
        for c, r in zip(current, reference):
            if c > 0.0:
                index += (c - r) * math.log(c / r)
        total += index
    return total / trials


def check_the_predicted_null_mean_matches_a_simulation():
    """The factor of two, pinned at four bin counts so a wrong constant cannot hide.

    One value of k would be satisfied by any formula passing through that point. Four
    separate the `(k - 1) / n` answer from `(k - 1) / (2n)`, from `k / n`, and from a
    constant. The tolerance is loose because 3,000 draws is a noisy mean and tight enough
    that a factor of two cannot pass: the two candidates differ by 100 percent and the
    band is 25.
    """
    n = 400
    for k in (2, 5, 10, 20):
        measured = uniform_psi_mean(k, n, trials=3000, seed=11 + k)
        predicted = drift.predicted_null_mean(k, n)
        assert abs(measured - predicted) < 0.25 * predicted, (k, measured, predicted)


def check_the_predicted_null_mean_is_not_the_one_directional_form():
    """Named on its own, because the check above would also pass on a day the tolerance
    got widened. The wrong answer is exactly half and this says so."""
    assert drift.predicted_null_mean(10, 200) == 9 / 200.0
    assert drift.predicted_null_mean(10, 200) != 9 / 400.0
    assert drift.predicted_null_mean(2, 100) == 1 / 100.0


def check_the_predicted_mean_refuses_a_nonsense_shape():
    for bins, n in ((10, 0), (10, -5), (0, 100), (-1, 100)):
        try:
            drift.predicted_null_mean(bins, n)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted bins={} n={}".format(bins, n))


def check_the_smallest_legal_shapes_are_accepted_rather_than_refused():
    """The other side of the guards above, which is the side nothing was testing.

    Two mutants widened those guards to refuse a window of one row and a profile of one
    bin, and both survived, because every case handed to them was already illegal. One bin
    is not a silly input: a constant profile has exactly one, and its expected null is
    zero because nothing in it can move.
    """
    assert drift.predicted_null_mean(10, 1) == 9.0
    assert drift.predicted_null_mean(1, 100) == 0.0
    assert drift.predicted_null_mean(1, 1) == 0.0


def check_ties_do_not_produce_a_bin_nothing_can_fall_into():
    """Ninety zeros and ten ones, which is what a count feature looks like here.

    Ten requested bins cannot be got out of two distinct values. An implementation that
    kept every quantile edge would return nine zeros, making eight bins with no reference
    mass and one that holds everything.
    """
    values = [0] * 90 + [1] * 10
    edges = drift.bin_edges(values, target=10)
    assert edges == [1], edges
    profile = drift.fit_profile("counts", values)
    assert profile.bins == 2, profile.bins
    assert all(share > 0.0 for share in profile.shares), profile.shares
    assert abs(sum(profile.shares) - 1.0) < 1e-12


def check_a_continuous_reference_gets_every_bin_it_asked_for():
    """The default bin count, and the fact that all of the edges are produced.

    Two mutants survived without this. One moved `BIN_TARGET` off 10, which changes the
    bin count of every profile in the repo and therefore every null floor, and nothing
    asserted the default. The other started the edge loop at the second quantile instead
    of the first, which silently drops the lowest edge and returns nine bins where the
    caller asked for ten. Both are invisible on a fixture with two distinct values,
    which is what the other binning checks here use.

    The shares are asserted too, because a bin count on its own would still pass if the
    edges were bunched at one end.
    """
    values = list(range(1000))
    profile = drift.fit_profile("continuous", values)
    assert drift.BIN_TARGET == 10
    assert profile.bins == drift.BIN_TARGET, profile.bins
    for share in profile.shares:
        assert abs(share - 0.1) < 0.02, profile.shares

    eight = drift.fit_profile("continuous", values, target=8)
    assert eight.bins == 8, eight.bins


def check_two_bins_is_a_legal_request():
    """The smallest split there is. Two mutants moved the guard so that a target of 2 was
    refused, and every check of that guard used 1 or 0 or a negative, so the boundary the
    guard sits on was never handed to it from the legal side."""
    profile = drift.fit_profile("two", list(range(1000)), target=2)
    assert profile.bins == 2, profile.bins
    assert len(drift.bin_edges(list(range(1000)), target=2)) == 1


def check_ties_in_the_middle_are_deduplicated_and_not_only_ties_at_the_bottom():
    """A repeated value that is not the smallest one.

    The other tie fixtures here put the repeats at the minimum, where the final filter
    removes them anyway, so a mutant accepting an edge equal to the previous edge survived
    all of them. Eighty rows sharing a middle value is where it shows: the duplicate edges
    are all above the minimum, the filter keeps them, and the profile comes back with eight
    bins that nothing can ever fall into.
    """
    values = [0] * 10 + [5] * 80 + [9] * 10
    edges = drift.bin_edges(values, target=10)
    assert edges == [5, 9], edges
    profile = drift.fit_profile("middle", values)
    assert profile.bins == 3, profile.bins
    assert all(share > 0.0 for share in profile.shares), profile.shares
    counts = drift.bin_counts(values, edges)
    assert counts == [10, 80, 10], counts


def check_a_reference_with_a_unique_smallest_value_still_keeps_its_first_edge():
    """One row below a wall of identical ones.

    The filter drops an edge that is not strictly above the smallest value. A mutant
    comparing against the second smallest instead survived everything, because every other
    fixture here has its first two values tied and the two are then the same number. Here
    they are not, and under the mutant the only edge is dropped and the whole feature is
    misreported as a constant.
    """
    values = [0] + [1] * 99
    profile = drift.fit_profile("one_low", values)
    assert not profile.is_constant(), "a reference holding 0 and 1 is not constant"
    assert profile.bins == 2, profile.bins
    assert drift.bin_edges(values, target=10) == [1]


def check_an_edge_on_the_smallest_value_is_dropped():
    """Binning is `value < edge`, so an edge equal to the minimum makes a first bin that
    nothing can ever fall into. The tie removal above does not catch this one, because the
    edge is distinct from the edges around it and still empty."""
    values = [5] * 60 + [6] * 20 + [7] * 20
    edges = drift.bin_edges(values, target=4)
    assert 5 not in edges, edges
    counts = drift.bin_counts(values, edges)
    assert all(c > 0 for c in counts), (edges, counts)


def check_a_reference_with_one_value_is_a_constant_rather_than_a_distribution():
    """A band around a signal that has never moved is degenerate. It is held as a constant
    and any change at all is the alert."""
    profile = drift.fit_profile("flat", [3] * 50)
    assert profile.is_constant()
    assert profile.constant == 3
    assert profile.bins == 1

    monitor = drift.Monitor(profile, 1.0, "constant")
    clean = monitor.judge([3] * 40)
    assert clean["verdict"] == drift.OK
    assert clean["emptied"] == 0 and clean["rows_off_the_constant"] == 0, clean
    moved = monitor.judge([3] * 39 + [4])
    assert moved["verdict"] == drift.CONSTANT_MOVED, moved
    assert moved["rows_off_the_constant"] == 1, moved
    # There are no bins on a constant profile, so nothing can have emptied.
    assert moved["emptied"] == 0, moved


def check_a_reference_of_one_row_is_a_constant():
    """The smallest reference there is. A mutant reading the second value out of the list
    survived, because every constant fixture here holds forty identical rows and the first
    and second are the same number. One row separates them, and it is not a silly input:
    an entity seen once is exactly what a cold start looks like."""
    profile = drift.fit_profile("one", [7])
    assert profile.is_constant()
    assert profile.constant == 7
    assert profile.n == 1


def check_a_hand_built_profile_with_a_dead_bin_does_not_divide_by_it():
    """`fit_profile` cannot produce a zero reference share, so the guard against one is
    unreachable through this module. It stays because `Profile` is public and a caller
    assembling one gets a division by zero without it, and that reason is only worth
    anything if something exercises it. A mutant narrowing the guard survived until this.
    """
    profile = drift.Profile("hand", edges=[10, 20], shares=[0.5, 0.0, 0.5], n=100)
    index, emptied, used = drift.psi(profile, [1] * 50 + [25] * 50)
    assert math.isfinite(index)
    # The dead bin is skipped rather than counted as emptied, because nothing was ever
    # expected there. Counting it would report drift on a bin the reference never used.
    assert emptied == 0, emptied
    assert used == 2, used


def check_a_constant_profile_has_no_index_and_says_so():
    profile = drift.fit_profile("flat", [1] * 40)
    for call in (lambda: drift.psi(profile, [1] * 40),
                 lambda: drift.null_floor(profile, 40, reference_values=[1] * 40)):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("a constant profile answered a question about spread")


def check_the_index_equals_a_value_worked_out_by_hand():
    """The arithmetic itself, on unequal reference shares, against a number computed off
    the page rather than off this code.

    The reference shares have to be unequal or this check is worthless, and that is the
    whole reason it looks like this. A mutant replacing `log(c / r)` with `log(c * r)`,
    which is the centre of the statistic, survived every other check in this file. The two
    forms differ by `sum((c - r) * 2 * log(r))`, and when every reference share is the same
    number that pulls out of the sum and leaves `2 * log(r) * sum(c - r)`, which is zero
    because both sides are shares. Quantile binning makes the reference shares equal by
    construction, so every fixture built from `fit_profile` on continuous data makes that
    mutant exactly equivalent.

    Ninety zeros and ten ones gives shares of 0.9 and 0.1. Against a window split evenly:

        (0.5 - 0.9) * ln(0.5 / 0.9) + (0.5 - 0.1) * ln(0.5 / 0.1)
          = -0.4 * -0.5877866649 + 0.4 * 1.6094379124
          =  0.2351146660 + 0.6437751650
          =  0.8788898310

    Under the product form the same input comes out at -0.8788898310, so the sign alone
    separates them.
    """
    profile = drift.fit_profile("skewed", [0] * 90 + [1] * 10)
    assert profile.shares == [0.9, 0.1], profile.shares

    index, emptied, used = drift.psi(profile, [0] * 50 + [1] * 50)
    assert abs(index - 0.8788898310) < 1e-9, index
    assert emptied == 0
    assert used == 2

    # The reverse direction is a different number, because this index is symmetric in the
    # pair and not in which side is the reference.
    other = drift.fit_profile("even", [0] * 50 + [1] * 50)
    back, _, _ = drift.psi(other, [0] * 90 + [1] * 10)
    assert abs(back - 0.8788898310) < 1e-9, back


def check_the_index_is_small_on_the_reference_and_large_on_a_shifted_window():
    """Both halves. An implementation returning zero always would pass the first."""
    rng = random.Random(3)
    reference = [rng.gauss(0.0, 1.0) for _ in range(4000)]
    profile = drift.fit_profile("g", reference)

    same, emptied, _ = drift.psi(profile, [rng.gauss(0.0, 1.0) for _ in range(4000)])
    assert same < 0.02, same
    assert emptied == 0

    shifted, _, _ = drift.psi(profile, [rng.gauss(1.5, 1.0) for _ in range(4000)])
    assert shifted > 1.0, shifted
    assert shifted > same * 50


def check_a_window_that_empties_a_bin_is_its_own_verdict_and_not_a_bigger_number():
    """An emptied bin needs `ln(0)`. Excluding it and counting it is the choice here, so
    the count has to reach the verdict rather than being folded into the index."""
    reference = list(range(100))
    profile = drift.fit_profile("r", reference)
    window = [0] * 50
    index, emptied, used = drift.psi(profile, window)
    # Exact counts rather than `>= 1`. Everything lands in the first bin, so every other
    # bin emptied and exactly one contributed to the sum. Mutants that multiplied the
    # emptied counter and that started the used counter at one both survived a check
    # phrased as a lower bound.
    assert emptied == profile.bins - 1, (emptied, profile.bins)
    assert used == 1, used
    assert emptied + used == profile.bins
    assert math.isfinite(index)

    monitor = drift.Monitor(profile, 10.0, "deliberately unreachable")
    verdict = monitor.judge(window)
    assert verdict["verdict"] == drift.BIN_EMPTIED, verdict
    assert drift.fired(verdict)


def check_a_bin_emptied_fires_even_when_the_index_is_under_the_threshold():
    """The threshold above is 10.0, which nothing here reaches. If the emptied branch were
    checked after the threshold rather than before it, this window would read as ok."""
    profile = drift.fit_profile("r", list(range(100)))
    monitor = drift.Monitor(profile, 10.0, "deliberately unreachable")
    verdict = monitor.judge([0] * 50)
    assert verdict["index"] < monitor.threshold, verdict
    assert verdict["verdict"] != drift.OK, verdict


def check_an_index_exactly_on_the_threshold_does_not_fire():
    """The comparison is strictly greater, so a window landing exactly on the line is ok.

    A mutant flipping it to `>=` survived, because no fixture had ever put an index on the
    boundary. The threshold here is the index itself, which is the only way to sit exactly
    on a float without guessing at one. This repo has already shipped one boundary
    comparison the wrong way round for the same reason, in the loader.
    """
    profile = drift.fit_profile("skewed", [0] * 90 + [1] * 10)
    window = [0] * 50 + [1] * 50
    index, _, _ = drift.psi(profile, window)

    on_the_line = drift.Monitor(profile, index, "exactly the index")
    assert on_the_line.judge(window)["verdict"] == drift.OK, index

    just_under = drift.Monitor(profile, index * 0.999999, "a hair below")
    assert just_under.judge(window)["verdict"] == drift.MOVED, index


def check_the_null_falls_as_one_over_the_window_size():
    """Doubling the window halves the index under no drift. A monitor whose threshold does
    not move with the window is a monitor about traffic volume."""
    rng = random.Random(5)
    reference = [rng.gauss(0.0, 1.0) for _ in range(3000)]
    profile = drift.fit_profile("g", reference)
    means = []
    for n in (100, 200, 400, 800):
        floor = drift.null_floor(
            profile, n, trials=120, rng=random.Random(9), reference_values=reference
        )
        means.append(floor["mean"])
    assert means == sorted(means, reverse=True), means
    # Each step is a doubling, so each ratio should sit near 2. The band is wide because
    # 120 resamples is a noisy mean, and it still excludes a flat sequence.
    for before, after in zip(means, means[1:]):
        assert 1.4 < before / after < 3.0, means


def check_the_null_reports_the_size_it_was_measured_at():
    """A floor with no n beside it is the defect this whole module is about."""
    reference = list(range(200))
    profile = drift.fit_profile("r", reference)
    floor = drift.null_floor(profile, 60, trials=40, rng=random.Random(1),
                             reference_values=reference)
    assert floor["n"] == 60
    assert floor["bins"] == profile.bins
    assert floor["trials"] == 40
    assert floor["quantile"] >= floor["mean"] * 0.5


class AlwaysTheFirstValue:
    """An rng that draws index zero every time.

    A stub rather than a seed, because the property below has to hold exactly rather than
    with high probability. With every draw landing on the same value, every resampled
    window falls in one bin and empties all the others, so the count of windows with an
    empty bin is the trial count and not a number that depends on luck.
    """

    def randrange(self, n):
        return 0


def check_the_null_reports_how_many_of_its_windows_emptied_a_bin():
    """Both ends of that counter, and both were needed.

    One mutant started it at one and another incremented it by two, and both survived a
    suite where nothing read the field at all. A resampled window that empties a bin is
    the case where the index is not comparable with the others, so a null summarised
    without it can be quietly built out of windows that were not measuring the same thing.
    """
    skewed = drift.fit_profile("skewed", [0] * 90 + [1] * 10)
    every = drift.null_floor(
        skewed, 40, trials=25, rng=AlwaysTheFirstValue(),
        reference_values=[0] * 90 + [1] * 10,
    )
    assert every["windows_with_an_empty_bin"] == 25, every

    # And a case where none of them can. Ten equal bins over five hundred draws leaves a
    # bin empty with probability about 1e-23, so zero here is a fact and not a coin flip.
    wide = list(range(1000))
    none = drift.null_floor(
        wide_profile(wide), 500, trials=20, rng=random.Random(4), reference_values=wide,
    )
    assert none["windows_with_an_empty_bin"] == 0, none


def wide_profile(values):
    return drift.fit_profile("wide", values)


def check_the_null_defaults_to_a_fixed_seed_and_a_stated_trial_count():
    """Two calls with nothing passed have to agree, and they have to agree with the seed
    the module documents. Three mutants lived here: one moved the default trial count, one
    moved the default seed, and one turned `rng or Random(0)` into `rng and Random(0)`,
    which silently discards the rng a caller passed and crashes when none is given. All
    three are invisible unless something calls this with the arguments left off.
    """
    reference = list(range(400))
    profile = drift.fit_profile("r", reference)
    default = drift.null_floor(profile, 50, reference_values=reference)
    explicit = drift.null_floor(
        profile, 50, trials=200, rng=random.Random(0), reference_values=reference
    )
    assert default["trials"] == 200, default["trials"]
    assert default["mean"] == explicit["mean"], (default["mean"], explicit["mean"])
    assert default["quantile"] == explicit["quantile"]

    # A caller supplied generator has to be the one used. Under the `and` mutant this
    # returns the seed 0 answer instead.
    mine = drift.null_floor(
        profile, 50, trials=30, rng=random.Random(12345), reference_values=reference
    )
    theirs = drift.null_floor(
        profile, 50, trials=30, rng=random.Random(0), reference_values=reference
    )
    assert mine["mean"] != theirs["mean"], "the supplied generator was ignored"


def check_the_null_refuses_to_guess_what_it_is_resampling_from():
    """Shares cannot be resampled at the right granularity, so the values are required.
    Defaulting to something derived from the profile would return a number that looks like
    a null and is not one."""
    profile = drift.fit_profile("r", list(range(100)))
    try:
        drift.null_floor(profile, 50, trials=5)
    except ValueError:
        pass
    else:
        raise AssertionError("resampled from nothing")


def check_a_window_below_the_floor_is_not_judged_rather_than_judged_badly():
    """`too_few` is not a drift verdict and it must not count in a fire rate. A monitor
    that judged a twenty row window would report the null floor at twenty rows as drift.

    The sizes are literals and the floor is asserted directly. Written as
    `MIN_WINDOW - 1` the window moves with the constant, so a mutant shifting the floor
    survived: the check was phrased in terms of the thing it was supposed to pin.
    """
    assert drift.MIN_WINDOW == 30
    profile = drift.fit_profile("r", list(range(200)))
    monitor = drift.Monitor(profile, 0.05, "test")

    verdict = monitor.judge(list(range(29)))
    assert verdict["verdict"] == drift.TOO_FEW, verdict
    assert verdict["index"] is None
    # A verdict that judged nothing has emptied no bins. The field is what a report
    # prints, and a mutant putting a 1 there survived until something read it.
    assert verdict["emptied"] == 0, verdict
    assert verdict["rows"] == 29, verdict
    assert not drift.fired(verdict)

    # And the first size that is judged, so the boundary is pinned from both sides.
    at_the_floor = monitor.judge(list(range(30)))
    assert at_the_floor["verdict"] != drift.TOO_FEW, at_the_floor
    assert at_the_floor["index"] is not None


def check_a_fire_rate_counts_only_the_windows_it_could_judge():
    """The denominator is the judged windows and not the supplied ones. A rate over the
    supplied count would fall every time a short window was added, which is a monitor that
    looks better the less traffic it sees."""
    reference = list(range(200))
    profile = drift.fit_profile("r", reference)
    monitor = drift.Monitor(profile, 0.05, "test")

    clean = [reference[:] for _ in range(4)]
    short = [[1, 2, 3]]
    result = drift.fire_rate(monitor, clean + short)
    assert result["windows"] == 5
    assert result["judged"] == 4, result
    assert result["fires"] == 0
    assert result["rate"] == 0.0
    assert result["lower_bound"] == 0.0


def check_a_fire_rate_carries_an_exact_lower_bound_rather_than_a_bare_share():
    """Three of ten and thirty of a hundred are the same share and not the same evidence.

    The windows are the reference shifted rather than the reference itself. Handed its own
    reference back this monitor returns an index of exactly zero, which no positive
    threshold can be under, so a fixture built that way reports nothing firing whatever the
    threshold is and the check passes for the wrong reason.
    """
    reference = list(range(200))
    profile = drift.fit_profile("r", reference)
    monitor = drift.Monitor(profile, 0.0001, "so low anything moving fires")
    windows = [[v + 40 for v in reference] for _ in range(10)]
    result = drift.fire_rate(monitor, windows)
    assert result["fires"] == 10 and result["rate"] == 1.0, result
    assert 0.7 < result["lower_bound"] < 1.0, result

    quiet = drift.fire_rate(drift.Monitor(profile, 0.0001, "same"), [reference[:]] * 10)
    assert quiet["fires"] == 0, quiet


def check_every_window_of_a_fire_rate_comes_back_with_a_verdict():
    """The verdicts are what a caller reports per window, and a rate with no per window
    detail behind it cannot be checked against anything."""
    reference = list(range(200))
    monitor = drift.Monitor(drift.fit_profile("r", reference), 0.05, "test")
    result = drift.fire_rate(monitor, [reference[:], [1, 2, 3]])
    assert len(result["verdicts"]) == 2
    assert [v["verdict"] for v in result["verdicts"]] == [drift.OK, drift.TOO_FEW]


def check_a_threshold_has_to_be_positive_and_carries_where_it_came_from():
    """A threshold copied from a rule of thumb and one measured off a null are different
    claims, and a monitor that cannot say which it holds lets a report imply the second."""
    profile = drift.fit_profile("r", list(range(100)))
    for bad in (0.0, -0.1):
        try:
            drift.Monitor(profile, bad, "test")
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a threshold of {!r}".format(bad))
    monitor = drift.Monitor(profile, 0.1, "null q95")
    assert monitor.source == "null q95"


def check_binning_puts_a_value_on_an_edge_in_the_upper_bin():
    """`value < edge` goes left, so a value equal to an edge goes right. Stated as a check
    because the two conventions differ on exactly the values a tied integer column is made
    of, and this repo has already shipped one boundary comparison the wrong way round."""
    edges = [10, 20]
    assert drift.bin_index(9, edges) == 0
    assert drift.bin_index(10, edges) == 1
    assert drift.bin_index(19, edges) == 1
    assert drift.bin_index(20, edges) == 2
    assert drift.bin_index(21, edges) == 2


def check_binning_with_no_edges_puts_everything_in_one_bin():
    assert drift.bin_counts([1, 2, 3], []) == [3]


def check_shares_and_profiles_refuse_an_empty_input():
    """A monitor fitted on nothing is the shape this repo keeps meeting, where a check
    reports clean because it was handed no input."""
    try:
        drift.to_shares([0, 0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("took shares of an empty count vector")
    try:
        drift.fit_profile("nothing", [])
    except ValueError:
        pass
    else:
        raise AssertionError("profiled nothing")


def check_bin_edges_refuses_fewer_than_two_bins():
    for target in (1, 0, -3):
        try:
            drift.bin_edges([1, 2, 3], target=target)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted target of {!r}".format(target))


def check_the_flat_signal_monitor_reports_which_features_moved():
    """Not just that something differs. A count with no feature name attached sends
    somebody to read three hundred thousand rows."""
    left = [{"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 1, "b": 2}]
    right = [{"a": 1, "b": 2}, {"a": 9, "b": 2}, {"a": 9, "b": 7}]
    out = drift.constant_agreement(left, right, ["a", "b"])
    assert out["cells"] == 6
    assert out["differing"] == 3
    assert out["by_feature"] == {"a": 2, "b": 1}, out
    assert out["verdict"] == drift.CONSTANT_MOVED


def check_the_flat_signal_monitor_is_clean_when_nothing_differs():
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    out = drift.constant_agreement(rows, list(rows), ["a", "b"])
    assert out["differing"] == 0 and out["verdict"] == drift.OK
    assert out["by_feature"] == {}
    assert out["cells"] == 4


def check_the_flat_signal_monitor_refuses_two_different_lengths():
    """Comparing three rows against two by zipping them silently drops the third, and the
    verdict that comes back is clean. That is the passes-on-less-than-it-was-given shape."""
    try:
        drift.constant_agreement([{"a": 1}] * 3, [{"a": 1}] * 2, ["a"])
    except ValueError:
        pass
    else:
        raise AssertionError("compared unequal lengths")
