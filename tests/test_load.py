"""Checks on the arithmetic the load probe publishes.

`featurestore/load.py` holds the quantiles, the control on the driver and the rule that
refuses a comparison. All three decide what a load table is allowed to say, so they live
in the package where a mutant can reach them rather than in `scripts/`.

Two of these fixtures exist because of things that have gone wrong here before. The
comparison fixtures carry arms that differ, because a check whose subject is "nothing
changes" passes whether or not the thing under test ran. And the quantile fixture pins the
values the moved implementation produced in its old home, so the move is a move rather than
a rewrite wearing the same name.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import load


def arm(durations, wall, workers, statuses=None):
    return load.summarise(durations, wall, workers, statuses or {200: len(durations)})


def check_quantiles_are_nearest_rank_and_not_interpolated():
    # 1..100. Nearest rank at index int(q*n) gives the 51st, 96th and 100th values, which
    # an interpolating definition would answer 50.5, 95.05 and 99.01 for. The two agree on
    # nothing here, which is the point of the fixture.
    values = [float(i) for i in range(1, 101)]
    p50, p95, p99, worst = load.quantiles(values)
    assert (p50, p95, p99, worst) == (51.0, 96.0, 100.0, 100.0), (p50, p95, p99, worst)


def check_quantiles_on_an_empty_list_are_zero_rather_than_an_error():
    assert load.quantiles([]) == (0.0, 0.0, 0.0, 0.0)


def check_quantiles_match_the_implementation_this_moved_from():
    # The values `scripts/serve_probe.quantiles` returned for this input before the
    # function moved here. Recorded so the move cannot quietly change a published p99.
    values = [3.0, 1.0, 2.0, 9.0, 5.0, 4.0, 8.0, 7.0, 6.0, 10.0]
    assert load.quantiles(values) == (6.0, 10.0, 10.0, 10.0)


def check_serve_probe_uses_this_quantile_and_does_not_carry_its_own():
    # Two modules reporting a p99 off two definitions is the failure this move prevents.
    # Reading the source rather than importing it, because the script pulls in fastapi.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(here, "scripts", "serve_probe.py")).read()
    assert "def quantiles(" not in text, "serve_probe grew its own quantile back"
    assert "load.quantiles" in text or "from featurestore.load import" in text


def check_offered_concurrency_is_the_mean_number_in_flight():
    # A wall clock of two seconds rather than one. At one second, dividing by the wall and
    # multiplying by it give the same answer, and a mutant that swapped them survived.
    # Twelve requests of 500 ms over two seconds is three in flight.
    got = load.offered_concurrency([500.0] * 12, 2.0)
    assert abs(got - 3.0) < 1e-9, got


def check_offered_concurrency_falls_as_the_wall_clock_grows():
    # The same work over twice the time is half the concurrency. Pins the direction, which
    # a single fixture at any one wall clock cannot.
    tight = load.offered_concurrency([100.0] * 10, 0.5)
    loose = load.offered_concurrency([100.0] * 10, 1.0)
    assert abs(tight - 2.0) < 1e-9 and abs(loose - 1.0) < 1e-9, (tight, loose)


def check_offered_concurrency_refuses_a_wall_clock_of_zero():
    try:
        load.offered_concurrency([1.0], 0.0)
    except ValueError as exc:
        assert "wall_s" in str(exc), str(exc)
    else:
        raise AssertionError("a zero wall clock gave a concurrency instead of raising")


def check_a_driver_that_kept_its_load_up_lands():
    # Eight workers, each request 100 ms, one second of wall. Eight in flight.
    landed, ratio = load.load_landed([100.0] * 80, 1.0, 8)
    assert landed is True and abs(ratio - 1.0) < 1e-9, (landed, ratio)


def check_a_driver_that_did_not_keep_its_load_up_fails_the_control():
    # Same eight workers but the requests only account for half the wall clock, so the
    # driver spent the rest somewhere other than waiting on the server.
    landed, ratio = load.load_landed([100.0] * 40, 1.0, 8)
    assert landed is False and abs(ratio - 0.5) < 1e-9, (landed, ratio)


def check_a_driver_sitting_exactly_on_the_floor_lands():
    # The floor is inclusive. Nothing else in this file sits on it, so a mutant turning the
    # comparison into a strict one survived every other fixture.
    landed, ratio = load.load_landed([90.0] * 80, 1.0, 8)
    assert abs(ratio - load.LANDED_FLOOR) < 1e-9, ratio
    assert landed is True, (landed, ratio)


def check_a_driver_just_under_the_floor_does_not_land():
    landed, ratio = load.load_landed([89.0] * 80, 1.0, 8)
    assert ratio < load.LANDED_FLOOR and landed is False, (landed, ratio)


def check_the_landed_floor_is_not_read_from_the_data():
    # A floor picked after seeing the arms could pick a winner. It is a module constant
    # and this pins it, so moving it is a visible edit rather than a tuning pass.
    assert load.LANDED_FLOOR == 0.90


def check_load_landed_refuses_a_worker_count_of_zero():
    try:
        load.load_landed([1.0], 1.0, 0)
    except ValueError as exc:
        assert "workers" in str(exc), str(exc)
    else:
        raise AssertionError("zero workers gave a ratio instead of raising")


def check_summarise_reports_throughput_over_wall_clock_and_not_over_the_sum():
    got = arm([10.0] * 100, 2.0, 4)
    assert got["n"] == 100
    assert abs(got["throughput_rps"] - 50.0) < 1e-9, got["throughput_rps"]


def check_summarise_carries_the_status_counts_it_was_given():
    got = arm([1.0, 1.0], 1.0, 1, {200: 1, 404: 1})
    assert got["statuses"] == {200: 1, 404: 1}


def check_an_arm_of_all_two_hundreds_is_ok():
    assert load.all_ok(arm([1.0] * 5, 1.0, 1)) is True


def check_an_arm_carrying_a_single_miss_is_not_ok():
    # A pass where a fraction of requests 404 is faster than one where they all score.
    mixed = arm([1.0] * 5, 1.0, 1, {200: 4, 404: 1})
    assert load.all_ok(mixed) is False


def check_an_arm_whose_statuses_do_not_add_up_to_its_own_count_is_not_ok():
    # Every request has to be accounted for. A count that disagrees with the sample size
    # means the two came from different places.
    wrong = arm([1.0] * 5, 1.0, 1, {200: 4})
    assert load.all_ok(wrong) is False


def check_compare_returns_the_ratios_when_both_arms_are_sound():
    fast = arm([10.0] * 100, 1.0, 1)
    slow = arm([20.0] * 50, 1.0, 1)
    verdict, detail = load.compare("fast", fast, "slow", slow)
    assert verdict == "ok", (verdict, detail)
    assert abs(detail["throughput_ratio"] - 2.0) < 1e-9, detail
    assert abs(detail["p99_ratio"] - 0.5) < 1e-9, detail


def check_compare_refuses_when_one_arm_did_not_keep_its_load_up():
    good = arm([100.0] * 80, 1.0, 8)
    starved = arm([100.0] * 20, 1.0, 8)
    verdict, detail = load.compare("good", good, "starved", starved)
    assert verdict == "driver_bound", (verdict, detail)
    assert "starved" in detail, detail


def check_compare_names_the_first_arm_that_failed_and_not_a_fixed_one():
    # Swapping which side is broken has to swap the name in the reason. A message that
    # always names the second argument would pass the check above and say nothing.
    starved = arm([100.0] * 20, 1.0, 8)
    good = arm([100.0] * 80, 1.0, 8)
    verdict, detail = load.compare("starved", starved, "good", good)
    assert verdict == "driver_bound" and "starved" in detail, (verdict, detail)


def check_compare_refuses_when_an_arm_served_something_other_than_a_score():
    # Both arms have to clear the driver control first or this reads driver_bound and the
    # status rule never runs. The first version of this fixture did exactly that, at an
    # offered ratio of 0.80 against a floor of 0.90.
    a = arm([100.0] * 80, 1.0, 8)
    b = arm([100.0] * 80, 1.0, 8, {200: 79, 404: 1})
    verdict, detail = load.compare("a", a, "b", b)
    assert verdict == "mixed_statuses", (verdict, detail)


def check_the_driver_control_is_checked_before_the_status_control():
    # An arm that is both starved and serving misses should report the starvation, because
    # a driver that never landed makes the status mix a fact about a pass that did not run.
    both = arm([100.0] * 20, 1.0, 8, {200: 19, 404: 1})
    good = arm([100.0] * 80, 1.0, 8)
    verdict, _ = load.compare("both", both, "good", good)
    assert verdict == "driver_bound", verdict


def check_compare_refuses_two_arms_run_at_different_concurrency():
    try:
        load.compare("a", arm([10.0] * 8, 1.0, 8), "b", arm([10.0] * 4, 1.0, 4))
    except ValueError as exc:
        assert "concurrency" in str(exc), str(exc)
    else:
        raise AssertionError("compared two arms at different worker counts")


def check_the_power_floor_is_two_over_two_to_the_n():
    # Five pairs cannot reach 0.05 at any effect size and six can. That is the whole reason
    # the probe defaults to six rounds, so it is pinned rather than left to arithmetic.
    assert load.p_floor(5) == 2.0 / 32
    assert load.p_floor(6) == 2.0 / 64
    assert load.p_floor(5) > load.ALPHA
    assert load.p_floor(6) < load.ALPHA


def check_the_power_floor_of_no_pairs_is_one():
    assert load.p_floor(0) == 1.0
    assert load.p_floor(-3) == 1.0


def check_the_sign_test_matches_arithmetic_that_can_be_done_by_hand():
    # Six of six one way is 2 * 1 / 64. Five of six is 2 * (6 + 1) / 64.
    assert abs(load.sign_test(6, 6) - 2.0 / 64) < 1e-12
    assert abs(load.sign_test(5, 6) - 14.0 / 64) < 1e-12
    assert load.sign_test(3, 6) == 1.0


def check_the_sign_test_is_two_sided_and_symmetric():
    # Six losses is as surprising as six wins. A one sided version would call one of them
    # certain, and the direction here is read off the medians rather than off the p value.
    for wins in range(7):
        assert load.sign_test(wins, 6) == load.sign_test(6 - wins, 6)


def check_the_sign_test_refuses_more_wins_than_pairs():
    try:
        load.sign_test(7, 6)
    except ValueError as exc:
        assert "wins" in str(exc), str(exc)
    else:
        raise AssertionError("accepted seven wins out of six pairs")


def reps(values, workers=8, statuses=None):
    # Durations chosen so every round clears the driver control at this worker count.
    return [arm([v] * 80, 1.0, workers, statuses) for v in values]


def check_compare_reps_decides_when_one_arm_wins_every_round():
    fast = reps([100.0, 101.0, 99.0, 100.5, 98.0, 102.0])
    slow = reps([200.0, 202.0, 198.0, 201.0, 196.0, 204.0])
    # Same request count both sides, so throughput ties. Compare on p99 where they differ.
    verdict, detail = load.compare_reps("fast", fast, "slow", slow, "p99_ms", False)
    assert verdict == "ok", (verdict, detail)
    assert detail["wins"] == 6 and detail["pairs"] == 6, detail
    assert detail["decided"] is True and detail["p"] < load.ALPHA, detail


def check_compare_reps_reports_not_decided_when_the_rounds_split():
    a = reps([100.0, 300.0, 100.0, 300.0, 100.0, 300.0])
    b = reps([200.0, 200.0, 200.0, 200.0, 200.0, 200.0])
    verdict, detail = load.compare_reps("a", a, "b", b, "p99_ms", False)
    assert verdict == "ok", (verdict, detail)
    assert detail["decided"] is False, detail
    assert detail["wins"] == 3 and detail["pairs"] == 6, detail


def check_compare_reps_refuses_five_rounds_before_looking_at_them():
    # Five rounds all one way still cannot clear the level, and the refusal has to name the
    # design rather than report a near miss the reader will round down.
    a = reps([100.0] * 5)
    b = reps([200.0] * 5)
    verdict, detail = load.compare_reps("a", a, "b", b, "p99_ms", False)
    assert verdict == "underpowered", (verdict, detail)
    assert "5 rounds" in detail, detail


def check_compare_reps_refuses_when_ties_take_the_pair_count_under_the_floor():
    # Six rounds of which four tie leaves two pairs. The rounds count cleared the floor and
    # the pair count does not, which is the case a check on rounds alone would miss.
    a = reps([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    b = reps([200.0, 300.0, 100.0, 100.0, 100.0, 100.0])
    verdict, detail = load.compare_reps("a", a, "b", b, "p99_ms", False)
    assert verdict == "underpowered", (verdict, detail)
    # The counts in the message, not just the word. A mutant added the two numbers instead
    # of subtracting them and the message still said "tied".
    assert "4 of 6 rounds tied" in detail, detail
    assert "leaving 2 pairs" in detail, detail


def check_compare_reps_defaults_to_throughput_where_more_is_better():
    # Every other check here passes both the key and the direction. A mutant flipped the
    # default direction to lower is better and survived all of them.
    # One worker, and each arm's durations fill its wall clock, so both land.
    fast = [arm([1000.0 / 120] * 120, 1.0, 1) for _ in range(6)]
    slow = [arm([1000.0 / 80] * 80, 1.0, 1) for _ in range(6)]
    verdict, detail = load.compare_reps("fast", fast, "slow", slow)
    assert verdict == "ok" and detail["key"] == "throughput_rps", (verdict, detail)
    assert detail["wins"] == 6 and detail["decided"] is True, detail


def check_compare_reps_reports_the_ratio_of_the_medians_and_not_their_product():
    # 120 requests a second against 80 is 1.5x. A mutant multiplied them, which gives 9600
    # and would sit in a throughput table looking like a large win.
    fast = [arm([1000.0 / 120] * 120, 1.0, 1) for _ in range(6)]
    slow = [arm([1000.0 / 80] * 80, 1.0, 1) for _ in range(6)]
    verdict, detail = load.compare_reps("fast", fast, "slow", slow)
    assert verdict == "ok", (verdict, detail)
    assert abs(detail["ratio_of_medians"] - 1.5) < 1e-9, detail["ratio_of_medians"]
    assert detail["median_a"] == 120.0 and detail["median_b"] == 80.0, detail


def check_an_arm_of_exactly_one_request_is_ok():
    # The guard against an empty arm has to stop at zero. A mutant moved it to one and
    # survived, because every other fixture here carries five requests or eighty.
    one = load.summarise([1.0], 1.0, 1, {200: 1})
    assert one["n"] == 1
    assert load.all_ok(one) is True


def check_an_arm_that_served_nothing_at_all_is_not_ok():
    # This found a real one. An arm of no requests has a status map of {200: 0}, which
    # matched the expected set and matched its own count, so a pass that served nothing
    # cleared the control that exists to say whether it served the right thing.
    empty = load.summarise([], 1.0, 1, {200: 0})
    assert empty["n"] == 0 and empty["throughput_rps"] == 0.0
    assert load.all_ok(empty) is False


def check_the_zero_denominator_guards_in_compare_cannot_be_reached():
    # `compare` guards its two divisions and both guards survived a mutation pass. They
    # cannot be reached, because an arm with a throughput of zero has no requests and fails
    # the status control first. Asserting the guarantee rather than pretending a fixture
    # can exercise the branch.
    empty = load.summarise([], 1.0, 8, {200: 0})
    good = arm([100.0] * 80, 1.0, 8)
    verdict, detail = load.compare("good", good, "empty", empty)
    assert verdict in ("driver_bound", "mixed_statuses"), (verdict, detail)


def check_compare_reps_refuses_a_round_where_the_driver_did_not_land():
    good = reps([100.0] * 6)
    bad = reps([100.0] * 6)
    bad[3] = arm([100.0] * 20, 1.0, 8)
    verdict, detail = load.compare_reps("good", good, "bad", bad, "p99_ms", False)
    assert verdict == "driver_bound", (verdict, detail)
    assert "round 3" in detail, detail


def check_compare_reps_refuses_a_round_that_served_a_miss():
    good = reps([100.0] * 6)
    bad = reps([100.0] * 6)
    bad[1] = arm([100.0] * 80, 1.0, 8, {200: 79, 404: 1})
    verdict, detail = load.compare_reps("good", good, "bad", bad, "p99_ms", False)
    assert verdict == "mixed_statuses", (verdict, detail)
    assert "round 1" in detail, detail


def check_compare_reps_refuses_arms_with_different_round_counts():
    try:
        load.compare_reps("a", reps([100.0] * 6), "b", reps([100.0] * 5), "p99_ms", False)
    except ValueError as exc:
        assert "rounds" in str(exc), str(exc)
    else:
        raise AssertionError("paired six rounds against five")


def check_the_median_of_an_even_count_is_the_mean_of_the_middle_two():
    assert load.median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert load.median([3.0, 1.0, 2.0]) == 2.0


def check_the_median_refuses_an_empty_list_rather_than_returning_zero():
    # Zero is a plausible looking throughput and it would sit in a table without comment.
    try:
        load.median([])
    except ValueError as exc:
        assert "median" in str(exc), str(exc)
    else:
        raise AssertionError("took the median of nothing")
