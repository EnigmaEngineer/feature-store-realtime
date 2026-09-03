"""Checks on the reference implementation.

The fixture is built so the four window readings disagree. A fixture with one event per
account, or with every event inside every window, would let all four readings agree and
the whole file would pass without exercising anything.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import compute, spec

DAY = 86400
# Midnight, then a spread across one day, chosen so the calendar and event_time readings
# land on different sets. T0 is a real midnight because floor_day works in UTC seconds.
T0 = 1750000000 - (1750000000 % DAY)

# r0 exists to separate the two alignments. Scoring at T0+10800 with a 1 day window puts
# the event_time lower bound at T0-75600 and the calendar one at T0-86400, so a row in
# between is the only thing that tells the two apart. Without it two of the four readings
# return the same events and the fixture cannot fail.
ROWS = [
    {"txn_id": "r1", "account_id": "a", "event_ts": T0 - 2 * DAY, "amount": 10.0, "merchant_id": "m1"},
    {"txn_id": "r0", "account_id": "a", "event_ts": T0 - 80000, "amount": 15.0, "merchant_id": "m1"},
    {"txn_id": "r2", "account_id": "a", "event_ts": T0 - 3600, "amount": 20.0, "merchant_id": "m2"},
    {"txn_id": "r3", "account_id": "a", "event_ts": T0 + 100, "amount": 30.0, "merchant_id": "m1"},
    {"txn_id": "r4", "account_id": "a", "event_ts": T0 + 7200, "amount": 40.0, "merchant_id": "m3"},
    {"txn_id": "r5", "account_id": "a", "event_ts": T0 + 10800, "amount": 50.0, "merchant_id": "m1"},
    {"txn_id": "r6", "account_id": "b", "event_ts": T0 + 10800, "amount": 999.0, "merchant_id": "m9"},
]

LOG = compute.EventLog(ROWS, "account_id", "event_ts")


def window(length, closed, align):
    return spec.parse_window({"length": length, "closed": closed, "align": align}, "test")


def ids(rows):
    return [r["txn_id"] for r in rows]


def slice_at(as_of, w, entity="a"):
    lo, hi, inc = compute.window_bounds(as_of, w)
    return ids(LOG.between(entity, lo, hi, inc))


def check_closed_left_excludes_the_scoring_row():
    at = T0 + 10800
    assert slice_at(at, window("24h", "left", "event_time")) == ["r2", "r3", "r4"]
    assert slice_at(at, window("24h", "right", "event_time")) == ["r2", "r3", "r4", "r5"]


def check_the_lower_bound_is_inclusive():
    """A row landing exactly on lo is in.

    The pair is the check. One second wider puts lo exactly on r3 and one second narrower
    puts it one past r3. An inclusive bound keeps r3 in the first and drops it in the
    second, and an exclusive one drops it from both.
    """
    at = T0 + 10800
    on_the_edge = slice_at(at, window("10700s", "left", "event_time"))
    just_past = slice_at(at, window("10699s", "left", "event_time"))
    assert on_the_edge == ["r3", "r4"], on_the_edge
    assert just_past == ["r4"], just_past


def check_calendar_alignment_moves_the_window_off_the_event():
    """The batch reading against the online reading, on one scoring moment."""
    at = T0 + 10800
    online = slice_at(at, window("1d", "left", "event_time"))
    batch = slice_at(at, window("1d", "left", "calendar_day"))
    assert online == ["r2", "r3", "r4"], online
    # The nightly job sees yesterday and nothing from today, however late today it is.
    # It also reaches further back, which is the half people forget.
    assert batch == ["r0", "r2"], batch
    assert online != batch


def check_calendar_right_reaches_back_and_forward():
    at = T0 + 10800
    both = slice_at(at, window("1d", "right", "calendar_day"))
    assert both == ["r0", "r2", "r3", "r4", "r5"], both


def check_the_four_readings_are_not_all_the_same():
    """The point of the fixture. If this fails the other checks prove nothing."""
    at = T0 + 10800
    seen = set()
    for align in ("event_time", "calendar_day"):
        for closed in ("left", "right"):
            seen.add(tuple(slice_at(at, window("1d", closed, align))))
    assert len(seen) == 4, "readings collapsed to {} distinct windows".format(len(seen))


def check_aggregates_over_a_real_window():
    at = T0 + 10800
    w = window("24h", "left", "event_time")
    lo, hi, inc = compute.window_bounds(at, w)
    rows = LOG.between("a", lo, hi, inc)
    assert ids(rows) == ["r2", "r3", "r4"]

    def f(agg, column, dtype):
        return spec.Feature("x", dtype, agg, column, w)

    assert compute.aggregate(rows, f("count", None, "int")) == 3
    assert compute.aggregate(rows, f("sum", "amount", "float")) == 90.0
    assert compute.aggregate(rows, f("mean", "amount", "float")) == 30.0
    assert compute.aggregate(rows, f("max", "amount", "float")) == 40.0
    assert compute.aggregate(rows, f("distinct_count", "merchant_id", "int")) == 3


def check_distinct_count_really_deduplicates():
    """m1 appears four times in this window. A fixture with no repeat in it would pass
    against a plain len() and say nothing about deduplication."""
    at = T0 + 4 * DAY
    w = window("7d", "left", "event_time")
    lo, hi, inc = compute.window_bounds(at, w)
    rows = LOG.between("a", lo, hi, inc)
    assert ids(rows) == ["r1", "r0", "r2", "r3", "r4", "r5"], ids(rows)
    merchants = [r["merchant_id"] for r in rows]
    assert merchants.count("m1") == 4
    feature = spec.Feature("x", "int", "distinct_count", "merchant_id", w)
    assert compute.aggregate(rows, feature) == 3
    counter = spec.Feature("x", "int", "count", None, w)
    assert compute.aggregate(rows, counter) == 6


def check_an_empty_window_gives_zero_and_the_declared_type():
    w = window("1h", "left", "event_time")
    for agg, column, dtype, want in (
        ("count", None, "int", 0),
        ("sum", "amount", "float", 0.0),
        ("max", "amount", "float", 0.0),
        ("mean", "amount", "float", 0.0),
        ("distinct_count", "merchant_id", "int", 0),
    ):
        value = compute.aggregate([], spec.Feature("x", dtype, agg, column, w))
        assert value == want
        assert isinstance(value, int if dtype == "int" else float)


def check_dtype_is_applied_and_not_merely_declared():
    w = window("24h", "left", "event_time")
    rows = [{"amount": 3.0}, {"amount": 4.0}]
    as_int = spec.Feature("x", "int", "mean", "amount", w)
    as_float = spec.Feature("x", "float", "mean", "amount", w)
    assert compute.aggregate(rows, as_int) == 3
    assert compute.aggregate(rows, as_float) == 3.5


def check_the_log_keeps_entities_apart():
    at = T0 + 10800
    w = window("30d", "left", "event_time")
    assert slice_at(at, w, "a") == ["r1", "r0", "r2", "r3", "r4"]
    assert slice_at(at, w, "b") == []
    assert compute.EventLog(ROWS, "account_id", "event_ts").rows_for("missing") == []


def check_snapshot_rows_are_constant_within_an_entity():
    """The whole reason the snapshot arm exists. If this ever fails the leakage probe
    is comparing two things that are not what it says they are."""
    fs = _small_spec()
    scoring = [dict(r, is_fraud=0) for r in ROWS if r["account_id"] == "a"]
    log = compute.EventLog(ROWS, "account_id", "event_ts")
    out = compute.snapshot_rows(log, fs, scoring, T0 + 4 * DAY)
    vectors = {tuple(r[n] for n in fs.feature_names()) for r in out}
    assert len(vectors) == 1, "snapshot produced {} vectors for one account".format(len(vectors))


def check_training_rows_vary_within_an_entity():
    fs = _small_spec()
    scoring = [dict(r, is_fraud=0) for r in ROWS if r["account_id"] == "a"]
    log = compute.EventLog(ROWS, "account_id", "event_ts")
    out = compute.training_rows(log, fs, scoring)
    vectors = {tuple(r[n] for n in fs.feature_names()) for r in out}
    assert len(vectors) > 1, "point in time features were constant, the fixture is wrong"
    assert all(r["is_fraud"] == 0 for r in out)
    assert [r["event_ts"] for r in out] == [r["event_ts"] for r in scoring]


def _small_spec():
    return spec.parse(
        {
            "feature_set": "t",
            "entity": "account_id",
            "event_time": "event_ts",
            "label": "is_fraud",
            "features": [
                {
                    "name": "n_24h",
                    "dtype": "int",
                    "aggregate": "count",
                    "window": {"length": "24h", "closed": "left", "align": "event_time"},
                },
                {
                    "name": "sum_24h",
                    "dtype": "float",
                    "aggregate": "sum",
                    "column": "amount",
                    "window": {"length": "24h", "closed": "left", "align": "event_time"},
                },
            ],
        }
    )


def check_floor_day_is_a_real_midnight():
    assert compute.floor_day(T0) == T0
    assert compute.floor_day(T0 + DAY - 1) == T0
    assert compute.floor_day(T0 + DAY) == T0 + DAY
