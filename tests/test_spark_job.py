"""Checks on the parts of the Spark job that decide anything.

The plan, the column list, the damage catalogue and the event time guard all run without
a session. That is why they live in `featurestore/spark_job.py` above the pyspark import
rather than inside it. The job running end to end is `scripts/oracle_check.py`,
because that needs a corpus and a JVM and neither belongs in a suite a clone runs.
"""

from featurestore import spark_job, spec

DOC = {
    "feature_set": "t",
    "entity": "account_id",
    "event_time": "event_ts",
    "label": "is_fraud",
    "features": [
        {
            "name": "txn_count_1h",
            "dtype": "int",
            "aggregate": "count",
            "window": {"length": "1h", "closed": "left", "align": "event_time"},
        },
        {
            "name": "amount_mean_7d",
            "dtype": "float",
            "aggregate": "mean",
            "column": "amount",
            "window": {"length": "7d", "closed": "left", "align": "event_time"},
        },
    ],
}


class FakeFrame:
    """Just enough of a DataFrame to answer `dtypes`."""

    def __init__(self, dtypes):
        self.dtypes = dtypes


def a_spec():
    return spec.parse(DOC)


def check_the_plan_carries_one_entry_per_feature():
    plan = spark_job.plan(a_spec())
    assert [e["name"] for e in plan] == ["txn_count_1h", "amount_mean_7d"]
    assert plan[0]["lower"] == -3600 and plan[0]["upper"] == -1
    assert plan[1]["lower"] == -604800 and plan[1]["upper"] == -1


def check_every_named_damage_changes_the_plan():
    # Driven off the catalogue rather than off a list beside it. A damage added later
    # and never wired up shows here rather than passing quietly.
    clean = spark_job.plan(a_spec())
    for name in spark_job.DAMAGES:
        damaged = spark_job.plan(a_spec(), damage=name)
        assert damaged != clean, "damage {!r} left the plan alone".format(name)


def check_the_two_damages_hit_opposite_ends_of_the_frame():
    clean = spark_job.plan(a_spec())
    closed = spark_job.plan(a_spec(), damage="ignore_closed")
    stretched = spark_job.plan(a_spec(), damage="stretch_lower_by_a_day")
    assert [e["lower"] for e in closed] == [e["lower"] for e in clean]
    assert [e["upper"] for e in closed] == [0, 0]
    assert [e["upper"] for e in stretched] == [e["upper"] for e in clean]
    assert [e["lower"] for e in stretched] == [
        e["lower"] - spark_job.DAY for e in clean
    ]


def check_an_unknown_damage_is_refused():
    try:
        spark_job.plan(a_spec(), damage="whatever")
    except spark_job.UnsupportedFeature as exc:
        assert "unknown damage" in str(exc)
        return
    raise AssertionError("an unknown damage name was accepted")


def check_the_default_output_columns_carry_nothing_extra():
    # The production default. Every caller in this repo passes carry=("txn_id",), so
    # without this the shape a caller who names nothing gets is unexercised.
    cols = spark_job.output_columns(a_spec())
    assert cols == [
        "account_id",
        "event_ts",
        "txn_count_1h",
        "amount_mean_7d",
        "is_fraud",
        "spec_fingerprint",
    ]


def check_carried_columns_come_first_and_are_not_duplicated():
    cols = spark_job.output_columns(a_spec(), carry=("txn_id",))
    assert cols[0] == "txn_id"
    assert len(cols) == len(set(cols))
    assert cols[1:] == spark_job.output_columns(a_spec())


def check_the_fingerprint_is_always_on_the_table():
    assert "spec_fingerprint" in spark_job.output_columns(a_spec())


def check_a_string_event_time_is_refused():
    df = FakeFrame([("account_id", "string"), ("event_ts", "string")])
    try:
        spark_job.check_event_time(df, a_spec())
    except spark_job.BadEventTime as exc:
        assert "whole seconds" in str(exc)
        return
    raise AssertionError("a string event time was accepted")


def check_a_double_event_time_is_refused():
    df = FakeFrame([("account_id", "string"), ("event_ts", "double")])
    try:
        spark_job.check_event_time(df, a_spec())
    except spark_job.BadEventTime:
        return
    raise AssertionError("a double event time was accepted and the frame is off by one")


def check_a_missing_event_time_column_is_refused_by_name():
    df = FakeFrame([("account_id", "string")])
    try:
        spark_job.check_event_time(df, a_spec())
    except spark_job.BadEventTime as exc:
        assert "event_ts" in str(exc)
        return
    raise AssertionError("a log with no event time column was accepted")


def check_a_bigint_event_time_is_accepted():
    df = FakeFrame([("account_id", "string"), ("event_ts", "bigint")])
    assert spark_job.check_event_time(df, a_spec()) == "bigint"


def check_a_calendar_aligned_spec_cannot_be_planned():
    doc = dict(DOC)
    doc["features"] = [
        dict(
            DOC["features"][0],
            window={"length": "1d", "closed": "left", "align": "calendar_day"},
        )
    ]
    try:
        spark_job.plan(spec.parse(doc))
    except Exception as exc:
        assert "not a constant offset" in str(exc)
        return
    raise AssertionError("a calendar aligned feature was planned as a fixed frame")
