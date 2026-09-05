"""The offline materialization job.

Reads the transaction log, builds one point in time feature row per scoring event, and
writes the training table. The whole job is generated from the spec. Nothing here knows
that `txn_count_24h` means anything.

`featurestore/compute.py` stays the oracle. Where this and that disagree, that one is
right until somebody proves otherwise, and `scripts/oracle_check.py` is what asks.

pyspark is imported inside the functions that need it. The frame arithmetic, the column
plan and the damage catalogue hold the decisions, so they have to run on a clone that
has installed `requirements.txt` and nothing else.
"""

from featurestore import frames

DAY = frames.DAY

INTEGRAL_TYPES = ("byte", "short", "int", "integer", "bigint", "long")

# Wrong versions of the same job, built by the same builder rather than written out
# beside it. A comparison that reports no differences looks identical whether the job is
# right or the two sides are reading different rows, so `scripts/oracle_check.py` runs
# every one of these and fails if any of them agrees with the reference.
#
# One damages each end of the frame. Without the second, a comparison would be blind to
# the lower bound and would still look convincing.
DAMAGES = {
    "ignore_closed": "admit the scoring event into its own window",
    "stretch_lower_by_a_day": "a window length that lost its unit conversion",
}


class UnsupportedFeature(ValueError):
    """An aggregate or a type the job cannot build a column for."""


class BadEventTime(TypeError):
    """The event time column is not whole seconds, so the frame arithmetic is unsound."""


def plan(feature_set, damage=None):
    """One entry per feature, naming the frame and the aggregate.

    Returned before any Spark object exists, so the decisions can be read and asserted
    without a session.
    """
    if damage is not None and damage not in DAMAGES:
        raise UnsupportedFeature(
            "unknown damage {!r}, expected one of {}".format(damage, sorted(DAMAGES))
        )

    out = []
    for feature in feature_set.features:
        lower, upper = frames.frame_offsets(feature.window)
        if damage == "ignore_closed":
            upper = 0
        elif damage == "stretch_lower_by_a_day":
            lower -= DAY
        out.append(
            {
                "name": feature.name,
                "aggregate": feature.aggregate,
                "column": feature.column,
                "dtype": feature.dtype,
                "lower": lower,
                "upper": upper,
            }
        )
    return out


def output_columns(feature_set, carry=()):
    """The training table's columns, in order.

    Separate from `materialize` so the default `carry` is exercised by a check. A
    caller who never names a carry column gets the table this returns, and that is the
    shape nothing would otherwise look at.
    """
    return (
        list(carry)
        + [feature_set.entity, feature_set.event_time]
        + feature_set.feature_names()
        + [feature_set.label, "spec_fingerprint"]
    )


def _agg_expression(entry, F):
    """The Spark expression for one aggregate, before the frame is applied.

    Returns the expression and whether an empty frame needs turning into a zero. An
    empty frame has to come back as 0 for every aggregate, because `compute.aggregate`
    does the same and the model has to be handed a number. Spark's sum, avg and max all
    return null on an empty frame. count returns 0 on its own and collect_set returns an
    empty array.
    """
    agg = entry["aggregate"]
    if agg == "count":
        return F.count(F.lit(1)), False
    col = F.col(entry["column"])
    if agg == "sum":
        return F.sum(col), True
    if agg == "mean":
        return F.avg(col), True
    if agg == "max":
        return F.max(col), True
    if agg == "distinct_count":
        # Spark has no count(distinct) over a window frame. collect_set does work over
        # one and it drops nulls, which is what compute.aggregate does by filtering.
        # The cost is that the whole set is held per row. A 7 day window on the busiest
        # account in this corpus is a few hundred merchant ids, so it is fine here and
        # it would not be on a real card portfolio.
        return F.collect_set(col), False
    raise UnsupportedFeature("no Spark expression for aggregate {!r}".format(agg))


def _cast(column, dtype):
    if dtype == "int":
        return column.cast("int")
    if dtype == "float":
        return column.cast("double")
    raise UnsupportedFeature("no Spark cast for dtype {!r}".format(dtype))


def check_event_time(df, feature_set):
    """Refuse a non integral event time column.

    `frames.frame_offsets` takes a second off an exclusive upper bound, which only lands
    on the same set of rows when the column holds whole seconds. On a double column an
    event 0.4 s before the scoring event would be admitted by the reference
    implementation and dropped here, and nothing would raise.
    """
    field = dict(df.dtypes).get(feature_set.event_time)
    if field is None:
        raise BadEventTime("the log has no column {!r}".format(feature_set.event_time))
    if field.lower() not in INTEGRAL_TYPES:
        raise BadEventTime(
            "event time column {!r} is {}, and the frame arithmetic needs whole "
            "seconds".format(feature_set.event_time, field)
        )
    return field


def materialize(df, feature_set, carry=(), damage=None, one_projection=True):
    """The training table for a feature set over a whole transaction log.

    Every row of `df` is a scoring event. `carry` names source columns to keep on the
    output, which the oracle check needs so it can pair rows by their id rather than by
    entity and timestamp. Two transactions on one account in the same second are
    otherwise indistinguishable.

    `one_projection=False` builds the same features by chaining a withColumn per feature,
    which is the obvious way to write this and is what `scripts/plan_probe.py` measures
    against. It is a parameter rather than a second function so the arm that loses comes
    out of this builder and not out of my memory of it.
    """
    from pyspark.sql import Window as W
    from pyspark.sql import functions as F

    check_event_time(df, feature_set)
    order = F.col(feature_set.event_time).cast("long")

    columns = []
    chained = df
    for entry in plan(feature_set, damage=damage):
        frame = (
            W.partitionBy(feature_set.entity)
            .orderBy(order)
            .rangeBetween(entry["lower"], entry["upper"])
        )
        expr, null_to_zero = _agg_expression(entry, F)
        value = expr.over(frame)
        if entry["aggregate"] == "distinct_count":
            value = F.size(value)
        if null_to_zero:
            value = F.coalesce(value, F.lit(0))
        cast = _cast(value, entry["dtype"])
        columns.append(cast.alias(entry["name"]))
        chained = chained.withColumn(entry["name"], cast)

    # One select rather than a withColumn per feature. Chaining withColumn produces one
    # nested projection per feature, and Spark only groups window expressions inside a
    # single projection, so the plan came out with seven Window operators and seven Sorts
    # of the same rows. In one projection it comes out with one of each, which is more
    # than I expected, because the grouping is by partition and ordering rather than by
    # frame. scripts/plan_probe.py is the measurement.
    out = df.select("*", *columns) if one_projection else chained

    # The fingerprint travels with the table. A training set whose provenance is a
    # sentence in a commit message cannot be checked against the store it will be served
    # from.
    out = out.withColumn("spec_fingerprint", F.lit(feature_set.fingerprint()))
    return out.select(*output_columns(feature_set, carry))


def read_log(spark, path):
    """The corpus, typed the way featurestore/corpus.py types it.

    An inferred schema would give event_ts a bigint here and a string somewhere else,
    and check_event_time would then pass or fail depending on which file it saw.
    """
    schema = (
        "txn_id string, account_id string, event_ts bigint, amount double, "
        "merchant_id string, is_fraud int"
    )
    return spark.read.csv(path, header=True, schema=schema)


def session(app_name="featurestore-materialize", cores=2):
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master("local[{}]".format(cores))
        .appName(app_name)
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        # The console progress bar writes carriage returns into stdout, which shreds any
        # transcript of a run and makes a printed figure unquotable.
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
