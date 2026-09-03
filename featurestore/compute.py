"""The reference implementation of a feature set.

This is plain Python over a list of dicts and it is deliberately slow. It exists to be
the thing the Spark job and the online loader are checked against, so it has to be
obviously correct rather than fast. When the two disagree, this is the one that is
right until somebody proves otherwise.

Everything here is point in time. A feature computed for an event at time t sees only
rows the window admits, and with closed=left that excludes the event itself.
"""

import bisect

DAY = 86400


def floor_day(ts):
    return (int(ts) // DAY) * DAY


class EventLog:
    """An event log indexed by entity, each entity's rows sorted by time.

    Sorting once and slicing with bisect is the only concession to speed here. A linear
    scan per feature per row made the leakage probe take minutes rather than seconds.
    """

    def __init__(self, rows, entity_field, time_field):
        self.entity_field = entity_field
        self.time_field = time_field
        buckets = {}
        for row in rows:
            buckets.setdefault(row[entity_field], []).append(row)
        self._rows = {}
        self._times = {}
        for eid, group in buckets.items():
            group.sort(key=lambda r: r[time_field])
            self._rows[eid] = group
            self._times[eid] = [r[time_field] for r in group]

    def entities(self):
        return list(self._rows)

    def rows_for(self, entity_id):
        return self._rows.get(entity_id, [])

    def between(self, entity_id, lo, hi, hi_inclusive):
        """Rows with lo <= t < hi, or lo <= t <= hi when hi_inclusive."""
        times = self._times.get(entity_id)
        if not times:
            return []
        start = bisect.bisect_left(times, lo)
        end = bisect.bisect_right(times, hi) if hi_inclusive else bisect.bisect_left(times, hi)
        return self._rows[entity_id][start:end]


def window_bounds(as_of, window):
    """The half open or closed interval a window admits at a given scoring time.

    Returns (lo, hi, hi_inclusive). Four readings of the same string like 7d come out
    of the two fields, and they are not interchangeable. See docs/adr-0002.
    """
    if window.align == "event_time":
        anchor = as_of
    elif window.align == "calendar_day":
        anchor = floor_day(as_of)
    else:
        raise ValueError("unknown align {!r}".format(window.align))

    lo = anchor - window.length_s
    if window.closed == "left":
        # Under calendar_day this is the nightly batch reading. The window ends at
        # midnight and the whole of today is invisible, however late in the day the
        # scoring event sits.
        return lo, anchor, False
    return lo, as_of, True


def aggregate(rows, feature):
    agg = feature.aggregate
    if agg == "count":
        value = len(rows)
    else:
        values = [r[feature.column] for r in rows if r.get(feature.column) is not None]
        if agg == "distinct_count":
            value = len(set(values))
        elif not values:
            # No rows in the window is not an error and it is not zero for every
            # aggregate. A sum of nothing is 0 and a max of nothing is not defined, so
            # both come back as 0.0 and the caller cannot tell them apart. Named in the
            # README limitations rather than solved with a null, because the model has
            # to be handed a number and choosing one here keeps that choice in one place.
            value = 0
        elif agg == "sum":
            value = sum(values)
        elif agg == "mean":
            value = sum(values) / len(values)
        elif agg == "max":
            value = max(values)
        else:
            raise ValueError("unknown aggregate {!r}".format(agg))

    return int(value) if feature.dtype == "int" else float(value)


def features_as_of(log, feature_set, entity_id, as_of):
    """One row of features for one entity at one moment."""
    out = {}
    for feature in feature_set.features:
        lo, hi, inclusive = window_bounds(as_of, feature.window)
        rows = log.between(entity_id, lo, hi, inclusive)
        out[feature.name] = aggregate(rows, feature)
    return out


def training_rows(log, feature_set, scoring_rows):
    """Point in time features for every row that will carry a label.

    scoring_rows are the events being scored. Each one supplies its own entity and its
    own timestamp, which is the only thing that makes this different from joining a
    feature snapshot onto a label table.
    """
    entity = feature_set.entity
    time_field = feature_set.event_time
    out = []
    for row in scoring_rows:
        vals = features_as_of(log, feature_set, row[entity], row[time_field])
        vals[entity] = row[entity]
        vals[time_field] = row[time_field]
        vals[feature_set.label] = row[feature_set.label]
        out.append(vals)
    return out


def snapshot_rows(log, feature_set, scoring_rows, as_of):
    """The wrong way, kept because it is the thing worth measuring against.

    One feature vector per entity computed at a single moment, then joined onto every
    labelled event for that entity. This is what a feature table built by a nightly job
    and joined on entity alone produces. scripts/leakage_probe.py measures what it buys.
    """
    entity = feature_set.entity
    time_field = feature_set.event_time
    cache = {}
    out = []
    for row in scoring_rows:
        eid = row[entity]
        if eid not in cache:
            cache[eid] = features_as_of(log, feature_set, eid, as_of)
        vals = dict(cache[eid])
        vals[entity] = eid
        vals[time_field] = row[time_field]
        vals[feature_set.label] = row[feature_set.label]
        out.append(vals)
    return out
