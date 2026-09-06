"""The online store. What gets written, under what key, and what comes back.

The batch job builds a training table with one row per scoring event. The online store
holds one row per entity, computed at a single moment, and the service reads it back at
request time. Same spec, same reference implementation, different shape.

Two things here carry the design.

The key comes from `spec.FeatureSet.online_key`, so it holds the fingerprint of the
definition that produced the row. A loader running an older spec writes to a different
keyspace, and a service asking for the newer one gets a miss rather than a plausible
answer built from a definition nobody is serving. `scripts/store_check.py` measures what
that is worth by turning it off.

A miss is an exception and never a zero vector. `compute.aggregate` returns 0 for an
empty window, so the training table already contains rows of zeros for an entity's first
transaction. If a miss also decoded to zeros then the service could not tell a new entity
from one the loader had seen, and the model would be handed a row that looks like real
history. The decision of what to do about a miss belongs to the caller and this module
refuses to make it quietly.
"""

from featurestore import compute

# The one field on the stored hash that is not a feature. A spec is free to name a
# feature anything, so the collision is checked rather than assumed.
AS_OF_FIELD = "as_of"

# Wrong loaders, built by the same builder rather than written out beside it. Each one
# has to disagree with the reference on at least one cell or the comparison in
# `scripts/store_check.py` has proved nothing.
DAMAGES = {
    "six_significant_digits": "a float encoder that formats instead of round tripping",
    "load_a_day_early": "a loader whose as_of is not the moment it was asked for",
}


class OnlineError(ValueError):
    """A payload that cannot be written or cannot be read back as declared."""


class Miss(LookupError):
    """No row for this entity under this definition.

    Deliberately not a KeyError. A caller writing `except KeyError` around a dict lookup
    somewhere else in its own code should not silently swallow this.
    """


def store_key(feature_set, entity_id, keyed_by_fingerprint=True):
    """The Redis key for one entity.

    `keyed_by_fingerprint=False` is the control arm and not a supported mode. It exists
    so the isolation the fingerprint buys can be measured rather than asserted, and it
    comes out of this function so the losing side is built by the shipped code.
    """
    if keyed_by_fingerprint:
        return feature_set.online_key(entity_id)
    return "fs:{}:{}".format(feature_set.name, entity_id)


def encode_value(value, dtype, damage=None):
    if dtype == "int":
        return str(int(value))
    if dtype == "float":
        if damage == "six_significant_digits":
            return "{:.6g}".format(float(value))
        # repr round trips a double exactly on Python 3 and str is the same function.
        # The damaged arm is here to put a number on what a formatted encoder costs,
        # because six significant digits is what somebody reaches for when they want the
        # stored value to look tidy in redis-cli.
        return repr(float(value))
    raise OnlineError("no online encoding for dtype {!r}".format(dtype))


def decode_value(text, dtype, field):
    try:
        if dtype == "int":
            return int(text)
        if dtype == "float":
            return float(text)
    except (TypeError, ValueError):
        raise OnlineError(
            "field {!r} holds {!r}, which does not read as {}".format(field, text, dtype)
        )
    raise OnlineError("no online decoding for dtype {!r}".format(dtype))


def encode_payload(values, feature_set, as_of, damage=None):
    """One entity's feature vector as a flat mapping of strings.

    Redis hash fields and values are bytes. Everything typed has to survive that round
    trip or training and serving disagree without anything raising.
    """
    schema = feature_set.payload_schema()
    if AS_OF_FIELD in schema:
        raise OnlineError(
            "a feature is named {!r}, which is the field carrying when the row was "
            "computed. Rename the feature or the store cannot tell one from the "
            "other.".format(AS_OF_FIELD)
        )

    missing = sorted(set(schema) - set(values))
    if missing:
        raise OnlineError("cannot write a payload missing {}".format(missing))

    out = {name: encode_value(values[name], dtype, damage) for name, dtype in schema.items()}
    out[AS_OF_FIELD] = str(int(as_of))
    return out


def decode_payload(raw, feature_set):
    """The typed feature vector and its metadata, or an error saying which field.

    Refuses a payload that is missing a declared feature and one carrying a field the
    spec does not declare. The second case is what a spec change looks like when the
    fingerprint is not in the key, which is the arm `scripts/store_check.py` runs.
    """
    schema = feature_set.payload_schema()
    fields = {k.decode() if isinstance(k, bytes) else k: v for k, v in raw.items()}
    fields = {k: v.decode() if isinstance(v, bytes) else v for k, v in fields.items()}

    missing = sorted(set(schema) - set(fields))
    if missing:
        raise OnlineError("stored payload is missing {}".format(missing))
    extra = sorted(set(fields) - set(schema) - {AS_OF_FIELD})
    if extra:
        raise OnlineError("stored payload carries undeclared fields {}".format(extra))
    if AS_OF_FIELD not in fields:
        # Every write puts it there, so a row without it was written by something else.
        # Returning None instead would hand the caller a payload whose age is unknown
        # and look exactly like a fresh one.
        raise OnlineError(
            "stored payload has no {!r}, so nothing can say how old it is".format(
                AS_OF_FIELD
            )
        )

    values = {name: decode_value(fields[name], dtype, name) for name, dtype in schema.items()}
    return values, decode_value(fields[AS_OF_FIELD], "int", AS_OF_FIELD)


def online_rows(log, feature_set, as_of):
    """One feature vector per entity the log has seen by `as_of`, all at that moment.

    Sorted by entity id so two loads of the same log write in the same order. Nothing
    downstream depends on that and a probe comparing two runs does.

    An entity with no event strictly before `as_of` is skipped rather than written. In
    production the log at load time simply would not contain it, so the filter costs
    nothing there. Every probe in this repo hands it a log that runs past `as_of`. Without
    this line the loader then writes that entity a vector of zeros, because every window is
    empty. That turns a cold start into a row that reads like real history and it makes any
    miss rate measured here a fiction.

    Strictly before, and the boundary is the whole point. Every window in this spec is
    `closed: left`, so a vector computed at `as_of` cannot contain an event that happened
    at `as_of`. Writing that entity produces the same row of zeros the filter exists to
    stop. The first version of this line compared with `>` and a mutation pass is what
    asked, because no fixture had put an entity's first event on the boundary.

    This is the same computation the leakage probe calls wrong, and here it is right. A
    snapshot joined onto labelled history is wrong because the labels sit at many times.
    A snapshot served to requests arriving now is what an online store is. What it costs
    is that the vector ages from the moment it lands, which is
    `scripts/staleness_probe.py`.
    """
    time_field = feature_set.event_time
    for entity_id in sorted(log.entities()):
        history = log.rows_for(entity_id)
        if not history or history[0][time_field] >= as_of:
            continue
        yield entity_id, compute.features_as_of(log, feature_set, entity_id, as_of)


def load(
    client,
    log,
    feature_set,
    as_of,
    damage=None,
    keyed_by_fingerprint=True,
    batch=500,
):
    """Fill the store from an event log. Returns how many entities were written.

    One function rather than a compute step and a write step, because the two damaged
    arms attack different halves and splitting them let `load_a_day_early` move the
    stored timestamp while the values were still computed at the right moment. That
    version passed its own comparison and the arm was measuring nothing.

    Batched through a pipeline because a round trip per entity is the difference between
    seconds and minutes at four thousand accounts, and this is the one path here that
    touches every key.
    """
    if damage is not None and damage not in DAMAGES:
        raise OnlineError(
            "unknown damage {!r}, expected one of {}".format(damage, sorted(DAMAGES))
        )
    if damage == "load_a_day_early":
        as_of = as_of - compute.DAY

    written = 0
    pending = 0
    pipe = client.pipeline()
    for entity_id, values in online_rows(log, feature_set, as_of):
        payload = encode_payload(values, feature_set, as_of, damage)
        pipe.hset(store_key(feature_set, entity_id, keyed_by_fingerprint), mapping=payload)
        pending += 1
        written += 1
        if pending >= batch:
            pipe.execute()
            pipe = client.pipeline()
            pending = 0
    if pending:
        pipe.execute()
    return written


def lookup(client, feature_set, entity_id, keyed_by_fingerprint=True):
    """The feature vector for one entity, typed, with the moment it was computed.

    Raises `Miss` when there is no row. See the module docstring for why that is not a
    vector of zeros.
    """
    raw = client.hgetall(store_key(feature_set, entity_id, keyed_by_fingerprint))
    if not raw:
        raise Miss(
            "no online row for {!r} under {} at {}".format(
                entity_id, feature_set.name, feature_set.fingerprint()
            )
        )
    return decode_payload(raw, feature_set)
