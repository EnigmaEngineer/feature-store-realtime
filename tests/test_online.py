"""Checks on the online store's encoding, its key, and what it refuses.

The fixture is deliberately lopsided. A payload with as many good fields as bad ones lets
an inverted rule keep the same failure count, and the check passes against code that says
the opposite of what it means to.

`RecordingStore` is a stub and not a Redis. It exists so a server is not needed to check
the batching, the key and the miss path. The one thing it does faithfully is stringify on
write, because that is the only Redis behaviour these functions have to survive. Everything
else about the round trip is graded against a real server by `scripts/store_check.py`,
which is what `tests/run_with_redis.py` runs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import compute, online, spec

DAY = 86400
T0 = 1750000000 - (1750000000 % DAY)

# Four features rather than two so a rule about choosing between fields has something to
# choose between, and two dtypes so a decoder that ignores the declared type is visible.
SPEC_DOC = {
    "feature_set": "fixture",
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
            "name": "txn_count_24h",
            "dtype": "int",
            "aggregate": "count",
            "window": {"length": "24h", "closed": "left", "align": "event_time"},
        },
        {
            "name": "amount_sum_1h",
            "dtype": "float",
            "aggregate": "sum",
            "column": "amount",
            "window": {"length": "1h", "closed": "left", "align": "event_time"},
        },
        {
            "name": "amount_mean_24h",
            "dtype": "float",
            "aggregate": "mean",
            "column": "amount",
            "window": {"length": "24h", "closed": "left", "align": "event_time"},
        },
    ],
}

FS = spec.parse(SPEC_DOC)

# Account c's one event lands after the load moment, so the loader has never seen it and
# a lookup has to miss. A fixture where every entity is active cannot tell a cold start
# from a quiet account, because both come out as a vector of zeros.
#
# Account a carries an event two days back as well as three recent ones. Without it the
# arm that loads a day early would find a with no history at all and skip it, and the
# check that the arm moves a value would fail for the wrong reason.
ROWS = [
    {"txn_id": "0", "account_id": "a", "event_ts": T0 - 2 * DAY, "amount": 5.0, "merchant_id": "m1"},
    {"txn_id": "1", "account_id": "a", "event_ts": T0 - 7200, "amount": 10.0, "merchant_id": "m1"},
    {"txn_id": "2", "account_id": "a", "event_ts": T0 - 1800, "amount": 33.333333333, "merchant_id": "m2"},
    {"txn_id": "3", "account_id": "a", "event_ts": T0 - 900, "amount": 1234.5678901, "merchant_id": "m1"},
    {"txn_id": "4", "account_id": "b", "event_ts": T0 - 600, "amount": 7.0, "merchant_id": "m3"},
    {"txn_id": "5", "account_id": "c", "event_ts": T0 + 5000, "amount": 99.0, "merchant_id": "m4"},
    # Account d's only event lands exactly on the load moment. Every window is closed on
    # the left, so a vector computed at T0 cannot contain it, and writing d produces the
    # same row of zeros the filter exists to stop. Without this row the comparison in
    # `online_rows` reads the same whether it is strict or not.
    {"txn_id": "6", "account_id": "d", "event_ts": T0, "amount": 42.0, "merchant_id": "m5"},
]

LOG = compute.EventLog(ROWS, "account_id", "event_ts")
AS_OF = T0


class RecordingStore:
    """A dict behind the two calls the loader and the lookup make.

    Values are stringified on write, which is what Redis does. That is the point of the
    stub. A store that handed a float straight back would let an encoder returning
    numbers pass, and the encoder is the thing under test.
    """

    def __init__(self):
        self.data = {}
        # How many rows each pipeline carried. A count of flushes cannot tell a first
        # batch of 499 from one of 500, and an off by one in the pending counter is
        # exactly that shape, so the sizes are kept and the count is len().
        self.batch_sizes = []

    def hset(self, name, mapping=None):
        row = self.data.setdefault(name, {})
        for key, value in mapping.items():
            row[str(key)] = str(value)

    def hgetall(self, name):
        return dict(self.data.get(name, {}))

    def pipeline(self):
        return _Pipe(self)


class _Pipe:
    def __init__(self, store):
        self.store = store
        self.queued = []

    def hset(self, name, mapping=None):
        self.queued.append((name, mapping))

    def execute(self):
        for name, mapping in self.queued:
            self.store.hset(name, mapping=mapping)
        self.store.batch_sizes.append(len(self.queued))
        self.queued = []


def loaded(**kwargs):
    store = RecordingStore()
    written = online.load(store, LOG, FS, AS_OF, **kwargs)
    return store, written


def many_entities(n):
    """A log of n entities with one transaction each, all before the load moment."""
    rows = [
        {
            "txn_id": "b{}".format(i),
            "account_id": "batch-{:05d}".format(i),
            "event_ts": T0 - 3600,
            "amount": 1.0,
            "merchant_id": "m1",
        }
        for i in range(n)
    ]
    return compute.EventLog(rows, "account_id", "event_ts")


def check_a_float_survives_the_round_trip_exactly():
    """Two of the four fixture values are ones a formatted encoder would round.

    33.333333333 and 1234.5678901 both lose digits at six significant figures. 10.0 and
    7.0 do not, so a check written only against those would pass under either encoder.
    """
    for value in (33.333333333, 1234.5678901, 10.0, 0.1 + 0.2):
        text = online.encode_value(value, "float")
        assert online.decode_value(text, "float", "f") == value, value


def check_the_damaged_encoder_really_loses_those_values():
    """The control for the check above.

    Without this, an encoder that happened to be exact on the fixture and lossy in
    general would look identical, and so would a comparison handed the same value twice.
    """
    lost = [
        value
        for value in (33.333333333, 1234.5678901, 10.0, 0.1 + 0.2)
        if online.decode_value(
            online.encode_value(value, "float", "six_significant_digits"), "float", "f"
        )
        != value
    ]
    assert lost == [33.333333333, 1234.5678901, 0.1 + 0.2], lost


def check_an_int_field_holding_a_float_string_is_refused():
    try:
        online.decode_value("3.0", "int", "txn_count_1h")
    except online.OnlineError as exc:
        assert "txn_count_1h" in str(exc), exc
        assert "int" in str(exc), exc
        return
    raise AssertionError("a float string decoded as an int")


def check_an_unknown_dtype_is_refused_on_both_sides():
    for call in (
        lambda: online.encode_value(1, "decimal"),
        lambda: online.decode_value("1", "decimal", "f"),
    ):
        try:
            call()
        except online.OnlineError:
            continue
        raise AssertionError("an unknown dtype was accepted")


def check_a_payload_missing_a_declared_feature_is_refused():
    values = {name: 1 for name in FS.feature_names()}
    values.pop("amount_mean_24h")
    try:
        online.encode_payload(values, FS, AS_OF)
    except online.OnlineError as exc:
        assert "amount_mean_24h" in str(exc), exc
        return
    raise AssertionError("a short payload was written")


def check_a_stored_payload_missing_a_feature_is_refused_on_read():
    raw = online.encode_payload({name: 1 for name in FS.feature_names()}, FS, AS_OF)
    del raw["txn_count_24h"]
    try:
        online.decode_payload(raw, FS)
    except online.OnlineError as exc:
        assert "txn_count_24h" in str(exc), exc
        return
    raise AssertionError("a short payload was read back")


def check_a_stored_payload_with_an_undeclared_field_is_refused():
    """This is what a spec change looks like when the key does not isolate it."""
    raw = online.encode_payload({name: 1 for name in FS.feature_names()}, FS, AS_OF)
    raw["txn_count_7d"] = "4"
    try:
        online.decode_payload(raw, FS)
    except online.OnlineError as exc:
        assert "txn_count_7d" in str(exc), exc
        return
    raise AssertionError("an undeclared field was read back as if it belonged")


def check_a_stored_payload_with_no_as_of_is_refused():
    """Every write puts it there, so a row without it came from something else.

    Returning None would hand the caller a payload whose age is unknown, and an unknown
    age is indistinguishable from a fresh one at the point it matters.
    """
    raw = online.encode_payload({name: 1 for name in FS.feature_names()}, FS, AS_OF)
    del raw["as_of"]
    try:
        online.decode_payload(raw, FS)
    except online.OnlineError as exc:
        assert "as_of" in str(exc), exc
        return
    raise AssertionError("a payload with no timestamp was read back as if it were fresh")


def check_the_as_of_field_is_metadata_and_not_a_feature():
    raw = online.encode_payload({name: 2 for name in FS.feature_names()}, FS, AS_OF)
    values, as_of = online.decode_payload(raw, FS)
    assert sorted(values) == sorted(FS.feature_names()), sorted(values)
    assert as_of == AS_OF, as_of


def check_a_feature_named_as_of_is_refused_rather_than_shadowing_the_metadata():
    doc = {k: v for k, v in SPEC_DOC.items()}
    doc["features"] = list(SPEC_DOC["features"]) + [
        {
            "name": "as_of",
            "dtype": "int",
            "aggregate": "count",
            "window": {"length": "1h", "closed": "left", "align": "event_time"},
        }
    ]
    clashing = spec.parse(doc)
    try:
        online.encode_payload({n: 1 for n in clashing.feature_names()}, clashing, AS_OF)
    except online.OnlineError as exc:
        assert "as_of" in str(exc), exc
        return
    raise AssertionError("a feature named as_of overwrote the metadata field")


def check_the_key_carries_the_fingerprint():
    key = online.store_key(FS, "a")
    assert key == "fs:fixture:{}:a".format(FS.fingerprint()), key
    assert FS.fingerprint() in key


def check_a_changed_spec_writes_to_a_different_key():
    doc = {k: v for k, v in SPEC_DOC.items()}
    widened = [dict(f) for f in SPEC_DOC["features"]]
    widened[0]["window"] = {"length": "2h", "closed": "left", "align": "event_time"}
    doc["features"] = widened
    other = spec.parse(doc)
    assert other.fingerprint() != FS.fingerprint()
    assert online.store_key(other, "a") != online.store_key(FS, "a")


def check_the_unkeyed_control_really_collides():
    """The isolation arm is worth nothing unless its control shares a key.

    Without this, `keyed_by_fingerprint=False` could quietly produce two different keys
    and the arm would report a miss for the wrong reason.
    """
    doc = {k: v for k, v in SPEC_DOC.items()}
    widened = [dict(f) for f in SPEC_DOC["features"]]
    widened[0]["window"] = {"length": "2h", "closed": "left", "align": "event_time"}
    doc["features"] = widened
    other = spec.parse(doc)
    assert online.store_key(other, "a", False) == online.store_key(FS, "a", False)


def check_a_miss_raises_rather_than_returning_zeros():
    store, _ = loaded()
    try:
        online.lookup(store, FS, "never-seen")
    except online.Miss as exc:
        assert "never-seen" in str(exc), exc
        return
    raise AssertionError("a miss came back as a value")


def check_a_miss_is_not_a_key_error():
    """A caller wrapping its own dict access in `except KeyError` must not swallow it."""
    assert not issubclass(online.Miss, KeyError)


def check_the_loader_skips_entities_it_could_not_have_seen():
    """Account c transacts after the load moment and account d exactly on it.

    Every window for either is empty, so writing them produces vectors of zeros that read
    exactly like an account with a quiet week. The loader has to leave both out and let
    the lookup miss. A fixture where every entity is active cannot tell the two cases
    apart, and one without d cannot tell a strict comparison from a loose one.
    """
    store, written = loaded()
    assert written == 2, written
    for entity_id in ("c", "d"):
        try:
            online.lookup(store, FS, entity_id)
        except online.Miss:
            continue
        raise AssertionError(
            "{} was written as zeros and the loader had seen nothing".format(entity_id)
        )


def check_the_round_trip_reproduces_the_reference_through_the_store():
    store, _ = loaded()
    for entity_id, expected in online.online_rows(LOG, FS, AS_OF):
        got, _ = online.lookup(store, FS, entity_id)
        assert got == expected, (entity_id, got, expected)


def check_online_rows_come_out_in_entity_order():
    ids = [eid for eid, _ in online.online_rows(LOG, FS, AS_OF)]
    assert ids == ["a", "b"], ids
    later = [eid for eid, _ in online.online_rows(LOG, FS, AS_OF + 10000)]
    assert later == ["a", "b", "c", "d"], later


def check_the_pipeline_flushes_a_partial_batch():
    """Three entities against a batch of two, so one execute carries a remainder.

    A batch size that divides the entity count exactly would let a loader that drops the
    tail write everything anyway.
    """
    store = RecordingStore()
    written = online.load(store, LOG, FS, AS_OF + 10000, batch=3)
    assert written == 4, written
    assert store.batch_sizes == [3, 1], store.batch_sizes
    assert len(store.data) == 4, sorted(store.data)


def check_the_default_batch_size_is_the_one_the_loader_really_uses():
    """501 entities against the default, so the first flush lands on the boundary.

    Every other check here names `batch`, which leaves the production default the one
    thing nothing exercises. 501 rather than 500 because at exactly the batch size the
    remainder is empty and a loader one out either way still flushes once.

    The sizes are asserted and not the number of flushes. A pending counter starting at
    one gives two flushes as well, carrying 499 rows and then 2.
    """
    store = RecordingStore()
    written = online.load(store, many_entities(501), FS, T0)
    assert written == 501, written
    assert store.batch_sizes == [500, 1], store.batch_sizes


def check_a_flush_resets_the_pending_counter():
    """Three flushes, because two cannot see it.

    A counter reset to one instead of zero still gives a first batch of the right size
    and a remainder of the right size, so the shape only goes wrong from the second full
    batch onward. Eight entities against a batch of three is the cheapest input that
    reaches it.
    """
    store = RecordingStore()
    written = online.load(store, many_entities(8), FS, T0, batch=3)
    assert written == 8, written
    assert store.batch_sizes == [3, 3, 2], store.batch_sizes


def check_loading_an_empty_log_writes_nothing_and_flushes_nothing():
    store = RecordingStore()
    written = online.load(store, compute.EventLog([], "account_id", "event_ts"), FS, T0)
    assert written == 0, written
    assert store.batch_sizes == [], store.batch_sizes


def check_an_unknown_damage_is_refused():
    try:
        online.load(RecordingStore(), LOG, FS, AS_OF, damage="no_such_arm")
    except online.OnlineError as exc:
        assert "no_such_arm" in str(exc), exc
        return
    raise AssertionError("an unknown damage was accepted")


def check_loading_a_day_early_moves_a_value():
    """The damage catalogue's second arm has to be able to fail the comparison."""
    clean, _ = loaded()
    early = RecordingStore()
    online.load(early, LOG, FS, AS_OF, damage="load_a_day_early")
    before, _ = online.lookup(clean, FS, "a")
    after, early_as_of = online.lookup(early, FS, "a")
    assert before != after, (before, after)
    # The stored as_of moves with the load, so a reader can tell which arm wrote the row.
    assert early_as_of == AS_OF - DAY, early_as_of


def check_bytes_from_a_client_that_does_not_decode_are_read():
    """redis-py returns bytes unless the connection was opened with decode_responses.

    A loader and a service configured differently is an ordinary mistake and it should
    not read as a missing field.
    """
    raw = online.encode_payload({n: 3 for n in FS.feature_names()}, FS, AS_OF)
    as_bytes = {k.encode(): v.encode() for k, v in raw.items()}
    values, as_of = online.decode_payload(as_bytes, FS)
    assert as_of == AS_OF, as_of
    assert values["txn_count_1h"] == 3, values
