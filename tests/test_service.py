"""Checks on the serving path, with no framework and no server.

Everything here runs under `tests/run_all.py` on a clone that installed only
`requirements.txt`. That is the point of `featurestore/service.py` importing nothing. What
`tests/run_with_api.py` adds is whether FastAPI really returns the status this module
asked for, which is a different question from whether the status is right.

The store is the same dict stub `tests/test_online.py` uses, for the same reason. It
stringifies on write, which is the one Redis behaviour these functions have to survive.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import model as model_mod
from featurestore import online, service, spec

DAY = 86400
T0 = 1750000000 - (1750000000 % DAY)

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
            "name": "amount_sum_1h",
            "dtype": "float",
            "aggregate": "sum",
            "column": "amount",
            "window": {"length": "1h", "closed": "left", "align": "event_time"},
        },
    ],
}

FS = spec.parse(SPEC_DOC)
NAMES = FS.feature_names()
COEFS = {"txn_count_1h": 0.25, "amount_sum_1h": -0.5}
VALUES = {"txn_count_1h": 3, "amount_sum_1h": 8.0}

MODEL = model_mod.LinearModel(NAMES, COEFS, -0.75, FS.fingerprint(), 0.4)


class DictStore:
    """The two calls the lookup makes, over a dict of dicts."""

    def __init__(self):
        self.data = {}

    def hset(self, key, mapping=None):
        self.data.setdefault(key, {}).update({k: str(v) for k, v in mapping.items()})

    def hgetall(self, key):
        return dict(self.data.get(key, {}))


def store_with(values=None, as_of=T0, key_entity="a"):
    store = DictStore()
    payload = online.encode_payload(values or VALUES, FS, as_of)
    store.hset(FS.online_key(key_entity), mapping=payload)
    return store


def build(store=None, policy=None, now=T0):
    return service.Service(
        store if store is not None else store_with(),
        FS,
        MODEL,
        policy,
        clock=lambda: now,
    )


def request(entity="a", ts=T0 + 60):
    return {"entity_id": entity, "event_ts": ts}


def check_a_good_request_is_scored_and_the_score_is_the_model_on_the_stored_vector():
    """Asserted against the model applied to the vector rather than against a literal.

    A literal here would pin the fixture's arithmetic and pass whatever vector the service
    fetched, including the wrong entity's.
    """
    result = build().handle(request())
    assert result.status == service.OK, (result.status, result.body)
    assert result.body["score"] == MODEL.score(VALUES), result.body
    assert result.log["outcome"] == "scored"


def check_every_scored_response_carries_the_age_of_what_it_was_built_from():
    """This is the whole of the `as_of` decision and it belongs on the response, not only
    in the log. A caller cannot act on a field it never receives."""
    result = build(now=T0 + 7200).handle(request())
    assert service.response_carries_its_age(result.body), result.body
    assert result.body["features_as_of"] == T0
    assert result.body["features_age_s"] == 7200


def check_the_age_is_measured_from_the_clock_and_not_from_the_request_timestamp():
    """A request carrying an event time far from now must not move the reported age. The
    two timestamps are different quantities and folding them makes the age unfalsifiable.
    """
    a = build(now=T0 + 100).handle(request(ts=T0 + 100))
    b = build(now=T0 + 100).handle(request(ts=T0 + 999999))
    assert a.body["features_age_s"] == b.body["features_age_s"] == 100


def check_an_unbounded_policy_says_unbounded_and_not_fresh():
    """`fresh` would be a claim nothing checked. A dashboard counting fresh responses
    must not be able to count these among them."""
    result = build(now=T0 + 10 * DAY).handle(request())
    assert result.status == service.OK
    assert result.body["freshness"] == service.UNBOUNDED, result.body


def check_a_policy_refuses_a_vector_past_its_threshold_and_serves_one_inside_it():
    policy = service.Policy(max_age_s=3600)
    inside = build(policy=policy, now=T0 + 3599).handle(request())
    assert inside.status == service.OK and inside.body["freshness"] == service.FRESH

    outside = build(policy=policy, now=T0 + 3601).handle(request())
    assert outside.status == service.GONE, (outside.status, outside.body)
    assert outside.log["outcome"] == "refused"
    assert outside.body["age_s"] == 3601 and outside.body["max_age_s"] == 3600


def check_the_threshold_is_inclusive_at_the_boundary():
    """A guard is a comparison and the boundary is where it goes wrong. An age of exactly
    max_age_s is inside, so `>` and `>=` are different code and one fixture sits on it."""
    policy = service.Policy(max_age_s=3600)
    on_it = build(policy=policy, now=T0 + 3600).handle(request())
    assert on_it.status == service.OK, (on_it.status, on_it.body)
    assert on_it.body["freshness"] == service.FRESH


def check_a_zero_or_negative_max_age_is_refused_at_construction():
    """A max_age_s of 0 refuses everything, which is a service that is off rather than a
    service that is strict. None is how you say no threshold."""
    for bad in (0, -1):
        try:
            service.Policy(max_age_s=bad)
        except ValueError:
            continue
        raise AssertionError("Policy accepted max_age_s={!r}".format(bad))
    assert service.Policy(max_age_s=None).judge(10 ** 9) == service.UNBOUNDED


def check_a_miss_is_a_404_and_never_a_vector_of_zeros():
    """The reason this matters is in compute.aggregate. An empty window is 0 for every
    aggregate, so a miss decoded to zeros is the same row a quiet account gets."""
    result = build().handle(request(entity="never_loaded"))
    assert result.status == service.NOT_FOUND, (result.status, result.body)
    assert result.log["outcome"] == "miss"
    assert "score" not in result.body and result.log["score"] is None


def check_a_stored_row_that_will_not_decode_is_a_503_naming_the_field():
    """Worse than a missing row, because something wrote it. The client should not get a
    traceback and the log should say which field."""
    store = DictStore()
    store.hset(FS.online_key("a"), mapping={"txn_count_1h": "3", "amount_sum_1h": "not a float", "as_of": str(T0)})
    result = build(store=store).handle(request())
    assert result.status == service.GONE, (result.status, result.body)
    assert result.log["outcome"] == "corrupt"
    assert "amount_sum_1h" in result.body["detail"], result.body


def check_a_stored_row_with_no_as_of_is_refused_rather_than_treated_as_fresh():
    store = DictStore()
    store.hset(FS.online_key("a"), mapping={"txn_count_1h": "3", "amount_sum_1h": "8.0"})
    result = build(store=store).handle(request())
    assert result.status == service.GONE, (result.status, result.body)
    assert "as_of" in result.body["detail"], result.body


def check_validation_collects_every_problem_rather_than_the_first():
    """One round trip per broken field is a bad interface. The fixture breaks two."""
    clean, problems = service.validate({"entity_id": "", "event_ts": 0})
    assert clean is None
    assert len(problems) == 2, problems


def check_a_missing_field_and_an_unknown_field_are_both_named():
    _, problems = service.validate({"event_ts": T0})
    assert any("entity_id" in p for p in problems), problems
    _, problems = service.validate({"entity_id": "a", "event_ts": T0, "amount": 5.0})
    assert any("amount" in p for p in problems), problems


def check_a_boolean_is_not_an_acceptable_event_ts():
    """bool is a subclass of int in Python, so True passes an isinstance int check and
    reads as epoch second 1. This is the case a reasonable implementation gets wrong."""
    _, problems = service.validate({"entity_id": "a", "event_ts": True})
    assert problems, "True was accepted as an epoch second"


def check_a_body_that_is_not_an_object_is_refused_without_raising():
    for bad in ([], "a string", 7, None):
        clean, problems = service.validate(bad)
        assert clean is None and problems, bad


def check_an_invalid_request_never_reaches_the_store():
    """A validation failure that still did a lookup would be paying for a round trip on
    every piece of junk that arrives, and the store is the expensive part."""

    class Exploding(DictStore):
        def hgetall(self, key):
            raise AssertionError("the store was read on an invalid request")

    result = build(store=Exploding()).handle({"entity_id": "a"})
    assert result.status == service.BAD_REQUEST
    assert result.log["outcome"] == "invalid"


def check_a_log_record_is_produced_on_every_outcome():
    """Including the ones that failed. A log that only records successes is the one that
    tells you nothing on the day something breaks."""
    cases = [
        (build().handle(request()), "scored"),
        (build().handle(request(entity="never_loaded")), "miss"),
        (build().handle({"entity_id": "a"}), "invalid"),
        (build(policy=service.Policy(60), now=T0 + 600).handle(request()), "refused"),
    ]
    seen = set()
    for result, expected in cases:
        assert result.log["outcome"] == expected, (result.log["outcome"], expected)
        assert "service_ms" in result.log, result.log
        assert result.log["fingerprint"] == FS.fingerprint()
        seen.add(expected)
    assert len(seen) == 4, seen


def check_the_log_carries_no_feature_values():
    """The record names the entity, the age and the score and not the vector.

    A feature value is derived from the account's own spending, so a log line holding
    seven of them is a partial transaction history sitting in whatever the log ships to.
    The store already holds them and it is the place with a key on it.
    """
    result = build().handle(request())
    flat = repr(sorted(result.log.items()))
    for name in NAMES:
        assert name not in flat, "the log record names {}".format(name)


def check_the_service_refuses_to_start_on_a_model_fitted_on_another_spec():
    """At construction and not on the first request. A process that comes up wrong should
    not come up."""
    wrong = model_mod.LinearModel(NAMES, COEFS, -0.75, "deadbeef0000", 0.4)
    try:
        service.Service(store_with(), FS, wrong)
    except model_mod.ModelError:
        return
    raise AssertionError("the service started with a model fitted on another spec")


def check_the_decision_flips_at_the_model_threshold_and_not_at_a_half():
    """The threshold is on the artefact and 0.5 is only its default. A service hard coding
    a half would serve every model the same way and the fixture sets 0.4 to catch it.
    """
    low = {"txn_count_1h": 0, "amount_sum_1h": 0.0}
    high = {"txn_count_1h": 40, "amount_sum_1h": 0.0}
    assert MODEL.score(low) < 0.4 <= MODEL.score(high), (MODEL.score(low), MODEL.score(high))
    assert build(store=store_with(low)).handle(request()).body["decision"] == "allow"
    assert build(store=store_with(high)).handle(request()).body["decision"] == "review"


def check_the_status_codes_are_the_http_numbers_and_not_whatever_the_module_says():
    """Every other check here compares a status against `service.OK` and friends, so both
    sides move together and a mutant renumbering them survives the whole suite.

    A status code is a fact about HTTP rather than a value this repo gets to choose, so
    the literals belong somewhere. Here, once, rather than scattered through the checks
    that read better with the names.
    """
    assert service.OK == 200
    assert service.BAD_REQUEST == 422
    assert service.NOT_FOUND == 404
    assert service.GONE == 503
    assert build().handle(request()).status == 200
    assert build().handle(request(entity="never_loaded")).status == 404
    assert build().handle({"entity_id": "a"}).status == 422


def check_the_smallest_legal_max_age_is_one_second():
    """`<= 0` and `<= 1` are different code and the fixture only had the refusing side.
    A one second threshold is a strange thing to configure and it is not an error."""
    assert service.Policy(max_age_s=1).judge(1) == service.FRESH
    assert service.Policy(max_age_s=1).judge(2) == service.REFUSED


def check_the_smallest_legal_event_ts_is_one():
    """Same shape one layer up. Nothing asserted that a positive epoch second is accepted,
    only that zero and negatives are refused, so the boundary was free to move."""
    clean, problems = service.validate({"entity_id": "a", "event_ts": 1})
    assert not problems, problems
    assert clean["event_ts"] == 1


def check_a_score_exactly_on_the_threshold_is_a_review():
    """The decision boundary, with a fixture sitting on it.

    The threshold is built from the score rather than chosen, because a logistic will not
    land on a round number and a fixture that never sits on the line cannot tell `>=` from
    `>`. Review at the threshold is the deliberate reading. A fraud score that has exactly
    reached the bar has reached it.
    """
    values = {"txn_count_1h": 2, "amount_sum_1h": 3.0}
    exact = model_mod.LinearModel(NAMES, COEFS, -0.75, FS.fingerprint(), 0.5)
    on_the_line = model_mod.LinearModel(
        NAMES, COEFS, -0.75, FS.fingerprint(), exact.score(values)
    )
    svc = service.Service(store_with(values), FS, on_the_line, clock=lambda: T0)
    body = svc.handle(request()).body
    assert body["score"] == on_the_line.threshold, (body["score"], on_the_line.threshold)
    assert body["decision"] == "review", body


def check_service_ms_is_measured_in_milliseconds():
    """A reporting field with no assertion on it is one whose arithmetic is free to move,
    and `scripts/serve_probe.py` publishes quantiles of this one.

    A store that sleeps is what gives the number a scale to be wrong against. Without it
    the value is a small positive float under a subtraction, an addition or a division by
    a thousand, and all three look equally plausible.
    """
    import time as time_mod

    class SlowStore(DictStore):
        def hgetall(self, key):
            time_mod.sleep(0.02)
            return super().hgetall(key)

        def hset(self, key, mapping=None):
            return super().hset(key, mapping=mapping)

    slow = SlowStore()
    slow.hset(FS.online_key("a"), mapping=online.encode_payload(VALUES, FS, T0))
    result = build(store=slow).handle(request())
    assert result.status == service.OK, result.body
    got = result.log["service_ms"]
    assert 15.0 < got < 500.0, "a 20 ms store gave service_ms of {}".format(got)


def check_the_response_names_the_fingerprint_it_was_served_under():
    """A skew investigation starts by asking which definition answered. Without it on the
    response that question needs the log, which is a different system."""
    body = build().handle(request()).body
    assert body["fingerprint"] == FS.fingerprint(), body
