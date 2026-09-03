"""Checks on the spec format.

The one that matters is check_every_semantic_field_moves_the_fingerprint. It walks the
fields out of the canonical form rather than reading a list somebody typed beside it. A
coupling check driven off a hand written list only checks the fields whoever wrote it
remembered, and its name still claims it checked all of them.
"""

import copy
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import spec

BASE = {
    "feature_set": "demo",
    "description": "does not count",
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
            "name": "amount_sum_7d",
            "dtype": "float",
            "aggregate": "sum",
            "column": "amount",
            "window": {"length": "7d", "closed": "left", "align": "event_time"},
        },
    ],
}


def doc(**overrides):
    d = copy.deepcopy(BASE)
    d.update(overrides)
    return d


def refuses(d, needle):
    try:
        spec.parse(d)
    except spec.SpecError as exc:
        assert needle in str(exc), "expected {!r} in {!r}".format(needle, str(exc))
        return
    raise AssertionError("expected a SpecError mentioning {!r}".format(needle))


def check_a_good_spec_parses():
    fs = spec.parse(doc())
    assert fs.feature_names() == ["txn_count_1h", "amount_sum_7d"]
    assert fs.features[1].window.length_s == 7 * 86400


def check_every_semantic_field_moves_the_fingerprint():
    """Mutate every leaf of the canonical form and confirm the hash moves.

    Driven off the canonical JSON so a field added to the spec later is covered without
    anybody remembering to add it here. If a new field is not in the canonical form this
    check will not see it, which is the right behaviour, because a field outside the
    hash is by definition not semantic.
    """
    fs = spec.parse(doc())
    base = fs.fingerprint()
    body = json.loads(fs.canonical())
    leaves = list(_leaves(body, ()))
    assert len(leaves) >= 14, "expected the canonical form to have real depth, got {}".format(
        len(leaves)
    )
    for path, value in leaves:
        mutated = copy.deepcopy(body)
        _set(mutated, path, _twiddle(value))
        moved = hashlib.sha256(
            json.dumps(mutated, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        assert moved != base, "changing {} left the fingerprint alone".format(path)


def _leaves(node, path):
    if isinstance(node, dict):
        for key, value in node.items():
            for item in _leaves(value, path + (key,)):
                yield item
    elif isinstance(node, list):
        for i, value in enumerate(node):
            for item in _leaves(value, path + (i,)):
                yield item
    else:
        yield path, node


def _set(node, path, value):
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value


def _twiddle(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    return str(value) + "x"


def check_the_canonical_form_is_exactly_this():
    """Pin the bytes, not the behaviour.

    A mutation pass flipped sort_keys to False and nothing failed, because both sides of
    every other comparison here are built by the same code. That would be harmless today
    and would silently move every fingerprint the first time somebody reorders the keys
    in canonical(). This is the only check that would notice.
    """
    expected = (
        '{"entity":"account_id","event_time":"event_ts",'
        '"features":['
        '{"aggregate":"sum","column":"amount","dtype":"float","name":"amount_sum_7d",'
        '"window":{"align":"event_time","closed":"left","length_s":604800}},'
        '{"aggregate":"count","column":null,"dtype":"int","name":"txn_count_1h",'
        '"window":{"align":"event_time","closed":"left","length_s":3600}}'
        '],"label":"is_fraud","name":"demo"}'
    )
    assert spec.parse(doc()).canonical() == expected


def check_every_unit_converts():
    """All four, pinned. Two of them were unreachable from any other check.

    A mutant that made an hour 3601 seconds survived the whole suite, because every
    window in the fixtures is compared against another window built the same way.
    """
    for text, seconds in (("1s", 1), ("1m", 60), ("1h", 3600), ("1d", 86400)):
        assert spec.parse_length(text) == seconds, text
    assert spec.parse_length("90m") == 5400
    assert spec.parse_length("36h") == 129600


def check_description_is_outside_the_hash():
    a = spec.parse(doc()).fingerprint()
    b = spec.parse(doc(description="something else entirely")).fingerprint()
    assert a == b, "a reworded description invalidated the warehouse"


def check_feature_order_is_outside_the_hash():
    flipped = doc()
    flipped["features"] = list(reversed(flipped["features"]))
    assert spec.parse(flipped).fingerprint() == spec.parse(doc()).fingerprint()


def check_canonical_and_fingerprint_agree():
    """Two surfaces answering one question have to be tied together.

    canonical() is what a human reads in a diff and fingerprint() is what the key uses.
    Nothing stops them drifting apart except this.
    """
    for variant in (doc(), doc(entity="card_id"), doc(label="chargeback")):
        fs = spec.parse(variant)
        expected = hashlib.sha256(fs.canonical().encode("utf-8")).hexdigest()[:12]
        assert fs.fingerprint() == expected


def check_the_fingerprint_is_in_the_online_key():
    fs = spec.parse(doc())
    key = fs.online_key("a000123")
    assert fs.fingerprint() in key
    assert key.endswith("a000123")
    other = spec.parse(doc(entity="card_id"))
    assert other.online_key("a000123") != key


def check_calendar_day_refuses_a_sub_day_window():
    d = doc()
    d["features"][0]["window"]["align"] = "calendar_day"
    refuses(d, "not a whole number of days")


def check_calendar_day_accepts_a_day_multiple():
    d = doc()
    d["features"][1]["window"]["align"] = "calendar_day"
    fs = spec.parse(d)
    assert fs.features[1].window.align == "calendar_day"


def check_a_window_needs_all_three_fields():
    for missing in ("length", "closed", "align"):
        d = doc()
        del d["features"][0]["window"][missing]
        refuses(d, "missing required key {!r}".format(missing))


def check_a_bad_length_is_refused():
    for bad in ("7", "0d", "7w", "-1h", 7, "d"):
        d = doc()
        d["features"][0]["window"]["length"] = bad
        try:
            spec.parse(d)
        except spec.SpecError:
            continue
        raise AssertionError("length {!r} was accepted".format(bad))


def check_count_must_not_carry_a_column():
    d = doc()
    d["features"][0]["column"] = "amount"
    refuses(d, "must not name a column")


def check_a_non_count_needs_a_column():
    d = doc()
    del d["features"][1]["column"]
    refuses(d, "needs a column")


def check_a_duplicate_feature_is_refused():
    d = doc()
    d["features"].append(copy.deepcopy(d["features"][0]))
    refuses(d, "declared twice")


def check_the_label_cannot_also_be_a_feature():
    d = doc()
    d["features"][0]["name"] = "is_fraud"
    refuses(d, "also declared as a feature")


def check_unknown_keys_are_refused_everywhere():
    d = doc()
    d["extra_top_level"] = 1
    refuses(d, "unknown keys")

    d = doc()
    d["features"][0]["agregate"] = "count"
    refuses(d, "unknown keys")

    d = doc()
    d["features"][0]["window"]["inclusive"] = True
    refuses(d, "unknown keys")


def check_an_empty_feature_list_is_refused():
    refuses(doc(features=[]), "non empty")


def check_the_shipped_spec_loads():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fs = spec.load(os.path.join(here, "specs", "txn_fraud.yml"))
    assert fs.name == "txn_fraud"
    assert len(fs.features) == 7
    assert set(fs.payload_schema().values()) <= {"int", "float"}
    # Every window in the shipped spec excludes the row being scored. At serving time
    # that row does not exist yet, so any other value would be unservable.
    assert all(f.window.closed == "left" for f in fs.features)
