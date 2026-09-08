"""Checks on the scorer and on the artefact it is loaded from.

The fixture coefficients are all different and none of them is 1.0 or 0.0. A model whose
weights are equal cannot catch a scorer that pairs the wrong coefficient with the wrong
feature, because the dot product comes out the same either way, and a weight of 1.0 hides
a missing multiplication.

The feature values are all different too, and none is 1.0, for the mirror reason.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import model as model_mod
from featurestore import spec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
NAMES = FS.feature_names()
COEFS = {"txn_count_1h": 0.25, "amount_sum_1h": -0.5, "amount_mean_24h": 2.0}
VALUES = {"txn_count_1h": 3, "amount_sum_1h": 8.0, "amount_mean_24h": 0.5}


def build(threshold=0.5, fingerprint=None):
    return model_mod.LinearModel(
        NAMES, COEFS, -0.75, fingerprint or FS.fingerprint(), threshold
    )


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as caught:
        return str(caught)
    raise AssertionError("expected {} and nothing was raised".format(exc.__name__))


def check_the_log_odds_are_the_dot_product_worked_out_by_hand():
    """Written out rather than looped, so a loop that skips a term is visible.

    0.25*3 + -0.5*8.0 + 2.0*0.5 + -0.75 comes to -3.0.
    """
    assert abs(build().log_odds(VALUES) - (-3.0)) < 1e-12, build().log_odds(VALUES)


def check_the_logistic_agrees_with_the_naive_form_on_a_large_negative():
    """The branch exists for overflow and the branch is what needs the check.

    math.exp(710) raises OverflowError, so the naive 1/(1+exp(-z)) form cannot be the
    oracle at that magnitude. It can at a moderate one, and the large one is asserted
    against the limit instead.
    """
    import math

    for z in (-30.0, -3.5, 0.0, 3.5, 30.0):
        naive = 1.0 / (1.0 + math.exp(-z))
        assert abs(model_mod.logistic(z) - naive) < 1e-15, z
    assert model_mod.logistic(-1000.0) == 0.0
    assert model_mod.logistic(1000.0) == 1.0
    assert model_mod.logistic(0.0) == 0.5


def check_the_branch_pivot_sits_at_zero_and_not_somewhere_else_harmless_looking():
    """Exact equality, because a tolerance is what hid this.

    The two forms are algebraically identical and they are not identical in a double. On
    a grid of a thousand points across [0, 1) they disagree in the last bit at 303 of
    them, and over the real corpus a pivot moved from 0 to 1 changes the served score on
    4,738 of 489,972 rows. The check above passes either way at 1e-15, which is the whole
    lesson. A tolerance on a comparison between two implementations of one function will
    accept a change in the one that ships.

    Nothing depends on the seventeenth digit of a fraud score. What depends on it is the
    claim this repo makes everywhere else, which is that two readers of one definition
    agree exactly rather than nearly.
    """
    import math

    for z in (0.007, 0.012, 0.013, 0.017, 0.5):
        assert model_mod.logistic(z) == 1.0 / (1.0 + math.exp(-z)), z
    for z in (-0.007, -0.5, -12.25):
        e = math.exp(z)
        assert model_mod.logistic(z) == e / (1.0 + e), z


def check_the_pivot_being_inclusive_or_strict_makes_no_difference():
    """The mirror of the check above, and it is a measurement rather than an argument.

    At exactly zero both forms give 0.5 in a double, so `>= 0` and `> 0` cannot be told
    apart by any input. Over the same 489,972 corpus rows that mutant moved zero bit
    patterns. It is equivalent and this is the line that says so, rather than leaving a
    survivor in a table with no explanation next to it.
    """
    import math

    e = math.exp(0.0)
    assert 1.0 / (1.0 + math.exp(-0.0)) == e / (1.0 + e) == 0.5


def check_a_vector_missing_a_feature_is_refused():
    short = {k: v for k, v in VALUES.items() if k != "amount_sum_1h"}
    msg = raises(model_mod.ModelError, build().score, short)
    assert "amount_sum_1h" in msg, msg


def check_a_vector_carrying_an_extra_feature_is_refused():
    """Silently ignoring it is the dangerous behaviour, because a renamed feature then
    scores against whatever the old name held rather than failing."""
    wide = dict(VALUES)
    wide["something_else"] = 1.0
    msg = raises(model_mod.ModelError, build().score, wide)
    assert "something_else" in msg, msg


def check_a_model_missing_a_coefficient_cannot_be_constructed():
    short = {k: v for k, v in COEFS.items() if k != "amount_mean_24h"}
    msg = raises(model_mod.ModelError, model_mod.LinearModel, NAMES, short, 0.0, "abc")
    assert "amount_mean_24h" in msg, msg


def check_a_threshold_outside_zero_and_one_is_refused():
    for bad in (0.0, 1.0, -0.1, 1.5):
        raises(model_mod.ModelError, model_mod.LinearModel, NAMES, COEFS, 0.0, "abc", bad)
    model_mod.LinearModel(NAMES, COEFS, 0.0, "abc", 0.999)


def check_binding_refuses_a_model_fitted_on_another_spec():
    """The failure this is for is a rebuilt store beside a stale model artefact. The key
    then agrees with itself and the coefficients belong to a definition nobody serves."""
    msg = raises(model_mod.ModelError, build(fingerprint="deadbeef0000").bind, FS)
    assert "deadbeef0000" in msg and FS.fingerprint() in msg, msg


def check_binding_refuses_a_model_whose_feature_names_do_not_match_the_spec():
    """A fingerprint collision is not the only way these two come apart. A model built by
    hand with the right fingerprint string and the wrong names would pass the first half."""
    wrong = model_mod.LinearModel(
        ["txn_count_1h", "amount_sum_1h"],
        {"txn_count_1h": 1.5, "amount_sum_1h": -2.5},
        0.0,
        FS.fingerprint(),
    )
    msg = raises(model_mod.ModelError, wrong.bind, FS)
    assert "amount_mean_24h" in msg, msg


def check_binding_returns_the_model_so_it_can_be_used_in_an_expression():
    assert build().bind(FS).fingerprint == FS.fingerprint()


def check_the_artefact_round_trips_through_json():
    """Values chosen so a float that goes through a formatter rather than repr comes back
    different. 0.1 + 0.2 is the standard case and the intercept here is that sum."""
    model = model_mod.LinearModel(NAMES, COEFS, 0.1 + 0.2, FS.fingerprint(), 0.37)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.json")
        model_mod.dump(model, path, trained={"rows": 5})
        back = model_mod.load(path)
    assert back.intercept == model.intercept, (back.intercept, model.intercept)
    assert back.threshold == model.threshold
    assert back.coefficients == model.coefficients
    assert back.score(VALUES) == model.score(VALUES)


def check_provenance_is_outside_everything_the_scorer_reads():
    """A rerun that moves an AUC must not be able to move a served score."""
    model = build()
    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "a.json")
        b = os.path.join(tmp, "b.json")
        model_mod.dump(model, a, trained={"test_auc": 0.8814})
        model_mod.dump(model, b, trained={"test_auc": 0.1})
        with open(a, encoding="utf-8") as fh:
            doc_a = json.load(fh)
        with open(b, encoding="utf-8") as fh:
            doc_b = json.load(fh)
        assert doc_a["trained"] != doc_b["trained"], "the fixture did not vary provenance"
        assert model_mod.load(a).score(VALUES) == model_mod.load(b).score(VALUES)


def check_the_artefact_is_written_in_a_stable_order():
    """The artefact is committed, so a retrain that reorders it is a diff nobody can read.

    The fixture builds the same model twice with the coefficient keys inserted in opposite
    orders. Without sorted keys the two files differ and every retrain shows as a change
    to a file whose numbers did not move.
    """
    forward = model_mod.LinearModel(NAMES, {k: COEFS[k] for k in NAMES}, 0.5, "abc")
    backward = model_mod.LinearModel(NAMES, {k: COEFS[k] for k in reversed(NAMES)}, 0.5, "abc")
    assert list(forward.coefficients) != list(backward.coefficients), "the fixture did not reorder"
    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "a.json")
        b = os.path.join(tmp, "b.json")
        model_mod.dump(forward, a)
        model_mod.dump(backward, b)
        with open(a, "rb") as fh:
            first = fh.read()
        with open(b, "rb") as fh:
            second = fh.read()
    assert first == second, "two orderings of the same model wrote different bytes"


def check_an_artefact_with_an_unknown_key_is_refused():
    """A key nobody reads is how a field gets renamed and quietly stops being applied.
    `threshold` becoming `thresholds` would otherwise fall back to 0.5 in silence."""
    doc = build().as_dict()
    doc["thresholds"] = 0.9
    msg = raises(model_mod.ModelError, model_mod.from_dict, doc)
    assert "thresholds" in msg, msg


def check_an_artefact_missing_a_required_key_is_refused():
    for key in ("feature_names", "coefficients", "intercept", "fingerprint"):
        doc = build().as_dict()
        del doc[key]
        msg = raises(model_mod.ModelError, model_mod.from_dict, doc)
        assert key in msg, (key, msg)


def check_the_shipped_artefact_matches_the_shipped_spec():
    """The one check here that reads the repo rather than a fixture.

    The corpus is gitignored and the model is fitted on it, so the artefact in the tree is
    a published figure whose input is not in the history. What makes it verifiable is that
    the corpus regenerates byte identically and `scripts/train_model.py` is named beside
    it. What this asserts is cheaper and catches the likelier mistake, which is editing
    the spec and not refitting. A changed window length moves the fingerprint and the
    committed model then belongs to a definition nothing serves.
    """
    path = os.path.join(REPO, "models", "txn_fraud.json")
    assert os.path.exists(path), "no model artefact at models/txn_fraud.json"
    feature_set = spec.load(os.path.join(REPO, "specs", "txn_fraud.yml"))
    model_mod.load(path).bind(feature_set)
