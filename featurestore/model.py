"""The scorer, as a handful of floats.

A linear model over the feature set, held as a coefficient per feature name and an
intercept. `scripts/train_model.py` fits it with scikit-learn and writes it out. Nothing
on the serving path imports scikit-learn, because the arithmetic is a dot product and a
logistic and both are three lines of standard library.

That is a deliberate choice and `docs/adr-0001` argues it. Two things it buys here.

The serving path has no model library in it at all, so a latency number is a statement
about the feature fetch rather than about somebody's inference runtime. And a scale error
stays attributable, because a feature arriving wrong by d moves the log odds by coef * d
and that is a number you can work out by hand.

The artefact carries the spec fingerprint it was trained under. A model fitted on one
definition and handed a vector built from another is the exact failure the key in the
online store exists to stop, and it can also arrive from this direction. `bind` is what
refuses it.
"""

import json
import math

# Two implementations of one function have to agree on more than the happy path, and the
# one that bites is a large negative log odds. math.exp overflows at about 710 and numpy
# returns 0.0 with a warning, so the branch keeps them equal rather than nearly equal.
# scripts/train_model.py asserts that against scikit-learn on every training row.
def logistic(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class ModelError(ValueError):
    """A model artefact that does not match the spec, or a vector that does not match it."""


class LinearModel:
    __slots__ = ("feature_names", "coefficients", "intercept", "fingerprint", "threshold")

    def __init__(self, feature_names, coefficients, intercept, fingerprint, threshold=0.5):
        missing = sorted(set(feature_names) - set(coefficients))
        if missing:
            raise ModelError("no coefficient for {}".format(missing))
        extra = sorted(set(coefficients) - set(feature_names))
        if extra:
            raise ModelError("coefficients for features the model does not name: {}".format(extra))
        if not 0.0 < threshold < 1.0:
            raise ModelError("threshold {!r} is not a probability strictly inside 0 and 1".format(threshold))
        self.feature_names = list(feature_names)
        self.coefficients = dict(coefficients)
        self.intercept = float(intercept)
        self.fingerprint = fingerprint
        self.threshold = float(threshold)

    def bind(self, feature_set):
        """Refuse to serve a feature set this model was not fitted on.

        The online key already stops a service reading rows written under an older spec.
        This is the same guard one layer up, because a rebuilt store and a stale model
        artefact leave the key agreeing with itself while the coefficients belong to a
        definition nobody is serving.
        """
        if feature_set.fingerprint() != self.fingerprint:
            raise ModelError(
                "model was fitted on spec {} and the service is serving {}".format(
                    self.fingerprint, feature_set.fingerprint()
                )
            )
        declared = set(feature_set.feature_names())
        if declared != set(self.feature_names):
            raise ModelError(
                "model names {} and the spec declares {}".format(
                    sorted(self.feature_names), sorted(declared)
                )
            )
        return self

    def log_odds(self, values):
        missing = sorted(set(self.feature_names) - set(values))
        if missing:
            raise ModelError("cannot score a vector missing {}".format(missing))
        extra = sorted(set(values) - set(self.feature_names))
        if extra:
            raise ModelError("vector carries features the model does not name: {}".format(extra))
        total = self.intercept
        for name in self.feature_names:
            total += self.coefficients[name] * float(values[name])
        return total

    def score(self, values):
        return logistic(self.log_odds(values))

    def as_dict(self):
        return {
            "feature_names": list(self.feature_names),
            "coefficients": dict(self.coefficients),
            "intercept": self.intercept,
            "fingerprint": self.fingerprint,
            "threshold": self.threshold,
        }

    def __repr__(self):
        return "LinearModel({} features, spec {})".format(len(self.feature_names), self.fingerprint)


def from_dict(doc):
    if not isinstance(doc, dict):
        raise ModelError("a model artefact must be a mapping at the top level")
    known = {"feature_names", "coefficients", "intercept", "fingerprint", "threshold", "trained"}
    extra = sorted(set(doc) - known)
    if extra:
        raise ModelError("model artefact has unknown keys {}".format(extra))
    for key in ("feature_names", "coefficients", "intercept", "fingerprint"):
        if key not in doc:
            raise ModelError("model artefact is missing required key {!r}".format(key))
    return LinearModel(
        doc["feature_names"],
        doc["coefficients"],
        doc["intercept"],
        doc["fingerprint"],
        doc.get("threshold", 0.5),
    )


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return from_dict(json.load(fh))


def dump(model, path, trained=None):
    """`trained` is provenance and is outside everything the scorer reads.

    It holds the row counts and the AUC the fit reported. Keeping it out of the scoring
    path means a rerun that moves an AUC by a decimal cannot change a served score.
    """
    doc = model.as_dict()
    if trained is not None:
        doc["trained"] = trained
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
