"""Feature set specs.

A spec is the one place a feature is defined. The batch job that builds the training
table reads it, the loader that fills the online store reads it, and the service that
answers a request reads it. Nothing else is allowed to know what `txn_count_1h` means.

The part that does the real work is `fingerprint`. It is a hash over the semantic
content of the spec and it goes into the online store key and onto the training table.
A spec change therefore writes to a different keyspace, so a model trained under one
definition cannot be served features built under another. It fails loudly instead of
quietly scoring against the wrong numbers.

Descriptions and ordering are deliberately outside the hash. Rewording a comment should
not invalidate a warehouse.
"""

import hashlib
import json
import os
import re

import yaml

DAY_S = 86400
UNITS = {"s": 1, "m": 60, "h": 3600, "d": DAY_S}
LENGTH_RE = re.compile(r"^([1-9][0-9]*)([smhd])$")

# closed=left excludes the scoring event itself. Getting this wrong is the whole reason
# the field is required rather than defaulted. See docs/adr-0002.
CLOSED = ("left", "right")

# event_time anchors the window on the scoring timestamp. calendar_day anchors it on
# midnight UTC, which is what a nightly batch job produces without anyone deciding to.
ALIGN = ("event_time", "calendar_day")

AGGREGATES = ("count", "sum", "mean", "max", "distinct_count")
DTYPES = ("int", "float")


class SpecError(ValueError):
    """Raised for anything a spec file gets wrong. Never returns a default."""


class Window:
    __slots__ = ("length_s", "closed", "align")

    def __init__(self, length_s, closed, align):
        self.length_s = length_s
        self.closed = closed
        self.align = align

    def as_dict(self):
        return {"length_s": self.length_s, "closed": self.closed, "align": self.align}

    def __repr__(self):
        return "Window({}s, {}, {})".format(self.length_s, self.closed, self.align)


class Feature:
    __slots__ = ("name", "dtype", "aggregate", "column", "window")

    def __init__(self, name, dtype, aggregate, column, window):
        self.name = name
        self.dtype = dtype
        self.aggregate = aggregate
        self.column = column
        self.window = window

    def as_dict(self):
        return {
            "name": self.name,
            "dtype": self.dtype,
            "aggregate": self.aggregate,
            "column": self.column,
            "window": self.window.as_dict(),
        }

    def __repr__(self):
        return "Feature({})".format(self.name)


class FeatureSet:
    __slots__ = ("name", "entity", "event_time", "features", "label", "source_path")

    def __init__(self, name, entity, event_time, features, label, source_path=None):
        self.name = name
        self.entity = entity
        self.event_time = event_time
        self.features = features
        self.label = label
        self.source_path = source_path

    def feature_names(self):
        return [f.name for f in self.features]

    def canonical(self):
        """The semantic content, as a stable JSON string.

        Feature order is sorted rather than preserved, so moving a block in the YAML
        does not invalidate a warehouse. Descriptions are absent because they carry no
        meaning for either code path.
        """
        body = {
            "name": self.name,
            "entity": self.entity,
            "event_time": self.event_time,
            "label": self.label,
            "features": sorted((f.as_dict() for f in self.features), key=lambda d: d["name"]),
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":"))

    def fingerprint(self):
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()[:12]

    def online_key(self, entity_id):
        """Redis key for one entity under this exact definition.

        The fingerprint sits in the key on purpose. A loader running an older spec
        cannot overwrite the rows a newer one wrote, and a service asking for the new
        fingerprint gets a miss rather than a stale answer.
        """
        return "fs:{}:{}:{}".format(self.name, self.fingerprint(), entity_id)

    def payload_schema(self):
        """What the service must have in hand before it is allowed to call the model."""
        return {f.name: f.dtype for f in self.features}

    def __repr__(self):
        return "FeatureSet({}, {} features, {})".format(
            self.name, len(self.features), self.fingerprint()
        )


def parse_length(text):
    if not isinstance(text, str):
        raise SpecError("window length must be a string like 7d, got {!r}".format(text))
    m = LENGTH_RE.match(text)
    if not m:
        raise SpecError(
            "window length {!r} is not <positive int><s|m|h|d>".format(text)
        )
    return int(m.group(1)) * UNITS[m.group(2)]


def _require(mapping, key, where):
    if key not in mapping:
        raise SpecError("{} is missing required key {!r}".format(where, key))
    return mapping[key]


def _one_of(value, allowed, key, where):
    if value not in allowed:
        raise SpecError(
            "{} has {}={!r}, expected one of {}".format(where, key, value, list(allowed))
        )
    return value


def parse_window(raw, where):
    if not isinstance(raw, dict):
        raise SpecError("{} window must be a mapping, got {!r}".format(where, type(raw).__name__))
    length = parse_length(_require(raw, "length", where))
    closed = _one_of(_require(raw, "closed", where), CLOSED, "closed", where)
    align = _one_of(_require(raw, "align", where), ALIGN, "align", where)
    extra = set(raw) - {"length", "closed", "align"}
    if extra:
        raise SpecError("{} window has unknown keys {}".format(where, sorted(extra)))

    # A calendar aligned window has to be a whole number of days or it means nothing.
    # `length: 1h` with `align: calendar_day` resolves to the hour before midnight on the
    # previous day, for every event in the day being scored. That is not a reading anybody
    # intended and the first version of this file accepted it. scripts/window_probe.py is
    # what found it, by producing a disagreement share for a window nobody would write.
    if align == "calendar_day" and length % DAY_S:
        raise SpecError(
            "{} aligns to calendar_day with length {}s, which is not a whole number of "
            "days. Use align=event_time or round the length.".format(where, length)
        )
    return Window(length, closed, align)


def parse_feature(raw, index):
    where = "feature[{}]".format(index)
    if not isinstance(raw, dict):
        raise SpecError("{} must be a mapping".format(where))
    name = _require(raw, "name", where)
    where = "feature {!r}".format(name)
    dtype = _one_of(_require(raw, "dtype", where), DTYPES, "dtype", where)
    aggregate = _one_of(_require(raw, "aggregate", where), AGGREGATES, "aggregate", where)
    window = parse_window(_require(raw, "window", where), where)

    # count needs no column and every other aggregate does. Letting count carry one
    # would make two specs that behave identically hash differently.
    column = raw.get("column")
    if aggregate == "count":
        if column is not None:
            raise SpecError("{} is a count and must not name a column".format(where))
    elif not column:
        raise SpecError("{} is a {} and needs a column".format(where, aggregate))

    known = {"name", "dtype", "aggregate", "column", "window", "description"}
    extra = set(raw) - known
    if extra:
        raise SpecError("{} has unknown keys {}".format(where, sorted(extra)))
    return Feature(name, dtype, aggregate, column, window)


def parse(doc, source_path=None):
    if not isinstance(doc, dict):
        raise SpecError("a spec must be a mapping at the top level")
    name = _require(doc, "feature_set", "spec")
    entity = _require(doc, "entity", "spec")
    event_time = _require(doc, "event_time", "spec")
    label = _require(doc, "label", "spec")

    raw_features = _require(doc, "features", "spec")
    if not isinstance(raw_features, list) or not raw_features:
        raise SpecError("spec needs a non empty features list")

    features = [parse_feature(raw, i) for i, raw in enumerate(raw_features)]

    seen = set()
    for f in features:
        if f.name in seen:
            raise SpecError("feature {!r} is declared twice".format(f.name))
        seen.add(f.name)

    if label in seen:
        raise SpecError("label {!r} is also declared as a feature".format(label))

    known = {"feature_set", "entity", "event_time", "label", "features", "description", "source"}
    extra = set(doc) - known
    if extra:
        raise SpecError("spec has unknown keys {}".format(sorted(extra)))

    return FeatureSet(name, entity, event_time, features, label, source_path)


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return parse(doc, source_path=os.path.abspath(path))
