"""What happens to one scoring request, with no web framework in it.

Validate. Fetch. Decide what the vector's age means. Score. Log. All five live here as
plain functions over plain dicts. `featurestore/api.py` is the FastAPI adapter and it is
about sixty lines, none of which make a decision.

The split is the same one `featurestore/online.py` makes about the Redis client. Every
rule worth arguing over is reachable from `tests/run_all.py`, which a clone runs with only
`requirements.txt` installed. What the fourth runner adds is whether the framework really
returns the status code this module asked for, and that is a different question from
whether the status code is right.

Three decisions are made here and none of them was obvious.

A miss is a 404 and not a zero vector. The store cannot write an entity it has not seen,
so a miss means a cold start, and `compute.aggregate` returns 0 for an empty window.
Handing the model zeros would score a brand new card as a quiet one.

A vector's age is attached to every answer rather than judged by default. The store is
loaded once and never refreshed, so any threshold short enough to protect the seven day
mean refuses every request this service will ever get. `scripts/serve_probe.py` measures
that, and it is why `max_age_s` defaults to None.

The log record is built whatever happens, including on a refusal. A request that failed is
the one you most want a line for.
"""

import time

from featurestore import online

# The verdicts an age can earn. `unbounded` is a distinct answer from `fresh` and not a
# synonym for it. It says no threshold was configured, so nothing judged the age. That is
# a fact about how the service was started and not a property of the row, and a dashboard
# counting `fresh` should not be able to count these among them.
UNBOUNDED = "unbounded"
FRESH = "fresh"
REFUSED = "refused"

OK = 200
BAD_REQUEST = 422
NOT_FOUND = 404
GONE = 503

REQUIRED = ("entity_id", "event_ts")


class Result:
    """A status, a body and the log line, together.

    One object because the three come out of the same pass, and a caller that can take the
    body without the log record will eventually do exactly that.
    """

    __slots__ = ("status", "body", "log")

    def __init__(self, status, body, log):
        self.status = status
        self.body = body
        self.log = log

    def __repr__(self):
        return "Result({}, {})".format(self.status, self.log.get("outcome"))


class Policy:
    """What the service does about a vector's age.

    Two states rather than three. Either a threshold was set and the age is inside it or
    outside it, or no threshold was set and nothing judged anything. A middle band would
    need an edge, and the only numbers available to put one on are the ones in the
    staleness table, which is the set this would then be judged against.

    `max_age_s=None` is the shipped default and it is a decision rather than a refusal to
    make one. `scripts/serve_probe.py` is the argument.
    """

    __slots__ = ("max_age_s",)

    def __init__(self, max_age_s=None):
        if max_age_s is not None and max_age_s <= 0:
            raise ValueError("max_age_s must be positive or None, got {!r}".format(max_age_s))
        self.max_age_s = max_age_s

    def judge(self, age_s):
        if self.max_age_s is None:
            return UNBOUNDED
        return REFUSED if age_s > self.max_age_s else FRESH


def validate(payload):
    """The request body, or a list of what is wrong with it.

    Returns (clean, problems). Both always, so a caller cannot read the first without the
    second being present and empty. Every problem is collected rather than raising on the
    first, because a client fixing one field at a time per round trip is a bad interface.
    """
    problems = []
    if not isinstance(payload, dict):
        return None, ["body must be an object"]

    for field in REQUIRED:
        if field not in payload:
            problems.append("missing {!r}".format(field))
    extra = sorted(set(payload) - set(REQUIRED))
    if extra:
        problems.append("unknown fields {}".format(extra))
    if problems:
        return None, problems

    entity_id = payload["entity_id"]
    if not isinstance(entity_id, str) or not entity_id.strip():
        problems.append("entity_id must be a non empty string")

    event_ts = payload["event_ts"]
    # bool is an int in Python, so True would otherwise read as a valid epoch second.
    if isinstance(event_ts, bool) or not isinstance(event_ts, int):
        problems.append("event_ts must be an integer epoch second")
    elif event_ts <= 0:
        problems.append("event_ts must be positive")

    if problems:
        return None, problems
    return {"entity_id": entity_id, "event_ts": event_ts}, []


class Service:
    """Everything a request touches, assembled once.

    The model is bound to the feature set at construction and not per request. A mismatch
    between the two is a deployment error, so it should stop the process coming up rather
    than produce a 500 on the first request that arrives.
    """

    def __init__(self, client, feature_set, model, policy=None, clock=time.time):
        self.client = client
        self.feature_set = feature_set
        self.model = model.bind(feature_set)
        self.policy = policy or Policy()
        self.clock = clock

    def handle(self, payload, request_id=None):
        started = time.perf_counter()
        log = {
            "request_id": request_id,
            "feature_set": self.feature_set.name,
            "fingerprint": self.feature_set.fingerprint(),
            "entity_id": None,
            "event_ts": None,
            "as_of": None,
            "age_s": None,
            "freshness": None,
            "score": None,
            "outcome": None,
        }

        clean, problems = validate(payload)
        if problems:
            log["outcome"] = "invalid"
            log["problems"] = problems
            body = {"error": "invalid request", "problems": problems}
            return self._done(Result(BAD_REQUEST, body, log), started)

        log["entity_id"] = clean["entity_id"]
        log["event_ts"] = clean["event_ts"]

        try:
            values, as_of = online.lookup(self.client, self.feature_set, clean["entity_id"])
        except online.Miss:
            # Not an error and not a zero vector. The store holds no row for this entity
            # under this definition, so the honest answer is that there is nothing to
            # score here. What to do about a new card belongs to the caller.
            log["outcome"] = "miss"
            body = {"error": "no features for this entity", "entity_id": clean["entity_id"]}
            return self._done(Result(NOT_FOUND, body, log), started)
        except online.OnlineError as exc:
            # A row that will not decode is worse than a missing one, because something
            # wrote it. Name the field rather than letting this reach the client as a 500.
            log["outcome"] = "corrupt"
            log["problems"] = [str(exc)]
            body = {"error": "stored features are unreadable", "detail": str(exc)}
            return self._done(Result(GONE, body, log), started)

        age_s = int(self.clock()) - as_of
        verdict = self.policy.judge(age_s)
        log["as_of"] = as_of
        log["age_s"] = age_s
        log["freshness"] = verdict

        if verdict == REFUSED:
            log["outcome"] = "refused"
            body = {
                "error": "stored features are older than this service will serve",
                "age_s": age_s,
                "max_age_s": self.policy.max_age_s,
            }
            return self._done(Result(GONE, body, log), started)

        score = self.model.score(values)
        log["score"] = score
        log["outcome"] = "scored"
        body = {
            "entity_id": clean["entity_id"],
            "score": score,
            "decision": "review" if score >= self.model.threshold else "allow",
            "features_as_of": as_of,
            "features_age_s": age_s,
            "freshness": verdict,
            "fingerprint": log["fingerprint"],
        }
        return self._done(Result(OK, body, log), started)

    def _done(self, result, started):
        # Measured around this module's own work and nothing else. It is called
        # service_ms rather than latency_ms so it cannot be read as the number a client
        # sees, which carries the framework and the socket on top of it.
        result.log["service_ms"] = round((time.perf_counter() - started) * 1000.0, 4)
        return result


def response_carries_its_age(body):
    """Every 200 says what moment it was built from and how long ago that was.

    A function rather than three literals inside one check, because the probe asserts it
    over forty thousand real responses and the suite asserts it over one. A response that
    dropped the age while the log kept it would pass a check that only read the log.
    """
    return all(key in body for key in ("features_as_of", "features_age_s", "freshness"))
