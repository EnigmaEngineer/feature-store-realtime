"""Every check in tests/, then the real FastAPI app over a real Redis.

    python tests/run_with_api.py

The fourth runner, and it is here for the same reason as the second and the third.
`tests/run_all.py` runs on a clone that installed only `requirements.txt`, so nothing
under `tests/` imports a framework or opens a socket. Everything that decides anything
lives in `featurestore/service.py` and is graded there.

What this adds is the half that file cannot answer. Whether FastAPI really returns the
status `service.Service` asked for. Whether a body that is not JSON reaches the handler at
all. Whether a response survives being serialised. Those are properties of the edge and
the edge is sixty lines of adapter that nothing else executes.

A missing fastapi, httpx, redis or redislite is a failure here and not a skip. A runner
whose second half quietly disappears reports the suite's number and reads exactly like a
full pass.

The corpus slice is small on purpose. This runs the whole loader per invocation and a
mutation harness pointed here pays that once per mutant:

    cp -r <repo> /tmp/mut/fsr
    cp /tmp/mut/fsr/tests/run_with_api.py /tmp/mut/fsr/tests/run_all.py
    python mutate.py --repo /tmp/mut/fsr --module featurestore/service.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from runner import run_checks  # noqa: E402  (tests/ is on the path by the line above)

ACCOUNTS = 40
PORT = "6403"


def build_store(tmpdir, feature_set):
    """A real Redis in this process, loaded from a slice of the real corpus.

    redislite starts a server on a TCP port and plain redis-py talks to it. The rdb
    directory has to exist first or the server refuses to start and the reason is one
    line above the traceback.
    """
    import redis
    import redislite

    from featurestore import compute, online
    from featurestore import corpus as corpus_io

    rdb = os.path.join(tmpdir, "api.rdb")
    os.makedirs(tmpdir, exist_ok=True)
    server = redislite.Redis(rdb, serverconfig={"port": PORT})

    rows = list(corpus_io.read_csv(os.path.join(ROOT, "data", "transactions.csv")))
    keep = set(sorted({r["account_id"] for r in rows})[:ACCOUNTS])
    rows = [r for r in rows if r["account_id"] in keep]
    log = compute.EventLog(rows, feature_set.entity, feature_set.event_time)

    times = sorted(r[feature_set.event_time] for r in rows)
    as_of = times[int(len(times) * 0.9)]

    client = redis.Redis(host="127.0.0.1", port=int(PORT))
    written = online.load(client, log, feature_set, as_of)
    entities = sorted({eid for eid, _ in online.online_rows(log, feature_set, as_of)})
    return server, client, as_of, written, entities


def api_arm():
    import tempfile

    from fastapi.testclient import TestClient

    from featurestore import api, model as model_mod, service, spec

    feature_set = spec.load(os.path.join(ROOT, "specs", "txn_fraud.yml"))
    model = model_mod.load(os.path.join(ROOT, "models", "txn_fraud.json"))

    tmpdir = os.path.join(tempfile.gettempdir(), "fsr_api_arm")
    server, client, as_of, written, entities = build_store(tmpdir, feature_set)
    try:
        emitted = []
        svc = service.Service(client, feature_set, model, service.Policy())
        app = api.build_app(svc, sink=emitted.append)
        http = TestClient(app)

        print("loaded {} entities at {} on port {}".format(written, as_of, PORT))

        health = http.get("/health")
        assert health.status_code == 200, health.status_code
        assert health.json()["fingerprint"] == feature_set.fingerprint(), health.json()
        assert health.json()["model_fingerprint"] == model.fingerprint, health.json()

        known = entities[0]
        got = http.post("/score", json={"entity_id": known, "event_ts": as_of + 60})
        assert got.status_code == 200, (got.status_code, got.text)
        body = got.json()
        assert service.response_carries_its_age(body), body
        assert body["features_as_of"] == as_of, body
        assert 0.0 <= body["score"] <= 1.0, body
        assert body["decision"] in ("allow", "review"), body

        # The service answer and the framework answer have to be the same number. A
        # serialiser that rounds a float would leave the log and the response disagreeing
        # and nothing else here would notice.
        direct = svc.handle({"entity_id": known, "event_ts": as_of + 60})
        assert direct.body["score"] == body["score"], (direct.body["score"], body["score"])

        miss = http.post("/score", json={"entity_id": "no_such_account", "event_ts": as_of})
        assert miss.status_code == 404, (miss.status_code, miss.text)

        bad = http.post("/score", json={"entity_id": "", "event_ts": 0})
        assert bad.status_code == 422, (bad.status_code, bad.text)
        assert len(bad.json()["problems"]) == 2, bad.json()

        # FastAPI must not reject this before the handler sees it. If pydantic validated
        # the body then the rules would have two implementations and the untested one
        # would be the one clients meet.
        extra = http.post("/score", json={"entity_id": known, "event_ts": as_of, "amount": 5})
        assert extra.status_code == 422, (extra.status_code, extra.text)
        assert "unknown fields" in " ".join(extra.json()["problems"]), extra.json()

        junk = http.post(
            "/score", content=b"{not json", headers={"content-type": "application/json"}
        )
        assert junk.status_code == 422, (junk.status_code, junk.text)

        refusing = service.Service(client, feature_set, model, service.Policy(max_age_s=60))
        strict = TestClient(api.build_app(refusing, sink=emitted.append))
        old = strict.post("/score", json={"entity_id": known, "event_ts": as_of})
        assert old.status_code == 503, (old.status_code, old.text)
        assert old.json()["max_age_s"] == 60, old.json()

        # One line per scoring request that arrived, including the one that never parsed.
        # A count that drifts from the request count is how a log stops being a record.
        #
        # Six and not seven. /health emits nothing, deliberately, because a readiness
        # probe firing every second would be most of the log and none of the information.
        # That is worth asserting rather than leaving as an off by one somebody later
        # 'fixes' by loosening the comparison.
        assert len(emitted) == 6, "{} log lines for 6 scoring requests".format(len(emitted))
        before = len(emitted)
        assert http.get("/health").status_code == 200
        assert len(emitted) == before, "/health wrote a log line"
        outcomes = [r["outcome"] for r in emitted]
        assert outcomes.count("unparseable") == 1, outcomes
        assert outcomes.count("scored") == 1, outcomes
        assert all("http_ms" in r for r in emitted), outcomes

        import json as json_mod

        for record in emitted:
            json_mod.dumps(record, sort_keys=True, default=str)

        print(
            "api arm agrees with the service, {} log lines from 6 scoring requests and "
            "none from health".format(len(emitted))
        )
    finally:
        server.shutdown()
    return 0


def main():
    suite = run_checks()
    if suite != 0:
        print("the suite failed, so the api arm was not run")
        return suite

    try:
        import fastapi  # noqa: F401
        import httpx  # noqa: F401
        import redis  # noqa: F401
        import redislite  # noqa: F401
    except ImportError as exc:
        print("{}, so this runner graded nothing beyond the suite".format(exc))
        return 2

    corpus = os.path.join(ROOT, "data", "transactions.csv")
    if not os.path.exists(corpus):
        print("no corpus at data/transactions.csv, run scripts/generate_corpus.py first")
        return 2

    return api_arm()


if __name__ == "__main__":
    sys.exit(main())
