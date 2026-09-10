"""The HTTP edge. FastAPI, and nothing that decides anything.

    uvicorn --factory featurestore.api:app

Every rule about what a request may contain, what a miss means and what to do with a
vector's age lives in `featurestore/service.py`, which imports no framework. This file
turns a `Result` into a response and a log record into a line of JSON on stdout.

fastapi is imported inside `build_app` rather than at the top, for the same reason
`featurestore/spark_job.py` defers pyspark. A clone that installed only
`requirements.txt` has to be able to import every module in this package, and
`tests/test_deps.py` asserts that arrangement rather than trusting it, because this
machine has fastapi installed and would never notice it breaking.

There is no pydantic request model, deliberately. The handler takes a raw `Request` and
passes the body to `service.validate`. If FastAPI rejected a bad body before that saw it
then the validation rules would have two implementations, one of which is untested and
returns a different shape. One implementation is the point and pydantic is not it.
"""

import json
import os
import sys
import time
import uuid

from featurestore import model as model_mod
from featurestore import service, spec

DEFAULT_SPEC = os.environ.get("FS_SPEC", "specs/txn_fraud.yml")
DEFAULT_MODEL = os.environ.get("FS_MODEL", "models/txn_fraud.json")
DEFAULT_REDIS_URL = os.environ.get("FS_REDIS_URL", "redis://127.0.0.1:6379/0")


def emit(record, stream=None):
    """One JSON object per line, no logging framework.

    A structured log is a shape, not a library. Sorting the keys means two lines from two
    processes diff cleanly, which is the whole reason to emit JSON rather than a sentence.
    """
    stream = stream or sys.stdout
    stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    stream.flush()


def build_app(svc, sink=emit, offload=True):
    """A FastAPI app over an already assembled `service.Service`.

    Taking the service rather than building one means the fourth runner can hand it a
    store it loaded itself, and nothing here reads an environment variable at request
    time.

    `offload` decides whether the blocking work runs on a worker thread or on the event
    loop. It exists so that `scripts/load_probe.py` can measure the arrangement this file
    does not use, and it defaults to the one it does. The alternative was writing the
    losing arm out by hand in the probe, which is how a comparison gets rigged without
    anybody meaning to.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="feature store scoring", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health():
        # Health says what this process is serving. A green tick that does not name the
        # spec and the model cannot tell you that half the fleet is on the old artefact.
        return {
            "status": "ok",
            "feature_set": svc.feature_set.name,
            "fingerprint": svc.feature_set.fingerprint(),
            "model_fingerprint": svc.model.fingerprint,
            "max_age_s": svc.policy.max_age_s,
        }

    @app.post("/score")
    async def score(request: Request):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        try:
            payload = await request.json()
        except Exception:
            # A body that is not JSON never reaches the service, so it needs its own log
            # line here or the count of requests received and the count of lines emitted
            # stop agreeing.
            record = {"request_id": request_id, "outcome": "unparseable", "score": None}
            record["http_ms"] = round((time.perf_counter() - started) * 1000.0, 4)
            sink(record)
            return JSONResponse(status_code=service.BAD_REQUEST, content={"error": "body is not JSON"})

        # Off the event loop. redis-py is synchronous, so calling svc.handle directly from
        # an async endpoint blocks the whole loop for the duration of the round trip, and
        # every other request in flight waits behind it. On one request at a time that is
        # invisible, which is why it survives into load testing and shows up there as a
        # throughput ceiling nobody can find in the code.
        #
        # The alternative is an async Redis client, which means the store path grows an
        # async twin and `featurestore/online.py` stops being one implementation.
        #
        # What that costs at one request at a time, and what it buys at many, is measured
        # by `scripts/load_probe.py` rather than argued here. Both answers are in
        # docs/adr-0006 and neither of them is the one this comment assumed.
        if offload:
            result = await run_in_threadpool(svc.handle, payload, request_id)
        else:
            result = svc.handle(payload, request_id)
        result.log["http_ms"] = round((time.perf_counter() - started) * 1000.0, 4)
        result.log["status"] = result.status
        sink(result.log)
        return JSONResponse(status_code=result.status, content=result.body)

    return app


def from_environment():
    """Assemble the service the way a container would.

    Raises rather than degrading. A process that comes up with no model, or with a model
    fitted on a different spec, is one that will answer requests with numbers nobody can
    account for, and it should fail at start.
    """
    import redis

    feature_set = spec.load(DEFAULT_SPEC)
    model = model_mod.load(DEFAULT_MODEL)
    client = redis.Redis.from_url(DEFAULT_REDIS_URL)
    max_age = os.environ.get("FS_MAX_AGE_S")
    policy = service.Policy(int(max_age) if max_age else None)
    return service.Service(client, feature_set, model, policy)


def app():
    """A factory, deferred so that importing this module does not open a socket.

    A module level `app = build_app(...)` would make every import of this file try to
    reach Redis, including the one `tests/test_deps.py` does when it walks the tree. The
    cost is that uvicorn needs telling, which is the `--factory` flag in the command at
    the top of this file.
    """
    return build_app(from_environment())
