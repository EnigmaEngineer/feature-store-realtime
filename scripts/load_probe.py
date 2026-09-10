"""What the service does when more than one request is in flight.

    python scripts/load_probe.py --accounts 0

Every latency figure this repo has published so far came off `fastapi.testclient`, which
sends one request at a time and never opens a socket. That is a floor and the README says
so twice. It is also blind to the one decision the HTTP edge makes, which is whether the
blocking work runs on a worker thread or on the event loop.

Three arms, and all three come out of `featurestore/api.build_app`. Writing the losing arm
out by hand in this file would let it lose for a reason nobody chose.

  offload   the shipped arrangement, `run_in_threadpool`
  inline    the same app with `offload=False`, blocking the loop
  cached    the offload arm against a store held in a dict rather than in Redis

The third arm is not a proposal. It is the ceiling on caching. A perfect cache is a dict
with every key already in it and no eviction and no round trip, so whatever it buys here
is the most any real cache could buy, before paying for staleness or invalidation. If the
ceiling is small the cache does not earn its place and this file is the argument.

The driver is a closed loop. One connection per worker, keep alive, and the next request
goes out as soon as the previous one returns. So the number of requests outstanding is the
worker count by construction, and `featurestore/load.offered_concurrency` checks that it
really was. A pass that fails that check is a measurement of this process trying to be N
clients, and `load.compare` refuses it rather than printing a ratio.

Locust is in the technology list for this project and it does not run in this sandbox.
`pip install --target` puts `zope.event` where `/usr/lib/python3/dist-packages/
zope.interface-5.4.0-nspkg.pth` cannot see it, because that file pins `zope.__path__` to
the system directory at startup, so gevent fails to import. Skipping site processing with
`-S` gets past that and then dies inside greenlet finalisation. The driver here is about a
hundred lines of the standard library and it gives exact control of the concurrency, which
is the thing being varied.
"""

import argparse
import http.client
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import api, compute, load, online, service, spec  # noqa: E402
from featurestore import corpus as corpus_io  # noqa: E402
from featurestore import model as model_mod  # noqa: E402


class DictStore:
    """Every row already in memory, under the same keys and the same bytes.

    Built by reading the real store out once, so the cached arm and the Redis arm decode
    identical payloads. A cache stocked from a different source would be measuring the
    other source.
    """

    def __init__(self, rows):
        self.rows = rows

    def hgetall(self, key):
        return self.rows.get(key, {})


class SlowStore:
    """A store with a wait in front of it, standing in for one that is not on loopback.

    `time.sleep` releases the GIL, which is the property that matters. A wait that held it
    would make the offload arm lose by construction and the sweep would prove nothing.

    This exists because the result on this machine is about this machine. Redis here is
    redislite over loopback and its round trip is a fraction of a millisecond, so there is
    almost nothing for a worker thread to overlap. Naming that as a caveat would be cheap.
    Measuring where it flips is the answer to it.
    """

    def __init__(self, inner, delay_s):
        self.inner = inner
        self.delay_s = delay_s

    def hgetall(self, key):
        time.sleep(self.delay_s)
        return self.inner.hgetall(key)


def snapshot(client, feature_set, entity_ids):
    """The whole store as a dict, read through the same key builder the service uses."""
    out = {}
    for entity_id in entity_ids:
        key = online.store_key(feature_set, entity_id)
        raw = client.hgetall(key)
        if raw:
            out[key] = raw
    return out


def serve(app, port):
    """uvicorn on a background thread, returning something that stops it.

    Signal handlers are suppressed because they can only be installed from the main
    thread. A real deployment wants them and this harness is not one.
    """
    import uvicorn

    class Quiet(uvicorn.Server):
        def install_signal_handlers(self):
            return None

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    server = Quiet(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30.0
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not come up on port {}".format(port))
        time.sleep(0.01)
    return server, thread


def stop(server, thread):
    server.should_exit = True
    thread.join(timeout=20.0)


def worker(port, bodies, count, start_gate, durations, statuses, lock):
    """One client. Its own connection, its own list, one request at a time."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    mine = []
    mine_status = {}
    start_gate.wait()
    for i in range(count):
        body = bodies[i % len(bodies)]
        began = time.perf_counter()
        conn.request("POST", "/score", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        mine.append((time.perf_counter() - began) * 1000.0)
        mine_status[resp.status] = mine_status.get(resp.status, 0) + 1
    conn.close()
    with lock:
        durations.extend(mine)
        for status, n in mine_status.items():
            statuses[status] = statuses.get(status, 0) + n


def drive(port, bodies, workers, per_worker):
    """One pass. Returns the summary `featurestore/load.py` builds."""
    durations = []
    statuses = {}
    lock = threading.Lock()
    gate = threading.Barrier(workers + 1)
    threads = [
        threading.Thread(
            target=worker,
            args=(port, bodies, per_worker, gate, durations, statuses, lock),
            daemon=True,
        )
        for _ in range(workers)
    ]
    for t in threads:
        t.start()
    gate.wait()
    began = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - began
    return load.summarise(durations, wall, workers, statuses)


def warm(port, bodies, calls):
    """Requests whose timings are thrown away.

    The first request through a fresh process pays for the route table, the encoder and,
    on the offload arm, for creating a worker thread. Charging that to whichever arm ran
    first is a mistake I have made three times.
    """
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    for i in range(calls):
        conn.request(
            "POST", "/score", body=bodies[i % len(bodies)],
            headers={"Content-Type": "application/json"},
        )
        conn.getresponse().read()
    conn.close()


def run_arm(svc, bodies, port, workers, per_worker, warmup, offload, sink):
    app = api.build_app(svc, sink=sink, offload=offload)
    server, thread = serve(app, port)
    try:
        warm(port, bodies, warmup)
        return drive(port, bodies, workers, per_worker)
    finally:
        stop(server, thread)


def row(label, arm):
    return "  {:<22} {:>7.0f} rps   p50 {:>6.2f}  p95 {:>6.2f}  p99 {:>6.2f}   offered {:>5.2f}".format(
        label, arm["throughput_rps"], arm["p50_ms"], arm["p95_ms"], arm["p99_ms"], arm["offered"]
    )


def delay_sweep(args, client, feature_set, model, bodies, report, sink):
    """Where a worker thread starts paying for itself, as a store round trip in ms.

    Only the two arrangement arms. The cache arms answer a different question and running
    them here would triple the time for nothing.
    """
    workers = int(args.workers.split(",")[0])
    per_worker = max(1, args.requests // workers)
    delays = [float(x) for x in args.store_delay_ms.split(",") if x.strip()]
    print("delay sweep at concurrency {}, {} requests per worker, {} rounds".format(
        workers, per_worker, args.rounds))
    print("best reachable p is {:.4f} against {}".format(load.p_floor(args.rounds), load.ALPHA))

    crossover = None
    for delay_ms in delays:
        store = SlowStore(client, delay_ms / 1000.0)
        pair = {
            "offload": (service.Service(store, feature_set, model), True),
            "inline": (service.Service(store, feature_set, model), False),
        }
        seen = {"offload": [], "inline": []}
        port = args.port + 1
        for rnd in range(args.rounds):
            order = ["offload", "inline"] if rnd % 2 == 0 else ["inline", "offload"]
            for name in order:
                svc, offload = pair[name]
                seen[name].append(
                    run_arm(svc, bodies, port, workers, per_worker, args.warmup, offload, sink)
                )
                port += 1

        med = {n: load.median([r["throughput_rps"] for r in seen[n]]) for n in seen}
        verdict, detail = load.compare_reps(
            "offload", seen["offload"], "inline", seen["inline"], "throughput_rps", True
        )
        print("\n  store delay {:.1f} ms".format(delay_ms))
        print("    offload {:>7.0f} rps   inline {:>7.0f} rps   ratio {:.2f}x".format(
            med["offload"], med["inline"], med["offload"] / med["inline"]))
        if verdict != "ok":
            print("    REFUSED, {}, {}".format(verdict, detail))
        else:
            print("    offload won {} of {} rounds, p {:.4f}, {}".format(
                detail["wins"], detail["pairs"], detail["p"],
                "decided" if detail["decided"] else "NOT DECIDED"))
            if detail["decided"] and detail["wins"] == detail["pairs"] and crossover is None:
                crossover = delay_ms
        report["levels"][delay_ms] = {
            "rounds": seen,
            "comparison": [verdict, detail],
            "medians": med,
        }

    if crossover is None:
        print("\nthe worker thread never won across {}".format(delays))
    else:
        print("\nthe worker thread first wins every round at a store delay of {:.1f} ms".format(
            crossover))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, default=str)
        print("report written to {}".format(args.json))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="specs/txn_fraud.yml")
    ap.add_argument("--model", default="models/txn_fraud.json")
    ap.add_argument("--corpus", default="data/transactions.csv")
    ap.add_argument("--accounts", type=int, default=200, help="0 means every account")
    ap.add_argument("--at", type=float, default=0.9, help="load the store this far through the corpus")
    ap.add_argument("--workers", default="1,8,32,64", help="concurrency levels to sweep")
    ap.add_argument("--requests", type=int, default=1000, help="requests per arm per round")
    ap.add_argument("--rounds", type=int, default=6, help="paired rounds per level, 6 is the power floor")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument(
        "--store-delay-ms",
        help="comma separated waits to put in front of the store. Runs the arrangement "
        "sweep at the first concurrency level instead of the four arm comparison.",
    )
    ap.add_argument("--rdb", default=os.path.join("/tmp", "fsr_load_probe"))
    ap.add_argument("--port", type=int, default=6405, help="redis port, http ports follow it")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args(argv)

    import redis
    import redislite

    feature_set = spec.load(args.spec)
    model = model_mod.load(args.model).bind(feature_set)
    entity = feature_set.entity
    time_field = feature_set.event_time

    rows = list(corpus_io.read_csv(args.corpus))
    if args.accounts:
        keep = sorted({r[entity] for r in rows})[: args.accounts]
        rows = [r for r in rows if r[entity] in set(keep)]
    log = compute.EventLog(rows, entity, time_field)
    times = sorted(r[time_field] for r in rows)
    as_of = times[int(len(times) * args.at)]

    levels = [int(x) for x in args.workers.split(",") if x.strip()]
    os.makedirs(args.rdb, exist_ok=True)
    server = redislite.Redis(os.path.join(args.rdb, "probe.rdb"), serverconfig={"port": str(args.port)})
    report = {"levels": {}, "requests_per_arm": args.requests, "warmup": args.warmup}
    try:
        client = redis.Redis(host="127.0.0.1", port=args.port)
        client.flushall()
        written = online.load(client, log, feature_set, as_of)

        # Only requests that will score. A pass where some fraction misses is faster than
        # one where they all score, and a latency table does not say which happened.
        served = sorted({eid for eid, _ in online.online_rows(log, feature_set, as_of)})
        arriving = [r for r in rows if r[time_field] >= as_of and r[entity] in set(served)]
        if not arriving:
            print("no request would score against this store, nothing to measure")
            return 2
        bodies = [
            json.dumps({"entity_id": r[entity], "event_ts": r[time_field]}).encode()
            for r in arriving[:512]
        ]

        cached_rows = snapshot(client, feature_set, served)
        print(
            "{} rows over {} entities, store written for {}, {} requests would score".format(
                len(rows), len({r[entity] for r in rows}), written, len(arriving)
            )
        )
        print("cache ceiling arm holds {} keys in a dict".format(len(cached_rows)))

        def drop(record):
            # The service still builds a log record on every request, which is work the
            # arms should pay for. Keeping forty thousand of them would measure a growing
            # list instead, so the sink is where they stop.
            return None

        # Four arms rather than three. The cache has to be measured inside each
        # arrangement, because a cache arm run one way and compared against a Redis arm run
        # the other is two changes at once and the table would credit the cache with both.
        # The first shape of this file did exactly that.
        arms = {
            "offload": (service.Service(client, feature_set, model), True),
            "inline": (service.Service(client, feature_set, model), False),
            "cached offload": (service.Service(DictStore(cached_rows), feature_set, model), True),
            "cached inline": (service.Service(DictStore(cached_rows), feature_set, model), False),
        }

        if args.store_delay_ms:
            return delay_sweep(args, client, feature_set, model, bodies, report, drop)

        names = ("offload", "inline", "cached offload", "cached inline")
        floor = load.p_floor(args.rounds)
        print("{} rounds per level, best reachable p is {:.4f} against {}".format(
            args.rounds, floor, load.ALPHA))
        if floor > load.ALPHA:
            print("that cannot decide anything. Raise --rounds to at least 6.")

        for workers in levels:
            per_worker = max(1, args.requests // workers)
            print("\nconcurrency {}, {} requests per worker".format(workers, per_worker))
            seen = {name: [] for name in names}
            port = args.port + 1
            for rnd in range(args.rounds):
                # Rotate which arm goes first. A fixed order charges whatever the machine
                # was doing at the start of a round to the same arm every time, which is
                # how a warmup cost ends up looking like a design difference.
                order = [names[(rnd + i) % len(names)] for i in range(len(names))]
                for name in order:
                    svc, offload = arms[name]
                    arm = run_arm(svc, bodies, port, workers, per_worker, args.warmup, offload, drop)
                    port += 1
                    seen[name].append(arm)

            for name in names:
                reps = seen[name]
                print(row(name, {
                    "throughput_rps": load.median([r["throughput_rps"] for r in reps]),
                    "p50_ms": load.median([r["p50_ms"] for r in reps]),
                    "p95_ms": load.median([r["p95_ms"] for r in reps]),
                    "p99_ms": load.median([r["p99_ms"] for r in reps]),
                    "offered": load.median([r["offered"] for r in reps]),
                }))
                print("      {} rounds, throughput {:.0f} to {:.0f} rps, p99 {:.2f} to {:.2f} ms".format(
                    len(reps),
                    min(r["throughput_rps"] for r in reps), max(r["throughput_rps"] for r in reps),
                    min(r["p99_ms"] for r in reps), max(r["p99_ms"] for r in reps)))

            comparisons = {}
            for label, a, b, key, higher in (
                ("threadpool costs, throughput", "offload", "inline", "throughput_rps", True),
                ("threadpool costs, p99", "offload", "inline", "p99_ms", False),
                ("cache ceiling on inline, throughput", "cached inline", "inline", "throughput_rps", True),
                ("cache ceiling on inline, p99", "cached inline", "inline", "p99_ms", False),
                ("cache ceiling on offload, throughput", "cached offload", "offload", "throughput_rps", True),
                ("cache ceiling on offload, p99", "cached offload", "offload", "p99_ms", False),
            ):
                verdict, detail = load.compare_reps(a, seen[a], b, seen[b], key, higher)
                comparisons[label] = [verdict, detail]
                if verdict != "ok":
                    print("  {}: REFUSED, {}, {}".format(label, verdict, detail))
                elif detail["decided"]:
                    print("  {}: {} of {} rounds, p {:.4f}, medians {:.2f}x".format(
                        label, detail["wins"], detail["pairs"], detail["p"],
                        detail["ratio_of_medians"]))
                else:
                    print("  {}: {} of {} rounds, p {:.4f}, NOT DECIDED, medians {:.2f}x".format(
                        label, detail["wins"], detail["pairs"], detail["p"],
                        detail["ratio_of_medians"]))

            report["levels"][workers] = {
                "rounds": {name: seen[name] for name in names},
                "comparisons": comparisons,
            }

        passes = [
            (w, name, i, arm)
            for w, level in report["levels"].items()
            for name, reps in level["rounds"].items()
            for i, arm in enumerate(reps)
        ]
        starved = [(w, name, i, arm["offered_ratio"]) for w, name, i, arm in passes if not arm["landed"]]
        if starved:
            print("\nthe driver failed its own control on {} of {} passes".format(
                len(starved), len(passes)))
            for w, name, i, ratio in starved:
                print("  {} round {} at {} workers offered {:.2f} of what it asked for".format(
                    name, i, w, ratio))
        else:
            print("\nthe driver kept its load up on every pass, {} of {}".format(
                len(passes), len(passes)))

        if args.json:
            with open(args.json, "w") as fh:
                json.dump(report, fh, indent=2, sort_keys=True, default=str)
            print("report written to {}".format(args.json))
        return 0
    finally:
        server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
