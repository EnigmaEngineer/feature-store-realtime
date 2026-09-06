"""Load the online store, read every row back, and grade the round trip.

    python scripts/store_check.py --accounts 400

What this grades is narrow and worth stating before the numbers. Both sides of the
comparison call `featurestore/compute.py`, so this says nothing about whether the feature
definitions are right. `scripts/oracle_check.py` is what asks that question, by putting a
second implementation next to the reference. This asks whether a value written to Redis
and read back is the value that was computed, which is a different question and has its
own way of going wrong.

The controls are not optional. A round trip reporting zero differences reads the same
whether the encoding is exact or both sides are being handed the same mistake. So the
same loader writes a damaged arm for every entry in `online.DAMAGES` and each one has to
disagree. The arms are read off the catalogue rather than listed here.

The isolation arm is separate because it damages no value. It reloads under a spec with
one window changed and counts what a service asking for the new definition gets back.
With the fingerprint in the key that is a miss on every entity. Without it the store
answers, and the answer is built from a definition nobody is serving.

The server is started inside this process on purpose. A backgrounded process does not
survive the call that starts it on the machine this was written for, so a script that
expects a Redis to already be running is a script that does not run here.
"""

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import compute
from featurestore import corpus as corpus_io
from featurestore import online, spec

PORT = 6399


def start_server(port):
    """A real Redis, not a stand in. Returns the redislite handle and a plain client."""
    import redis
    import redislite

    scratch = tempfile.mkdtemp(prefix="store-check-")
    server = redislite.Redis(
        os.path.join(scratch, "store.rdb"), serverconfig={"port": str(port)}
    )
    client = redis.Redis(host="127.0.0.1", port=port, decode_responses=True)
    client.ping()
    return server, client


def pick_rows(rows, n_accounts):
    """Whole accounts, taken in id order so a re-run reads the same rows.

    Half an account's history gives the reference a different answer, because every
    window is per entity.
    """
    if n_accounts is None:
        return rows
    keep = set(sorted({r["account_id"] for r in rows})[:n_accounts])
    return [r for r in rows if r["account_id"] in keep]


def variant_spec(feature_set):
    """The same feature set with one window widened, so its fingerprint moves.

    Built from the loaded spec rather than from a second file. A second file could drift
    into agreeing with the first, and then the isolation arm silently tests nothing.
    """
    changed = []
    for feature in feature_set.features:
        window = spec.Window(
            feature.window.length_s + 3600, feature.window.closed, feature.window.align
        )
        changed.append(
            spec.Feature(
                feature.name, feature.dtype, feature.aggregate, feature.column, window
            )
        )
    return spec.FeatureSet(
        feature_set.name,
        feature_set.entity,
        feature_set.event_time,
        changed,
        feature_set.label,
    )


def read_back(client, feature_set, entity_ids, keyed_by_fingerprint=True):
    out = {}
    misses = 0
    for entity_id in entity_ids:
        try:
            values, _ = online.lookup(
                client, feature_set, entity_id, keyed_by_fingerprint
            )
        except online.Miss:
            misses += 1
            continue
        out[entity_id] = values
    return out, misses


def compare(reference, actual, feature_set):
    report = {name: {"compared": 0, "exact": 0, "differing": 0} for name in feature_set.feature_names()}
    examples = []
    for entity_id in sorted(set(reference) & set(actual)):
        want = reference[entity_id]
        got = actual[entity_id]
        for name in feature_set.feature_names():
            cell = report[name]
            cell["compared"] += 1
            if want[name] == got[name]:
                cell["exact"] += 1
            else:
                cell["differing"] += 1
                if len(examples) < 5:
                    examples.append(
                        {
                            "entity": entity_id,
                            "feature": name,
                            "computed": want[name],
                            "round_tripped": got[name],
                        }
                    )
    return {
        "entities_paired": len(set(reference) & set(actual)),
        "entities_missing": len(set(reference) - set(actual)),
        "by_feature": report,
        "examples": examples,
    }


def totals(result):
    out = {"compared": 0, "exact": 0, "differing": 0}
    for cell in result["by_feature"].values():
        for k in out:
            out[k] += cell[k]
    return out


def print_arm(label, result):
    t = totals(result)
    print(
        "{:<24} {:>6} entities  {:>7} cells  exact {:>7}  differing {:>7}".format(
            label, result["entities_paired"], t["compared"], t["exact"], t["differing"]
        )
    )


def measure_lookup(client, feature_set, entity_ids, repeats):
    """Wall clock for one lookup, which is a decode on top of one HGETALL.

    Loopback on one machine with the server in the same process. There is no network and
    no model here, so this is a floor under the serving latency and not a serving
    latency. The first pass is discarded because the connection pool opens its socket on
    the first call and that cost lands on whichever lookup runs first.
    """
    for entity_id in entity_ids[:50]:
        online.lookup(client, feature_set, entity_id)

    samples = []
    for _ in range(repeats):
        for entity_id in entity_ids:
            started = time.perf_counter()
            online.lookup(client, feature_set, entity_id)
            samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return {
        "n": len(samples),
        "p50": samples[len(samples) // 2],
        "p95": samples[int(len(samples) * 0.95)],
        "p99": samples[int(len(samples) * 0.99)],
        "max": samples[-1],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="specs/txn_fraud.yml")
    ap.add_argument("--corpus", default="data/transactions.csv")
    ap.add_argument("--accounts", type=int, default=400, help="0 means every account")
    ap.add_argument("--repeats", type=int, default=3, help="latency passes over every entity")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args(argv)

    feature_set = spec.load(args.spec)
    rows = pick_rows(list(corpus_io.read_csv(args.corpus)), args.accounts or None)
    log = compute.EventLog(rows, feature_set.entity, feature_set.event_time)
    as_of = max(r[feature_set.event_time] for r in rows) + 1
    entity_ids = sorted(log.entities())

    print(
        "{} rows over {} entities, spec {} at {}".format(
            len(rows), len(entity_ids), feature_set.name, feature_set.fingerprint()
        )
    )
    print("loading at as_of {}, one second past the last event in the slice".format(as_of))

    reference = {eid: vals for eid, vals in online.online_rows(log, feature_set, as_of)}

    server, client = start_server(args.port)
    results = {}
    try:
        # Before and after, because the store has to be sized before anybody proposes
        # putting a real customer base in it. This is one hash of seven small strings per
        # entity and Redis charges its own overhead on top, so the figure is not the one
        # a reader would get by adding up the values.
        empty_bytes = client.info("memory")["used_memory"]
        started = time.time()
        written = online.load(client, log, feature_set, as_of)
        load_s = time.time() - started
        loaded_bytes = client.info("memory")["used_memory"]
        per_entity = (loaded_bytes - empty_bytes) / written if written else 0.0
        print(
            "wrote {} entities in {:.2f}s, {:.0f} entities per second".format(
                written, load_s, written / load_s
            )
        )
        print(
            "the store grew from {} to {} bytes, {:.0f} bytes per entity".format(
                empty_bytes, loaded_bytes, per_entity
            )
        )

        actual, misses = read_back(client, feature_set, entity_ids)
        results["shipped"] = compare(reference, actual, feature_set)
        results["shipped"]["misses"] = misses

        for name in sorted(online.DAMAGES):
            client.flushall()
            online.load(client, log, feature_set, as_of, damage=name)
            damaged, _ = read_back(client, feature_set, entity_ids)
            results[name] = compare(reference, damaged, feature_set)

        # The isolation arm. Load under the shipped spec, then ask under a spec whose
        # windows moved, which is what a deploy of a changed definition looks like from
        # the service's side.
        variant = variant_spec(feature_set)
        isolation = {}
        for keyed in (True, False):
            client.flushall()
            online.load(client, log, feature_set, as_of, keyed_by_fingerprint=keyed)
            answered, missed = read_back(client, variant, entity_ids, keyed)
            wrong = compare(
                {eid: vals for eid, vals in online.online_rows(log, variant, as_of)},
                answered,
                variant,
            )
            counts = totals(wrong)
            isolation["keyed" if keyed else "unkeyed"] = {
                "answered": len(answered),
                "misses": missed,
                "cells": counts["compared"],
                "cells_differing_from_the_new_definition": counts["differing"],
                # Computed here rather than written into a sentence somewhere. A share
                # that is arithmetic over two published counts goes wrong silently the
                # first time either count is corrected.
                "share_wrong": (
                    counts["differing"] / counts["compared"] if counts["compared"] else 0.0
                ),
            }

        client.flushall()
        online.load(client, log, feature_set, as_of)
        latency = measure_lookup(client, feature_set, entity_ids, args.repeats)
    finally:
        server.shutdown()

    print()
    print_arm("shipped", results["shipped"])
    for name in sorted(online.DAMAGES):
        print_arm(name, results[name])
    for ex in results["shipped"]["examples"]:
        print("  disagreement {}".format(ex))

    print()
    print("a spec change, asked for by a service running the new definition")
    for arm in ("keyed", "unkeyed"):
        row = isolation[arm]
        # A share of 0.000 over zero cells reads as "nothing is wrong" and means
        # "nothing was compared". On the arm that misses everything, which is the arm
        # that is supposed to win, that is the more misleading of the two readings.
        share = "{:.3f}".format(row["share_wrong"]) if row["cells"] else "no cells"
        print(
            "  fingerprint in the key: {:<5}  answered {:>5}  missed {:>5}  cells wrong "
            "{:>7} of {:>7}  share {:>8}".format(
                "yes" if arm == "keyed" else "no",
                row["answered"],
                row["misses"],
                row["cells_differing_from_the_new_definition"],
                row["cells"],
                share,
            )
        )

    print()
    print(
        "lookup over {} calls  p50 {:.3f} ms  p95 {:.3f} ms  p99 {:.3f} ms  max "
        "{:.3f} ms".format(
            latency["n"], latency["p50"], latency["p95"], latency["p99"], latency["max"]
        )
    )

    report = {
        "rows": len(rows),
        "entities": len(entity_ids),
        "as_of": as_of,
        "fingerprint": feature_set.fingerprint(),
        "variant_fingerprint": variant.fingerprint(),
        "load_seconds": round(load_s, 3),
        "bytes_empty": empty_bytes,
        "bytes_loaded": loaded_bytes,
        "bytes_per_entity": round(per_entity, 1),
        "arms": results,
        "isolation": isolation,
        "lookup_ms": latency,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)

    blind = [name for name in online.DAMAGES if totals(results[name])["differing"] == 0]
    if blind:
        print()
        print(
            "CONTROL FAILED for {}. A damaged loader round tripped every cell, so this "
            "comparison is not sensitive to that mistake.".format(", ".join(sorted(blind)))
        )
        return 2

    if isolation["keyed"]["answered"] or isolation["keyed"]["misses"] != len(entity_ids):
        print()
        print("CONTROL FAILED. A spec change was answered from the old keyspace.")
        return 2

    shipped = totals(results["shipped"])
    if shipped["differing"] or results["shipped"]["misses"] or results["shipped"]["entities_missing"]:
        print()
        print("the round trip lost something, see above")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
