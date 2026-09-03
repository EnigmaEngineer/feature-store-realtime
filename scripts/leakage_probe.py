"""What a feature table joined on entity alone is worth, against one joined point in time.

    python scripts/leakage_probe.py

Both arms use the same spec, the same corpus, the same split and the same model. The only
difference is when the features were computed. The point in time arm asks the log what it
knew at the moment of each transaction. The snapshot arm computes one vector per account
over the whole history and joins it onto every labelled row for that account, which is
what a nightly feature table and a join on account_id produce.

A third arm is the reason the first two mean anything. The oracle scores every row with
its own account's fraud rate in the test set, which is the best any account constant
score can reach. The snapshot arm is account constant by construction, so the oracle is
its ceiling and not 1.0.

Neither AUC is an accuracy claim about fraud detection. The corpus is synthetic, so both
are statements about the generator. The comparison between them is a statement about the
join.

TODO: a third snapshot arm taken near the label rather than at the end of the corpus.
That is the version of this mistake that inflates a metric instead of destroying it, and
it is the one a nightly feature table joined to same day labels actually produces.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import compute, corpus, spec


def to_matrix(rows, names):
    return [[float(r[n]) for n in names] for r in rows]


def entity_constant_ceiling(rows, entity, label):
    """The best AUC any score that is constant within an entity can reach here.

    The snapshot arm hands every transaction on an account the same feature vector, so
    whatever it learns, it can only rank accounts. This scores each row with its own
    account's fraud rate in the same set, using the labels, which is the oracle version
    of that. Without it there is no way to tell an arm that is failing from an arm that
    is at its own ceiling.
    """
    from sklearn.metrics import roc_auc_score

    total = {}
    positive = {}
    for r in rows:
        eid = r[entity]
        total[eid] = total.get(eid, 0) + 1
        positive[eid] = positive.get(eid, 0) + r[label]
    scores = [positive[r[entity]] / total[r[entity]] for r in rows]
    return roc_auc_score([r[label] for r in rows], scores)


def constant_within_entity(rows, entity, names):
    """How many entities have more than one distinct feature vector."""
    seen = {}
    varying = set()
    for r in rows:
        key = tuple(r[n] for n in names)
        eid = r[entity]
        if eid in seen and seen[eid] != key:
            varying.add(eid)
        seen[eid] = key
    return len(seen), len(varying)


def split_by_time(rows, time_field, share):
    times = sorted(r[time_field] for r in rows)
    cut = times[int(len(times) * share)]
    train = [r for r in rows if r[time_field] < cut]
    test = [r for r in rows if r[time_field] >= cut]
    return train, test, cut


def fit_and_score(train, test, names, label):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    x_train = to_matrix(train, names)
    y_train = [r[label] for r in train]
    x_test = to_matrix(test, names)
    y_test = [r[label] for r in test]

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        raise SystemExit("one side of the split has a single class, nothing to measure")

    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(x_train), y_train)
    scores = model.predict_proba(scaler.transform(x_test))[:, 1]
    return roc_auc_score(y_test, scores), model, scaler


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="specs/txn_fraud.yml")
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--train-share", type=float, default=0.7)
    args = ap.parse_args(argv)

    fs = spec.load(args.spec)
    events = list(corpus.read_csv(args.data))
    log = compute.EventLog(events, fs.entity, fs.event_time)
    names = fs.feature_names()

    t0 = time.perf_counter()
    pit = compute.training_rows(log, fs, events)
    pit_build_s = time.perf_counter() - t0

    # The snapshot is taken at the last event in the corpus, which is the most generous
    # reading of the mistake. A real nightly job would snapshot earlier and leak less.
    as_of = max(r[fs.event_time] for r in events)
    t0 = time.perf_counter()
    snap = compute.snapshot_rows(log, fs, events, as_of)
    snap_build_s = time.perf_counter() - t0

    pit_train, pit_test, cut = split_by_time(pit, fs.event_time, args.train_share)
    snap_train, snap_test, _ = split_by_time(snap, fs.event_time, args.train_share)

    pit_auc, model, _ = fit_and_score(pit_train, pit_test, names, fs.label)
    snap_auc, _, _ = fit_and_score(snap_train, snap_test, names, fs.label)

    ceiling = entity_constant_ceiling(snap_test, fs.entity, fs.label)
    entities, varying = constant_within_entity(snap_test, fs.entity, names)
    pit_entities, pit_varying = constant_within_entity(pit_test, fs.entity, names)

    out = {
        "spec": fs.name,
        "fingerprint": fs.fingerprint(),
        "rows": len(events),
        "train_rows": len(pit_train),
        "test_rows": len(pit_test),
        "test_positives": sum(r[fs.label] for r in pit_test),
        "split_ts": cut,
        "point_in_time_auc": round(pit_auc, 4),
        "entity_join_snapshot_auc": round(snap_auc, 4),
        "entity_constant_oracle_auc": round(ceiling, 4),
        "coin_flip_auc": 0.5,
        "point_in_time_build_s": round(pit_build_s, 2),
        "snapshot_build_s": round(snap_build_s, 2),
        "test_entities": entities,
        "snapshot_entities_with_more_than_one_vector": varying,
        "point_in_time_entities_with_more_than_one_vector": pit_varying,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    print("")
    print("coefficients of the point in time model, standardised inputs")
    for name, coef in sorted(zip(names, model.coef_[0]), key=lambda p: -abs(p[1])):
        print("  {:<24} {:+.3f}".format(name, coef))
    print("")
    print("these are not importances and the two largest are not opposed in the way they")
    print("look. a burst raises the hourly count and the hourly merchant count together,")
    print("so the model is fitting their difference and splitting one effect across two")
    print("coefficients with opposite signs. read them as a pair or not at all.")
    print("")
    print("both AUCs are about a synthetic corpus. the comparison is about the join.")
    print("the oracle is the best any account constant score can do on this test set.")
    print("it is high, so the snapshot arm is not capped by its own shape. it misses a")
    print("ceiling that was available to it, because a snapshot taken at one moment")
    print("carries nothing about a burst that happened somewhere else in time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
