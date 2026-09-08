"""Fit the scorer and write it out as a handful of floats.

    python scripts/train_model.py --accounts 0 --out models/txn_fraud.json

Point in time features for every transaction, a split on the clock rather than at random,
and a logistic regression. The artefact is JSON and the serving path reads it with
`featurestore/model.py`, which imports nothing.

The part of this worth having is the last check. scikit-learn scores a row through numpy
and the service scores the same row through `math.exp` in `featurestore/model.py`. Those
are two implementations of one function, and this program has been bitten more than once
by two implementations that were each correct and disagreed. So the fit is not written out
until every training row has been pushed through both and the largest gap between them
reported.

A time split and not a random one. A random split puts a transaction from Tuesday in
training and one from Monday in test for the same account, which is the join mistake
`scripts/leakage_probe.py` measures, arriving through the split instead of through the
features.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import compute
from featurestore import corpus as corpus_io
from featurestore import model as model_mod
from featurestore import spec


def pick_rows(rows, n_accounts):
    if n_accounts is None:
        return rows
    keep = set(sorted({r["account_id"] for r in rows})[:n_accounts])
    return [r for r in rows if r["account_id"] in keep]


def split_on_the_clock(table, time_field, train_share):
    times = sorted(r[time_field] for r in table)
    cut = times[int(len(times) * train_share)]
    train = [r for r in table if r[time_field] < cut]
    test = [r for r in table if r[time_field] >= cut]
    return train, test, cut


def as_matrix(table, names):
    return [[float(row[name]) for name in names] for row in table]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="specs/txn_fraud.yml")
    ap.add_argument("--corpus", default="data/transactions.csv")
    ap.add_argument("--accounts", type=int, default=400, help="0 means every account")
    ap.add_argument("--train-share", type=float, default=0.7)
    ap.add_argument("--out", default="models/txn_fraud.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    feature_set = spec.load(args.spec)
    names = feature_set.feature_names()
    time_field = feature_set.event_time
    label = feature_set.label

    rows = pick_rows(list(corpus_io.read_csv(args.corpus)), args.accounts or None)
    log = compute.EventLog(rows, feature_set.entity, time_field)
    table = compute.training_rows(log, feature_set, rows)
    train, test, cut = split_on_the_clock(table, time_field, args.train_share)

    print("{} rows over {} accounts, spec {} at {}".format(
        len(table), len(log.entities()), feature_set.name, feature_set.fingerprint()))
    print("split at {}, {} training rows and {} test rows".format(cut, len(train), len(test)))
    print("{} fraud in training, {} in test".format(
        sum(r[label] for r in train), sum(r[label] for r in test)))

    if not sum(r[label] for r in test):
        # An AUC needs both classes. Reporting one computed on a single class would be a
        # number with no meaning, and the failure is silent otherwise.
        print("the test half holds no fraud, so there is no AUC to report")
        return 2

    # class_weight balanced because the corpus is 0.08 percent fraud and an unweighted fit
    # on that predicts the majority class everywhere. The coefficients are what the
    # service uses, so this is a serving decision and not a reporting one.
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(as_matrix(train, names), [r[label] for r in train])

    train_auc = roc_auc_score([r[label] for r in train], clf.predict_proba(as_matrix(train, names))[:, 1])
    test_auc = roc_auc_score([r[label] for r in test], clf.predict_proba(as_matrix(test, names))[:, 1])
    print("train AUC {:.4f}".format(train_auc))
    print("test AUC  {:.4f}".format(test_auc))

    coefficients = {name: float(c) for name, c in zip(names, clf.coef_[0])}
    model = model_mod.LinearModel(
        names, coefficients, float(clf.intercept_[0]), feature_set.fingerprint(), args.threshold
    )

    # The check this script exists for. Same rows, two scorers, one of which is the one
    # that will answer a request at 3am.
    sk = clf.predict_proba(as_matrix(table, names))[:, 1]
    worst = 0.0
    worst_row = None
    for row, expected in zip(table, sk):
        got = model.score({name: row[name] for name in names})
        gap = abs(got - float(expected))
        if gap > worst:
            worst = gap
            worst_row = row
    print("largest disagreement with scikit-learn over {} rows: {:.3e}".format(len(table), worst))
    if worst > 1e-12:
        print("the serving scorer does not reproduce the fit, so nothing was written")
        print("worst row: {}".format({name: worst_row[name] for name in names}))
        return 2

    print()
    print("{:<24} {:>12}".format("feature", "coefficient"))
    for name in names:
        print("{:<24} {:>12.4f}".format(name, coefficients[name]))
    print("{:<24} {:>12.4f}".format("intercept", model.intercept))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    model_mod.dump(
        model,
        args.out,
        trained={
            "rows": len(table),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_auc": round(train_auc, 4),
            "test_auc": round(test_auc, 4),
            "split_at": cut,
            "corpus": os.path.basename(args.corpus),
        },
    )
    print()
    print("wrote {}".format(args.out))
    print(json.dumps({"test_auc": round(test_auc, 4), "fingerprint": model.fingerprint}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
