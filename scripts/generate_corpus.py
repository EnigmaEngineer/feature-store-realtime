"""Build the transaction corpus and print what it actually contains.

    python scripts/generate_corpus.py

Writes data/transactions.csv, which is gitignored, so every figure taken off it has to
name this command beside it or nobody can check the figure.

The summary is not decoration. The first version of this generator drew an account at
random for every transaction and produced a corpus where no account ever had a trailing
window, which read fine in the source and was obvious in one line of the summary.
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import corpus as corpus_io
from featurestore import generate


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/transactions.csv")
    ap.add_argument("--accounts", type=int, default=4000)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args(argv)

    corpus = generate.generate(n_accounts=args.accounts, days=args.days, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    corpus_io.write_csv(corpus.transactions, args.out)

    summary = corpus.summary()
    summary["file"] = args.out
    summary["sha256_12"] = digest(args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["skipped_too_thin"]:
        print(
            "note: {} of {} picked accounts were too quiet to carry a burst and were "
            "skipped".format(summary["skipped_too_thin"], summary["compromise_target"])
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
