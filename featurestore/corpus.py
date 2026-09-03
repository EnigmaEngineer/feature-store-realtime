"""Reading and writing the transaction corpus.

Small enough to look pointless and it is not. The types are the point. `event_ts` has to
come back as an int and `amount` as a float, because everything downstream sorts on one
and sums the other. A CSV round trip that hands both back as strings is the first place
training and serving drift apart, and it does it without raising anything.
"""

import csv

FIELDS = ["txn_id", "account_id", "event_ts", "amount", "merchant_id", "is_fraud"]

TYPES = {
    "txn_id": str,
    "account_id": str,
    "event_ts": int,
    "amount": float,
    "merchant_id": str,
    "is_fraud": int,
}


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_row(row):
    return {name: TYPES[name](row[name]) for name in FIELDS}


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("corpus is missing columns {}".format(sorted(missing)))
        for row in reader:
            yield parse_row(row)
