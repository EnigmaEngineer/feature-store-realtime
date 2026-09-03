"""A synthetic card transaction log.

Nothing in this repo runs on real payment data and nothing pretends to. Every number
this project publishes about accuracy is a number about this generator, and the README
says so next to the numbers rather than in a footnote.

The corpus exists because the thing under test needs event timestamps and an entity
history. A feature store is only interesting when a feature has to be recomputed as of
a moment in the past, and the small public tabular datasets have no such moment in them.

Shape of the process. Each account transacts at its own rate with a diurnal swing. A
minority of accounts get compromised once, and a compromise is a short burst of unusually
frequent transactions at unfamiliar merchants. The features the model sees are counts
and sums over a trailing window, which is not the form the generator uses, so a good fit
is not automatic.
"""

import math
import random

HOUR = 3600
DAY = 86400

# Rough diurnal multiplier by hour of day, quiet overnight and a bump at lunch and in
# the evening. Hand chosen. It is here so the calendar_day window reading has something
# to disagree with, not because it is a claim about real card traffic.
DIURNAL = [
    0.25, 0.15, 0.10, 0.10, 0.15, 0.30, 0.60, 0.95,
    1.10, 1.15, 1.20, 1.45, 1.60, 1.35, 1.20, 1.25,
    1.40, 1.65, 1.70, 1.50, 1.20, 0.90, 0.60, 0.40,
]
DIURNAL_MAX = max(DIURNAL)


class Corpus:
    """What a run produced, as opposed to what it was asked for.

    Every count here is measured off the rows. The compromise target is a request and
    `compromised_accounts` is what the draw actually produced, because an account with
    too few transactions to hold a burst is skipped rather than forced.
    """

    def __init__(self, transactions, params, compromised, skipped_thin):
        self.transactions = transactions
        self.params = params
        self.compromised_accounts = compromised
        self.skipped_thin = skipped_thin

    def summary(self):
        rows = self.transactions
        fraud = [r for r in rows if r["is_fraud"] == 1]
        per_account = {}
        for r in rows:
            per_account[r["account_id"]] = per_account.get(r["account_id"], 0) + 1
        counts = sorted(per_account.values())

        bursts = {}
        for r in fraud:
            bursts.setdefault(r["account_id"], []).append(r["event_ts"])
        burst_sizes = sorted(len(v) for v in bursts.values())
        first_of_burst = len(bursts)

        return {
            "accounts": len(per_account),
            "transactions": len(rows),
            "fraud_transactions": len(fraud),
            "fraud_rate": len(fraud) / len(rows) if rows else 0.0,
            "compromised_accounts": self.compromised_accounts,
            "compromise_target": self.params["compromised_target"],
            "skipped_too_thin": self.skipped_thin,
            "median_txns_per_account": _median(counts),
            "min_txns_per_account": counts[0] if counts else 0,
            "max_txns_per_account": counts[-1] if counts else 0,
            "median_burst_size": _median(burst_sizes),
            # The first transaction of a burst has an ordinary trailing window behind it.
            # No count or sum over prior events can separate it from a normal purchase,
            # so this is a floor on what the model is going to miss.
            "fraud_that_opens_a_burst": first_of_burst,
            "share_of_fraud_that_opens_a_burst": first_of_burst / len(fraud) if fraud else 0.0,
            "span_days": self.params["days"],
        }


def _median(values):
    if not values:
        return 0
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _thinned_arrivals(rng, rate_per_day, start, end):
    """Non homogeneous Poisson by thinning against the diurnal curve."""
    out = []
    peak = rate_per_day * DIURNAL_MAX / DAY
    if peak <= 0:
        return out
    t = start
    while True:
        t += rng.expovariate(peak)
        if t >= end:
            return out
        hour = int(t // HOUR) % 24
        if rng.random() < DIURNAL[hour] / DIURNAL_MAX:
            out.append(t)


def generate(
    n_accounts=4000,
    days=60,
    seed=20260903,
    compromise_share=0.02,
    burst_minutes=90,
    burst_rate_multiplier=40.0,
    n_merchants=600,
    start_ts=1750000000,
):
    rng = random.Random(seed)
    merchants = ["m{:04d}".format(i) for i in range(n_merchants)]
    end_ts = start_ts + days * DAY

    rows = []
    accounts = []
    for i in range(n_accounts):
        aid = "a{:06d}".format(i)
        rate = math.exp(rng.gauss(0.5, 0.8))  # transactions per day, heavy right tail
        favourites = rng.sample(merchants, rng.randint(3, 9))
        amount_mu = rng.gauss(3.2, 0.5)
        accounts.append((aid, rate, favourites, amount_mu))
        for t in _thinned_arrivals(rng, rate, start_ts, end_ts):
            rows.append(
                {
                    "txn_id": None,
                    "account_id": aid,
                    "event_ts": t,
                    "amount": round(math.exp(rng.gauss(amount_mu, 0.6)), 2),
                    "merchant_id": rng.choice(favourites),
                    "is_fraud": 0,
                }
            )

    target = int(round(n_accounts * compromise_share))
    picked = rng.sample(accounts, min(target, len(accounts)))
    compromised = 0
    skipped_thin = 0
    burst_s = burst_minutes * 60

    for aid, rate, favourites, amount_mu in picked:
        # A burst needs somewhere to sit and it needs a bit of prior history behind it,
        # otherwise the trailing window is empty for a reason that has nothing to do
        # with fraud. Accounts too quiet to give it that are skipped and counted.
        if rate < 0.5:
            skipped_thin += 1
            continue
        burst_start = rng.uniform(start_ts + 5 * DAY, end_ts - burst_s)
        arrivals = _thinned_arrivals(
            rng, rate * burst_rate_multiplier, burst_start, burst_start + burst_s
        )
        if len(arrivals) < 2:
            skipped_thin += 1
            continue
        compromised += 1
        for t in arrivals:
            rows.append(
                {
                    "txn_id": None,
                    "account_id": aid,
                    "event_ts": t,
                    # A compromised card gets spent harder and in places the account
                    # has never used.
                    "amount": round(math.exp(rng.gauss(amount_mu + 0.9, 0.7)), 2),
                    "merchant_id": rng.choice(merchants),
                    "is_fraud": 1,
                }
            )

    rows.sort(key=lambda r: (r["event_ts"], r["account_id"]))
    for i, r in enumerate(rows):
        r["txn_id"] = "t{:08d}".format(i)
        r["event_ts"] = int(r["event_ts"])

    params = {
        "n_accounts": n_accounts,
        "days": days,
        "seed": seed,
        "compromise_share": compromise_share,
        "compromised_target": target,
        "burst_minutes": burst_minutes,
        "burst_rate_multiplier": burst_rate_multiplier,
        "n_merchants": n_merchants,
        "start_ts": start_ts,
        "end_ts": end_ts,
    }
    return Corpus(rows, params, compromised, skipped_thin)
