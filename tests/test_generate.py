"""Checks on the corpus generator.

Reading a generator does not tell you what it produces. These assert on the summary,
because the first version of this file looked correct and produced a corpus where no
account ever had more than one event, which the summary said in one line.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featurestore import compute, corpus, generate

SMALL = dict(n_accounts=300, days=20, seed=7)


def check_the_same_seed_gives_the_same_corpus():
    a = generate.generate(**SMALL).transactions
    b = generate.generate(**SMALL).transactions
    assert a == b
    c = generate.generate(n_accounts=300, days=20, seed=8).transactions
    assert a != c


def check_accounts_have_histories_rather_than_one_event_each():
    """The defect the first generator had. One event per account makes every trailing
    window empty, and every feature in this project is a trailing window."""
    s = generate.generate(**SMALL).summary()
    assert s["median_txns_per_account"] > 5, s["median_txns_per_account"]
    assert s["min_txns_per_account"] >= 1


def check_the_compromise_target_is_reported_and_not_clamped():
    """A knob the draw cannot always hit reports what it achieved.

    Accounts too quiet to hold a burst are skipped rather than forced, so the achieved
    count is below the target and the difference is counted rather than hidden.
    """
    s = generate.generate(**SMALL).summary()
    assert s["compromised_accounts"] + s["skipped_too_thin"] == s["compromise_target"]
    assert s["compromised_accounts"] <= s["compromise_target"]


def check_fraud_is_rare_and_bursty():
    s = generate.generate(**SMALL).summary()
    assert 0 < s["fraud_rate"] < 0.02, s["fraud_rate"]
    assert s["median_burst_size"] >= 2


def check_the_share_that_opens_a_burst_is_reported():
    """The first transaction of a burst has ordinary history behind it, so no trailing
    count can catch it. That share is a floor on what the model will miss and it belongs
    in the summary rather than in a sentence somebody wrote once."""
    s = generate.generate(**SMALL).summary()
    assert s["fraud_that_opens_a_burst"] == s["compromised_accounts"]
    assert 0 < s["share_of_fraud_that_opens_a_burst"] < 1


def check_fraud_rows_really_sit_in_bursts():
    """Assert on the rows, not on the summary that was computed from them."""
    c = generate.generate(**SMALL)
    fraud = [r for r in c.transactions if r["is_fraud"] == 1]
    assert fraud
    by_account = {}
    for r in fraud:
        by_account.setdefault(r["account_id"], []).append(r["event_ts"])
    for times in by_account.values():
        span = max(times) - min(times)
        assert span <= 90 * 60, span


def check_a_compromised_account_has_normal_traffic_too():
    """Otherwise the trailing window before a burst is empty for the wrong reason and
    the model would be separating accounts with history from accounts without."""
    c = generate.generate(**SMALL)
    fraud_accounts = {r["account_id"] for r in c.transactions if r["is_fraud"] == 1}
    clean_counts = {}
    for r in c.transactions:
        if r["is_fraud"] == 0 and r["account_id"] in fraud_accounts:
            clean_counts[r["account_id"]] = clean_counts.get(r["account_id"], 0) + 1
    assert set(clean_counts) == fraud_accounts
    assert min(clean_counts.values()) > 0


def check_rows_are_sorted_and_ids_are_unique():
    rows = generate.generate(**SMALL).transactions
    times = [r["event_ts"] for r in rows]
    assert times == sorted(times)
    assert len({r["txn_id"] for r in rows}) == len(rows)


def check_the_csv_round_trip_keeps_the_types():
    """A corpus that comes back as strings sorts wrong and sums wrong, and raises
    nothing on the way."""
    import tempfile

    rows = generate.generate(n_accounts=40, days=5, seed=3).transactions
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.csv")
        corpus.write_csv(rows, path)
        back = list(corpus.read_csv(path))
    assert len(back) == len(rows)
    assert isinstance(back[0]["event_ts"], int)
    assert isinstance(back[0]["amount"], float)
    assert isinstance(back[0]["is_fraud"], int)
    assert [r["txn_id"] for r in back] == [r["txn_id"] for r in rows]


def check_a_corpus_missing_a_column_is_refused():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.csv")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("txn_id,account_id\nt1,a1\n")
        try:
            list(corpus.read_csv(path))
        except ValueError as exc:
            assert "missing columns" in str(exc)
            return
    raise AssertionError("a corpus missing four columns was accepted")


def check_the_median_helper():
    """Two published figures come out of this and nothing was pinning it.

    A mutation pass changed `n // 2` to `n // 3` and `values[mid - 1]` to `values[mid - 2]`
    and the whole suite stayed green, because every other check reads the median through
    an inequality. The even case needs two different middles or the mutant survives here
    too.
    """
    assert generate._median([]) == 0
    assert generate._median([5]) == 5
    assert generate._median([1, 2, 3]) == 2
    assert generate._median([1, 2, 3, 9]) == 2.5
    assert generate._median([1, 2, 3, 4, 5, 6]) == 3.5
    assert generate._median([0, 0, 0, 10]) == 0


def check_the_summary_counts_what_is_in_front_of_it():
    """A hand built corpus with counts nobody has to derive.

    Three accounts. `a` has four rows and two of them are fraud, `b` has two rows and
    `c` has one. So the counts are 7 and 2, the median is 2 and the range is 1 to 4.
    """
    rows = (
        [_row("a", 100 + i, i % 2) for i in range(4)]
        + [_row("b", 200 + i, 0) for i in range(2)]
        + [_row("c", 300, 0)]
    )
    c = generate.Corpus(rows, {"compromised_target": 5, "days": 3}, 4, 1)
    s = c.summary()
    assert s["accounts"] == 3
    assert s["transactions"] == 7
    assert s["fraud_transactions"] == 2
    assert s["min_txns_per_account"] == 1
    assert s["max_txns_per_account"] == 4
    assert s["median_txns_per_account"] == 2
    assert s["fraud_rate"] == 2 / 7
    assert s["compromised_accounts"] == 4
    assert s["compromise_target"] == 5
    assert s["skipped_too_thin"] == 1
    assert s["span_days"] == 3
    # One account carries both fraud rows, so there is one burst of size two and it
    # opens with one of them.
    assert s["median_burst_size"] == 2
    assert s["fraud_that_opens_a_burst"] == 1
    assert s["share_of_fraud_that_opens_a_burst"] == 0.5


def check_the_summary_survives_an_empty_corpus():
    """The zero fallbacks in summary() were unreachable from any other check."""
    s = generate.Corpus([], {"compromised_target": 0, "days": 1}, 0, 0).summary()
    assert s["accounts"] == 0
    assert s["transactions"] == 0
    assert s["fraud_rate"] == 0.0
    assert s["share_of_fraud_that_opens_a_burst"] == 0.0
    assert s["median_txns_per_account"] == 0
    assert s["min_txns_per_account"] == 0
    assert s["max_txns_per_account"] == 0


def _row(account, ts, fraud):
    return {
        "txn_id": "{}{}".format(account, ts),
        "account_id": account,
        "event_ts": ts,
        "amount": 1.0,
        "merchant_id": "m",
        "is_fraud": fraud,
    }


def check_the_corpus_spans_the_days_it_was_asked_for():
    """Pins the day constant. Without it a mutant making a day 86401 seconds long
    changes the horizon and nothing notices."""
    c = generate.generate(n_accounts=200, days=10, seed=11)
    times = [r["event_ts"] for r in c.transactions]
    span = max(times) - min(times)
    assert 9 * 86400 <= span <= 10 * 86400, span
    assert min(times) >= c.params["start_ts"]
    assert max(times) < c.params["end_ts"]
    # The horizon exactly, not an inequality. Asserting only that no row lands past the
    # end lets a day be 86401 seconds long, because nothing ever lands in the last ten.
    assert c.params["end_ts"] - c.params["start_ts"] == 10 * 86400


def check_a_fixed_seed_produces_exactly_this_corpus():
    """A golden value over the whole draw, deliberately brittle.

    Every other check here is an inequality, so the arrival process could be rewritten
    underneath them and stay green. A mutation pass moved the thinning acceptance test,
    the diurnal ratio and the burst floor and none of the inequalities noticed.

    If this fails after an intentional change to the generator, that is the check working.
    Every published figure comes off this draw, so a change to it invalidates all of them
    and the new values go in here at the same time as the figures get re-run.
    """
    import hashlib

    c = generate.generate(n_accounts=200, days=10, seed=11)
    s = c.summary()
    assert s["transactions"] == 3948, s["transactions"]
    assert s["fraud_transactions"] == 18, s["fraud_transactions"]
    assert s["compromised_accounts"] == 4
    assert s["skipped_too_thin"] == 0
    assert s["max_txns_per_account"] == 97
    assert s["median_txns_per_account"] == 16.0

    blob = "".join(
        "{}|{}|{}|{}|{}".format(
            r["txn_id"], r["account_id"], r["event_ts"], r["amount"], r["merchant_id"]
        )
        for r in c.transactions
    )
    assert hashlib.sha256(blob.encode()).hexdigest()[:16] == "f7e027e063f318b1"


def check_the_defaults_are_the_ones_the_figures_were_measured_on():
    """The published corpus is built by calling generate() with no arguments.

    Nothing else in this file exercises a default, because every check passes explicit
    arguments to keep itself fast. So a silent edit to the seed or the horizon would move
    every number this repo publishes with nothing to say it had.
    """
    import inspect

    defaults = {
        name: p.default
        for name, p in inspect.signature(generate.generate).parameters.items()
    }
    assert defaults == {
        "n_accounts": 4000,
        "days": 60,
        "seed": 20260903,
        "compromise_share": 0.02,
        "burst_minutes": 90,
        "burst_rate_multiplier": 40.0,
        "n_merchants": 600,
        "start_ts": 1750000000,
    }, defaults


def check_a_zero_rate_returns_nothing_rather_than_dividing_by_zero():
    """The guard in _thinned_arrivals is unreachable from the shipped call sites.

    Every rate the generator passes comes out of math.exp, so it is always positive.
    Removing the guard changes nothing today and turns a silent empty result into a
    ZeroDivisionError the moment somebody calls the helper with a rate of nought. It
    stays, and this is what holds it.
    """
    import random

    assert generate._thinned_arrivals(random.Random(1), 0, 0, 10000) == []
    assert generate._thinned_arrivals(random.Random(1), -1, 0, 10000) == []


def check_the_event_log_indexes_what_the_generator_produced():
    c = generate.generate(**SMALL)
    log = compute.EventLog(c.transactions, "account_id", "event_ts")
    assert len(log.entities()) == c.summary()["accounts"]
    busiest = max(log.entities(), key=lambda e: len(log.rows_for(e)))
    rows = log.rows_for(busiest)
    assert [r["event_ts"] for r in rows] == sorted(r["event_ts"] for r in rows)
