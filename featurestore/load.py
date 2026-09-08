"""Turning a pile of finished requests into a latency number, and checking the load landed.

`scripts/load_probe.py` drives the service and this module grades what came back. The split
is the one the rest of this repo makes. A decision rule that a published figure rests on
lives in the package where a mutant can reach it, and the script holds the sockets and the
printing.

Three things here are worth reading.

`quantiles` moved here out of `scripts/serve_probe.py`. Two modules were about to report a
p99 and only one of them had a definition of it. A quantile has more than one correct
definition and two of them meeting in one table is the kind of disagreement nobody looks
for, because each half is right on its own.

`offered_concurrency` is the control on the driver rather than a result about the server. A
closed loop driver keeps exactly one request in flight per worker, so the mean number in
flight has to come back at the worker count. If it comes back low then the driver spent its
time somewhere other than waiting on the server, and the arms are being compared on the
driver instead. Little's law gives the number for free from timestamps already collected.

`compare` refuses rather than reporting when either arm failed that control. A throughput
ratio measured off a driver that could not keep the load up is a number about this machine's
GIL, and it would read exactly like a number about the server.

`compare_reps` is the one that decides anything. The first shape of this file compared one
pass of each arm and the ratio it printed was smaller than the spread between two passes of
the same arm. So the comparison is paired across repetitions and it is a sign test, and
`p_floor` is checked before any of it runs. Six pairs is the smallest number that can reach
a p below 0.05, so five repetitions is a comparison that cannot be won at any effect size.
"""

import math

# A driver that keeps this fraction of its workers in flight is driving the server. Below
# it, the thing being measured is the driver. Not tuned. It is a round number chosen before
# the first arm ran, and both arms are held to it equally so it cannot pick a winner.
LANDED_FLOOR = 0.90

# The level a paired comparison has to reach before it is reported as decided. Fixed here
# rather than passed in, so no caller can lower it after seeing a result.
ALPHA = 0.05


def quantiles(values):
    """Three quantiles and the largest value. Nearest rank.

    No interpolation. The samples are milliseconds off a clock with far more resolution
    than the spread between neighbouring points, so an interpolated quantile would be
    inventing precision the measurement does not carry.
    """
    ordered = sorted(values)

    def at(q):
        # No clamp on the index. For any q below 1, int(q * n) is an integer strictly less
        # than n, so it is already inside the list. The first version of this held a
        # min(n - 1, ...) and a mutant turned it into min(n + 1, ...) and survived every
        # check, because the clamp could never bind and was doing nothing. The largest
        # value is returned separately below rather than as a quantile at 1.
        if not ordered:
            return 0.0
        return ordered[int(q * len(ordered))]

    return at(0.50), at(0.95), at(0.99), ordered[-1] if ordered else 0.0


def offered_concurrency(durations_ms, wall_s):
    """Mean requests in flight, from the durations and the wall clock.

    Little's law. The sum of the time each request spent outstanding, over the time the
    whole pass took. A closed loop driver with N workers should land on N.
    """
    if wall_s <= 0:
        raise ValueError("wall_s must be positive, got {!r}".format(wall_s))
    return sum(durations_ms) / 1000.0 / wall_s


def load_landed(durations_ms, wall_s, workers, floor=LANDED_FLOOR):
    """Did the driver really keep `workers` requests outstanding.

    Returns (landed, ratio). The ratio is what was achieved over what was asked for. A pass
    that comes back at 0.4 was not a test of the server at 32 clients. It was a test of one
    process trying to be 32 clients.
    """
    if workers <= 0:
        raise ValueError("workers must be positive, got {!r}".format(workers))
    ratio = offered_concurrency(durations_ms, wall_s) / workers
    return ratio >= floor, ratio


def summarise(durations_ms, wall_s, workers, statuses=None):
    """One arm of a load pass, as numbers.

    `throughput_rps` is requests completed over wall clock. It is the figure a capacity
    argument is made from and the quantiles are the figure a latency budget is made from,
    and the two answer different questions.
    """
    p50, p95, p99, worst = quantiles(durations_ms)
    # This raises on a wall clock of zero, which is why the division below has no guard on
    # it. A guard there would be unreachable and a mutant removing it survived.
    landed, ratio = load_landed(durations_ms, wall_s, workers)
    return {
        "n": len(durations_ms),
        "workers": workers,
        "wall_s": wall_s,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": worst,
        "throughput_rps": len(durations_ms) / wall_s,
        "offered": offered_concurrency(durations_ms, wall_s),
        "offered_ratio": ratio,
        "landed": landed,
        "statuses": dict(statuses or {}),
    }


def all_ok(arm, ok_status=200):
    """Every request in this arm came back with the status the comparison assumes.

    A load pass where a fraction of requests 404 is faster than one where they all score,
    and nothing in a latency table says which happened. This is asserted rather than
    reported, because an arm that served errors is not an arm.

    An arm of no requests is not ok either, and the first version of this said it was. With
    zero requests the status map is `{200: 0}`, which matched the expected set and matched
    the count, so a pass that served nothing at all cleared the control that exists to say
    whether it served the right thing. That is the same shape as a checker reporting clean
    because it was pointed at no files.
    """
    statuses = arm.get("statuses") or {}
    if arm["n"] <= 0:
        return False
    return set(statuses) == {ok_status} and statuses.get(ok_status) == arm["n"]


def compare(name_a, arm_a, name_b, arm_b):
    """Two arms, or a refusal naming which control failed.

    Returns (verdict, detail). The verdict is `ok`, `driver_bound` or `mixed_statuses`.
    Anything other than `ok` means the numbers underneath it are not a statement about the
    server, and the caller prints the reason instead of the ratio.
    """
    if arm_a["workers"] != arm_b["workers"]:
        raise ValueError("arms ran at different concurrency, nothing to compare")

    for name, arm in ((name_a, arm_a), (name_b, arm_b)):
        if not arm["landed"]:
            return "driver_bound", "{} offered {:.2f} of {} workers".format(
                name, arm["offered"], arm["workers"]
            )
    for name, arm in ((name_a, arm_a), (name_b, arm_b)):
        if not all_ok(arm):
            return "mixed_statuses", "{} served {}".format(name, arm["statuses"])

    return "ok", {
        "throughput_ratio": arm_a["throughput_rps"] / arm_b["throughput_rps"]
        if arm_b["throughput_rps"] > 0
        else float("inf"),
        "p99_ratio": arm_a["p99_ms"] / arm_b["p99_ms"] if arm_b["p99_ms"] > 0 else float("inf"),
    }


def p_floor(pairs):
    """The smallest p value this many pairs could ever return.

    A sign test on n pairs is decided by n bits, so the two sided tail cannot go below
    2 / 2**n however large the effect is. Checking it before collecting anything is how you
    find out that a comparison is unwinnable while it is still free to fix.
    """
    if pairs <= 0:
        return 1.0
    return min(1.0, 2.0 / (2 ** pairs))


def sign_test(wins, pairs):
    """Exact two sided sign test. Ties are dropped by the caller before this is called."""
    if pairs <= 0:
        return 1.0
    if not 0 <= wins <= pairs:
        raise ValueError("wins must be between 0 and pairs, got {} of {}".format(wins, pairs))
    edge = max(wins, pairs - wins)
    tail = sum(math.comb(pairs, i) for i in range(edge, pairs + 1))
    return min(1.0, 2.0 * tail / float(2 ** pairs))


def median(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values to take a median of")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def compare_reps(name_a, reps_a, name_b, reps_b, key="throughput_rps", higher_is_better=True):
    """Two arms over repeated rounds, paired by round.

    Returns (verdict, detail). `ok` means the sign test cleared ALPHA and the direction is
    in the detail. `underpowered` means the pair count could not have reached it, which is a
    fact about the design rather than about the arms. `driver_bound` and `mixed_statuses`
    mean a control failed on some round and nothing here is a statement about the server.

    Pairing by round rather than pooling, because the arms in one round ran on the same
    machine in the same minute and the machine is the thing that moves between rounds.
    """
    if len(reps_a) != len(reps_b):
        raise ValueError("arms ran a different number of rounds, nothing to pair")

    for name, reps in ((name_a, reps_a), (name_b, reps_b)):
        for i, arm in enumerate(reps):
            if not arm["landed"]:
                return "driver_bound", "{} round {} offered {:.2f} of {} workers".format(
                    name, i, arm["offered"], arm["workers"]
                )
            if not all_ok(arm):
                return "mixed_statuses", "{} round {} served {}".format(name, i, arm["statuses"])

    rounds = len(reps_a)
    if p_floor(rounds) > ALPHA:
        return "underpowered", (
            "{} rounds can reach p {:.4f} at best, which is above {}".format(
                rounds, p_floor(rounds), ALPHA
            )
        )

    wins = 0
    pairs = 0
    for a, b in zip(reps_a, reps_b):
        if a[key] == b[key]:
            continue
        pairs += 1
        better = a[key] > b[key] if higher_is_better else a[key] < b[key]
        if better:
            wins += 1

    if p_floor(pairs) > ALPHA:
        return "underpowered", (
            "{} of {} rounds tied, leaving {} pairs which can reach p {:.4f} at best".format(
                rounds - pairs, rounds, pairs, p_floor(pairs)
            )
        )

    p = sign_test(wins, pairs)
    med_a = median([r[key] for r in reps_a])
    med_b = median([r[key] for r in reps_b])
    return "ok", {
        "key": key,
        "rounds": rounds,
        "pairs": pairs,
        "wins": wins,
        "p": p,
        "decided": p < ALPHA,
        "median_a": med_a,
        "median_b": med_b,
        "ratio_of_medians": med_a / med_b if med_b else float("inf"),
        "spread_a": (min(r[key] for r in reps_a), max(r[key] for r in reps_a)),
        "spread_b": (min(r[key] for r in reps_b), max(r[key] for r in reps_b)),
    }
