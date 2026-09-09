"""Whether a feature's distribution has moved, and what a threshold on that is worth.

Three things can be called skew here and only one of them moves.

The Spark job and the reference implementation agree cell for cell on the whole corpus.
That has never once differed, so a monitor over it is a constant and it is handled as one:
`constant_agreement` fires on any disagreement at all rather than on a distance. A band
around a series that has never moved is degenerate, and a degenerate band either refuses to
judge or fires on everything.

The store against the truth at request time does move, and it is the population every
number in this module is measured on. A vector written once and read for hours is wrong on
a growing share of cells, which is not a bug to be detected. It is what a batch loaded
online store is.

The third is the feature distribution itself moving between one window of traffic and the
next, which is what a drift monitor usually means.

The statistic is the population stability index. It is used here because everybody uses it
and because the thresholds everybody uses with it, 0.1 and 0.25, are the thing worth
measuring rather than the thing to copy.

    PSI = sum over bins of (c - r) * ln(c / r)

Two properties decide the whole design.

**Its null value is not zero and it depends on the window size.** Two samples from one
distribution give a positive PSI purely from sampling, and the expected value under no
drift at all is about `(k - 1) / n`. At ten bins and two hundred rows that is 0.045, and at
twenty bins and two hundred rows it is 0.095, which is the industry threshold for moderate
drift reached by having a small window and nothing else. `null_floor` measures it rather
than trusting the algebra, and `predicted_null_mean` is the algebra so the two can be
compared. They disagreed by a factor of two the first time they were put side by side, and
the algebra was the half that was wrong. See that function.

**Its bin count is not the bin count you asked for.** Every count feature in this spec is a
small integer with heavy ties, so quantile edges collapse onto each other and ten requested
bins come back as two or three. Since the null floor scales with `k`, a threshold set per
feature is a different number from a threshold set per project, and the difference is not
small.

An empty current bin is not given an epsilon. The usual fix replaces a zero share with
something like 1e-6, which makes the answer a function of the number somebody picked. A bin
that emptied is reported as its own count and its own verdict.
"""

import math
import random

from featurestore import stats

BIN_TARGET = 10

# Verdicts. `moved` and `bin_emptied` are both drift. They are separate because one is a
# distance and the other is a category disappearing, and a distance cannot express the
# second without an arbitrary constant.
OK = "ok"
MOVED = "moved"
BIN_EMPTIED = "bin_emptied"
CONSTANT_MOVED = "constant_moved"

# Not drift. These say the monitor could not judge, which is a fact about the fit or about
# the window and gets reported once rather than raised as an alert every window. That is
# the answer this repo arrived at after a monitor emitted two alerts per partition
# forever, each one restating that the same signal could not be banded.
UNPROFILED = "unprofiled"
TOO_FEW = "too_few"

DRIFT_VERDICTS = (MOVED, BIN_EMPTIED, CONSTANT_MOVED)

# A window smaller than this is not judged. Chosen before any threshold was fitted, and it
# is here because the null floor at ten bins and twenty rows is 0.45, which is already well
# above the level anybody would set an alert at. Below this the statistic is noise whatever
# threshold sits under it.
MIN_WINDOW = 30


class Profile:
    """What one feature looked like over the reference window.

    `edges` are interior boundaries, so `k` bins need `k - 1` of them. A feature whose
    reference held one distinct value has no edges and carries `constant` instead.
    """

    __slots__ = ("name", "edges", "shares", "n", "constant")

    def __init__(self, name, edges, shares, n, constant=None):
        self.name = name
        self.edges = edges
        self.shares = shares
        self.n = n
        self.constant = constant

    @property
    def bins(self):
        return len(self.shares)

    def is_constant(self):
        return self.constant is not None

    def as_dict(self):
        return {
            "name": self.name,
            "edges": list(self.edges),
            "shares": list(self.shares),
            "n": self.n,
            "bins": self.bins,
            "constant": self.constant,
        }

    def __repr__(self):
        if self.is_constant():
            return "Profile({}, constant {})".format(self.name, self.constant)
        return "Profile({}, {} bins over {} rows)".format(self.name, self.bins, self.n)


def bin_edges(values, target=BIN_TARGET):
    """Interior quantile edges, with duplicates removed.

    Deduplication is the whole content of this function. `txn_count_1h` is zero on most
    rows here, so nine quantile edges come back as nine zeros, and left in place they make
    eight bins that can never hold anything. A bin with no reference mass divides by zero
    in the statistic, and a bin with no mass on either side quietly inflates the requested
    `k` that the null floor is computed from. Both failures are silent.

    The returned list can be empty, which means the values could not be split at all and
    the caller has a constant.
    """
    if target < 2:
        raise ValueError("need at least 2 bins, got {!r}".format(target))
    ordered = sorted(values)
    if not ordered:
        return []
    seen = []
    for i in range(1, target):
        edge = ordered[int(i * len(ordered) / target)]
        if not seen or edge > seen[-1]:
            seen.append(edge)
    # An edge equal to the smallest value makes a first bin that nothing can fall into,
    # because binning is `value < edge`. Same silent empty bin from the other end.
    return [e for e in seen if e > ordered[0]]


def bin_index(value, edges):
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if value < edges[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def bin_counts(values, edges):
    counts = [0] * (len(edges) + 1)
    for value in values:
        counts[bin_index(value, edges)] += 1
    return counts


def to_shares(counts):
    total = sum(counts)
    if total == 0:
        raise ValueError("cannot take shares of an empty count vector")
    return [c / total for c in counts]


def fit_profile(name, values, target=BIN_TARGET):
    """The reference distribution for one feature.

    Raises on an empty reference rather than returning a profile that judges nothing. A
    monitor fitted on no data is the failure this repo keeps meeting from other directions,
    where a check reports clean because it was handed nothing.
    """
    values = list(values)
    if not values:
        raise ValueError("cannot profile {!r} from no values".format(name))
    edges = bin_edges(values, target)
    if not edges:
        return Profile(name, [], [1.0], len(values), constant=values[0])
    counts = bin_counts(values, edges)
    return Profile(name, edges, to_shares(counts), len(values))


def psi(profile, values):
    """The index, the number of bins that emptied, and the bin count it was taken over.

    Bins with no current mass are excluded from the sum and counted. Including them needs
    `ln(0)`, and the epsilon everybody substitutes decides the answer: on the same input a
    floor of 1e-6 and a floor of 1e-4 differ by the ratio of their logarithms, which is a
    factor the data had no say in. `scripts/drift_probe.py` measures that rather than
    describing it.
    """
    if profile.is_constant():
        raise ValueError("{!r} is a constant profile and has no index".format(profile.name))
    counts = bin_counts(values, profile.edges)
    shares = to_shares(counts)
    total = 0.0
    emptied = 0
    used = 0
    for current, reference in zip(shares, profile.shares):
        if reference <= 0.0:
            # `bin_edges` makes this unreachable on a profile this module fitted, because
            # every surviving edge is strictly above the one before it and strictly above
            # the smallest value. It stays because `Profile` is constructible directly and
            # a caller building one by hand would otherwise get a division rather than a
            # count.
            continue
        if current <= 0.0:
            emptied += 1
            continue
        total += (current - reference) * math.log(current / reference)
        used += 1
    return total, emptied, used


def predicted_null_mean(bins, n):
    """What the index reads on a window of `n` rows drawn from the reference itself.

    `(k - 1) / n`, and the first version of this line said `(k - 1) / (2n)`.

    The wrong one is what you get by writing down the standard result and stopping. Twice a
    likelihood ratio statistic is asymptotically chi square on `k - 1` degrees of freedom,
    so a one directional divergence has a mean of `(k - 1) / (2n)`. This index is not one
    directional. Expanding it,

        sum (c - r) ln(c / r)  =  sum c ln(c / r)  +  sum r ln(r / c)

    which is the divergence in both directions added together, so it is twice the standard
    result and not equal to it.

    Nothing about reading the code says which of those two it is. What said so was
    `null_floor` coming back at almost exactly twice the prediction on all seven features
    at once. A multinomial with a known bin count then agreed with `(k - 1) / n` to within
    a few percent at five bin counts and two window sizes. A factor of two here is not
    cosmetic. A threshold set from the wrong form sits at half the level the window will
    produce with nothing wrong at all.

    The mean is here and the upper quantile is not, because that needs a chi square
    critical value and writing one from memory is how a fabricated constant lands in a
    fixture. `null_floor` measures the quantile instead.
    """
    if n <= 0:
        raise ValueError("n must be positive, got {!r}".format(n))
    if bins < 1:
        raise ValueError("bins must be positive, got {!r}".format(bins))
    return (bins - 1) / float(n)


def null_floor(profile, n, trials=200, level=0.95, rng=None, reference_values=None):
    """The index's own distribution when nothing has drifted.

    Windows of `n` rows are resampled from the reference and scored against the reference
    shares. Every value that comes back is sampling noise, because the two sides are the
    same distribution by construction.

    `reference_values` is required and is not the profile, because a profile holds shares
    and shares cannot be resampled at the right granularity. Passing the values in keeps
    the resampling honest about what it is drawing from.
    """
    if profile.is_constant():
        raise ValueError("{!r} is constant and has no null distribution".format(profile.name))
    if reference_values is None:
        raise ValueError("null_floor needs the reference values to resample from")
    values = list(reference_values)
    if not values:
        raise ValueError("cannot resample {!r} from no values".format(profile.name))
    rng = rng or random.Random(0)

    observed = []
    emptied_any = 0
    for _ in range(trials):
        window = [values[rng.randrange(len(values))] for _ in range(n)]
        index, emptied, _ = psi(profile, window)
        observed.append(index)
        if emptied:
            emptied_any += 1
    return {
        "n": n,
        "trials": trials,
        "bins": profile.bins,
        "mean": sum(observed) / len(observed),
        "predicted_mean": predicted_null_mean(profile.bins, n),
        "level": level,
        "quantile": stats.quantile(observed, level),
        "max": max(observed),
        "windows_with_an_empty_bin": emptied_any,
    }


class Monitor:
    """A profile, a threshold, and where the threshold came from.

    `source` is not decoration. A threshold copied from a rule of thumb, a threshold taken
    off the null distribution, and a threshold fitted on the windows it will judge are
    three different claims, and only the first two can be stated without also stating that
    the evidence is in sample.
    """

    __slots__ = ("profile", "threshold", "source")

    def __init__(self, profile, threshold, source):
        if threshold <= 0.0:
            raise ValueError("threshold must be positive, got {!r}".format(threshold))
        self.profile = profile
        self.threshold = threshold
        self.source = source

    def judge(self, values):
        """One verdict for one window. Never raises on a window it cannot judge."""
        values = list(values)
        if len(values) < MIN_WINDOW:
            return {
                "feature": self.profile.name,
                "verdict": TOO_FEW,
                "rows": len(values),
                "index": None,
                "emptied": 0,
            }
        if self.profile.is_constant():
            moved = sum(1 for v in values if v != self.profile.constant)
            return {
                "feature": self.profile.name,
                "verdict": CONSTANT_MOVED if moved else OK,
                "rows": len(values),
                "index": None,
                "emptied": 0,
                "rows_off_the_constant": moved,
            }
        index, emptied, used = psi(self.profile, values)
        if emptied:
            verdict = BIN_EMPTIED
        elif index > self.threshold:
            verdict = MOVED
        else:
            verdict = OK
        return {
            "feature": self.profile.name,
            "verdict": verdict,
            "rows": len(values),
            "index": index,
            "emptied": emptied,
            "bins_used": used,
        }


def fired(verdict):
    return verdict["verdict"] in DRIFT_VERDICTS


def fire_rate(monitor, windows):
    """How often a monitor fires over a list of windows, with an exact lower bound.

    `windows` are lists of values. Whether they are the windows the threshold was fitted
    on is the caller's business and it is the only thing that makes this number mean
    anything, which is why `holdout` is a required argument rather than a default. A rate
    measured on the data a threshold was chosen from is zero by construction often enough
    that it has cost three separate days of work elsewhere.
    """
    judged = [monitor.judge(w) for w in windows]
    scored = [v for v in judged if v["verdict"] not in (TOO_FEW, UNPROFILED)]
    fires = sum(1 for v in scored if fired(v))
    return {
        "feature": monitor.profile.name,
        "windows": len(windows),
        "judged": len(scored),
        "fires": fires,
        "rate": fires / len(scored) if scored else None,
        "lower_bound": stats.binomial_lower_bound(fires, len(scored)) if scored else None,
        "verdicts": judged,
    }


def constant_agreement(left, right, names):
    """The monitor for a signal that has never moved: any disagreement is the alert.

    `left` and `right` are lists of feature dicts in the same order, one per row. This is
    the Spark job against the reference implementation, and the whole corpus has come back
    identical every time it has been run. A distance on that is a band around a constant,
    which either refuses to judge or fires on everything, and neither is useful. The
    signals that come out flat are the ones whose first movement matters most.
    """
    if len(left) != len(right):
        raise ValueError(
            "cannot compare {} rows against {}".format(len(left), len(right))
        )
    cells = 0
    differing = 0
    by_feature = {}
    for a, b in zip(left, right):
        for name in names:
            cells += 1
            if a[name] != b[name]:
                differing += 1
                by_feature[name] = by_feature.get(name, 0) + 1
    return {
        "cells": cells,
        "differing": differing,
        "by_feature": by_feature,
        "verdict": OK if differing == 0 else CONSTANT_MOVED,
    }
