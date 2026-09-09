"""Exact binomial intervals, and the two functions underneath them.

This was inside `scripts/serve_probe.py` until the drift monitor needed the same interval
to say what a held out fire rate is worth. Two copies of a Clopper Pearson interval is the
shape where each half is right on its own and the pair disagrees on some input nobody
tried. It is also the rule this repo already applied to `quantiles`, which moved into
`featurestore/load.py` when a second module wanted a p99.

There is a second reason and it is the stronger one. An interval here decides whether a
threshold ships, so it is a measurement that decides a design, and those live where a
mutation pass can reach them rather than in a script.

No scipy. It is a scikit-learn dependency rather than one this repo declares, and the
probes that print these numbers run on a clone that installed `requirements.txt` alone.
"""

import math


def betainc(a, b, x):
    """The regularised incomplete beta function, by continued fraction.

    Here rather than a direct binomial sum because the direct sum underflows. The first
    version of this summed the binomial tail term by term starting from (1-p)**n, which
    is zero in a double for n of 48,998, so every evaluation returned 0 and the interval
    came back as 1.0000 to 1.0000 on every row with a large denominator. It printed a
    plausible number on the small rows, which is why it survived a first look.

    Lentz's method on the standard continued fraction.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - betainc(b, a, 1.0 - x)

    tiny = 1e-30
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        step = c * d
        f *= step
        if abs(1.0 - step) < 1e-14:
            return front * (f - 1.0) / a
    # Silently returning an unconverged value is how a numeric routine ships a wrong
    # number that looks like a right one. This one already did that once, in a different
    # way, and the interval it feeds is printed in a table.
    raise ArithmeticError(
        "the incomplete beta did not converge at a={}, b={}, x={}".format(a, b, x)
    )


def beta_quantile(p, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def binomial_interval(k, n, level=0.95):
    """Clopper Pearson, through the beta quantile it is exactly equal to.

    A threshold that serves forty requests reports a rate whose third decimal is noise,
    and printing 0.0370 beside 0.0785 invites a reader to treat the gap as settled. The
    interval is what the count licenses and nothing more.
    """
    if n == 0:
        return 0.0, 1.0
    alpha = 1.0 - level
    low = 0.0 if k == 0 else beta_quantile(alpha / 2.0, k, n - k + 1)
    high = 1.0 if k == n else beta_quantile(1.0 - alpha / 2.0, k + 1, n - k)
    return low, high


def binomial_lower_bound(k, n, level=0.95):
    """The one sided exact lower bound on a rate.

    Separate from `binomial_interval` because the two answer different questions and
    reading the lower end of a two sided interval as a one sided bound is off by the
    tail. A fire rate measured on ten windows is the case this exists for. Three fires
    out of ten prints as 0.300, which invites a reader to dismiss it, and the one sided
    bound is what says three out of ten still rules out a quiet monitor.
    """
    if n == 0:
        return 0.0
    if k == 0:
        return 0.0
    return beta_quantile(1.0 - level, k, n - k + 1)


def quantile(values, q):
    """Nearest rank, on an already materialised list.

    `featurestore.load.quantiles` returns a fixed three for a latency table. This one
    takes the level, because a null distribution is summarised at whatever level the
    threshold is being set at. The two share a definition on purpose: no interpolation,
    index `int(q * n)` with q strictly below 1.
    """
    if not values:
        raise ValueError("no values to take a quantile of")
    if not 0.0 <= q < 1.0:
        raise ValueError("q must be in [0, 1), got {!r}".format(q))
    ordered = sorted(values)
    return ordered[int(q * len(ordered))]
