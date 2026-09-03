# Choosing the corpus and the model

## Status

Accepted.

## The corpus

A feature store is only interesting when a feature has to be recomputed as of a moment in
the past. That needs an entity with a history and an event with a timestamp. The small
public tabular datasets are one row per subject with no clock in them, so none of them can
exercise the thing this repo is about.

So the corpus is generated. `featurestore/generate.py` produces a card transaction log
where each account transacts at its own rate with a diurnal swing, and a minority of
accounts get compromised once. A compromise is a short burst of unusually frequent
transactions at merchants the account has never used.

Every accuracy number here is therefore a number about that generator. The README says so
next to the figures rather than in a footnote at the bottom.

Two things keep this from being circular.

The generator is multiplicative. A rate is scaled by a diurnal factor and thinned. The
features are counts and sums over a trailing window, which is a different functional form,
so a good fit is not automatic.

The summary is the check on the generator, not the source. Reading the code does not tell
you what it produces. An earlier draft drew an account at random for every transaction and
looked correct, and it produced a corpus where almost no account had a trailing window at
all. One line of the summary said so.

The corpus is gitignored, so it carries the command that rebuilds it wherever a figure
taken off it appears.

    python scripts/generate_corpus.py

## The model

Logistic regression, from scikit-learn.

The subject here is the serving path, not the model. Three reasons that points at a linear
model rather than a gradient boosted tree.

Inference has to be cheap enough that the latency budget is dominated by the feature fetch.
If the model is the expensive part then every latency number becomes a statement about the
model library instead of about the store.

A scale error stays attributable. If a feature arrives at serving time with the wrong
scale, the score moves by the coefficient times the difference, and that is a number you
can compute and check. A tree ensemble absorbs the same error into a split and the score
moves by an amount nobody can derive by hand.

It serialises to a handful of floats, so there is no model artefact version story sitting
between training and serving to confuse a skew investigation.

## What is wrong with both choices

A gradient boosted tree would beat this on real fraud data and probably on this corpus too.
That comparison is not run, because on a synthetic corpus it would measure the generator
rather than the models.

The corpus makes compromise an account level property. An oracle that scores every
transaction with its own account's fraud rate reaches 0.9982 AUC on the held out split,
measured by `scripts/leakage_probe.py`. So any feature that leaks account identity is
worth a great deal here, more than it would be on a real portfolio where accounts are
compromised and recover. That inflates the value of the point in time discipline this repo
is about, and it is named here rather than left to be discovered.
