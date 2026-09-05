"""Turning a window from the spec into a Spark frame, or refusing to.

A Spark RANGE frame is two constant offsets from the current row's ordering value. A
window in this spec is a wider thing than that, so the translation is not always
possible, and the gap is real rather than theoretical.

The offsets are therefore not written out by hand here. They are read back out of
`compute.window_bounds`, which is the reference implementation, at several probe
timestamps. If every probe yields the same pair of offsets the window is a fixed offset
frame and the pair is what Spark gets. If the pair moves between probes then the window
is anchored on something other than the scoring event, and no constant frame expresses
it.

Writing the offsets by hand would have made the Spark job and the oracle two
independent readings of one English sentence. Then a comparison between them measures
whether I made the same mistake twice.
"""

from featurestore import compute

DAY = compute.DAY

# Probes have to disagree on the time of day or a calendar aligned window looks like a
# fixed offset. Probes an exact day apart are the trap, because they agree on the time
# of day and look varied. `probes_differ_in_time_of_day` is what stops a later edit
# quietly removing the only thing that makes the refusal work.
PROBES = (
    1756944000,  # midnight UTC
    1756944000 + 1,
    1756944000 + 3600,
    1756944000 + 43200,
    1756944000 + DAY - 1,
    1756944000 + DAY + 17,
)


class UnsupportedWindow(ValueError):
    """The window is legal in the spec and has no constant frame behind it."""


def probes_differ_in_time_of_day(probes):
    """True when the probes sit at more than one time of day.

    A probe set that fails this cannot tell a calendar aligned window from an event
    aligned one, so `frame_offsets` refuses rather than returning a pair that happens
    to be right for one moment.
    """
    return len({p % DAY for p in probes}) >= 2


def frame_disagreement(window, probes=PROBES):
    """Every frame the probes produced, keyed by frame and listing the probes that gave it.

    The arithmetic lives here and nowhere else. It was written out twice at first, once
    here and once in `frame_offsets`, and a mutation pass then killed the copy under the
    heavier tests while the other one sat there with the same subtractions in it.
    """
    seen = {}
    for as_of in probes:
        lo, hi, hi_inclusive = compute.window_bounds(as_of, window)
        # Spark's rangeBetween includes both ends. The reference implementation's upper
        # bound is exclusive under closed=left, so one second comes off it. That is only
        # sound on whole seconds, which is why `spark_job.check_event_time` refuses a
        # non integral column rather than trusting the caller.
        upper = hi - as_of if hi_inclusive else hi - as_of - 1
        seen.setdefault((lo - as_of, upper), []).append(as_of)
    return seen


def frame_offsets(window, probes=PROBES):
    """The one (lower, upper) RANGE offset pair Spark needs, both inclusive.

    Raises when the probes did not agree, which is the answer for anything anchored on
    something other than the scoring event.
    """
    if not probes_differ_in_time_of_day(probes):
        raise ValueError(
            "probe set sits at one time of day, so it cannot tell a calendar aligned "
            "window from an event aligned one"
        )

    seen = frame_disagreement(window, probes)
    if len(seen) != 1:
        spread = sorted(seen)
        raise UnsupportedWindow(
            "{!r} is not a constant offset from the scoring event. Probes produced {} "
            "different frames, from {} to {}. A Spark RANGE frame takes literals, so "
            "this window needs a join rather than a frame.".format(
                window, len(seen), spread[0], spread[-1]
            )
        )

    return next(iter(seen))
