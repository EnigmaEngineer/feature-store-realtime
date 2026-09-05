"""Checks on the window to frame translation.

No Spark here on purpose. The arithmetic is the part that decides whether the training
table is right, and it should be checkable on a clone that installed requirements.txt
and nothing else.
"""

from featurestore import compute, frames, spec


def window(length_s, closed, align):
    return spec.Window(length_s, closed, align)


def check_event_time_left_ends_one_second_before_the_row():
    assert frames.frame_offsets(window(3600, "left", "event_time")) == (-3600, -1)


def check_event_time_right_includes_the_row():
    assert frames.frame_offsets(window(3600, "right", "event_time")) == (-3600, 0)


def check_the_offsets_scale_with_the_length():
    assert frames.frame_offsets(window(86400, "left", "event_time")) == (-86400, -1)
    assert frames.frame_offsets(window(7 * 86400, "left", "event_time")) == (-604800, -1)


def check_calendar_day_is_refused():
    try:
        frames.frame_offsets(window(86400, "left", "calendar_day"))
    except frames.UnsupportedWindow as exc:
        assert "not a constant offset" in str(exc)
        # The endpoints get quoted in the README, so they are a published figure and
        # they need pinning. A mutant reading the second widest frame instead of the
        # widest survived every other check here.
        assert "6 different frames" in str(exc)
        assert "from (-172799, -86400) to (-86400, -1)" in str(exc)
        return
    raise AssertionError("a calendar aligned window was accepted as a fixed frame")


def check_calendar_day_is_refused_under_closed_right_too():
    try:
        frames.frame_offsets(window(86400, "right", "calendar_day"))
    except frames.UnsupportedWindow:
        return
    raise AssertionError("closed=right hid the alignment problem")


def check_exactly_two_frames_is_already_a_refusal():
    # Two probes one second apart give a calendar aligned window exactly two frames.
    # That is the smallest disagreement there is, and without it the count can be
    # raised to three and every real refusal still happens for the wrong reason.
    two = (frames.PROBES[0], frames.PROBES[0] + 1)
    seen = frames.frame_disagreement(window(86400, "left", "calendar_day"), probes=two)
    assert len(seen) == 2, "this probe pair no longer gives two frames, rewrite it"
    try:
        frames.frame_offsets(window(86400, "left", "calendar_day"), probes=two)
    except frames.UnsupportedWindow:
        return
    raise AssertionError("two disagreeing frames were accepted as one")


def check_the_offsets_do_not_depend_on_which_probes_are_used():
    """The claim behind leaving the probe constants alone.

    Mutating any single timestamp in PROBES changes no answer, and that is either an
    equivalent mutant or a hole. This says which, by asking an unrelated probe set the
    same question. If a probe value ever starts to matter, this fails.
    """
    other = (0, 7, 4000, 50000, 86399, 200000, 1000000000 + 61)
    assert frames.probes_differ_in_time_of_day(other)
    for w in (
        window(3600, "left", "event_time"),
        window(3600, "right", "event_time"),
        window(604800, "left", "event_time"),
    ):
        assert frames.frame_offsets(w, probes=other) == frames.frame_offsets(w)


def check_the_refusal_names_how_far_apart_the_frames_are():
    seen = frames.frame_disagreement(window(86400, "left", "calendar_day"))
    lowers = [k[0] for k in seen]
    assert len(seen) > 1
    assert max(lowers) - min(lowers) == frames.DAY - 1


def check_an_event_aligned_window_gives_one_frame_at_every_probe():
    seen = frames.frame_disagreement(window(3600, "left", "event_time"))
    assert len(seen) == 1
    assert sum(len(v) for v in seen.values()) == len(frames.PROBES)


def check_probes_a_whole_day_apart_do_not_count_as_varied():
    # This is the trap. Three probes on three different days look varied and agree on
    # the time of day, so a calendar aligned window resolves to one frame under them and
    # the refusal silently stops working.
    same_hour = (100, 100 + frames.DAY, 100 + 2 * frames.DAY)
    assert not frames.probes_differ_in_time_of_day(same_hour)
    # Exactly two, which is the smallest set that should count. Without this case the
    # threshold can be raised to three and nothing notices.
    assert frames.probes_differ_in_time_of_day((100, 200))
    assert frames.probes_differ_in_time_of_day((100, 200, 300))
    assert frames.probes_differ_in_time_of_day(frames.PROBES)


def check_a_probe_set_at_one_time_of_day_is_refused_rather_than_used():
    same_hour = (frames.PROBES[0], frames.PROBES[0] + frames.DAY)
    try:
        frames.frame_offsets(window(3600, "left", "event_time"), probes=same_hour)
    except ValueError as exc:
        assert "one time of day" in str(exc)
        return
    raise AssertionError("a probe set that cannot tell the alignments apart was used")


def check_that_probe_set_really_would_have_hidden_the_problem():
    """The reason the guard exists, shown rather than asserted.

    Without the guard, probes a whole day apart resolve a calendar aligned window to a
    single frame, so `frame_offsets` would return a plausible pair and the Spark job
    would silently build the wrong feature.
    """
    same_hour = (frames.PROBES[0], frames.PROBES[0] + frames.DAY)
    seen = frames.frame_disagreement(window(86400, "left", "calendar_day"), probes=same_hour)
    assert len(seen) == 1, "the probe set no longer hides it, rewrite this check"


def check_the_frame_admits_exactly_what_the_reference_admits():
    """The claim the whole translation rests on, checked against real rows.

    A frame is a pair of offsets and the reference is a bisect over a sorted list. This
    walks both over the same events and asserts they select the same rows, which is the
    only thing that makes the offsets more than a plausible pair of numbers.
    """
    # 3600 appears three times and 0 twice. A frame boundary is exactly where a tie
    # decides the answer, and the generated corpus has 18 tied rows out of 489,972, so
    # relying on it to exercise this would be relying on almost nothing.
    times = [0, 0, 10, 3599, 3600, 3600, 3600, 3601, 7200, 7201]
    rows = [{"e": "a", "t": t} for t in times]
    log = compute.EventLog(rows, "e", "t")
    for w in (
        window(3600, "left", "event_time"),
        window(3600, "right", "event_time"),
        window(60, "left", "event_time"),
    ):
        lower, upper = frames.frame_offsets(w)
        for as_of in times:
            lo, hi, inclusive = compute.window_bounds(as_of, w)
            by_reference = [r["t"] for r in log.between("a", lo, hi, inclusive)]
            by_frame = [t for t in times if as_of + lower <= t <= as_of + upper]
            assert by_reference == by_frame, (w, as_of, by_reference, by_frame)
