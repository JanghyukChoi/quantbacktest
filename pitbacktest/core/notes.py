"""Warnings kept on the result.

A warning that scrolls past in a terminal is lost to a program that reads the result (a notebook export, a report, an AI assistant fetching the numbers over a tool
call). Library code raises its warnings through `warn` here instead of `warnings.warn`: it does what `warnings.warn` does (same filters, same `-W error`, the same
`stacklevel` meaning, a warning that raises stops the run where it happened) and also writes the text on a list that belongs to the running engine call. `keep_warnings`
opens that list around an engine and puts it on the result as `result.notes`, with what the panel said when it was built (prices set to missing, a `periods_per_year`
that does not fit the dates).

The list lives in a `ContextVar`, so two engine calls in two threads (or two asyncio tasks) never see each other's warnings, and nothing global is changed or restored:
an earlier version swapped the process-wide warning filters for the duration of the call and was not safe to run in parallel."""
from __future__ import annotations

import functools
import sys
import warnings
from contextvars import ContextVar

_sink: ContextVar[list | None] = ContextVar("pitbacktest_notes", default=None)


def warn(message, category=UserWarning, stacklevel: int = 1) -> None:
    """`warnings.warn`, plus the text goes to the running engine call's notes. `stacklevel` counts as in `warnings.warn`; the decorator's own frame is skipped."""
    sink = _sink.get()
    if sink is not None and issubclass(category, UserWarning) and str(message) not in sink:
        sink.append(str(message))
    try:
        f = sys._getframe(stacklevel)
    except ValueError:
        f = sys._getframe(0)
    while f.f_back is not None and f.f_code.co_filename == __file__:
        f = f.f_back
    warnings.warn_explicit(message, category, f.f_code.co_filename, f.f_lineno, module=f.f_globals.get("__name__", "<unknown>"),
                           registry=f.f_globals.setdefault("__warningregistry__", {}))


def keep_warnings(fn):
    """Decorator for an engine: collect the warnings raised through `warn` during the call and store them in `result.notes`."""
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        notes: list[str] = []
        token = _sink.set(notes)
        try:
            out = fn(*args, **kwargs)
        finally:
            _sink.reset(token)
        if out is not None and hasattr(out, "notes"):
            panel = args[0] if args else kwargs.get("panel")
            built = list(getattr(panel, "meta", {}).get("construction_notes", [])) if panel is not None else []
            out.notes = built + [n for n in notes if n not in built]
        return out
    return inner
