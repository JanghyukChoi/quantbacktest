"""Warnings raised while an engine runs, kept on the result.

A warning that scrolls past in a terminal is lost to a program that reads the result (a notebook export, a report, an AI assistant fetching the numbers over a
tool call). `keep_warnings` stores the text of every `UserWarning` the call raised in `result.notes` and raises the same warnings again, attributed to the
caller's line, so the console and `-W error` behave as before."""
from __future__ import annotations

import functools
import sys
import warnings


def keep_warnings(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        notes: list[str] = []
        out = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = fn(*args, **kwargs)
        finally:                                                       # also when the engine raised: the warnings that came first are still worth showing
            caller = sys._getframe(1)
            registry = caller.f_globals.setdefault("__warningregistry__", {})
            for w in caught:
                warnings.warn_explicit(w.message, w.category, caller.f_code.co_filename, caller.f_lineno,
                                       module=caller.f_globals.get("__name__", "<unknown>"), registry=registry)
                if issubclass(w.category, UserWarning) and str(w.message) not in notes:
                    notes.append(str(w.message))
        if out is not None and hasattr(out, "notes"):
            out.notes = notes
        return out
    return inner
