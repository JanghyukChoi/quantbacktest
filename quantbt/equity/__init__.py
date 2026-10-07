"""Equity survivorship tools that work without paid data.

None of this removes survivorship bias from a survivors-only price source; it measures how much of the real
universe is missing and shows, by explicit scenarios, how much that could matter. Real removal needs point-in-time data
(see `quantbt.adapters.long_format` for a strict way to plug it in, and `quantbt.adapters.krx` for Korea).
"""
from .master import load_us_master, universe_coverage
from .scenarios import inject_delistings, survivorship_scenarios, survivors_only

__all__ = ["load_us_master", "universe_coverage", "inject_delistings", "survivorship_scenarios", "survivors_only"]
