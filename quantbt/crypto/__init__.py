"""Crypto perpetual-futures data and panel construction (Binance USDT-M).

The point of this subpackage is to remove, as far as public data allows, three biases that crypto backtests
usually carry: survivorship (delisted contracts are included), universe look-ahead (eligibility uses only data up
to the signal date) and ignored funding (the daily funding rate is charged to the position).
"""
from .binance_archive import ArchiveStore, fetch_all
from .panel import build_panel
from .costs import liquidity_cost_bp, participation_report

__all__ = ["ArchiveStore", "fetch_all", "build_panel", "liquidity_cost_bp", "participation_report"]
