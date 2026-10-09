"""pitbacktest: a backtest toolkit that measures its own biases (US, Korea, crypto).

Three entry points
    screen()             A. Factor screening with neutralisation          (research stage)
    backtest_event()     C. Event signals                                  (alert products)
    backtest_portfolio() B. Portfolio alpha (CAGR, MDD, Sharpe)

Review: one call that runs a strategy through the tests an institution would ask for, and a page that shows them
    review_portfolio()   factor portfolio: luck (permutation, t-tests), market-relative, factor decomposition, walk-forward, settings, cost
    review_event()       event signal: win rate against a random pick, cost, years, permutation, the signal held as a portfolio
    robustness           the building blocks (walk_forward, factor_permutation, signal_permutation, market_relative, bh_fdr, ...)

Design principles
    - The core is plain pandas and numpy; data sources live in `adapters`.
    - There is one timing convention and it is enforced with an assert.
    - Controls for firm characteristics are the default.
    - The statistics that decide a verdict are computed with controls; uncontrolled figures are returned next to them, not in their place.
"""

from .core.panel import Panel, build_pit_eligible
from .core.gates import GateConfig
from .event import backtest_event
from .portfolio import backtest_portfolio, assert_timing
from .screen import screen
from . import validation, analytics, execution, robustness
from .ledger import Ledger
from .weights import backtest_weights, capacity_curve, ImpactModel
from .shorting import shortable_from_bans
from .review import review_portfolio, review_event, event_weights
from ._version import __version__

__all__ = ["shortable_from_bans", "execution", "validation", "analytics", "Ledger", "backtest_weights", "capacity_curve", "ImpactModel", "Panel", "build_pit_eligible", "GateConfig", "screen",
           "backtest_event", "backtest_portfolio", "assert_timing", "robustness", "review_portfolio", "review_event", "event_weights"]
