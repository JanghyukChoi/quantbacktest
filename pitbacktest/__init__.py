"""pitbacktest: a backtest toolkit that measures its own biases (US, Korea, crypto).

Three entry points
    screen()             A. Factor screening with neutralisation          (research stage)
    backtest_event()     C. Event signals                                  (alert products)
    backtest_portfolio() B. Portfolio alpha (CAGR, MDD, Sharpe)

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
from . import validation, analytics, execution
from .ledger import Ledger
from .weights import backtest_weights, capacity_curve, ImpactModel
from .shorting import shortable_from_bans

__version__ = "0.2.1"
__all__ = ["shortable_from_bans", "execution", "validation", "analytics", "Ledger", "backtest_weights", "capacity_curve", "ImpactModel", "Panel", "build_pit_eligible", "GateConfig", "screen",
           "backtest_event", "backtest_portfolio", "assert_timing"]
