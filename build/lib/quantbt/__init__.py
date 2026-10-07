"""quantbt — KR/US 통합 퀀트 백테스트 프레임워크.

세 진입점
  screen()             A. 팩터 전수 스크리닝 + 중립화     (리서치 단계)
  backtest_event()     C. 이벤트형 시그널 (알림 제품)
  backtest_portfolio() B. 포트폴리오 알파 (CAGR·MDD·Sharpe)

설계 원칙
  · 코어는 순수 pandas/numpy. 데이터 소스는 adapters 로 분리.
  · 시점 규약은 하나뿐이고 assert 로 강제한다.
  · 재무 특성 통제가 기본값이다.
  · 무통제 결과는 요약에서 제외한다.
"""

from .core.panel import Panel, build_pit_eligible
from .core.gates import GateConfig
from .event import backtest_event
from .portfolio import backtest_portfolio, assert_timing
from .screen import screen
from . import validation

__version__ = "0.1.0"
__all__ = ["validation", "Panel", "build_pit_eligible", "GateConfig", "screen",
           "backtest_event", "backtest_portfolio", "assert_timing"]
