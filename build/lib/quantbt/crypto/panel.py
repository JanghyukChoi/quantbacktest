"""Build a point-in-time (PIT) Panel from the Binance USDT-M archive.

Bias controls, in order of importance
  1. Survivorship: the universe is every contract that ever traded, delisted ones included. `survivors_only=True`
     reproduces the usual shortcut (only contracts that still trade today) so the bias can be measured, not guessed.
  2. Universe look-ahead: a contract is eligible on day t only if, using data up to and including day t, it has
     enough history and enough trailing dollar volume. Nothing from t+1 is used.
  3. Stale prices: bars with zero volume (a frozen last price after a halt or a delisting) are dropped so they
     cannot create fake zero returns or fake liquidity.
  4. Funding: the daily funding rate is attached to the panel and charged by `backtest_portfolio`.
  5. Delisting: the last real bar of a contract that stopped trading is marked in `delist_after`, so the engine can
     apply an explicit delisting return instead of silently assuming the position was closed at the last price.

Prices are daily UTC closes. Volumes are converted so that `close * volume` equals the USDT turnover.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..core.panel import Panel
from .binance_archive import ArchiveStore

# Pegged coins, fiat pairs and index perpetuals are not tradable risk assets for a cross-sectional study.
DEFAULT_EXCLUDE = re.compile(
    r"^(USDC|BUSD|TUSD|FDUSD|USDP|DAI|USTC|EUR|GBP|AUD|TRY|BRL)USDT$|^(BTCDOM|DEFI|FOOTBALL|BLUEBIRD|ALL)USDT$|DOMUSDT$"
)


def build_panel(store: ArchiveStore | None = None, *, symbols: list[str] | None = None,
                start: str = "2020-06-01", end: str | None = None,
                adv_window: int = 30, min_adv_usd: float = 5e6, min_age_days: int = 60,
                top_n: int | None = None, survivors_only: bool = False,
                exclude: re.Pattern | None = DEFAULT_EXCLUDE, entry_lag: int = 1,
                live: set[str] | None = None) -> Panel:
    """Return a Panel with PIT eligibility, funding and delisting information.

    min_adv_usd    trailing median USDT turnover needed to be eligible (median, so one spike cannot qualify a coin)
    min_age_days   bars a contract must already have, so the first weeks after a listing are excluded
    top_n          optionally keep only the top-N by trailing turnover among eligible contracts, each day
    survivors_only keep only contracts still trading today (for measuring survivorship bias)
    """
    store = store or ArchiveStore()
    syms = symbols or store.symbols()
    if exclude is not None:
        syms = [s for s in syms if not exclude.search(s)]
    if survivors_only and live is None:
        live = store.trading_now()
    live = live or set()
    bars: dict[str, pd.DataFrame] = {}
    fund: dict[str, pd.Series] = {}
    for s in syms:
        f = store.dir / "daily" / f"{s}.pkl"
        d = pd.read_pickle(f) if f.exists() else store.fetch_daily(s, False)
        if d is None or d.empty:
            continue
        if survivors_only and s not in live:
            continue
        d = d[d["quote_volume"] > 0]                      # drop frozen, zero-volume bars (halts, delisting tail)
        d = d[d["close"] > 0]
        if d.empty:
            continue
        bars[s] = d
        ff = store.dir / "funding" / f"{s}.pkl"
        fund[s] = pd.read_pickle(ff) if ff.exists() else store.fetch_funding(s, False)
    if not bars:
        raise ValueError("no data: run quantbt.crypto.fetch_all() first")

    idx = pd.date_range(min(d.index[0] for d in bars.values()), max(d.index[-1] for d in bars.values()), freq="D")
    cols = sorted(bars)

    def mat(col: str) -> pd.DataFrame:
        return pd.DataFrame({s: bars[s][col] for s in cols}).reindex(idx)

    close, open_, high, low = mat("close"), mat("open"), mat("high"), mat("low")
    qv = mat("quote_volume")
    funding = pd.DataFrame({s: fund[s] for s in cols if fund.get(s) is not None and len(fund[s])}).reindex(idx)
    funding = funding.reindex(columns=cols)

    # --- point-in-time eligibility: only information up to and including day t ---
    age = close.notna().cumsum()
    adv = qv.rolling(adv_window, min_periods=adv_window).median()
    ok = close.notna() & (age >= min_age_days) & (adv >= min_adv_usd)
    if top_n:
        rank = adv.where(ok).rank(axis=1, ascending=False, method="first")
        ok = ok & (rank <= top_n)

    # --- delisting: last real bar of a contract that ended well before the panel end ---
    last = close.apply(lambda c: c.last_valid_index())
    delist_after = pd.DataFrame(False, index=idx, columns=cols)
    for s in cols:
        lv = last[s]
        if lv is not None and lv < idx[-1] - pd.Timedelta(days=3):
            delist_after.loc[lv, s] = True

    # --- crypto-specific controls (replace the equity ROA / book-to-market controls) ---
    ret = close.pct_change(fill_method=None)
    btc = ret["BTCUSDT"] if "BTCUSDT" in ret.columns else ret.median(axis=1)
    beta = ret.rolling(60, min_periods=40).cov(btc).div(btc.rolling(60, min_periods=40).var(), axis=0)
    chars = {"log_dollar_vol": np.log(qv.rolling(adv_window, min_periods=adv_window).mean().clip(lower=1)),
             "btc_beta_60": beta}

    vol_base = qv / close                                   # so that close * volume == USDT turnover
    keep = idx >= pd.Timestamp(start)
    if end:
        keep &= idx <= pd.Timestamp(end)
    sl = lambda df: df.loc[keep]
    p = Panel(close=sl(close), eligible=sl(ok), open=sl(open_), high=sl(high), low=sl(low), volume=sl(vol_base),
              chars={k: sl(v) for k, v in chars.items()}, market="CRYPTO", entry_lag=entry_lag, periods_per_year=365,
              funding=sl(funding), delist_after=sl(delist_after))
    n = p.eligible.sum(axis=1)
    p.meta = {
        "symbols_in_archive": len(store.symbols()), "symbols_used": len(cols),
        "survivors_only": survivors_only, "delisted_in_panel": int(delist_after.values.any(axis=0).sum()),
        "eligible_median": float(n[n > 0].median()), "min_adv_usd": min_adv_usd, "min_age_days": min_age_days,
        "funding_coverage": float(funding.reindex(index=p.close.index).notna().values[p.close.notna().values].mean()),
    }
    return p
