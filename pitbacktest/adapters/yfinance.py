"""Public-data adapter: build a Panel from yfinance.

Dependency: `pip install yfinance` (not needed by the core, used only by this adapter)

Usage
    from pitbacktest.adapters.yfinance import load_panel
    panel = load_panel(["AAPL", "MSFT", "NVDA"], "2018-01-01", "2024-12-31")

Warning: the limits of free data
    - **Survivorship bias**: a ticker list given as of today leaves out securities delisted since.
        It is not a point-in-time universe, so performance is biased upward. Use it for exploration only.
    - **Adjusted prices**: auto_adjust=True applies dividends and splits. Without it a split day
        creates a fake return.
    - Firm characteristics (ROA, book-to-market, ...) are not provided. Running without chars raises a warning,
        and it is a legitimate one: "price controls alone cannot filter out someone else's alpha".
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..core.panel import Panel, build_pit_eligible


def load_panel(tickers: list[str], start: str, end: str, *,
               min_dollar_volume: float = 1e6,
               market: str = "US", entry_lag: int = 1,
               with_market_cap: bool = True,
               with_chars: bool = False,
               shares_outstanding: dict[str, float] | None = None) -> Panel:
    """Build a (date x ticker) panel from yfinance.

    tickers            securities to look up. **Passing only securities listed today creates survivorship bias.**
    min_dollar_volume  minimum average traded value for eligibility (20 days)
    with_market_cap    market cap from get_shares_full (share count over time) x close.
                       Switches on the size and turnover controls. One API call per security, so somewhat slow.
    with_chars         builds the 5 firm characteristics from quarterly financials (log_size, log_bm, momentum, roa, asset_growth).
                       **The key input of the neutralisation test.** Two more API calls per security.
    shares_outstanding manual {ticker: shares}. A fallback when with_market_cap fails.
    """
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "yfinance is required: pip install yfinance"
        ) from e

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column")
    if raw is None or raw.empty:
        raise ValueError("yfinance returned no data (check the tickers and the period)")

    def pick(field: str) -> pd.DataFrame | None:
        if isinstance(raw.columns, pd.MultiIndex):
            if field not in raw.columns.get_level_values(0):
                return None
            d = raw[field]
        else:
            if field not in raw.columns:
                return None
            d = raw[[field]].rename(columns={field: tickers[0]})
        return d.sort_index()

    close = pick("Close")
    if close is None:
        raise ValueError("there is no Close column")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume = pick("Volume")
    high, low, open_ = pick("High"), pick("Low"), pick("Open")
    for d in (volume, high, low, open_):
        if d is not None:
            d.index = close.index

    mkt_cap = None
    if with_market_cap:
        mkt_cap = _market_cap(yf, list(close.columns), close)
    if mkt_cap is None and shares_outstanding:
        so = pd.Series(shares_outstanding).reindex(close.columns)
        if so.notna().any():
            mkt_cap = close.mul(so, axis=1)
    if mkt_cap is None:
        warnings.warn("Market cap could not be built: the size and turnover controls are switched off", stacklevel=2)

    chars = {}
    if with_chars:
        if mkt_cap is None:
            warnings.warn("There is no market cap, so firm characteristics cannot be built", stacklevel=2)
        else:
            chars = _characteristics(yf, close, mkt_cap)

    eligible = build_pit_eligible(close, min_adv=min_dollar_volume, volume=volume)
    n_drop = int((~eligible).sum().sum())
    if eligible.values.sum() == 0:
        raise ValueError("eligible is all False: lower min_dollar_volume")

    warnings.warn(
        f"The yfinance panel is **not a point-in-time universe**. Only tickers listed today are looked up, so "
        f"delisted securities are missing and performance is biased upward. ({n_drop:,} cells excluded)",
        stacklevel=2)

    ppy = 365 if market.upper() == "CRYPTO" else 252   # a 24/7 market has 365 trading days a year
    n_ended = sum(1 for c in close.columns if close[c].last_valid_index() is not None
                  and close[c].last_valid_index() < close.index[-1] - pd.Timedelta(days=30))
    if len(close.columns) >= 30 and n_ended == 0:
        warnings.warn("No security in this panel has a price that stops inside the period: it holds only securities alive today, "
                      "so it has survivorship bias (yfinance hardly returns past prices of delisted or acquired securities). "
                      "See docs/survivorship.md.", stacklevel=2)
    return Panel(close=close, eligible=eligible, open=open_, high=high, low=low,
                 volume=volume, mkt_cap=mkt_cap, chars=chars,
                 market=market, entry_lag=entry_lag, periods_per_year=ppy)


def _market_cap(yf, tickers: list[str], close: pd.DataFrame) -> pd.DataFrame | None:
    """Share count over time x close. get_shares_full reflects changes in shares outstanding (buybacks, issuance)."""
    cols = {}
    for t in tickers:
        try:
            sh = yf.Ticker(t).get_shares_full(start=str(close.index[0].date()))
        except Exception:  # noqa: BLE001, PERF203
            continue
        if sh is None or not len(sh):
            continue
        s = pd.Series(sh)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        cols[t] = s.reindex(close.index, method="ffill")
    if not cols:
        return None
    shares = pd.DataFrame(cols).reindex(columns=close.columns)
    got = int(shares.notna().any().sum())
    if got < len(tickers):
        warnings.warn(f"Received shares for only {got}/{len(tickers)} securities: "
                      f"for the rest the market cap is NaN, so they are left out of the controls", stacklevel=3)
    return close * shares


def _characteristics(yf, close: pd.DataFrame, mkt_cap: pd.DataFrame) -> dict:
    """The five firm characteristics: standard Fama-French style definitions.

    Warning: yfinance's financial coverage is short: 5 to 7 quarters and 5 years of annual data only.
          A long backtest such as 2018 to 2024 would be empty for most of the period, so quarterly and annual data are combined and
          **a characteristic with too little coverage is dropped altogether** (an empty column ruins the regression).
          For long studies attach a paid financial source (Compustat, Sharadar and the like).

    Disclosure lag: yfinance gives no filing date, so **period end + 45 business days** is used as an approximation.
    """
    LAG = 45
    MIN_COVER = 0.30      # a characteristic is dropped if its share of valid cells is below this
    ta, be, ni = {}, {}, {}
    for t in close.columns:
        try:
            tk = yf.Ticker(t)
            bs, inc = tk.quarterly_balance_sheet, tk.quarterly_income_stmt
        except Exception:  # noqa: BLE001, PERF203
            continue
        if bs is None or bs.empty:
            continue

        def grab(df, keys):
            if df is None or df.empty:
                return None
            for k in keys:
                hit = [i for i in df.index if k in str(i)]
                if hit:
                    s = df.loc[hit[0]].dropna()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    return s.sort_index()
            return None

        # combine quarterly (last 5 to 7) and annual (last 5 years) to widen the coverage
        try:
            abs_, ainc = tk.balance_sheet, tk.income_stmt
        except Exception:  # noqa: BLE001
            abs_ = ainc = None

        def merge(q_, a_):
            parts = [x for x in (q_, a_) if x is not None and len(x)]
            if not parts:
                return None
            m = pd.concat(parts)
            return m[~m.index.duplicated(keep="first")].sort_index()

        a = merge(grab(bs, ["Total Assets"]), grab(abs_, ["Total Assets"]))
        e = merge(grab(bs, ["Stockholders Equity", "Common Stock Equity"]),
                  grab(abs_, ["Stockholders Equity", "Common Stock Equity"]))
        n = merge(grab(inc, ["Net Income From Continuing Operation Net Minority Interest",
                             "Net Income"]),
                  grab(ainc, ["Net Income From Continuing Operation Net Minority Interest",
                              "Net Income"]))
        for src, dst in ((a, ta), (e, be), (n, ni)):
            if src is not None and len(src):
                dst[t] = src

    def to_daily(d: dict) -> pd.DataFrame | None:
        if not d:
            return None
        df = pd.DataFrame(d)
        # use from the period end + LAG business days (approximating the disclosure lag)
        df.index = df.index + pd.tseries.offsets.BDay(LAG)
        return df.reindex(close.index.union(df.index)).ffill().reindex(close.index)\
                 .reindex(columns=close.columns)

    TA, BE, NI = to_daily(ta), to_daily(be), to_daily(ni)
    if TA is None and BE is None:
        warnings.warn("Quarterly financials could not be fetched, so the firm characteristics are empty", stacklevel=3)
        return {}
    out = {"log_size": np.log(mkt_cap.clip(lower=1)),
           "momentum": close.pct_change(252) - close.pct_change(21)}
    if BE is not None:
        out["log_bm"] = np.log((BE / mkt_cap.replace(0, np.nan)).clip(lower=1e-6))
    if NI is not None and TA is not None:
        out["roa"] = NI / TA.replace(0, np.nan)
    if TA is not None:
        out["asset_growth"] = TA.pct_change(252)
    # drop characteristics with low coverage: an empty column wipes out the regression sample
    kept, dropped = {}, {}
    for k, v in out.items():
        cover = float(v.notna().values.mean())
        (kept if cover >= MIN_COVER else dropped)[k] = cover
        if cover >= MIN_COVER:
            kept[k] = v
    dropped_names = {k: f"{c*100:.0f}%" for k, c in dropped.items()}
    kept_out = {k: out[k] for k in out if k not in dropped}
    if dropped_names:
        warnings.warn(
            f"Firm characteristics {list(dropped_names)} were dropped because their coverage ({dropped_names}) is low "
            f"(yfinance provides only 5 to 7 quarters and 5 years of annual data). "
            f"For long backtests attach a paid financial source.", stacklevel=3)
    got = {k: f"{float(v.notna().values.mean())*100:.0f}%" for k, v in kept_out.items()}
    warnings.warn(f"Firm characteristic valid cells: {got} | period-end + {LAG} business-day lag applied", stacklevel=3)
    return kept_out


def sp500_tickers() -> list[str]:
    """Wikipedia's current S&P 500 constituents. **A list as of today, so it has survivorship bias.**"""
    try:
        t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return sorted(t["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist())
    except Exception as e:  # pragma: no cover  # noqa: BLE001
        raise RuntimeError(f"Could not fetch the S&P 500 list: {e}") from e
