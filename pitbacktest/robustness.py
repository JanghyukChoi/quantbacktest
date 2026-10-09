"""Is the result luck, and does it survive being tested the way an institution would test it?

Every function here answers one question about a strategy's return series or about the panel and signal that produced it. They take plain series and matrices, so they can be
used on their own, and `pitbacktest.review` runs them together and puts them in the report.

    market_relative      against the benchmark: excess growth, how often it beat it, up and down capture, beta, tracking error, information ratio
    mean_tests           is the mean return different from zero: Newey-West t, the Sharpe t-statistic, a sign test on the hit rate
    subperiods           year by year and by halves: does the edge show in most of them or in one
    walk_forward         choose the best setting on the past, run it on the next stretch, repeat: the out-of-sample record of the *selection*
    parameter_plateau    is the best setting a hill or a spike among its neighbours
    factor_permutation   rerun the whole backtest with the factor shuffled across securities each day: how often does noise do as well
    signal_permutation   the same for an event signal: the same number of fires each day, picked at random
    bh_fdr, holm         p-values corrected for how many things were tried
    style_factor_returns, decompose   how much of the return is size, momentum, reversal, volatility, liquidity and the market
    attribution          where the return came from: long or short leg, which securities, small or large, liquid or not, which group, and what costs took

Nothing here recommends a strategy or a threshold. Each returns the numbers and what they mean, and says how uncertain they are."""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from .core.estimators import newey_west_t, shuffle_columns
from .core.notes import warn


def _ppy(ppy) -> int:
    if isinstance(ppy, bool) or not isinstance(ppy, (int, np.integer)) or ppy < 1:
        raise ValueError(f"periods_per_year must be a whole number of at least 1, got {ppy!r}")
    return int(ppy)


def _series(x, name: str) -> pd.Series:
    if not isinstance(x, pd.Series):
        raise ValueError(f"{name} must be a pandas Series indexed by date, got {type(x).__name__}")
    return x.astype(float)


def _max_drawdown(r: np.ndarray) -> float:
    eq = np.cumprod(1.0 + r)
    return float((eq / np.maximum(np.maximum.accumulate(eq), 1.0) - 1.0).min()) if len(eq) else float("nan")


def _cagr(r: np.ndarray, ppy: int) -> float:
    n = len(r)
    if n == 0:
        return float("nan")
    g = float(np.prod(1.0 + r))
    return g ** (ppy / n) - 1.0 if g > 0 else -1.0


def _sharpe(r: np.ndarray, ppy: int) -> float:
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    return float(np.mean(r) / sd * math.sqrt(ppy)) if sd > 1e-12 else float("nan")


# ---------------------------------------------------------------------------------------------------------------------------- against the market
def market_relative(net: pd.Series, bench: pd.Series, *, periods_per_year: int = 252) -> dict:
    """The strategy against a benchmark on the dates they share.

    excess_cagr           strategy CAGR minus benchmark CAGR (fractions per year)
    hit_rate_month/year   share of calendar months / years in which the strategy's compounded return beat the benchmark's (years with a part-year are counted)
    up_capture, down_capture   the strategy's mean return on days the benchmark was up (down) divided by the benchmark's mean on those days: above 1 up and below 1 down is
                          what a good long-biased strategy looks like; a market-neutral one has both near 0
    beta, correlation     of daily returns
    tracking_error        annualised standard deviation of the daily difference
    information_ratio     annualised mean difference over its standard deviation
    max_relative_drawdown the worst fall of the strategy's equity relative to the benchmark's (their ratio), as a negative fraction
    """
    ppy = _ppy(periods_per_year)
    a, b = _series(net, "net"), _series(bench, "bench")
    df = pd.concat([a.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
    if len(df) < 20:
        raise ValueError(f"only {len(df)} dates in common: at least 20 are needed")
    s, m = df["s"].to_numpy(), df["b"].to_numpy()
    d = s - m
    idx = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index

    def hit(freq: str) -> float:
        g = ((1.0 + df).groupby(idx.to_period(freq)).prod() - 1.0)
        return float((g["s"] > g["b"]).mean())

    up, dn = m > 0, m < 0
    ratio = np.cumprod(1.0 + s) / np.cumprod(1.0 + m)
    var_m = float(np.var(m, ddof=1))
    sd_d = float(np.std(d, ddof=1))
    sd_s = float(np.std(s, ddof=1))
    return {
        "n": int(len(df)), "excess_cagr": _cagr(s, ppy) - _cagr(m, ppy), "strategy_cagr": _cagr(s, ppy), "benchmark_cagr": _cagr(m, ppy),
        "hit_rate_month": hit("M"), "hit_rate_year": hit("Y"),
        "up_capture": float(s[up].mean() / m[up].mean()) if up.any() else float("nan"),
        "down_capture": float(s[dn].mean() / m[dn].mean()) if dn.any() else float("nan"),
        "beta": float(np.cov(s, m, ddof=1)[0, 1] / var_m) if var_m > 1e-18 else float("nan"),
        "correlation": float(np.corrcoef(s, m)[0, 1]) if sd_s > 1e-12 and var_m > 1e-18 else float("nan"),
        "tracking_error": sd_d * math.sqrt(ppy),
        "information_ratio": float(d.mean() / sd_d * math.sqrt(ppy)) if sd_d > 1e-12 else float("nan"),
        "max_relative_drawdown": float((ratio / np.maximum(np.maximum.accumulate(ratio), 1.0) - 1.0).min()),
    }


# ---------------------------------------------------------------------------------------------------------------------------- is the mean different from zero
def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided p-value of k successes in n fair coin tosses (doubled smaller tail, at most 1)."""
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / (1 << n)          # integer division: 2.0 ** n overflows above n = 1023
    return min(1.0, 2.0 * tail)


def _norm_two_sided(t: float) -> float:
    return math.erfc(abs(t) / math.sqrt(2.0))


def mean_tests(net: pd.Series, *, periods_per_year: int = 252, lag: int | None = None) -> dict:
    """Plain tests that the mean return is not zero.

    mean_daily, nw_t, nw_p     the mean return per bar, its Newey-West t-statistic (overlapping or autocorrelated returns widen the standard error) and the two-sided
                               normal p-value. The lag is the Newey-West (1994) plug-in `int(4 (T/100)^(2/9))` unless given. Finite-sample t-values over-reject: read
                               |t| below 2.5 as no evidence.
    sharpe_t                   Sharpe ratio x sqrt(years): the t-statistic of the mean for independent returns, the number Harvey, Liu and Zhu compare with 3 for a new factor
    win_rate, sign_p           share of bars with a positive return and the exact two-sided sign-test p-value against 50 percent
    win_ci                     the 95 percent Wilson interval of the win rate
    skew, kurtosis             of the returns (kurtosis 3 is normal): fat tails make every t-value above too optimistic
    """
    ppy = _ppy(periods_per_year)
    x = _series(net, "net").dropna().to_numpy()
    T = len(x)
    if T < 30:
        raise ValueError(f"only {T} observations: at least 30 are needed")
    L = int(4 * (T / 100.0) ** (2.0 / 9.0)) if lag is None else int(lag)
    mean, se, t, _ = newey_west_t(x, L)
    pos = int((x > 0).sum())
    nz = int((x != 0).sum())
    z = 1.959963984540054
    p_hat = pos / T
    den = 1 + z * z / T
    centre = (p_hat + z * z / (2 * T)) / den
    half = z * math.sqrt(p_hat * (1 - p_hat) / T + z * z / (4 * T * T)) / den
    sd = float(x.std(ddof=1))
    zz = (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0)
    return {"n": T, "lag": L, "mean_daily": float(mean), "nw_t": float(t), "nw_p": _norm_two_sided(t) if math.isfinite(t) else float("nan"),
            "sharpe": _sharpe(x, ppy), "sharpe_t": _sharpe(x, ppy) * math.sqrt(T / ppy) if sd > 1e-12 else float("nan"),
            "win_rate": float(p_hat), "sign_p": _binom_two_sided(int((x > 0).sum()), nz), "win_ci": (float(centre - half), float(centre + half)),
            "skew": float((zz ** 3).mean()), "kurtosis": float((zz ** 4).mean())}


# ---------------------------------------------------------------------------------------------------------------------------- stability over time
def subperiods(net: pd.Series, *, periods_per_year: int = 252) -> dict:
    """Calendar years and the two halves of the sample.

    `years` has one row per year (return, volatility, Sharpe, worst drawdown from the year before's end, bars). `share_positive` is the share of years with a positive return and
    `worst_year` the lowest. `halves` are the Sharpe ratios of the first and second half of the bars: an edge that lives in only one of them is a regime, not a property.
    """
    ppy = _ppy(periods_per_year)
    s = _series(net, "net").dropna()
    if len(s) < 40:
        raise ValueError(f"only {len(s)} observations: at least 40 are needed")
    idx = s.index.tz_localize(None) if getattr(s.index, "tz", None) is not None else s.index
    rows, start = [], 1.0
    for y, g in s.groupby(idx.year):
        eq = (1.0 + g).cumprod() * start
        rows.append({"year": int(y), "ret": float(eq.iloc[-1] / start - 1.0), "vol": float(g.std(ddof=1) * math.sqrt(ppy)) if len(g) > 1 else float("nan"),
                     "sharpe": _sharpe(g.to_numpy(), ppy), "max_dd": float((eq / np.maximum(eq.cummax(), start) - 1.0).min()), "bars": int(len(g))})
        start = float(eq.iloc[-1])
    tbl = pd.DataFrame(rows).set_index("year")
    h = len(s) // 2
    return {"years": tbl, "share_positive": float((tbl["ret"] > 0).mean()), "worst_year": float(tbl["ret"].min()), "best_year": float(tbl["ret"].max()),
            "halves": (_sharpe(s.to_numpy()[:h], ppy), _sharpe(s.to_numpy()[h:], ppy))}


# ---------------------------------------------------------------------------------------------------------------------------- walk-forward
def walk_forward(R: pd.DataFrame, *, train: int, test: int, step: int | None = None, expanding: bool = True, embargo: int = 0,
                 periods_per_year: int = 252, select: Callable[[np.ndarray], float] | None = None) -> dict:
    """Walk-forward evaluation of a *selection*.

    `R` has one column per candidate setting (a parameter, a lookback, a variant) and one row per bar: the net return series each setting would have produced over the whole sample.
    Because every setting's signal is computed from past data only, a column does not depend on which setting is chosen later, so the walk-forward can be run on the matrix: for
    each fold the setting with the best score over the *training* bars is chosen and its return over the next `test` bars is kept. Stitched together those are returns that no
    choice ever saw. Compare them with the in-sample best: the gap is what the selection cost.

    train      bars in the first training window (the window grows with `expanding`, otherwise it slides at this length)
    test       bars in each test stretch; `step` (default `test`) is how far the window moves
    embargo    bars dropped between the end of training and the start of the test: holding periods that straddle the boundary would otherwise leak (use hold + entry_lag)
    select     score of a training column (default: Sharpe ratio); a larger score is better

    Returns `oos` (the stitched out-of-sample returns), `folds` (a table: training and test dates, the chosen setting, its in-sample and out-of-sample Sharpe),
    `oos_sharpe`, `mean_is_sharpe` (average in-sample Sharpe of the chosen settings), `efficiency` (oos over in-sample Sharpe: near 1 means the selection transfers, near 0 or below
    means it did not), `share_folds_positive`, and `chosen_counts`.
    """
    ppy = _ppy(periods_per_year)
    if not isinstance(R, pd.DataFrame) or R.shape[1] < 2:
        raise ValueError("R must be a DataFrame with at least two columns (one per candidate setting)")
    for name, v in (("train", train), ("test", test)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 10:
            raise ValueError(f"{name} must be a whole number of bars, at least 10, got {v!r}")
    step = int(test) if step is None else int(step)
    if step < 1 or embargo < 0:
        raise ValueError("step must be at least 1 and embargo not negative")
    X = R.astype(float)
    T = len(X)
    if train + embargo + test > T:
        raise ValueError(f"the sample has {T} bars but one fold needs {train + embargo + test}")
    score = select or (lambda col: _sharpe(col, ppy))
    folds, oos, k0 = [], [], int(train)
    while k0 + embargo + test <= T:
        tr_lo = 0 if expanding else k0 - train
        tr = X.iloc[tr_lo:k0]
        te = X.iloc[k0 + embargo:k0 + embargo + test]
        sc = np.array([score(tr[c].to_numpy()) if tr[c].notna().sum() >= 10 else np.nan for c in tr.columns], dtype=float)
        if not np.isfinite(sc).any():
            k0 += step
            continue
        best = tr.columns[int(np.nanargmax(sc))]
        r_te = te[best].dropna()
        folds.append({"train_start": tr.index[0], "train_end": tr.index[-1], "test_start": te.index[0], "test_end": te.index[-1], "chosen": best,
                      "is_sharpe": float(np.nanmax(sc)), "oos_sharpe": _sharpe(r_te.to_numpy(), ppy), "oos_return": float(np.prod(1.0 + r_te.to_numpy()) - 1.0)})
        oos.append(r_te)
        k0 += step
    if not folds:
        raise ValueError("no fold could be run: no training window had a usable score")
    out = pd.concat(oos)
    out = out[~out.index.duplicated(keep="first")] if step < test else out          # overlapping tests would count a bar twice
    fd = pd.DataFrame(folds)
    is_mean = float(fd["is_sharpe"].mean())
    oos_sh = _sharpe(out.to_numpy(), ppy)
    return {"oos": out, "folds": fd, "oos_sharpe": oos_sh, "mean_is_sharpe": is_mean,
            "efficiency": oos_sh / is_mean if math.isfinite(oos_sh) and is_mean > 1e-9 else float("nan"),
            "share_folds_positive": float((fd["oos_return"] > 0).mean()), "chosen_counts": fd["chosen"].astype(str).value_counts().to_dict(), "n_folds": int(len(fd))}


# ---------------------------------------------------------------------------------------------------------------------------- spike or hill
def parameter_plateau(table: pd.DataFrame | pd.Series, *, rel: float = 0.5) -> dict:
    """Is the best setting a hill or a spike?

    `table` holds one score (for example the Sharpe ratio) per setting: a DataFrame for two parameters (rows and columns in their natural order) or a Series for one.
    For the best cell the function looks at the cells next to it (one step in each direction, diagonals included) and reports `neighbour_ratio`, their mean score over the best
    score, and `share_within`, the share of neighbours that keep at least `rel` of the best score. A spike (isolated, ratio near 0 or below) is a sign the best was found by
    searching, not by the strategy; a plateau (ratio near 1) means the neighbours agree. `grid_positive` is the share of all cells with a positive score.
    """
    if isinstance(table, pd.Series):
        table = table.to_frame("v")                                      # one parameter: neighbours are the cells before and after
    T = table.astype(float).to_numpy()
    if T.size < 3:
        raise ValueError("the table needs at least 3 cells")
    if not np.isfinite(T).any():
        raise ValueError("the table has no finite score")
    i, j = np.unravel_index(int(np.nanargmax(np.where(np.isfinite(T), T, -np.inf))), T.shape)
    best = float(T[i, j])
    nb = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (di or dj) and 0 <= i + di < T.shape[0] and 0 <= j + dj < T.shape[1] and np.isfinite(T[i + di, j + dj]):
                nb.append(float(T[i + di, j + dj]))
    ratio = float(np.mean(nb) / best) if nb and best > 0 else float("nan")
    native = lambda v: v.item() if hasattr(v, "item") else v                  # noqa: E731
    return {"best": best, "best_at": (native(table.index[i]), native(table.columns[j])), "n_neighbours": len(nb), "neighbour_ratio": ratio,
            "neighbour_mean": float(np.mean(nb)) if nb else float("nan"),
            "share_within": float(np.mean([v >= rel * best for v in nb])) if nb and best > 0 else float("nan"),
            "grid_positive": float((T[np.isfinite(T)] > 0).mean()),
            "verdict": "plateau" if (math.isfinite(ratio) and ratio >= 0.7) else ("spike" if math.isfinite(ratio) and ratio < 0.3 else ("hill" if math.isfinite(ratio) else "undetermined"))}


# ---------------------------------------------------------------------------------------------------------------------------- permutation nulls
def factor_permutation(panel, factor: pd.DataFrame, *, n: int = 100, seed: int = 0, **backtest_kwargs) -> dict:
    """How often does noise do as well? The whole backtest rerun with the factor shuffled across securities within each date.

    A shuffled factor keeps the factor's cross-sectional distribution and coverage and destroys the link between a security's score and its future return: the null is that the
    ranking carries no information. **The comparison is before every cost** (spread, side costs, borrow, funding and impact are switched off, whatever `backtest_kwargs` say).
    Shuffling each date independently also destroys the persistence of the ranking, so a shuffled portfolio turns over far more than a slow factor does; with costs in, the shuffles
    would lose to the costs and any slow factor, informative or not, would beat them (a pure-noise factor with persistent scores got p = 0.02 that way). Costs are judged separately
    (the cost table). `backtest_kwargs` go to `backtest_portfolio` (long_q, short_q, hold, weighting...); `benchmark` and `grid` are switched off. Returns the real gross Sharpe,
    the shuffled Sharpes' mean and 95th/99th percentile, and `p = (1 + number of shuffles at least as good) / (n + 1)`. With n = 100 the smallest p is about 0.01. Slow: it runs `n`
    backtests.
    """
    from .portfolio import backtest_portfolio
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 10:
        raise ValueError(f"n must be a whole number of at least 10, got {n!r}")
    kw = {**backtest_kwargs, "benchmark": None, "grid": False, "spread_bp": 0.0, "buy_bp": 0.0, "sell_bp": 0.0, "funding": False}
    for k in ("ledger", "impact", "family", "name"):
        kw.pop(k, None)
    real = backtest_portfolio(panel, factor, **kw).metrics["Sharpe"]
    if not math.isfinite(real):
        raise ValueError("the real backtest has no Sharpe ratio (no variation, or too few bars)")
    rng = np.random.default_rng(seed)
    el = panel.eligible
    base = factor.reindex(index=panel.dates, columns=panel.tickers)
    nulls = []
    for _ in range(int(n)):
        sh = backtest_portfolio(panel, shuffle_columns(base, el, rng), **kw).metrics["Sharpe"]
        nulls.append(sh if math.isfinite(sh) else 0.0)
    a = np.array(nulls)
    return {"real_sharpe": float(real), "n": int(n), "p": float((1 + (a >= real).sum()) / (n + 1)), "null_mean": float(a.mean()), "null_std": float(a.std(ddof=1)),
            "null_p95": float(np.percentile(a, 95)), "null_p99": float(np.percentile(a, 99)), "null": a}


def signal_permutation(panel, signal: pd.DataFrame, horizon: int, cost_bp: float = 0.0, *, n: int = 200, seed: int = 0) -> dict:
    """The same question for an event signal: does the *selection* beat picking at random?

    On every date the signal fires on some number of securities. The null picks that many at random among the eligible securities of the same date, n times, and records the win
    rate (share of trades above zero after the round-trip `cost_bp`) and the mean return. It keeps how often and how many times the signal fires and destroys which securities it
    picks. `p_win` and `p_mean` are the shares of random draws at least as good as the real signal (with the +1 correction). This is the question "does the filter have information"
    that a test against zero cannot answer: in a rising market almost any pick has a positive mean.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError(f"horizon must be a whole number of at least 1, got {horizon!r}")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 20:
        raise ValueError(f"n must be a whole number of at least 20, got {n!r}")
    if not (cost_bp >= 0 and math.isfinite(cost_bp)):
        raise ValueError(f"cost_bp must be finite and not negative, got {cost_bp!r}")
    fwd = panel.forward(int(horizon)).to_numpy(np.float64)
    fire = signal.reindex(index=panel.dates, columns=panel.tickers).fillna(False).to_numpy(bool) & panel.eligible.to_numpy(bool)
    usable = panel.eligible.to_numpy(bool) & np.isfinite(fwd)
    fire &= usable
    k = fire.sum(axis=1)
    days = k > 0
    if k.sum() < 30:
        raise ValueError(f"the signal has only {int(k.sum())} fires with a measured outcome: at least 30 are needed")
    cost = cost_bp / 1e4
    real_ret = fwd[fire] - cost
    real_win, real_mean = float((real_ret > 0).mean()), float(real_ret.mean())
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(days)
    U, F, KK = usable[idx], fwd[idx], k[idx]
    w_null, m_null = np.empty(int(n)), np.empty(int(n))
    for b in range(int(n)):
        r = np.where(U, rng.random(U.shape), np.inf)
        rank = np.argsort(np.argsort(r, axis=1), axis=1)                # each row's random order of the securities; the usable ones come first
        pick = rank < KK[:, None]
        v = F[pick] - cost
        w_null[b], m_null[b] = (v > 0).mean(), v.mean()
    return {"n_fires": int(k.sum()), "real_win_rate": real_win, "real_mean_bp": real_mean * 1e4, "n": int(n),
            "random_win_rate_mean": float(w_null.mean()), "random_win_rate_p95": float(np.percentile(w_null, 95)),
            "random_mean_bp_mean": float(m_null.mean() * 1e4), "random_mean_bp_p95": float(np.percentile(m_null, 95) * 1e4),
            "p_win": float((1 + (w_null >= real_win).sum()) / (n + 1)), "p_mean": float((1 + (m_null >= real_mean).sum()) / (n + 1))}


# ---------------------------------------------------------------------------------------------------------------------------- many tests
def bh_fdr(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values): reject those at or below the false discovery rate you accept. Order of the input is kept."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("p must be a non-empty one-dimensional array")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p-values must lie in [0, 1] and not be NaN")
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    adj[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(adj, 1.0)


def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values: controls the chance of *any* false discovery, stricter than `bh_fdr`."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("p must be a non-empty one-dimensional array")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p-values must lie in [0, 1] and not be NaN")
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    adj[order] = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    return adj


# ---------------------------------------------------------------------------------------------------------------------------- what explains the return
def style_factor_returns(panel, *, hold: int = 5, q: float = 0.3) -> pd.DataFrame:
    """Daily long-short returns of the usual styles, built from the panel itself, to explain a strategy's return with.

    Each style ranks the eligible securities on a past-only characteristic, goes long the top `q` and short the bottom `q` (equal weight, gross of cost, held `hold` bars):
    `market` (the equal-weight return of the eligible universe, not long-short), `size` (small minus big: needs `mkt_cap`), `momentum` (12 months skipping the last month: needs
    more than a year of bars), `reversal` (the last 5 bars, losers minus winners), `lowvol` (low minus high 60-bar volatility) and `illiquidity` (illiquid minus liquid: needs
    `volume`). A style whose input is missing is left out. These are the panel's own factors, not published factor returns: use `alpha_beta` with Fama-French or similar when
    you have them.
    """
    from .portfolio import backtest_portfolio
    c = panel.close
    ppy = panel.periods_per_year
    mom_lb = max(20, int(round(ppy)))
    skip = max(2, int(round(ppy / 12)))
    cand = {}
    if panel.mkt_cap is not None:
        cand["size"] = -panel.mkt_cap.where(panel.mkt_cap > 0)
    if len(c) > mom_lb + skip + 30:
        cand["momentum"] = c.shift(skip) / c.shift(mom_lb) - 1.0
    cand["reversal"] = -c.pct_change(5, fill_method=None)
    cand["lowvol"] = -c.pct_change(fill_method=None).rolling(60, min_periods=40).std()
    if panel.volume is not None:
        dv = (c * panel.volume).rolling(30, min_periods=20).mean()
        cand["illiquidity"] = -dv.where(dv > 0)
    out = {}
    for name, f in cand.items():
        try:
            r = backtest_portfolio(panel, f, long_q=q, short_q=q, hold=hold, spread_bp=0.0, benchmark=None, grid=False, funding=False)
            out[name] = r.net_returns
        except ValueError as e:                                  # a style that cannot be built on this panel is left out, said once
            warn(f"style factor {name!r} left out: {str(e)[:100]}", stacklevel=2)
    frame = pd.DataFrame(out)
    # the strategies' returns are indexed by signal date d and earn close(d+lag) -> close(d+lag+1): the market is the equal-weight mean of exactly that return
    # over the securities eligible on d (`forward(1)` also carries a delisting loss instead of dropping the name)
    mret = panel.forward(1).where(panel.eligible).mean(axis=1).reindex(frame.index) if len(frame) else pd.Series(dtype=float)
    frame.insert(0, "market", mret.astype(float))
    return frame.dropna(how="all")


def decompose(net: pd.Series, factors: pd.DataFrame, *, periods_per_year: int = 252, lag: int | None = None) -> dict:
    """Regress the strategy's returns on factor returns: how much is market, size, momentum, reversal...

    A thin wrapper of `analytics.alpha_beta` that also reports, per factor, its t-statistic and its *contribution* to the annual return (beta x the factor's mean x periods per
    year), and `unexplained_share`: `1 - R^2`. The `alpha_annual` left over is what none of the supplied factors explains; with only the panel's own style factors it is the return
    that is not simple size, momentum, reversal, volatility, liquidity or market exposure.
    """
    from .analytics import alpha_beta
    ppy = _ppy(periods_per_year)
    F = factors.copy()
    F = F.loc[:, [c for c in F.columns if F[c].notna().sum() > 60 and F[c].std() > 1e-12]]
    if F.shape[1] == 0:
        raise ValueError("no usable factor column")
    ab = alpha_beta(net, F, periods_per_year=ppy, lag=lag)
    contrib = {}
    for name, b in ab["betas"].items():
        contrib[name] = float(b["beta"] * F[name].mean() * ppy)
    return {"alpha_annual": ab["alpha_annual"], "alpha_t": ab["alpha_t"], "r2": ab["r2"], "unexplained_share": float(1.0 - ab["r2"]), "n": ab["n"],
            "factors": {k: {"beta": v["beta"], "t": v["t"], "contribution_annual": contrib[k]} for k, v in ab["betas"].items()}}



# ---------------------------------------------------------------------------------------------------------------------------- where the return came from
def _terciles(values: pd.DataFrame, eligible: pd.DataFrame) -> np.ndarray:
    """Per date, 0 / 1 / 2 for the lowest, middle and highest third of the eligible securities by `values` (-1 where unknown)."""
    rk = values.where(eligible).rank(axis=1, pct=True, na_option="keep").to_numpy(np.float64)
    out = np.full(rk.shape, -1, dtype=int)
    ok = np.isfinite(rk)
    out[ok] = np.clip(np.ceil(rk[ok] * 3).astype(int) - 1, 0, 2)
    return out


def attribution(panel, result, *, groups: pd.Series | None = None, n_top: int = 10) -> dict:
    """Where a backtest's return came from.

    Works on the result's `holdings` (weights per unit of capital) and the one-bar returns the engine used, so its pieces add up to the engine's own gross return exactly.
    All figures are arithmetic returns per year per unit of capital (what the engine's `*_annual_bp` costs are measured against).

    gross_annual              mean gross return per bar x periods per year
    long_annual, short_annual contribution of the long and the short positions (the short's is positive when the shorted securities fell); they add up to `gross_annual`
    names                     the `n_top` best and worst securities by contribution (`top`, `bottom`), the number of securities with a positive total, `top_share` (the share of the
                              positive contributions that came from the `n_top` best: 1 means the return rests on a few names) and `best_days_share` (the share of the total that
                              came from the five best bars)
    by_size, by_liquidity     contribution and average gross exposure by lowest, middle and highest third of market capitalisation (needs `mkt_cap`) and of 30-bar average traded
                              value (needs `volume`), the thirds taken among the eligible securities on each signal date
    by_group                  the same by a label you supply for each security (`groups`: a Series indexed by ticker, for example the sector)
    waterfall                 from gross to net, in basis points per year: the gross return, each cost the engine charged, what is left, and `residual` (the part the engine's net
                              return has that this accounting does not: a freeze markdown, the effect of a ruin)
    """
    from .portfolio import _forward_arrays
    H = result.holdings
    if H is None or result.net_returns is None:
        raise ValueError("the result has no holdings or net returns to attribute")
    n_top = int(n_top)
    if n_top < 1:
        raise ValueError(f"n_top must be at least 1, got {n_top!r}")
    n = len(result.net_returns)
    ppy = panel.periods_per_year
    years = n / ppy
    H = np.asarray(H, dtype=np.float64)[:n]
    fwd, _, _ = _forward_arrays(panel, False, result.spec.get("delist_return"))
    C = H * fwd[:n]
    per_bar = C.sum(axis=1)
    long_c = float(np.where(H > 0, C, 0.0).sum() / years)
    short_c = float(np.where(H < 0, C, 0.0).sum() / years)
    names = pd.Series(C.sum(axis=0) / years, index=panel.tickers)
    pos = names[names > 0].sort_values(ascending=False)
    top = names.sort_values(ascending=False).head(n_top)
    bottom = names.sort_values().head(n_top)
    best5 = np.sort(per_bar)[::-1][:5].sum()
    tot = float(per_bar.sum())
    out = {"gross_annual": float(per_bar.mean() * ppy), "long_annual": long_c, "short_annual": short_c,
           "names": {"top": {str(k): float(v) for k, v in top.items()}, "bottom": {str(k): float(v) for k, v in bottom.items()}, "n_positive": int((names > 0).sum()),
                     "n_held": int((np.abs(H).sum(axis=0) > 0).sum()), "top_share": float(pos.head(n_top).sum() / pos.sum()) if len(pos) else float("nan"),
                     "best_days_share": float(best5 / tot) if tot > 1e-12 else float("nan")}}

    def by_bucket(bucket: np.ndarray, labels) -> dict:
        b = bucket[:n]
        gross_exp = np.abs(H).sum()
        return {lab: {"contribution_annual": float(C[b == k].sum() / years), "exposure_share": float(np.abs(H)[b == k].sum() / gross_exp) if gross_exp > 0 else float("nan")}
                for k, lab in enumerate(labels)}

    if panel.mkt_cap is not None:
        out["by_size"] = by_bucket(_terciles(panel.mkt_cap.where(panel.mkt_cap > 0), panel.eligible), ("smallest third", "middle third", "largest third"))
    if panel.volume is not None:
        out["by_liquidity"] = by_bucket(_terciles(panel.adv(30), panel.eligible), ("least liquid third", "middle third", "most liquid third"))
    if groups is not None:
        if not isinstance(groups, pd.Series):
            raise ValueError("groups must be a pandas Series indexed by ticker")
        lab = groups.reindex(panel.tickers)
        res = {}
        for g in sorted(lab.dropna().unique(), key=str):
            m = (lab == g).to_numpy()
            res[str(g)] = {"contribution_annual": float(C[:, m].sum() / years), "exposure_share": float(np.abs(H[:, m]).sum() / np.abs(H).sum()) if np.abs(H).sum() > 0 else float("nan")}
        if lab.isna().any():
            m = lab.isna().to_numpy()
            res["(no group)"] = {"contribution_annual": float(C[:, m].sum() / years), "exposure_share": float(np.abs(H[:, m]).sum() / np.abs(H).sum()) if np.abs(H).sum() > 0 else float("nan")}
        out["by_group"] = res
    mt = result.metrics
    finite = lambda k: k in mt and isinstance(mt[k], (int, float)) and math.isfinite(mt[k])          # noqa: E731
    if "spread_annual_bp" in mt:                                                      # backtest_weights reports every cost on its own
        keys = (("spread_annual_bp", "spread"), ("side_cost_annual_bp", "buy/sell side costs"), ("borrow_annual_bp", "short borrow"),
                ("impact_annual_bp", "market impact"), ("funding_annual_bp", "funding"))
    else:                                                                             # backtest_portfolio: the spread and the side costs are one figure
        keys = (("cost_annual_bp", "spread and buy/sell side costs"), ("impact_annual_bp", "market impact"), ("funding_annual_bp", "funding"))
    costs = [(label, float(mt[k])) for k, label in keys if finite(k)]
    gross_bp = out["gross_annual"] * 1e4
    net_bp = float(result.net_returns.mean() * ppy * 1e4)
    spent = sum(v for _, v in costs)
    out["waterfall"] = {"gross_bp": gross_bp, "costs": costs, "net_bp": net_bp, "residual_bp": float(net_bp - (gross_bp - spent))}
    return out
