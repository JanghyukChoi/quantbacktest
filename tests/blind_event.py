# Reference implementation written from the specification alone by an assistant that had not seen the library (docs/blind_review/README.md); unchanged except for this line.
# tests/test_blind_reference.py compares it with the library.
"""Blind implementation of spec area C (event engine + gates). numpy/pandas only."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

MIN_TRADES = 100      # per horizon, spec: "유효 발화가 100건 미만이면 그 기간은 제외"
MIN_POOL = 20         # spec: "같은 날 적격 종목이 20개 미만이면 기저승률에서 빠져"
MIN_DAYS_NET = 100    # spec: below this backtest_event only warns


# ---------------------------------------------------------------- Newey-West
def newey_west_t(series, lag):
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    T = int(len(x))
    if T < 2:
        return (float("nan"), float("nan"), float("nan"), T)
    m = float(x.mean())
    e = x - m
    L = int(max(0, min(int(lag), T - 1)))
    var = float(np.dot(e, e)) / T
    for k in range(1, L + 1):
        g = float(np.dot(e[k:], e[:-k])) / T
        var += 2.0 * (1.0 - k / (L + 1.0)) * g
    if not np.isfinite(var) or var <= 0:
        return (m, float("nan"), float("nan"), T)
    se = math.sqrt(var / T)
    return (m, se, m / se, T)


# ---------------------------------------------------------------- event stats
def event_stats(close, eligible, signal, horizon, cost_bp, entry_lag=1):
    lag = int(entry_lag)
    h = int(horizon)
    nan = float("nan")
    fwd = close.shift(-(lag + h)) / close.shift(-lag) - 1.0
    elig = eligible.reindex_like(close).fillna(False).astype(bool)
    sig = signal.reindex_like(close).fillna(False).astype(bool)
    valid = fwd.notna()
    fire = sig & elig & valid
    pool = elig & valid

    fv = fwd.values
    fire_v = fire.values
    pool_v = pool.values
    cost = cost_bp / 1e4

    pnl = (fv[fire_v] - cost) * 1e4          # bp, net of round-trip cost
    n_trades = int(pnl.size)
    out = dict(n_trades=n_trades, horizon=h, excluded=False)

    # --- day level
    n_fire_d = fire_v.sum(1)
    n_pool_d = pool_v.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd0 = np.where(pool_v, fv, 0.0)
        pool_mean = fwd0.sum(1) / n_pool_d
        fire_mean = np.where(fire_v, fv, 0.0).sum(1) / n_fire_d
        pool_win = (np.where(pool_v, (fv - cost) > 0, False)).sum(1) / n_pool_d
    day_ok = (n_fire_d >= 1) & (n_pool_d >= MIN_POOL)
    excess = (fire_mean - pool_mean)[day_ok] * 1e4
    excess_dates = close.index[day_ok]

    if n_trades < MIN_TRADES:
        out.update(excluded=True, win_rate=nan, base_rate=nan, lift_pp=nan, mean_bp=nan,
                   median_bp=nan, avg_win_bp=nan, avg_loss_bp=nan, payoff=nan, p25_bp=nan,
                   p75_bp=nan, skew_warning=False, excess_bp=nan, t=nan, n_days=int(excess.size),
                   net_bp=nan, daily_excess=excess, dates=excess_dates, short_series_warning=False)
        return out

    win = pnl > 0
    win_rate = 100.0 * win.mean()
    # base rate: per-day eligible win share weighted by that day's fire count
    w = n_fire_d[day_ok].astype(float)
    if w.sum() > 0:
        base = 100.0 * float((pool_win[day_ok] * w).sum() / w.sum())
    else:
        base = nan
    wins = pnl[win]
    losses = pnl[~win]
    avg_win = float(wins.mean()) if wins.size else nan
    avg_loss = float(losses.mean()) if losses.size else nan
    payoff = avg_win / abs(avg_loss) if (wins.size and losses.size and avg_loss != 0) else nan
    mean_bp = float(pnl.mean())
    med = float(np.median(pnl))

    lagnw = max(h, 21)
    if excess.size:
        ex_mean, _, t, _ = newey_west_t(excess, lagnw)
    else:
        ex_mean, t = nan, nan
    out.update(
        win_rate=float(win_rate), base_rate=base,
        lift_pp=float(win_rate - base) if np.isfinite(base) else nan,
        mean_bp=mean_bp, median_bp=med, avg_win_bp=avg_win, avg_loss_bp=avg_loss, payoff=payoff,
        p25_bp=float(np.percentile(pnl, 25)), p75_bp=float(np.percentile(pnl, 75)),
        skew_warning=bool(mean_bp > 0 and med < 0),
        excess_bp=float(ex_mean), t=float(t), n_days=int(excess.size),
        net_bp=float(ex_mean - cost_bp) if excess.size else nan,
        daily_excess=excess, dates=excess_dates,
        short_series_warning=bool(excess.size < MIN_DAYS_NET),
    )
    return out


# ---------------------------------------------------------------- fire structure
def fire_structure(fire, dates, tickers):
    F = np.asarray(fire, dtype=bool)
    nd = F.shape[0]
    tickers = pd.Index(tickers)
    per = F.sum(0)
    out = dict(n_days=int(nd), n_fire=int(F.sum()))
    if nd == 0 or F.shape[1] == 0 or per.max() == 0:
        out.update(top_name=None, top_name_share=0.0 if nd else float("nan"), longest_run=0,
                   longest_run_name=None, turnover=float("nan"), n_turnover_pairs=0)
        return out
    j = int(np.argmax(per))
    out["top_name"] = tickers[j]
    out["top_name_share"] = float(per[j] / nd)
    best, best_j = 0, 0
    for c in range(F.shape[1]):
        col = F[:, c]
        if not col.any():
            continue
        # longest True run
        d = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        r = int((ends - starts).max())
        if r > best:
            best, best_j = r, c
    out["longest_run"] = best
    out["longest_run_name"] = tickers[best_j]
    A, B = F[:-1], F[1:]
    inter = (A & B).sum(1)
    uni = (A | B).sum(1)
    ok = uni > 0
    out["n_turnover_pairs"] = int(ok.sum())
    out["turnover"] = float(np.mean(1.0 - inter[ok] / uni[ok])) if ok.any() else float("nan")
    return out


# ---------------------------------------------------------------- sub-results
def _clean(values, dates):
    v = np.asarray(values, dtype=float)
    d = pd.DatetimeIndex(dates)
    assert len(v) == len(d)
    return v, d


def _sgn(x):
    return int(np.sign(x)) if np.isfinite(x) else None


def subperiod(values, dates, lag=21):
    v, d = _clean(values, dates)
    n = len(v)
    h = n // 2
    a, b = v[:h], v[h:]
    am, _, at, an = newey_west_t(a, lag)
    bm, _, bt, bn = newey_west_t(b, lag)
    sa, sb = _sgn(am), _sgn(bm)
    return dict(first_mean=am, first_t=at, first_n=an, second_mean=bm, second_t=bt, second_n=bn,
                split_date=d[h] if n and h < n else None,
                same_sign=bool(sa is not None and sb is not None and sa == sb))


def yearly(values, dates, lag=21):
    v, d = _clean(values, dates)
    rows = []
    for y in sorted(set(d.year)):
        x = v[d.year == y]
        m, _, t, T = newey_west_t(x, lag)
        if T == 0:
            continue
        rows.append(dict(year=int(y), mean=m, t=t, n_days=T, positive=bool(m > 0)))
    ny = len(rows)
    npos = sum(r["positive"] for r in rows)
    return dict(years=rows, n_years=ny, n_positive=npos,
                positive_share=(npos / ny) if ny else float("nan"))


def oos_holdout(values, dates, months=12, lag=21):
    v, d = _clean(values, dates)
    if len(d) == 0:
        return dict(mean=float("nan"), t=float("nan"), n_days=0, start=None, is_mean=float("nan"), is_n_days=0)
    cut = d[-1] - pd.DateOffset(months=months)
    m = d > cut
    om, _, ot, on = newey_west_t(v[m], lag)
    im, _, it, inn = newey_west_t(v[~m], lag)
    return dict(mean=om, t=ot, n_days=on, start=d[m][0] if m.any() else None, cutoff=cut,
                positive=bool(om > 0) if np.isfinite(om) else False,
                is_mean=im, is_t=it, is_n_days=inn)


# ---------------------------------------------------------------- gates
DEFAULT_CFG = dict(null_threshold=None, fallback_t=3.0, require_net_positive=True,
                   require_subperiod_sign=True, min_monotonicity=0.5, min_yearly_positive=0.6,
                   max_single_name_share=0.5, min_turnover=0.05)


def run_gates(daily_excess, dates, t_stat, net_bp, rho, fire=None, tickers=None, cfg=None, lag=21):
    c = dict(DEFAULT_CFG)
    if cfg:
        unknown = set(cfg) - set(c)
        if unknown:
            raise KeyError(unknown)
        c.update(cfg)
    thr = c["null_threshold"] if c["null_threshold"] is not None else c["fallback_t"]
    de = np.asarray(daily_excess, dtype=float)
    sp = subperiod(de, dates, lag)
    yr = yearly(de, dates, lag)
    oos = oos_holdout(de, dates, 12, lag)
    st = None
    if fire is not None:
        tk = tickers if tickers is not None else pd.RangeIndex(np.asarray(fire).shape[1])
        st = fire_structure(fire, dates, tk)

    def fin(x):
        return x is not None and np.isfinite(x)

    steps = []
    steps.append(dict(name="G1 statistical threshold", passed=bool(fin(t_stat) and abs(t_stat) > thr),
                      value=t_stat, threshold=thr))
    steps.append(dict(name="G2 net profit after costs",
                      passed=bool((not c["require_net_positive"]) or (fin(net_bp) and net_bp > 0)),
                      value=net_bp, threshold=0.0))
    steps.append(dict(name="G3 sub-period sign",
                      passed=bool((not c["require_subperiod_sign"]) or sp["same_sign"]),
                      value=(sp["first_mean"], sp["second_mean"]), threshold=None))
    steps.append(dict(name="G4 decile monotonicity",
                      passed=bool(fin(rho) and abs(rho) >= c["min_monotonicity"]),
                      value=rho, threshold=c["min_monotonicity"]))
    steps.append(dict(name="G5 yearly consistency",
                      passed=bool(fin(yr["positive_share"]) and yr["positive_share"] >= c["min_yearly_positive"]),
                      value=yr["positive_share"], threshold=c["min_yearly_positive"]))
    if st is None:
        g6 = False
    else:
        g6 = bool(fin(st["top_name_share"]) and st["top_name_share"] <= c["max_single_name_share"]
                  and fin(st["turnover"]) and st["turnover"] >= c["min_turnover"])
    steps.append(dict(name="G6 firing structure", passed=g6,
                      value=None if st is None else (st["top_name_share"], st["turnover"]),
                      threshold=(c["max_single_name_share"], c["min_turnover"])))
    failed = next((s["name"] for s in steps if not s["passed"]), None)
    return dict(passed=failed is None, failed_at=failed, steps=steps, subperiod=sp, yearly=yr,
                oos=oos, oos_holdout=oos, structure=st, threshold=thr)
