"""추정량 — Fama-MacBeth · Newey-West · 십분위 · 셔플 귀무.

중첩 수익률 주의
  h일 보유 수익률을 매일 계산하면 h-1 일치 구간이 겹친다. 독립 관측이 실제로는 훨씬 적으므로
  보정 없이는 t 가 몇 배로 부풀려진다. 모든 t 는 Newey-West 로 낸다.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

EPS = 1e-12


def newey_west_t(series: np.ndarray, lag: int) -> tuple[float, float, float, int]:
    """중첩 수익률용 NW 보정 t. 반환 (mean, se, t, T)."""
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 30:
        return float("nan"), float("nan"), float("nan"), T
    mu = x.mean()
    xd = x - mu
    s = (xd @ xd) / T
    for lg in range(1, min(lag, T - 1) + 1):
        w = 1.0 - lg / (lag + 1.0)
        s += 2.0 * w * (xd[lg:] @ xd[:-lg]) / T
    if s <= 0:
        return float(mu), float("nan"), float("nan"), T
    se = np.sqrt(s / T)
    return float(mu), float(se), float(mu / se), T


def fama_macbeth(factor: pd.DataFrame, fwd: dict[int, pd.DataFrame],
                 controls: dict[str, pd.DataFrame], eligible: pd.DataFrame,
                 *, min_stocks: int | None = None) -> dict:
    """팩터 1개 × 호라이즌 여러개의 FM 계수 시계열 → NW t.

    QR 분해로 통제변수 공간에 직교화한 뒤 계수를 뽑는다.
    통제 있는 t 와 없는 t(t_raw)를 함께 내지만, **보고는 통제 후만 하도록** 호출측에서 강제한다.
    """
    fv = factor.values.astype(np.float64)
    ev = eligible.values
    cvs = [c.values.astype(np.float64) for c in controls.values()]
    hs = sorted(fwd)

    # 표본 하한을 고정하면 소형 유니버스(예: 20종목)에서 전 날짜가 버려진다.
    # 회귀에 필요한 최소치는 (통제변수 + 절편 + 팩터) 이므로 그 두 배를 하한으로 삼되,
    # 유니버스 중앙 크기의 절반을 넘지 않게 한다.
    if min_stocks is None:
        need = 2 * (len(cvs) + 2)
        med = float(np.median(ev.sum(axis=1)))
        min_stocks = int(max(need, min(50, med * 0.5)))
        if med < need:
            warnings.warn(
                f"유니버스 중앙 {med:.0f}종목인데 통제변수가 {len(cvs)}개라 "
                f"회귀 자유도가 부족합니다(최소 {need}종목 필요). "
                f"통제를 줄이거나 종목을 늘리십시오.", stacklevel=2)
    rv = {h: fwd[h].values.astype(np.float64) for h in hs}
    n = fv.shape[0]
    betas = {h: np.full(n, np.nan) for h in hs}
    raws = {h: np.full(n, np.nan) for h in hs}
    n_eligible_days = 0      # 표본은 충분했던 날
    n_collinear = 0          # 그런데 잔차가 0 이라 버린 날 (통제변수와 완전공선)

    for i in range(n):
        f = fv[i]
        ok = ev[i] & np.isfinite(f)
        for c in cvs:
            ok &= np.isfinite(c[i])
        if ok.sum() < min_stocks:
            continue
        fo = f[ok]
        Z = np.column_stack([np.ones(ok.sum())] + [c[i][ok] for c in cvs])
        try:
            Q, _ = np.linalg.qr(Z)
        except np.linalg.LinAlgError:
            continue
        fp = fo - Q @ (Q.T @ fo)
        den = fp @ fp
        foc = fo - fo.mean()
        den_r = foc @ foc
        n_eligible_days += 1
        if den <= EPS or den_r <= EPS:
            n_collinear += 1
            continue
        for h in hs:
            r = rv[h][i]
            okh = ok & np.isfinite(r)
            if okh.sum() < min_stocks:
                continue
            if okh.sum() != ok.sum():
                fo2 = f[okh]
                Z2 = np.column_stack([np.ones(okh.sum())] + [c[i][okh] for c in cvs])
                try:
                    Q2, _ = np.linalg.qr(Z2)
                except np.linalg.LinAlgError:
                    continue
                fp2 = fo2 - Q2 @ (Q2.T @ fo2)
                d2 = fp2 @ fp2
                if d2 <= EPS:
                    continue
                rr = r[okh]
                betas[h][i] = (fp2 @ (rr - Q2 @ (Q2.T @ rr))) / d2
                f2c = fo2 - fo2.mean()
                raws[h][i] = (f2c @ (rr - rr.mean())) / (f2c @ f2c)
            else:
                rr = r[ok]
                betas[h][i] = (fp @ (rr - Q @ (Q.T @ rr))) / den
                raws[h][i] = (foc @ (rr - rr.mean())) / den_r

    out = {}
    for h in hs:
        mu, se, t, T = newey_west_t(betas[h], lag=max(h, 21))
        _, _, t63, _ = newey_west_t(betas[h], lag=max(h, 63))
        mu_r, _, t_r, _ = newey_west_t(raws[h], lag=max(h, 21))
        out[h] = {"coef_bp": mu * 1e4 if np.isfinite(mu) else np.nan, "t": t,
                  "t_nw63": t63, "n_days": T,
                  "coef_bp_raw": mu_r * 1e4 if np.isfinite(mu_r) else np.nan, "t_raw": t_r}

    # 완전공선 진단 — 팩터가 통제변수와 사실상 같으면 잔차가 0 이 되어 전 날짜가 버려진다.
    # 조용히 NaN 을 내면 원인을 알 수 없으므로 명시한다.
    if n_eligible_days and n_collinear / n_eligible_days > 0.5:
        warnings.warn(
            f"팩터가 통제변수와 거의 완전공선입니다 "
            f"({n_collinear}/{n_eligible_days}일에서 잔차≈0). "
            f"통제 후 정보가 남지 않아 t 가 NaN 입니다. "
            f"팩터 정의를 바꾸거나 해당 통제변수를 빼십시오.",
            stacklevel=2)
    for h in hs:
        out[h]["collinear_pct"] = (100.0 * n_collinear / n_eligible_days
                                   if n_eligible_days else np.nan)
    return out


def decile_profile(factor: pd.DataFrame, fwd_h: pd.DataFrame, eligible: pd.DataFrame,
                   *, lag: int = 21, n_bins: int = 10) -> dict:
    """십분위 프로파일. 단조성이 없으면 극단만 다른 것이라 신호로 보기 어렵다."""
    rk = factor.where(eligible).rank(axis=1, pct=True, na_option="keep")
    c = fwd_h.values.astype(np.float64)
    ev = eligible.values
    prof = []
    for d in range(n_bins):
        lo, hi = d / n_bins, (d + 1) / n_bins + (0.01 if d == n_bins - 1 else 0)
        sel = ((rk >= lo) & (rk < hi) & eligible).values
        v = []
        for i in range(c.shape[0]):
            s = sel[i] & np.isfinite(c[i])
            pool = ev[i] & np.isfinite(c[i])
            if s.sum() >= 3 and pool.sum() >= 30:
                v.append(c[i][s].mean() - c[i][pool].mean())
        prof.append(float(np.mean(v) * 1e4) if v else np.nan)
    # 스피어만 상관 = 순위에 대한 피어슨 상관. method="spearman" 은 scipy 가 필요해서 쓰지 않는다
    # (이 패키지의 코어 의존성은 pandas, numpy 둘뿐이다).
    rho = pd.Series(prof).rank().corr(pd.Series(range(n_bins), dtype=float).rank())
    hi_lo = [x for x in (prof[-1], prof[0]) if np.isfinite(x)]
    return {"decile_bp": prof, "monotonicity_rho": float(rho) if pd.notna(rho) else np.nan,
            "spread_bp": float(prof[-1] - prof[0]) if len(hi_lo) == 2 else np.nan}


def shuffle_null(factor_builder, eligible: pd.DataFrame, fwd: dict[int, pd.DataFrame],
                 controls: dict[str, pd.DataFrame], *, n_rep: int = 3,
                 seed: int = 0) -> dict:
    """셔플 귀무 — 다중검정 문턱을 **실측**한다.

    날짜별로 종목 배열만 무작위 치환한다. 날짜 구조·팩터 분포는 보존되고
    종목-팩터 연결만 파괴되므로, 여기서 나온 |t| 분포가 곧 '우연의 크기'다.
    Bonferroni 같은 이론적 보정은 검정 간 상관을 무시해 과도하게 보수적이다.

    factor_builder: (rng) -> DataFrame  — 셔플된 원천으로 팩터를 재구성하는 콜백
    """
    rng = np.random.default_rng(seed)
    ts = []
    for _ in range(n_rep):
        f = factor_builder(rng)
        r = fama_macbeth(f, fwd, controls, eligible)
        ts.extend(abs(v["t"]) for v in r.values() if np.isfinite(v["t"]))
    a = np.array(ts)
    if not len(a):
        return {"n": 0, "p95": np.nan, "max": np.nan}
    return {"n": int(len(a)), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


def shuffle_columns(df: pd.DataFrame, eligible: pd.DataFrame, rng) -> pd.DataFrame:
    """날짜별 eligible 종목 안에서만 값을 치환. shuffle_null 의 표준 셔플러."""
    a = df.values.copy()
    ev = eligible.values
    for i in range(a.shape[0]):
        j = np.where(ev[i])[0]
        if len(j) > 1:
            a[i, j] = a[i, rng.permutation(j)]
    return pd.DataFrame(a, index=df.index, columns=df.columns)


def paired_diff(base_mask: np.ndarray, alt_mask: np.ndarray, cum: np.ndarray,
                *, lag: int = 21, min_n: int = 5) -> dict:
    """짝지은 비교 — 같은 날짜에서 조건을 추가했을 때의 순수 기여.

    조건을 붙이면 헤드라인 숫자는 거의 항상 좋아진다. 그게 진짜 개선인지 보려면
    **같은 날짜·같은 표본**에서 짝지어 비교해야 한다.
    """
    d = []
    for i in range(cum.shape[0]):
        a = base_mask[i] & np.isfinite(cum[i])
        b = alt_mask[i] & np.isfinite(cum[i])
        if a.sum() >= min_n and b.sum() >= min_n:
            d.append(cum[i][b].mean() - cum[i][a].mean())
    if len(d) < 30:
        return {"diff_bp": np.nan, "t": np.nan, "n_days": len(d)}
    mu, _, t, n = newey_west_t(np.array(d), lag=lag)
    return {"diff_bp": float(mu * 1e4), "t": float(t), "n_days": n}
