# Ambiguities in B_statistics_spec.md (implementer B)

Likelihood = chance a different careful reader picks something else.
"Textbook" = I used my own knowledge to fill the gap (stated per entry).

## newey_west_t / mean_tests / alpha_beta

1. **Long-run variance normalisation.** Spec: "Newey-West (Bartlett kernel, no small-sample correction)". I used
   S = g0 + 2*sum_{l=1..L} (1 - l/(L+1)) g_l with g_l = (1/T) sum u_t u_{t-l}, demeaned u, se = sqrt(S/T). This equals statsmodels
   HAC with use_correction=False and maxlags=L (checked to 1e-12). Spec only states this for alpha_beta; for `newey_west_t` and
   `mean_tests` it says nothing about demeaning or the divisor. Likelihood: low for alpha_beta, **medium for newey_west_t**
   (a reader could use ddof=1 variance or skip demeaning).
2. **Meaning of `lag` in the Bartlett weight.** I read "lag" as L = max lag included, weights 1 - l/(L+1) (statsmodels `maxlags`).
   Another reading is weights 1 - l/L with l up to L-1. Likelihood: low-medium. lag=0 means plain variance (population, /T).
3. **T used in the plug-in rule** `int(4 (T/100)^(2/9))`: I used T after dropping NaN/inf (and after alignment in alpha_beta). Zero bars
   are included in T for mean_tests (they are only excluded from win_rate/sign test). Likelihood: low.
4. **Lag larger than T-1**: I silently cap the summation at T-1. Spec silent. Low.
5. **nw_p**: "two-sided normal p-value" -> erfc(|t|/sqrt2). Clear. alpha_p the same (spec says compared with a normal).
6. **mean_tests extra keys.** The task lists 7 keys; I also return lag, sharpe_t (= annualised Sharpe * sqrt(T/ppy), Sharpe with ddof=1),
   skew, kurtosis. Spec for skew/kurtosis does not say biased or bias-corrected; I used population (biased) moments, Pearson
   kurtosis (normal = 3). Likelihood of a different choice (pandas .skew()/.kurt() bias-corrected, or excess): **medium-high** for skew/kurtosis values.
   Note: "years" in sharpe_t = T/periods_per_year (assumed).
7. **mean_tests with no non-zero bars**: win_rate/sign_p = nan; win_ci = (nan,nan). Spec silent. Low importance.
8. **alpha_beta residual vol**: "annualised residual volatility" - I used std(resid, ddof=1)*sqrt(ppy). Alternatives: ddof=0, or
   ddof=k (degrees-of-freedom corrected). Likelihood: **medium**.
9. **alpha_beta factor names** in `betas`: DataFrame column names; a Series uses its `.name`, or "factor" if it has none.
   Spec silent for unnamed Series. Likelihood of mismatch: medium (could be e.g. 'beta', 'market', 0).
10. **min_obs rule**: "less than max(min_obs, 3*number of coefficients)": I count the intercept in the coefficient count
    (k = n_factors + 1). Could be excluded. Low impact (only matters for >19 factors).
11. **Constant / collinear detection**: constant = max-min == 0 exactly; collinear = matrix_rank(X) < k (numpy default tolerance).
    Error type ValueError (spec says only "오류"). Low.
12. **r2** = 1 - SSR/SST (centred). Not defined in spec; textbook.
13. Spec text (8.2) says "|t| below 2.5 is no evidence" - no code consequence.

## deflated_sharpe

14. **Formula not given.** I used Bailey & Lopez de Prado (2014) from memory: SR0 = sd(SR across columns) *
    ((1-g) Phi^-1(1-1/N) + g Phi^-1(1-1/(N e))), g = Euler-Mascheroni, N = trials;
    DSR = Phi((SR_best - SR0) sqrt(T-1) / sqrt(1 - skew*SR_best + (kurt-1)/4 SR_best^2)). The spec only says "열 사이 샤프의 분산으로 우연
    수준을 재므로" and that one column gives benchmark 0. Likelihood that the reference differs in some detail: **medium-high**
    (e.g. sqrt(T) instead of sqrt(T-1); the (kurt-1)/4 term; whether N appears as N or N-1).
15. **Variance across columns: ddof.** I used ddof=1 (sample variance of per-bar Sharpes). np.var default would be ddof=0. Likelihood: **medium**
    (2.5 percent shift in SR0 for N=20; changes dsr in the 3rd decimal).
16. **Units of returned `sharpe` and `sharpe_luck_benchmark`.** Spec: "periods_per_year: 보고용 연율화(통계 자체는 봉 단위)". I compute dsr per bar
    and report `sharpe` and `sharpe_luck_benchmark` multiplied by sqrt(ppy). It might be that only `sharpe` is annualised, or neither. Likelihood: **medium**.
17. **`best`**: I return the integer column position (argmax of per-bar Sharpe), also for DataFrame input. Could be a label. Medium-low.
18. **Skew / kurtosis** of the best column: population moments, Pearson kurtosis (3 = normal), consistent with "kurtosis 3 is normal" in
    mean_tests. Whether the DSR formula gets biased or unbiased moments is unspecified. Medium.
19. **trials < 2 with >=2 columns**: Phi^-1(0) is -inf; I set benchmark to 0 when trials < 2. Spec silent. Low.
    **trials < number of columns**: used as given (no max()). Low-medium.
20. **Constant columns** (zero std): Sharpe nan, excluded via nanstd/nanargmax. Spec silent.
21. Spec sentence "열이 1개면 무시되고 우연 수준은 0" - implemented. Spec claims no >0.95 in 400 noise sets with 20 cols x 750; mine gave 0/100.
    Cannot verify the 400 exactly.

## pbo_cscv

22. **Block cutting when T % blocks != 0.** Spec silent. I use L = T // blocks and drop the remainder rows at the START (keep latest).
    Alternatives: drop at the end, np.array_split with uneven blocks. Likelihood of different choice: **medium-high** when T is not divisible
    (identical result when divisible).
23. **Tie handling in rank**: average rank for ties; omega = rank/(N+1); counted when omega <= 0.5 (spec: "at or below the median").
    For N even, "at or below median" = rank <= N/2 ... with N+1 denominator, rank N/2 gives omega < 0.5, rank N/2+1 gives >0.5. Same thing. For odd N the median rank
    (N+1)/2 gives omega exactly 0.5 and is counted. A reader using rank <= N/2 (no +1) would differ for odd N. Medium-low.
24. **Sharpe used inside CSCV** is per-column mean/std(ddof=1) over the concatenated half; ranking is invariant to ddof and annualisation.
    The two reported means are multiplied by sqrt(ppy). Spec says only "보고용 연율화". Low.
25. **Enumeration.** "every way of choosing half" -> all C(S, S/2) combinations (both a split and its complement are counted separately, so 12870 for S=16).
    The original CSCV paper does this too; a reader might dedupe complements (6435). Medium-low. `splits` = number evaluated.
26. **max_splits sampling**: random sample of distinct combinations with numpy default_rng(seed); not reproducible vs another implementation. Unavoidable.
    If max_splits >= total, full enumeration. Odd `blocks` raises ValueError (spec: "짝수 권장" i.e. recommended even, not required - I made it an error). Low-medium.
27. If every IS Sharpe is nan the split is skipped.

## binom / wilson / multiple testing

28. **binom_two_sided**: min(1, 2*P(X <= min(k, n-k))), X~Bin(n, 0.5). Matches scipy binomtest for p = 0.5 (checked n up to 51). n=0 -> 1.0. Low.
29. **wilson**: z = Phi^-1(0.975) = 1.959964 (exact), not 1.96. Differences ~1e-5. Likelihood a reference uses 1.96: **medium** (only matters at tolerance < 1e-4).
    Interval clipped to [0,1]. n = 0 -> (nan, nan).
30. **bh_fdr / holm**: standard step-up/step-down with monotonic envelope and clip at 1; verified vs statsmodels multipletests. NaN p-values are not handled.
    Low ambiguity.

## risk_profile

31. **Quantile definition**: np.quantile linear interpolation (type 7). Spec says only "the return that 5 percent of bars fell below". Likelihood of a different
    method (lower/nearest/pandas default is also linear): low-medium.
32. **CVaR**: mean of returns <= VaR (inclusive); spec: "the mean return of the bars below it". With interpolated quantile the difference exists only when the quantile
    equals a data point. Low-medium.
33. **EWMA / filtered historical simulation (largest gap).** Spec: "returns scaled to today's volatility (an exponentially weighted estimate, ewma_lambda...)".
    I implemented: zero-mean RiskMetrics recursion s2[t] = lam*s2[t-1] + (1-lam)*r[t-1]^2, initialised s2[0] = sample variance (ddof=1) of the whole series;
    standardised z_t = r_t / sqrt(s2[t]) (one-step-ahead forecast, so no look-ahead for the bar itself); `vol_now` = sqrt(lam*s2[T-1] + (1-lam)*r[T-1]^2), the
    forecast for the NEXT bar; var_*_now = quantile(z) * vol_now. Per-bar (not annualised) vol_now. Alternatives: s2 includes r_t itself (filtered not forecast),
    different initial value (first return squared, first 20 bars), mean-demeaned returns, annualised vol_now. Likelihood of a different choice: **very high**; expect numbers
    to disagree unless the reference matches these exactly.
34. **var_95_normal** = mean + Phi^-1(0.05) * std(ddof=1). ddof could be 0. Low-medium.
35. **tail_ratio** = q95/|q05| with the same quantile method. Clear.
36. **max_dd_bars**: longest run of consecutive bars with equity < running peak (start capital 1 counts as peak). Hand case [-.1,.05,.1] gives 2. Alternative reading counts
    from the peak bar to the recovery bar (3), or counts bars from the peak. Likelihood: **medium**. Also plain float comparison, no tolerance (a recovery that lands
    1e-16 below the peak counts as underwater).
37. **share_underwater** = fraction of the n bars with equity < peak. Low.
38. **ulcer_index** = sqrt(mean(dd^2)) over the n bars (the initial point not included), dd = equity/peak - 1. Alternative: include the t=0 point (adds a zero). Low-medium.
39. **gain_to_pain** = sum(r) / sum(|r| for r<0) (note: "sum of returns", not sum of gains). If no losses: +inf if sum>0, else nan. Spec silent on no-loss. Low-medium.
40. **total_return** = prod(1+r)-1. Clear. **ulcer_performance** CAGR/ulcer with CAGR = eq^(ppy/n)-1 (bar-count based). Not in required keys.
41. best_month/worst_month only when the index is a DatetimeIndex (calendar-month compounding); otherwise omitted. Spec lists them; the task key list does not.
42. NaN/inf bars are dropped. Spec silent.

## market_relative

43. **CAGR**: eq_end ** (ppy / n) - 1 using the number of shared bars n, not calendar years. A reference using (last date - first date)/365.25 would differ. Likelihood: **medium-high**
    (the spec's signature takes periods_per_year, which supports my reading, but "CAGR" is never defined).
44. **Alignment**: inner join on index, then drop rows with NaN/inf in either. Non-Series inputs aligned positionally. Spec: "on the dates they share". Low.
45. **hit_rate_month/year**: calendar groups via index.to_period; compounded return strictly greater than the benchmark's; partial years/months counted (spec says so). Ties are
    not wins. Needs DatetimeIndex, else nan. Low-medium (strict vs >=).
46. **up/down capture**: bench > 0 and bench < 0 strictly (zero days excluded); ratio of means as described. Low.
47. **beta** = cov/var (ddof cancels); **correlation** Pearson; **tracking_error** = std(diff, ddof=1)*sqrt(ppy) (ddof could be 0: medium-low); **information_ratio** = mean(diff)/std(diff, ddof=1)*sqrt(ppy).
    Spec: "annualised mean difference over its standard deviation" - IR is therefore mean*ppy / (std*sqrt(ppy)) = mean/std*sqrt(ppy); I read the denominator as the annualised std,
    which is the only sensible one; a reading with mean*ppy / std(daily) would be wrong dimensionally but possible. Low.
48. **max_relative_drawdown**: ratio equity_s/equity_b including a starting ratio of 1 as peak; min of ratio/peak-1. Low-medium (whether the start counts).

## Possible contradictions / suspicious passages

- Spec header for `deflated_sharpe` says "보수적이다 ... 0.95 초과 0회" and also "시도 횟수가 열 수보다 크면 trials로 넘겨야 한다", while `trials` is documented to have effect "only when >=2 columns";
  fine, but nothing says what a trials LOWER than the number of columns does (see 19).
- `mean_tests` docstring lists keys (sharpe_t, skew, kurtosis) beyond the task's required list; `risk_profile` docstring lists best_month, worst_month, skew, kurtosis, ulcer_performance not in the
  required key list. Not a contradiction, but the task list and the docstrings differ.
- `alpha_beta` docs: "unless lag is given" with int(4*(T/100)^(2/9)); the docstring says alpha_annual is "arithmetic" while `appraisal` is "alpha_annual over the annualised residual volatility" - the
  residual vol is not defined (ddof) (see 8).
- In `pbo_cscv`, the parameter doc says "짝수 권장 (recommended even)" while the docstring needs a half of the blocks; for odd S the "half" is undefined. I raise an error.
- `market_relative`: "years with a part-year are counted" - consistent with my implementation, but the CAGR of a part-year is not discussed.
- The spec gives few numeric examples; nothing in it could be reproduced numerically except qualitative claims (HAC 9.4 percent, DSR 0/400 on noise: mine 0/100 on noise).
- Statement "var_95_now ... after a calm stretch the plain historical figure understates the risk of the next bar" means vol_now is a forecast of the NEXT bar, supporting my reading in 33.
- Section 8.3: "alpha_beta = statsmodels HAC with coefficient 1e-12, t 1e-9" supports reading 1 (use_correction False) - which my implementation matches to 1e-8..1e-12.
