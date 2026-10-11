# Ambiguities and suspected spec problems (implementer A)

Likelihood = chance that another careful reader chose differently.

## Genuine gaps
1. **turnover_daily is not defined anywhere in the spec** (only "비용·회전율이 계산된다(순비중이 아니라 레그별 변화량의 합)", sec. backtest_portfolio / 보유). I defined it as the mean over signal-date rows 1..n-1 of sum over both legs of sum_i |dW| (n = T-lag-1; row 0 excluded because "첫 행에는 비용을 매기지 않는다"). Not halved, not annualised, long-only has only one leg. HIGH that others differ (include row 0; annualise; halve; mean over all rows; one-way vs two-way).
2. **Ruin inside metrics.** The task text says metrics "includes what a ruin does: see stop_at_ruin", but the spec text of `metrics` never mentions ruin. I apply `stop_at_ruin(net, len(net))` inside `metrics_from_returns` before computing anything. MEDIUM (a reader could keep metrics raw; then a day <= -100% gives negative equity and CAGR NaN via negative base).
3. **Meaning of `cut` in stop_at_ruin.** "within the first `cut` rows" -> I search net[:cut] (indices 0..cut-1). A ruin at index >= cut is ignored (no change at all, returns None). LOW-MEDIUM. Also: after a ruin at row i, I zero ALL later rows, including rows beyond `cut`. MEDIUM (could stop at cut).
4. **Ruin threshold.** "loses 100 percent or more" -> net <= -1.0 (equality counts). Row i set to exactly -1.0. NaN never triggers. LOW.
5. **Short-sample metrics.** n < ann//2: CAGR, MDD, Sharpe = NaN + warning. "(나머지 키는 없다)" is odd Korean; the docstring says "NaN for CAGR, MDD and Sharpe only", so I compute Sortino, vol, pos_days normally. Calmar then is NaN (derived from NaN CAGR/MDD) although not listed. MEDIUM on Calmar; "나머지 키는 없다" can literally read "the other keys are absent" -> LOW-MEDIUM that others return a dict with fewer keys.
6. **Calmar when MDD == 0** (no drawdown): spec silent; I return NaN (not inf). MEDIUM.
7. **CAGR with non-positive terminal wealth**: after stop_at_ruin terminal wealth is 0 -> CAGR = -1 naturally. Negative wealth cannot occur after the stop. n/a mostly.
8. **vol ddof.** Spec says ddof=1 for Sharpe, only "std" for vol. I use ddof=1 for both. LOW-MEDIUM.
9. **Sortino NaN rule**: "표준편차나 하방편차가 사실상 0 (<=1e-12) -> Sharpe and Sortino NaN" is ambiguous: does a zero *std* also NaN the Sortino and a zero *downside deviation* NaN the Sharpe? I read it as each ratio checking its own denominator (Sharpe: std, Sortino: downside dev). Consequence: all-positive series gives Sortino NaN (no downside) but Sharpe finite. MEDIUM. Also compares "<= 1e-12" on std itself, not annualised.
10. **Single observation / empty net**: not specified; n==0 all NaN, n==1 std NaN (ddof=1). LOW (edge).
11. **NaN inside net** not specified; not handled (propagates).
12. **Rank percentile.** "순위 백분위" -> `rank(pct=True)` among eligible & non-NaN names, ties averaged ("average"), computed per date; long: pct >= 1-long_q, short: pct <= short_q, raw float comparison (no tolerance). Example: 4 names, q=.25 -> long set has TWO names (ranks .75 and 1.0) because 1-q=.75 is included, while short set has one. I consider this consistent with the "boundary included" docstring. Floating-point at the boundary (e.g. 1-0.3 vs 21/30) is a risk. MEDIUM that others use tolerance / method='first' / (rank-1)/(n-1) percentiles.
13. **Empty-leg days inside the tranche average.** If a signal day has no eligible names, its leg is all zeros and still counts as one of the `hold` tranches (denominator min(d+1, hold) counts days, not non-empty tranches). The spec says "최근 h개 신호일의 레그를 평균". MEDIUM.
14. **Warm-up average.** "표본 앞 h-1일은 있는 만큼" -> divide by the number of available tranches (min(d+1,h)), so the book is fully invested from day 0 (leg sums to 1), not 1/h invested. The earlier sentence "매일 자본의 1/h를 ... 새로 진입" could be read as sum/h (partially invested in warm-up). MEDIUM.
15. **Costs on drifting weights**: weights are treated as target weights with no drift; cost = sum |W[d]-W[d-1]| over both legs separately, charged on row d (signal-date index), first row zero. Cost on row d is thus attributed to the same row's return (not shifted). MEDIUM-LOW.
16. **Leg magnitude convention.** Short weights stored as positive magnitudes; net = long gross - short gross. Spec says "롱·숏 레그는 각각 합이 1" so gross exposure 2 for long-short. LOW.
17. **Missing prices.** ret[t] = close[t]/close[t-1]-1, NaN -> 0 (so a jump over a gap vanishes, as the spec warns). Infinite returns (zero price) not handled. LOW.
18. **Leading rows**: rows before any eligible name give 0.0 returns (not NaN). Spec says only the last lag+1 rows are cut; I never emit NaN. LOW-MEDIUM.
19. **weighting="signal"/"rank"**: signal = |score| normalised within leg (a leg with all-zero scores becomes empty, not equal-weighted); rank weights use the same percentile r, long r-(1-q), short q-r, so names exactly on the boundary get weight 0 (a leg consisting only of such names is empty). Fallback for a zero-sum leg not specified. MEDIUM. Not part of the required example numbers.
20. **Validation**: long_q+short_q>1 raises, short_q=0 raises, hold<1 raises, entry_lag<0 raises, boolean factor raises, spread_bp negative/non-finite raises. Raising ValueError vs other types unspecified. Irrelevant to numbers.
21. **Eligible/factor reindexing**: eligible reindexed to close, NaN->False (spec 3.1); factor reindexed with NaN. LOW.
22. **Return type/index**: `pd.Series` of float indexed by close.index[:T-lag-1]; empty series if T-lag-1 <= 0. LOW.

## Possible inconsistencies / suspicious points in the spec
- turnover_daily is requested but absent from the excerpt (see 1).
- `metrics` docstring says keys include "years" and drawdown dates, and `mdd_peak`="start"; the task asks for 7 keys only, so I omit these.
- 3.3 says "매일 자본의 1/h를 ... 새로 진입" and "롱·숏 레그는 각각 합이 1": consistent only in steady state; warm-up behaviour is described only by "있는 만큼" (see 14).
- "(나머지 키는 없다)" in the metrics paragraph vs docstring "NaN ... only": possible mistranslation of "the other keys are not affected".
- 3.4 "스칼라 spread_bp = 왕복" so cost per unit traded is spread/2 per side; "첫 행 비용 없음" means first *signal-date* row; with hold>1 the first rows are partially built but still charged since only row 0 is exempt (my reading).
- Percentile boundary: long_q=short_q=.25 on 4 names gives asymmetric legs (2 long, 1 short); docstring ("comparisons include the boundary") confirms it is intended, but looks odd.
- stop_at_ruin docstring: "The row is a signal date; the loss itself is booked entry_lag + 1 rows later" - consistent with indexing by signal date, no action needed here.
