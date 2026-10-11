# AMBIGUITIES (spec area C, blind implementation)

Likelihood = chance a different careful reader chose otherwise.

## event_stats
1. **Win definition and base-rate cost.** Spec: "손익 `cum - cost`를 모아 승률", "기저승률은 같은 날 적격 종목 전체의 승률". I made win = (fwd - cost) > 0 for fires AND for the eligible pool (net of cost for both, so lift is not biased by the cost). Alternative: pool win gross (fwd>0). Likelihood: **high**.
2. **Which days enter the base rate.** "발화 수로 가중", "적격 20개 미만이면 기저승률에서 빠져". I weight each day's pool win share by that day's fire count, over days with >=1 fire and pool>=20; `win_rate` still uses ALL valid fires (including fires on days dropped from the base). Alternative: restrict win_rate to the same days, or give NaN lift if any day is dropped. Likelihood: **medium-high**.
3. **Does the 20-pool rule also apply to the day-level excess series?** Spec mentions 20 only for base rate. I applied it to the daily excess too (days with pool<20 dropped), and require >=1 fire. Alternative: excess uses every day with a fire. Likelihood: **high**.
4. **Pool definition.** Pool = eligible AND finite forward return on that day; fires with NaN forward are dropped ("유효 발화"). The pool for excess includes the fired names ("적격 전체 평균"). Likelihood: low-medium.
5. **Newey-West lag for t in event_stats.** Spec gives no lag for backtest_event; only `fama_macbeth` says `max(h, 21)` and run_gates defaults lag=21. I used max(horizon, 21). Alternative: fixed 21, or horizon alone. Likelihood: **high**.
6. **Excess is gross, net_bp = excess_bp - cost_bp.** Spec: "비용 뒤 순수익(초과수익 bp - 왕복 비용 bp)". Daily excess itself has no cost (it cancels between fire group and pool). Likelihood: low.
7. **Sign/units.** avg_loss_bp reported negative (mean of the losing pnl), payoff = avg_win / |avg_loss|. Losses = pnl <= 0 (complement of win). Alternative: avg_loss as a positive magnitude; ties pnl==0 excluded. Likelihood: **medium**.
8. **Percentile method.** p25/p75 with numpy linear interpolation default. Low-medium.
9. **Exclusion behaviour.** "유효 발화 100건 미만이면 그 기간은 제외". For a single horizon I return the dict with `excluded=True` and NaN stats instead of raising (ValueError belongs to the all-excluded case at the multi-horizon level). n_days/daily_excess are still filled. Whether the threshold is `<100` on valid fires (as I did) vs raw fires: medium.
10. **Fewer than 100 excess days**: `backtest_event` only warns, so net_bp stays finite; I set a flag `short_series_warning`. Low.
11. **Percent scale.** win_rate/base_rate 0..100, lift_pp their difference. Low.
12. `signal` NaN treated as False; eligible NaN as False. Low.
13. Extra returned keys (`daily_excess`, `dates`, `excluded`, `horizon`) are my additions.

## newey_west_t
14. Spec only says "Newey-West corrected t". I used Bartlett weights 1 - k/(L+1), gamma_k divided by T (not T-k), no small-sample correction, variance of the mean = long-run var / T, NaNs dropped, lag capped at T-1. t=NaN if variance <=0 or T<2. Alternatives: statsmodels-style `maxlags` convention (same weights, matches mine), divide by T-1 in gamma0 (ddof). Likelihood of difference in the 2nd-3rd digit: **medium-high**.
   `se` returned = standard error of the mean.

## run_gates
15. **Strictness.** G1 `|t| > threshold` (spec: "보다 크다"); G2 `net > 0`; G4 `|rho| >= min`; G5 `share >= min`; G6 share `<= max`, turnover `>= min` (all from the table wording). NaN fails any gate that is active.
16. **Gate evaluation.** I evaluate all six and set `failed_at` to the first failing gate's full name (e.g. "G2 net profit after costs"). Alternative: stop at first failure; short names ("G2"). Likelihood: **medium** for the name format.
17. **fire=None** : G6 cannot be evaluated; I treat it as failed (unknown is not permission). Alternative: skipped/pass. Likelihood: **medium-high**.
18. **G3 when require_subperiod_sign=False** passes automatically; G5 has no switch.
19. G3 split: first half = first n//2 observations, second = the rest (positional, not calendar). "Same sign" = signs of the two means equal (both zero counts equal). Likelihood: medium.
20. **G5 years**: calendar years of `dates`, every year with >=1 non-NaN observation counts (partial years too, no minimum day count); a year is positive if its mean > 0. Likelihood: **medium-high** (a min-days rule could exist).
21. `oos` is computed and returned but is not a gate (spec lists no gate for it). Keys `oos` and `oos_holdout` both returned. t_stat/net_bp/rho are taken as given (not recomputed from daily_excess).
22. cfg overrides: dict with GateConfig field names; unknown key raises KeyError. null_threshold None -> fallback_t.
23. Result key names (`steps` entries: name/passed/value/threshold; `subperiod`, `yearly`, `oos`) are my choice.

## fire_structure / subperiod / yearly / oos_holdout
24. **top_name_share denominator**: all days in `dates` ("전체 기간 중 발화한 날의 비율"). Alternative: days with any fire. Likelihood: medium.
25. **Turnover**: mean over consecutive day pairs of 1 - |A∩B|/|A∪B|; pairs where both sets are empty are skipped (not 0, not 1); pair with one empty set gives 1. Alternative: empty-empty counts as 0. Likelihood: **medium-high**.
26. longest run is per ticker consecutive rows (no calendar-gap logic); returned as `longest_run` with the ticker. The name `top_name_share` etc. are guesses; spec gives dict but not keys.
27. **oos_holdout**: last `months` months = dates > last_date - DateOffset(months); returns t via NW(lag), n_days, mean, positive flag, plus in-sample mean/t. No minimum-sample rule. Likelihood: medium.
28. yearly/subperiod/oos call newey_west_t with `lag` (default 21) for their t; the spec doesn't say that `lag` is used that way (only that it's a parameter). Medium.

## Spec oddities / possible errors
- No numeric worked example in this excerpt for these functions, so nothing could be reproduced; I hand-checked a 3x5 panel instead.
- The doc says base rate is "적격 종목 전체의 승률" but never says whether cost is subtracted there; (1) above is a genuine hole.
- "기간마다 유효 발화가 100건 미만" vs "일별 초과수익이 100일 미만": two different 100s; easy to mix up.
- Spec says G1 for backtest_event uses NW t "of daily excess" without giving the lag; the fama_macbeth max(h,21) rule is in a different section.
- `lift_pp` "NaN일 수 있다" when pool<20: it only becomes NaN in my code if NO day qualifies; if the spec means any dropped day voids it, differs.
- G6 description in the table ("한 종목이 발화의 max_single_name_share 이하") says share of firings, while GateConfig says share of days fired. I followed GateConfig (days in whole period); the table wording would give a different denominator (total fire count). Self-contradiction, likelihood of other reading: **medium-high**.
- Section 3.5 says a missing end price drops the observation, but delisted tickers are held at the last price; ignored here since delist flags were out of scope.
