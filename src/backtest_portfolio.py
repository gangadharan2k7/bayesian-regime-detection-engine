import json
import numpy as np
import pandas as pd
from scipy.stats import norm

REGIME_ORDER = ["Risk-Off", "Post-Shock", "Transitional", "Late-Cycle", "Risk-On"]

def ranked_probability_score(probs: np.ndarray, y_true: np.ndarray, K: int = 5) -> float:
    onehot = np.eye(K)[y_true]
    cdf_pred = np.cumsum(probs, axis=1)
    cdf_obs = np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cdf_pred - cdf_obs) ** 2, axis=1)) / (K - 1))

def brier_score(probs: np.ndarray, y_true: np.ndarray, K: int = 5) -> float:
    onehot = np.eye(K)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

def multiclass_log_loss(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.mean(np.log(p)))

def deflated_sharpe_ratio(sharpe_daily: float, n_obs: int, n_trials: int = 12, skew: float = -0.3, kurt: float = 5.0) -> float:
    e_max = np.sqrt(2.0 * np.log(n_trials)) if n_trials > 1 else 0.0
    sr_std = np.sqrt((1.0 - skew * sharpe_daily + ((kurt - 1.0) / 4.0) * (sharpe_daily ** 2)) / max(n_obs - 1, 1))
    z = (sharpe_daily - e_max * sr_std) / max(sr_std, 1e-12)
    return float(norm.cdf(z))

def run_regime_mc(P: np.ndarray, mu: np.ndarray, sig: np.ndarray, init_dist: np.ndarray, horizon: int = 252, n_sims: int = 10000):
    rng = np.random.default_rng(42)
    K = len(mu)
    s0 = rng.choice(K, size=n_sims, p=init_dist)
    states = np.zeros((n_sims, horizon), dtype=int)
    states[:, 0] = s0
    for t in range(1, horizon):
        probs = P[states[:, t - 1]]
        cum = probs.cumsum(axis=1)
        u = rng.random(n_sims)
        states[:, t] = (u[:, None] < cum).argmax(axis=1)
    rets = mu[states] + sig[states] * rng.standard_normal((n_sims, horizon))
    paths = np.exp(np.cumsum(np.log1p(np.clip(rets, -0.5, 0.5)), axis=1))
    final_ret = paths[:, -1] - 1.0
    var_95 = float(np.percentile(final_ret, 5))
    cvar_95 = float(final_ret[final_ret <= var_95].mean())
    return {
        "horizon_days": horizon,
        "n_sims": n_sims,
        "median_1y_return": round(float(np.median(final_ret)), 4),
        "p05_1y_return": round(var_95, 4),
        "p95_1y_return": round(float(np.percentile(final_ret, 95)), 4),
        "cvar_95_1y": round(cvar_95, 4),
        "prob_negative_1y": round(float((final_ret < 0).mean()), 4),
    }

def sebi_riskometer(p_risk_off: float, vol_21d: float) -> str:
    score = 0.6 * p_risk_off + 0.4 * np.clip(vol_21d / 0.35, 0.0, 1.0)
    if score < 0.20:
        return "Moderate"
    elif score < 0.35:
        return "Moderately High"
    elif score < 0.55:
        return "High"
    return "Very High"

if __name__ == "__main__":
    df = pd.read_csv("data/indian_market_pit.csv", parse_dates=["Date"], index_col="Date")
    feat = pd.read_csv("data/features_pit.csv", parse_dates=["Date"], index_col="Date")
    hmm_df = pd.read_csv("artifacts/hmm_regime_probs.csv", parse_dates=["Date"], index_col="Date")
    rsv_df = pd.read_csv("artifacts/rsvar_regime_probs.csv", parse_dates=["Date"], index_col="Date")
    ens_df = pd.read_csv("artifacts/ensemble_regimes.csv", parse_dates=["Date"], index_col="Date")

    df = df.loc[ens_df.index]
    feat = feat.loc[ens_df.index]

    # Forward 5-day realised directional regime label for strict out-of-sample scoring
    fwd_5d = df["NiftyClose"].pct_change(5).shift(-5).fillna(0.0)
    vol_21 = feat["vol_21d"].values
    y_eval = np.full(len(df), 2, dtype=int)  # default Transitional
    y_eval[(fwd_5d < -0.025) | (vol_21 > 0.26)] = 0          # Risk-Off
    y_eval[(fwd_5d > 0.015) & (vol_21 > 0.20)] = 1           # Post-Shock
    y_eval[(fwd_5d > 0.002) & (vol_21 <= 0.20) & (feat["pct_above_50dma"] <= 62)] = 3  # Late-Cycle
    y_eval[(fwd_5d > 0.008) & (vol_21 <= 0.18) & (feat["pct_above_50dma"] > 62)] = 4   # Risk-On

    oos_start = int(len(df) * 0.50)
    y_oos = y_eval[oos_start:]

    ens_probs = ens_df[[f"ensemble_p_{k}" for k in range(5)]].values[oos_start:]
    hmm_probs = hmm_df[[f"hmm_bayes_p_{k}" for k in range(5)]].values[oos_start:]
    rsv_probs = rsv_df[[f"rsvar_p_{k}" for k in range(5)]].values[oos_start:]
    pf_probs = ens_df[[f"pf_p_{k}" for k in range(5)]].values[oos_start:]

    # Naive benchmarks: Climatology & Persistence
    clim_probs = np.tile(np.bincount(y_eval[:oos_start], minlength=5) / oos_start, (len(y_oos), 1))
    pers_probs = np.eye(5)[np.roll(y_oos, 1)]
    pers_probs[0] = clim_probs[0]

    scoring_table = {}
    for name, P_mat in [
        ("Ensemble_v3.2", ens_probs),
        ("Bayesian_HMM", hmm_probs),
        ("Bayesian_RSVAR", rsv_probs),
        ("Particle_Filter", pf_probs),
        ("Climatology_Baseline", clim_probs),
        ("Persistence_Baseline", pers_probs),
    ]:
        rps = ranked_probability_score(P_mat, y_oos, K=5)
        brier = brier_score(P_mat, y_oos, K=5)
        ll = multiclass_log_loss(P_mat, y_oos)
        scoring_table[name] = {
            "RPS": round(rps, 4),
            "Brier": round(brier, 4),
            "LogLoss": round(ll, 4),
            "RPS_Skill_vs_Climatology": round(1.0 - rps / ranked_probability_score(clim_probs, y_oos, K=5), 4),
        }

    # Walk-Forward Regime-Conditioned Allocation Backtest (Long-Only Indian Equity Scheme)
    # Target equity weight per regime: Risk-Off=0.45, Post-Shock=0.95, Transitional=0.75, Late-Cycle=0.80, Risk-On=1.00
    regime_eq_target = np.array([0.45, 0.95, 0.75, 0.80, 1.00])
    all_ens = ens_df[[f"ensemble_p_{k}" for k in range(5)]].values
    conviction = 1.0 - (ens_df["conformal_set_size"].values - 1.0) / 4.0

    raw_target_w = all_ens @ regime_eq_target
    # Conviction-scaled tilt around 85% baseline equity allocation
    target_w = 0.85 + conviction * (raw_target_w - 0.85)
    target_w = np.clip(target_w, 0.40, 1.00)

    # Apply 1.5% no-trade band + 1-day execution lag (strictly point-in-time)
    exec_w = np.zeros(len(df))
    exec_w[0] = 0.85
    for t in range(1, len(df)):
        desired = target_w[t - 1]
        if abs(desired - exec_w[t - 1]) >= 0.015:
            exec_w[t] = desired
        else:
            exec_w[t] = exec_w[t - 1]

    nifty_ret = df["NiftyClose"].pct_change().fillna(0.0).values
    mid_ret = df["MidcapClose"].pct_change().fillna(0.0).values
    cash_ret = (df["Gilt10Y"].values / 100.0) / 252.0

    # Blend 75% Large-Cap + 25% Mid-Cap for active equity sleeve, scaled down in Risk-Off
    eq_sleeve_ret = 0.75 * nifty_ret + 0.25 * mid_ret
    turnover = np.abs(np.diff(exec_w, prepend=exec_w[0]))
    tx_cost = turnover * 0.0012  # 12 bps STT + brokerage + impact

    strat_ret = exec_w * eq_sleeve_ret + (1.0 - exec_w) * cash_ret - tx_cost
    bench_ret = 0.75 * nifty_ret + 0.25 * mid_ret
    active_ret = strat_ret - bench_ret

    def perf_stats(r: np.ndarray, b: np.ndarray):
        ann_ret = float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)
        ann_vol = float(np.std(r) * np.sqrt(252.0))
        sharpe = float((np.mean(r) - np.mean(cash_ret)) / (np.std(r) + 1e-12) * np.sqrt(252.0))
        eq_curve = np.cumprod(1.0 + r)
        max_dd = float(np.min(eq_curve / np.maximum.accumulate(eq_curve) - 1.0))
        ir = float(np.mean(r - b) / (np.std(r - b) + 1e-12) * np.sqrt(252.0))
        return {
            "CAGR": round(ann_ret, 4),
            "Ann_Vol": round(ann_vol, 4),
            "Sharpe": round(sharpe, 3),
            "Max_Drawdown": round(max_dd, 4),
            "Information_Ratio": round(ir, 3),
        }

    full_strat = perf_stats(strat_ret, bench_ret)
    full_bench = perf_stats(bench_ret, bench_ret)
    oos_strat = perf_stats(strat_ret[oos_start:], bench_ret[oos_start:])
    oos_bench = perf_stats(bench_ret[oos_start:], bench_ret[oos_start:])

    dsr = deflated_sharpe_ratio(np.mean(strat_ret[oos_start:]) / (np.std(strat_ret[oos_start:]) + 1e-12), len(y_oos), n_trials=12)

    # Part C Historical Crisis Episode Attribution
    episodes = {
        "2008_GFC": ("2008-01-01", "2009-03-31"),
        "2013_Taper_Tantrum": ("2013-05-01", "2013-09-30"),
        "2018_ILFS_Credit_Shock": ("2018-08-15", "2018-11-30"),
        "2020_COVID_Crash_Recovery": ("2020-02-01", "2020-12-31"),
        "2024_Election_Shock": ("2024-05-15", "2024-06-30"),
    }
    bt_df = pd.DataFrame(
        {"strat_ret": strat_ret, "bench_ret": bench_ret, "equity_weight": exec_w, "regime": ens_df["ensemble_regime_label"].values},
        index=df.index,
    )
    bt_df["strat_cum"] = (1.0 + bt_df["strat_ret"]).cumprod()
    bt_df["bench_cum"] = (1.0 + bt_df["bench_ret"]).cumprod()
    bt_df["sebi_riskometer"] = [sebi_riskometer(ens_df["ensemble_p_0"].iloc[i], feat["vol_21d"].iloc[i]) for i in range(len(df))]
    bt_df.to_csv("artifacts/backtest_timeseries.csv")

    crisis_report = {}
    for ep_name, (s_dt, e_dt) in episodes.items():
        sub = bt_df.loc[s_dt:e_dt]
        if len(sub) > 5:
            sc = (1.0 + sub["strat_ret"]).cumprod()
            bc = (1.0 + sub["bench_ret"]).cumprod()
            crisis_report[ep_name] = {
                "strat_return": round(float(sc.iloc[-1] - 1.0), 4),
                "bench_return": round(float(bc.iloc[-1] - 1.0), 4),
                "strat_max_dd": round(float((sc / sc.cummax() - 1.0).min()), 4),
                "bench_max_dd": round(float((bc / bc.cummax() - 1.0).min()), 4),
                "avg_equity_weight": round(float(sub["equity_weight"].mean()), 3),
            }

    # 10,000-path Regime-Conditioned Monte Carlo Projection from latest state
    with open("artifacts/hmm_diagnostics.json", "r", encoding="utf-8") as f:
        hmm_diag = json.load(f)
    P_mc = np.array(hmm_diag["bayesian_hmm_mcmc_diagnostics"]["P_post_mean"])
    mu_mc = np.array([-0.0018, 0.0012, 0.0002, 0.0005, 0.0009])
    sig_mc = np.array([0.021, 0.015, 0.011, 0.009, 0.008])
    mc_summary = run_regime_mc(P_mc, mu_mc, sig_mc, all_ens[-1], horizon=252, n_sims=10000)

    final_report = {
        "proper_scoring_rules_oos": scoring_table,
        "full_sample_performance": {"Regime_Overlay": full_strat, "Benchmark_75_25": full_bench},
        "out_of_sample_performance": {"Regime_Overlay": oos_strat, "Benchmark_75_25": oos_bench},
        "overfitting_controls": {
            "deflated_sharpe_prob": round(dsr, 4),
            "annualised_turnover": round(float(turnover.mean() * 252.0), 2),
            "total_tx_cost_bps_pa": round(float(tx_cost.mean() * 252.0 * 10000), 1),
        },
        "crisis_episode_attribution": crisis_report,
        "regime_conditioned_mc_1y": mc_summary,
        "latest_regime_contract": {
            "date": str(df.index[-1].date()),
            "dominant_regime": str(ens_df["ensemble_regime_label"].iloc[-1]),
            "probabilities": {REGIME_ORDER[k]: round(float(all_ens[-1, k]), 4) for k in range(5)},
            "conformal_set": str(ens_df["conformal_pred_set"].iloc[-1]),
            "meta_confidence": float(ens_df["meta_confidence"].iloc[-1]),
            "recommended_equity_weight": round(float(exec_w[-1]), 4),
            "sebi_riskometer": bt_df["sebi_riskometer"].iloc[-1],
        },
    }
    with open("artifacts/backtest_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("OOS Proper Scoring Rules:", json.dumps(scoring_table["Ensemble_v3.2"]))
    print("OOS Overlay vs Benchmark:", oos_strat, "vs", oos_bench)
    print("Deflated Sharpe Probability:", round(dsr, 4))
    print("Latest Regime Contract:", json.dumps(final_report["latest_regime_contract"]))
