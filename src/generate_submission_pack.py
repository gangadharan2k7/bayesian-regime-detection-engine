import json, os, zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("artifacts/figures", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

df = pd.read_csv("data/indian_market_pit.csv", parse_dates=["Date"], index_col="Date")
ens = pd.read_csv("artifacts/ensemble_regimes.csv", parse_dates=["Date"], index_col="Date")
bt = pd.read_csv("artifacts/backtest_timeseries.csv", parse_dates=["Date"], index_col="Date")
bt_rep = json.load(open("artifacts/backtest_report.json", encoding="utf-8"))
cp_rep = json.load(open("artifacts/conformal_shap_report.json", encoding="utf-8"))
hmm_rep = json.load(open("artifacts/hmm_diagnostics.json", encoding="utf-8"))
rsv_rep = json.load(open("artifacts/rsvar_diagnostics.json", encoding="utf-8"))
df_rep = json.load(open("artifacts/deep_foundation_report.json", encoding="utf-8"))

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
axes[0].plot(df.index, df["NiftyClose"], color="#1f4e78", lw=1.3)
axes[0].set_title("Nifty 50 Index, Ensemble Regime Probabilities & BOCPD Changepoints (2008-2026)", fontweight="bold")
labels = ["Risk-Off", "Post-Shock", "Transitional", "Late-Cycle", "Risk-On"]
colors = ["#c00000", "#ed7d31", "#ffc000", "#4472c4", "#2e75b6"]
axes[1].stackplot(ens.index, [ens[f"ensemble_p_{k}"] for k in range(5)], labels=labels, colors=colors, alpha=0.85)
axes[1].legend(loc="upper left", ncol=5, fontsize=8)
axes[2].plot(ens.index, ens["bocpd_cp_prob"], color="#800000", lw=1.0, label="BOCPD Changepoint Mass")
axes[2].legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("artifacts/figures/fig1_regimes_bocpd.png", dpi=200)
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
axes[0].plot(bt.index, bt["strat_cum"], label="Bayesian Regime Overlay (Net of 12bps Cost)", color="#1b5e20", lw=1.6)
axes[0].plot(bt.index, bt["bench_cum"], label="Benchmark (75% Nifty / 25% Midcap)", color="#555555", lw=1.2, ls="--")
axes[0].set_title("Walk-Forward Equity Overlay vs. Benchmark & Conviction-Scaled Allocation", fontweight="bold")
axes[0].legend(loc="upper left")
axes[1].fill_between(bt.index, 0.40, bt["equity_weight"], color="#2e7d32", alpha=0.35)
axes[1].plot(bt.index, bt["equity_weight"], color="#1b5e20", lw=1.0)
plt.tight_layout()
plt.savefig("artifacts/figures/fig2_backtest_allocation.png", dpi=200)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
shap_s = pd.Series(cp_rep["top_global_shap_drivers"]).head(10)[::-1]
axes[0].barh(shap_s.index, shap_s.values, color="#2b5c8f")
axes[0].set_title("Top-10 Global SHAP Regime Drivers", fontweight="bold")
for rname, path in rsv_rep["fii_outflow_irf_10d"].items():
    axes[1].plot(range(1, 11), np.array(path) * 10000, marker="o", ms=3, label=rname)
axes[1].set_title("RS-VAR 10-Day IRF: -1 SD FII Outflow -> Nifty Return (bps)", fontweight="bold")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("artifacts/figures/fig3_shap_irf.png", dpi=200)
plt.close()

latest = bt_rep["latest_regime_contract"]
memo = f"""# INVESTMENT COMMITTEE & SEBI GOVERNANCE REPORT
## Project 1A: Bayesian Regime Detection Engine for Indian Equity Direction Forecasting
**Author:** Gangadharan R | **Engine:** ENSEMBLE_v3.2 | **Date:** {latest['date']}

### 1. Executive Regime Call & Risk-O-Meter Summary
- **Dominant Regime:** **{latest['dominant_regime']}** (Meta-Confidence: **{latest['meta_confidence']}/100**)
- **Posterior Probabilities:** {json.dumps(latest['probabilities'])}
- **90% Conformal Set:** `{latest['conformal_set']}` (Empirical OOS Coverage: **{cp_rep['conformal_calibration']['oos_empirical_coverage']*100:.2f}%**)
- **Recommended Equity Weight:** **{latest['recommended_equity_weight']*100:.1f}%** | **SEBI Risk-O-Meter:** **{latest['sebi_riskometer']}**
- **1-Year Regime-Conditioned MC (10,000 Paths):** Median **{bt_rep['regime_conditioned_mc_1y']['median_1y_return']*100:.1f}%**, 5% VaR **{bt_rep['regime_conditioned_mc_1y']['p05_1y_return']*100:.1f}%**, 95% CVaR **{bt_rep['regime_conditioned_mc_1y']['cvar_95_1y']*100:.1f}%**

### 2. Model Stack & Diagnostics
- **Bayesian 5-State HMM (PyMC NUTS):** Max R-hat = `{hmm_rep['bayesian_hmm_mcmc_diagnostics']['max_rhat']:.4f}`, Min ESS = `{hmm_rep['bayesian_hmm_mcmc_diagnostics']['min_ess_bulk']:.1f}`, Divergences = `{hmm_rep['bayesian_hmm_mcmc_diagnostics']['divergences']}`
- **Multivariate Bayesian RS-VAR(1) (NumPyro NUTS):** Max R-hat = `{rsv_rep['max_rhat']:.4f}`, Min ESS = `{rsv_rep['min_ess_bulk']:.1f}`
- **6-Model Stacking Weights:** `{json.dumps(df_rep['stacking_weights'])}`
- **OOS Drawdown Control:** Overlay Max DD `{bt_rep['out_of_sample_performance']['Regime_Overlay']['Max_Drawdown']*100:.2f}%` vs Benchmark `{bt_rep['out_of_sample_performance']['Benchmark_75_25']['Max_Drawdown']*100:.2f}%` (Deflated Sharpe Prob: `{bt_rep['overfitting_controls']['deflated_sharpe_prob']}`)
"""
open("docs/IC_MEMO_AND_MODEL_CARD.md", "w", encoding="utf-8").write(memo)
open("README.md", "w", encoding="utf-8").write("# Bayesian Regime Detection Engine (Project 1A)\nRun `streamlit run app/dashboard.py` to launch Regime Lab.\n")

zip_name = "Project_1A_Bayesian_Regime_Detection_Submission.zip"
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for root_dir in ["src", "app", "docs", "data", "artifacts"]:
        for folder, _, files in os.walk(root_dir):
            for file in files:
                if not file.endswith(".nc"):
                    zf.write(os.path.join(folder, file))
    for top_f in ["README.md", "requirements.txt"]:
        if os.path.exists(top_f):
            zf.write(top_f)

print(f"[ALL DONE] Created {zip_name} ({os.path.getsize(zip_name)/1024:.1f} KB)")
