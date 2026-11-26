import json
import os
from fpdf import FPDF

bt_rep = json.load(open("artifacts/backtest_report.json", encoding="utf-8"))
cp_rep = json.load(open("artifacts/conformal_shap_report.json", encoding="utf-8"))
hmm_rep = json.load(open("artifacts/hmm_diagnostics.json", encoding="utf-8"))
rsv_rep = json.load(open("artifacts/rsvar_diagnostics.json", encoding="utf-8"))
df_rep = json.load(open("artifacts/deep_foundation_report.json", encoding="utf-8"))
flt_rep = json.load(open("artifacts/filtering_diagnostics.json", encoding="utf-8"))
lin_rep = json.load(open("artifacts/data_lineage.json", encoding="utf-8"))

class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "Project 1A: Bayesian Regime Detection Engine for Indian Equity Direction Forecasting (ENSEMBLE_v3.2)", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"Prepared by Gangadharan R | Multi-Asset Solutions Desk | Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(24, 58, 98)
        self.cell(0, 8, title, ln=True)
        self.set_draw_color(24, 58, 98)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, label, val):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(20, 20, 20)
        self.cell(62, 5.5, f"- {label}:")
        self.set_font("Helvetica", "", 10)
        self.multi_cell(self.epw - (self.get_x() - self.l_margin), 5.5, str(val))

pdf = ReportPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title Banner
pdf.set_fill_color(24, 58, 98)
pdf.rect(10, 18, 190, 28, "F")
pdf.set_y(21)
pdf.set_font("Helvetica", "B", 15)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 8, "BAYESIAN REGIME DETECTION ENGINE (PROJECT 1A)", align="C", ln=True)
pdf.set_font("Helvetica", "", 10.5)
pdf.cell(0, 6, "Calibrated Probabilistic Forecasting, Conformal Uncertainty & SEBI Risk-O-Meter Governance", align="C", ln=True)
pdf.set_font("Helvetica", "I", 9.5)
pdf.cell(0, 6, "Author: Gangadharan R | Financial Data Analyst (Multi-Asset Solutions Desk)", align="C", ln=True)
pdf.ln(8)

# 1. Executive Regime Call
latest = bt_rep["latest_regime_contract"]
pdf.section_title("1. Executive Regime Call & Audit-Defensible Output Contract")
pdf.body_text(
    "This engine classifies the Indian equity market (2008-2026, N = 4,670 point-in-time NSE trading sessions) "
    "into five economic states: Risk-Off, Post-Shock, Transitional, Late-Cycle, and Risk-On. Rather than fragile "
    "point price targets, it emits calibrated posterior probabilities, split-conformal uncertainty sets, epistemic "
    "disagreement scores, and SEBI Risk-O-Meter classifications."
)
pdf.bullet("Evaluation Date & Snapshot", f"{latest['date']} (SHA-256: {lin_rep['sha256'][:16]})")
pdf.bullet("Dominant Regime & Confidence", f"{latest['dominant_regime']} (Meta-Confidence: {latest['meta_confidence']}/100)")
pdf.bullet("Posterior Probabilities", ", ".join([f"{k}: {v*100:.1f}%" for k, v in latest["probabilities"].items()]))
pdf.bullet("90% Split-Conformal Set", f"{{{latest['conformal_set']}}} (OOS Empirical Coverage: {cp_rep['conformal_calibration']['oos_empirical_coverage']*100:.2f}%, Mean Set Size: {cp_rep['conformal_calibration']['oos_mean_set_size']})")
pdf.bullet("Recommended Equity Weight", f"{latest['recommended_equity_weight']*100:.1f}% Equity / {(1-latest['recommended_equity_weight'])*100:.1f}% Liquid-Gilt Buffer")
pdf.bullet("SEBI Risk-O-Meter Rating", latest["sebi_riskometer"])
mc = bt_rep["regime_conditioned_mc_1y"]
pdf.bullet("1-Yr Regime MC (10,000 Paths)", f"Median: {mc['median_1y_return']*100:.1f}% | 5% VaR: {mc['p05_1y_return']*100:.1f}% | 95% CVaR: {mc['cvar_95_1y']*100:.1f}%")
pdf.ln(3)

# 2. Multi-Model Architecture & Bayesian MCMC Diagnostics
pdf.section_title("2. Six-Model Complementary Architecture & MCMC Convergence")
pdf.body_text(
    "To eliminate single-model fragility, ENSEMBLE_v3.2 integrates six complementary probabilistic and deep-learning architectures "
    "combined via out-of-fold simplex log-likelihood stacking:"
)
hmm_d = hmm_rep["bayesian_hmm_mcmc_diagnostics"]
dur_d = hmm_rep["duration_hypothesis_test"]
pdf.bullet("1. Bayesian 5-State Gaussian HMM", f"PyMC NUTS (4 chains x 1,000 draws): Max R-hat = {hmm_d['max_rhat']:.4f}, Min ESS = {hmm_d['min_ess_bulk']:.1f}, Divergences = {hmm_d['divergences']}. Geometric duration null rejected (p = {dur_d['p_value']:.4f}).")
pdf.bullet("2. Multivariate Bayesian RS-VAR(1)", f"Hierarchical NumPyro XLA NUTS over Nifty return, India VIX, 50-DMA breadth, FII flows & USD/INR: Max R-hat = {rsv_rep['max_rhat']:.4f}, Min ESS = {rsv_rep['min_ess_bulk']:.1f}.")
pdf.bullet("3. Sequential Particle Filter & BOCPD", f"Bootstrap SMC (N = 1,000 particles): Mean ESS = {flt_rep['mean_pf_ess']:.1f}, Mean Wasserstein-1 vs RS-VAR = {flt_rep['mean_wasserstein_pf_rsvar']:.4f}. Offline PELT detected {flt_rep['pelt_count']} structural breaks.")
pdf.bullet("4. MC-Dropout Bayesian Neural Net", f"50 stochastic forward passes separating epistemic vs aleatoric uncertainty (Mean Epistemic Std = {df_rep['mean_bnn_epistemic_uncertainty']:.4f}).")
pdf.bullet("5-6. Foundation Model Heads", "Chronos quantile-tokenized transformer + Lag-Llama multi-lag dilated causal encoder (32-dim latent embeddings).")
pdf.bullet("Learned Stacking Weights", ", ".join([f"{k}: {v:.3f}" for k, v in df_rep["stacking_weights"].items()]))

# Embed Figure 1
pdf.add_page()
pdf.section_title("3. Historical Regime Detection & BOCPD Changepoints (2008-2026)")
pdf.body_text("Figure 1 illustrates Nifty 50 index levels alongside the 5-state posterior probability stack and BOCPD changepoint hazard spikes across the 2008 GFC, 2013 Taper Tantrum, 2018 IL&FS crisis, 2020 COVID crash, and 2024 Election shock.")
if os.path.exists("artifacts/figures/fig1_regimes_bocpd.png"):
    pdf.image("artifacts/figures/fig1_regimes_bocpd.png", x=10, w=190)
pdf.ln(3)

# 4. Out-of-Sample Proper Scoring Rules & Backtest Performance
pdf.section_title("4. Out-of-Sample Proper Scoring Rules & Walk-Forward Overlay")
oos_s = bt_rep["out_of_sample_performance"]["Regime_Overlay"]
oos_b = bt_rep["out_of_sample_performance"]["Benchmark_75_25"]
psr = bt_rep["proper_scoring_rules_oos"]["Ensemble_v3.2"]
pdf.bullet("OOS Proper Scoring Rules", f"Ranked Probability Score (RPS) = {psr['RPS']} | Brier = {psr['Brier']} | LogLoss = {psr['LogLoss']}")
pdf.bullet("OOS Regime Overlay (Net 12bps)", f"CAGR: {oos_s['CAGR']*100:.2f}% | Ann. Vol: {oos_s['Ann_Vol']*100:.2f}% | Sharpe: {oos_s['Sharpe']} | Max Drawdown: {oos_s['Max_Drawdown']*100:.2f}%")
pdf.bullet("OOS 75/25 Nifty-Midcap Benchmark", f"CAGR: {oos_b['CAGR']*100:.2f}% | Ann. Vol: {oos_b['Ann_Vol']*100:.2f}% | Sharpe: {oos_b['Sharpe']} | Max Drawdown: {oos_b['Max_Drawdown']*100:.2f}%")
pdf.bullet("Drawdown Reduction & DSR", f"Max Drawdown cut by {(abs(oos_b['Max_Drawdown']) - abs(oos_s['Max_Drawdown']))*100:.2f}% | Deflated Sharpe Probability: {bt_rep['overfitting_controls']['deflated_sharpe_prob']}")
if os.path.exists("artifacts/figures/fig2_backtest_allocation.png"):
    pdf.image("artifacts/figures/fig2_backtest_allocation.png", x=10, w=190)

# 5. SHAP Feature Attribution & RS-VAR Impulse Responses
pdf.add_page()
pdf.section_title("5. SHAP Explainability & Cross-Asset FII Shock Transmission")
top_shap = list(cp_rep["top_global_shap_drivers"].items())[:8]
pdf.bullet("Top Global SHAP Drivers", ", ".join([f"{k} ({v:.4f})" for k, v in top_shap]))
abl = cp_rep["feature_ablation_study"]
pdf.bullet("Feature Ablation OOS Accuracy", f"Full: {abl['Full_Model_OOS_Accuracy']} | No VIX: {abl['No_India_VIX']['oos_accuracy']} | No FII/DII: {abl['No_FII_DII_Flows']['oos_accuracy']} | No Breadth: {abl['No_Market_Breadth']['oos_accuracy']}")
if os.path.exists("artifacts/figures/fig3_shap_irf.png"):
    pdf.image("artifacts/figures/fig3_shap_irf.png", x=10, w=190)
pdf.ln(4)

pdf.section_title("6. Historical Indian Market Crisis Attribution")
for ep_name, ep_d in bt_rep["crisis_episode_attribution"].items():
    pdf.bullet(ep_name.replace("_", " "), f"Overlay Return: {ep_d['strat_return']*100:+.1f}% (Max DD {ep_d['strat_max_dd']*100:.1f}%) vs Benchmark: {ep_d['bench_return']*100:+.1f}% (Max DD {ep_d['bench_max_dd']*100:.1f}%) | Avg Eq Wt: {ep_d['avg_equity_weight']*100:.1f}%")

out_pdf = "Project_1A_Bayesian_Regime_Detection_Report.pdf"
pdf.output(out_pdf)
print(f"[SUCCESS] Generated {out_pdf} ({os.path.getsize(out_pdf)/1024:.1f} KB)")
