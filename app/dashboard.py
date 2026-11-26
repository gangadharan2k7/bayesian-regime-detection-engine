import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Regime Lab — Bayesian Indian Equity Engine", layout="wide")
st.title("Regime Lab: Bayesian Regime Detection & SEBI Risk-O-Meter Engine")

with open("artifacts/backtest_report.json", "r", encoding="utf-8") as f:
    bt = json.load(f)
with open("artifacts/conformal_shap_report.json", "r", encoding="utf-8") as f:
    cp = json.load(f)

ens = pd.read_csv("artifacts/ensemble_regimes.csv", parse_dates=["Date"], index_col="Date")
curve = pd.read_csv("artifacts/backtest_timeseries.csv", parse_dates=["Date"], index_col="Date")

c1, c2, c3, c4 = st.columns(4)
latest = bt["latest_regime_contract"]
c1.metric("Current Dominant Regime", latest["dominant_regime"])
c2.metric("Meta-Confidence", f"{latest['meta_confidence']}/100")
c3.metric("Recommended Equity Weight", f"{latest['recommended_equity_weight']*100:.1f}%")
c4.metric("SEBI Risk-O-Meter", latest["sebi_riskometer"])

st.subheader("Posterior Regime Probabilities (Trailing 500 Sessions)")
prob_df = ens[[f"ensemble_p_{k}" for k in range(5)]].tail(500).copy()
prob_df.columns = ["Risk-Off", "Post-Shock", "Transitional", "Late-Cycle", "Risk-On"]
st.area_chart(prob_df)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Cumulative Performance: Regime Overlay vs. 75/25 Benchmark")
    st.line_chart(curve[["strat_cum", "bench_cum"]])
with col_b:
    st.subheader("Top Global SHAP Regime Drivers")
    st.bar_chart(pd.Series(cp["top_global_shap_drivers"]))
