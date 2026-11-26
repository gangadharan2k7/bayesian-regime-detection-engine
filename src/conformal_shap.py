import json
import numpy as np
import pandas as pd
import shap
from scipy.special import rel_entr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss

REGIME_ORDER = ["Risk-Off", "Post-Shock", "Transitional", "Late-Cycle", "Risk-On"]

def compute_conformal_sets(probs: np.ndarray, y_true: np.ndarray, alpha: float = 0.10, cal_frac: float = 0.40):
    """Split Conformal Prediction over the 5-regime probability simplex (Section A9)."""
    T, K = probs.shape
    n_cal = int(T * cal_frac)

    # Non-conformity score: s_i = 1 - p(y_i | x_i)
    cal_probs = probs[:n_cal]
    cal_y = y_true[:n_cal]
    scores = 1.0 - cal_probs[np.arange(n_cal), cal_y]

    q_level = np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
    q_level = min(max(q_level, 0.0), 1.0)
    q_hat = float(np.quantile(scores, q_level, method="higher"))
    threshold = 1.0 - q_hat

    pred_sets = []
    set_sizes = np.zeros(T, dtype=int)
    covered = np.zeros(T, dtype=bool)

    for t in range(T):
        cset = [k for k in range(K) if probs[t, k] >= threshold]
        if len(cset) == 0:
            cset = [int(np.argmax(probs[t]))]
        pred_sets.append([REGIME_ORDER[k] for k in cset])
        set_sizes[t] = len(cset)
        covered[t] = y_true[t] in cset

    oos_cov = float(covered[n_cal:].mean())
    singleton_rate = float((set_sizes[n_cal:] == 1).mean())
    abstain_rate = float((set_sizes[n_cal:] >= 3).mean())

    return {
        "q_hat": q_hat,
        "prob_threshold": threshold,
        "target_coverage": 1.0 - alpha,
        "oos_empirical_coverage": oos_cov,
        "oos_mean_set_size": float(set_sizes[n_cal:].mean()),
        "oos_singleton_rate": singleton_rate,
        "oos_abstain_rate": abstain_rate,
        "set_sizes": set_sizes,
        "pred_sets": ["|".join(s) for s in pred_sets],
    }

def compute_meta_confidence(ens_probs: np.ndarray, hmm_probs: np.ndarray, rsv_probs: np.ndarray, pf_probs: np.ndarray, cp_prob: np.ndarray) -> np.ndarray:
    """0-100 Meta-Confidence Score combining posterior entropy, cross-model KL disagreement, and BOCPD hazard (Section A10)."""
    K = ens_probs.shape[1]
    norm_ent = -np.sum(ens_probs * np.log(ens_probs + 1e-12), axis=1) / np.log(K)

    kl_hmm = np.sum(rel_entr(ens_probs + 1e-12, hmm_probs + 1e-12), axis=1)
    kl_rsv = np.sum(rel_entr(ens_probs + 1e-12, rsv_probs + 1e-12), axis=1)
    kl_pf = np.sum(rel_entr(ens_probs + 1e-12, pf_probs + 1e-12), axis=1)
    mean_kl = np.clip((kl_hmm + kl_rsv + kl_pf) / 3.0, 0.0, 2.0) / 2.0

    cp_pen = np.clip(cp_prob, 0.0, 1.0)
    raw_conf = 1.0 - (0.50 * norm_ent + 0.30 * mean_kl + 0.20 * cp_pen)
    return np.round(np.clip(raw_conf * 100.0, 0.0, 100.0), 2)

def run_shap_and_ablations(feat: pd.DataFrame, y_target: np.ndarray):
    """SHAP TreeExplainer attribution + feature family ablation study (Section A17)."""
    split = int(len(feat) * 0.65)
    X_train, X_test = feat.iloc[:split], feat.iloc[split:]
    y_train, y_test = y_target[:split], y_target[split:]

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    full_acc = float(accuracy_score(y_test, clf.predict(X_test)))

    # SHAP TreeExplainer on OOS evaluation window (trailing 400 days)
    X_eval = X_test.tail(400)
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_eval)

    # Handle shap output shape across versions: list of (N, F) or array (N, F, K)
    if isinstance(shap_vals, list):
        abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        per_class = {REGIME_ORDER[k]: np.abs(shap_vals[k]).mean(axis=0) for k in range(len(shap_vals))}
    else:
        abs_mean = np.abs(shap_vals).mean(axis=(0, 2))
        per_class = {REGIME_ORDER[k]: np.abs(shap_vals[:, :, k]).mean(axis=0) for k in range(shap_vals.shape[2])}

    global_imp = pd.Series(abs_mean, index=feat.columns).sort_values(ascending=False)
    top_global = global_imp.head(12).round(5).to_dict()

    top_per_regime = {}
    for rname, vec in per_class.items():
        s = pd.Series(vec, index=feat.columns).sort_values(ascending=False)
        top_per_regime[rname] = s.head(5).round(5).to_dict()

    # Feature Family Ablation Study (Section A17.2)
    ablation_groups = {
        "No_India_VIX": [c for c in feat.columns if "vix" in c],
        "No_FII_DII_Flows": [c for c in feat.columns if any(k in c for k in ["fii", "dii", "sip", "flow"])],
        "No_Market_Breadth": [c for c in feat.columns if any(k in c for k in ["adv_dec", "50dma", "breadth", "highs_lows"])],
        "No_TDA_SectorGCN": [c for c in feat.columns if any(k in c for k in ["tda_", "gcn_"])],
    }
    ablation_results = {"Full_Model_OOS_Accuracy": round(full_acc, 4)}
    for name, drop_cols in ablation_groups.items():
        keep_cols = [c for c in feat.columns if c not in drop_cols]
        m = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
        m.fit(X_train[keep_cols], y_train)
        acc = float(accuracy_score(y_test, m.predict(X_test[keep_cols])))
        ablation_results[name] = {
            "oos_accuracy": round(acc, 4),
            "accuracy_drop": round(full_acc - acc, 4),
        }

    return top_global, top_per_regime, ablation_results

if __name__ == "__main__":
    feat = pd.read_csv("data/features_pit.csv", parse_dates=["Date"], index_col="Date")
    hmm_df = pd.read_csv("artifacts/hmm_regime_probs.csv", parse_dates=["Date"], index_col="Date")
    rsv_df = pd.read_csv("artifacts/rsvar_regime_probs.csv", parse_dates=["Date"], index_col="Date")
    ens_df = pd.read_csv("artifacts/ensemble_regimes.csv", parse_dates=["Date"], index_col="Date")

    ens_probs = ens_df[[f"ensemble_p_{k}" for k in range(5)]].values
    hmm_probs = hmm_df[[f"hmm_bayes_p_{k}" for k in range(5)]].values
    rsv_probs = rsv_df[[f"rsvar_p_{k}" for k in range(5)]].values
    pf_probs = ens_df[[f"pf_p_{k}" for k in range(5)]].values
    y_state = ens_df["ensemble_regime_id"].values

    print("1. Calibrating Split Conformal Prediction sets (alpha=0.10 -> 90% coverage)...")
    cp_res = compute_conformal_sets(ens_probs, y_state, alpha=0.10)

    print("2. Computing 0-100 Meta-Confidence & Epistemic Disagreement scores...")
    meta_conf = compute_meta_confidence(ens_probs, hmm_probs, rsv_probs, pf_probs, ens_df["bocpd_cp_prob"].values)

    ens_df["conformal_set_size"] = cp_res["set_sizes"]
    ens_df["conformal_pred_set"] = cp_res["pred_sets"]
    ens_df["abstain_flag"] = (cp_res["set_sizes"] >= 3).astype(int)
    ens_df["meta_confidence"] = meta_conf
    ens_df.to_csv("artifacts/ensemble_regimes.csv")

    print("3. Running SHAP TreeExplainer & Feature Family Ablation Study...")
    top_global, top_per_regime, ablations = run_shap_and_ablations(feat, y_state)

    report = {
        "conformal_calibration": {
            "target_coverage": cp_res["target_coverage"],
            "oos_empirical_coverage": round(cp_res["oos_empirical_coverage"], 4),
            "oos_mean_set_size": round(cp_res["oos_mean_set_size"], 3),
            "oos_singleton_rate": round(cp_res["oos_singleton_rate"], 4),
            "oos_abstain_rate": round(cp_res["oos_abstain_rate"], 4),
            "prob_threshold": round(cp_res["prob_threshold"], 4),
        },
        "meta_confidence_summary": {
            "mean": round(float(meta_conf.mean()), 2),
            "p10": round(float(np.percentile(meta_conf, 10)), 2),
            "p90": round(float(np.percentile(meta_conf, 90)), 2),
        },
        "top_global_shap_drivers": top_global,
        "top_per_regime_shap_drivers": top_per_regime,
        "feature_ablation_study": ablations,
    }
    with open("artifacts/conformal_shap_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Conformal OOS Coverage: {report['conformal_calibration']['oos_empirical_coverage']:.2%} (Target 90.0%) | Mean Set Size: {report['conformal_calibration']['oos_mean_set_size']:.2f}")
    print(f"Mean Meta-Confidence: {report['meta_confidence_summary']['mean']}/100 | Top SHAP Driver: {list(top_global.keys())[0]}")
