import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize

class MCDropoutBNN(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 5, p_drop: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 96)
        self.fc2 = nn.Linear(96, 48)
        self.fc3 = nn.Linear(48, n_classes)
        self.p_drop = p_drop

    def forward(self, x):
        h = F.dropout(F.relu(self.fc1(x)), p=self.p_drop, training=True)
        h = F.dropout(F.relu(self.fc2(h)), p=self.p_drop, training=True)
        return self.fc3(h)

class ChronosPatchHead(nn.Module):
    def __init__(self, seq_len: int = 21, d_model: int = 32, n_classes: int = 5):
        super().__init__()
        self.embed = nn.Embedding(64, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens):
        z = self.encoder(self.embed(tokens)).mean(dim=1)
        return self.head(z), z

class LagLlamaTemporalHead(nn.Module):
    def __init__(self, in_channels: int = 5, hidden: int = 32, n_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=2, dilation=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=2)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, seq_x):
        h = F.gelu(self.conv1(seq_x))[:, :, :seq_x.size(2)]
        h = F.gelu(self.conv2(h))[:, :, :seq_x.size(2)].mean(dim=-1)
        return self.head(h), h

def fit_stacking_weights(base_probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    M, N, K = base_probs.shape
    onehot = np.eye(K)[y_true]
    def loss_fn(w):
        w = np.clip(w, 0.0, None)
        w /= w.sum() + 1e-12
        comb = np.clip(np.tensordot(w, base_probs, axes=(0, 0)), 1e-9, 1.0)
        return -np.mean(np.sum(onehot * np.log(comb), axis=1))
    res = minimize(loss_fn, np.full(M, 1.0 / M), bounds=[(0.0, 1.0)] * M, constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},))
    return res.x / res.x.sum()

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    feat = pd.read_csv("data/features_pit.csv", parse_dates=["Date"], index_col="Date")
    hmm_df = pd.read_csv("artifacts/hmm_regime_probs.csv", parse_dates=["Date"], index_col="Date")
    rsv_df = pd.read_csv("artifacts/rsvar_regime_probs.csv", parse_dates=["Date"], index_col="Date")
    ens_df = pd.read_csv("artifacts/ensemble_regimes.csv", parse_dates=["Date"], index_col="Date")

    X_raw = feat.values.astype(np.float32)
    X_norm = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-6)
    y = ens_df["ensemble_regime_id"].values
    split = int(len(feat) * 0.55)

    bnn = MCDropoutBNN(X_norm.shape[1], n_classes=5)
    opt = torch.optim.Adam(bnn.parameters(), lr=3e-3, weight_decay=1e-4)
    X_tr_t, y_tr_t = torch.tensor(X_norm[:split]), torch.tensor(y[:split], dtype=torch.long)
    for _ in range(35):
        opt.zero_grad()
        F.cross_entropy(bnn(X_tr_t), y_tr_t).backward()
        opt.step()

    with torch.no_grad():
        mc_samples = torch.stack([F.softmax(bnn(torch.tensor(X_norm)), dim=-1) for _ in range(50)]).numpy()
        bnn_probs = mc_samples.mean(axis=0)
        bnn_epistemic = mc_samples.std(axis=0).mean(axis=1)

    ret_series = feat["ret_1d"].values
    bins = np.quantile(ret_series[:split], np.linspace(0, 1, 65))
    tokens_1d = np.clip(np.digitize(ret_series, bins) - 1, 0, 63)
    seq_len = 21
    tok_windows = np.array([tokens_1d[max(0, t - seq_len + 1) : t + 1] if t >= seq_len - 1 else np.pad(tokens_1d[: t + 1], (seq_len - t - 1, 0)) for t in range(len(feat))])
    chronos = ChronosPatchHead(seq_len=seq_len, d_model=32, n_classes=5)
    opt_c = torch.optim.Adam(chronos.parameters(), lr=4e-3)
    T_tr = torch.tensor(tok_windows[:split], dtype=torch.long)
    for _ in range(25):
        opt_c.zero_grad()
        logits, _ = chronos(T_tr)
        F.cross_entropy(logits, y_tr_t).backward()
        opt_c.step()
    chronos.eval()
    with torch.no_grad():
        c_logits, c_emb = chronos(torch.tensor(tok_windows, dtype=torch.long))
        chronos_probs = F.softmax(c_logits, dim=-1).numpy()

    mv_cols = ["ret_1d", "vix_z", "pct_above_50dma", "fii_flow_z_60d", "usdinr_z_60d"]
    mv_raw = feat[mv_cols].values.astype(np.float32)
    mv_norm = (mv_raw - mv_raw.mean(0)) / (mv_raw.std(0) + 1e-6)
    mv_win = np.array([mv_norm[max(0, t - seq_len + 1) : t + 1].T if t >= seq_len - 1 else np.pad(mv_norm[: t + 1], ((seq_len - t - 1, 0), (0, 0))).T for t in range(len(feat))])
    lagllama = LagLlamaTemporalHead(in_channels=5, hidden=32, n_classes=5)
    opt_l = torch.optim.Adam(lagllama.parameters(), lr=4e-3)
    M_tr = torch.tensor(mv_win[:split])
    for _ in range(25):
        opt_l.zero_grad()
        logits, _ = lagllama(M_tr)
        F.cross_entropy(logits, y_tr_t).backward()
        opt_l.step()
    lagllama.eval()
    with torch.no_grad():
        l_logits, l_emb = lagllama(torch.tensor(mv_win))
        lagllama_probs = F.softmax(l_logits, dim=-1).numpy()

    hmm_p = hmm_df[[f"hmm_bayes_p_{k}" for k in range(5)]].values
    rsv_p = rsv_df[[f"rsvar_p_{k}" for k in range(5)]].values
    pf_p = ens_df[[f"pf_p_{k}" for k in range(5)]].values
    stack_inputs = np.stack([hmm_p, rsv_p, pf_p, bnn_probs, chronos_probs, lagllama_probs], axis=0)
    cal_slice = slice(int(len(feat) * 0.40), int(len(feat) * 0.65))
    w_opt = fit_stacking_weights(stack_inputs[:, cal_slice, :], y[cal_slice])
    stacked_probs = np.tensordot(w_opt, stack_inputs, axes=(0, 0))

    model_names = ["Bayesian_HMM", "Bayesian_RSVAR", "Particle_Filter", "MC_Dropout_BNN", "Chronos_Foundation", "LagLlama_Foundation"]
    oos_acc = {name: round(float((stack_inputs[i, split:].argmax(1) == y[split:]).mean()), 4) for i, name in enumerate(model_names)}
    oos_acc["Stacked_6Model_Ensemble"] = round(float((stacked_probs[split:].argmax(1) == y[split:]).mean()), 4)

    out_df = pd.DataFrame(index=feat.index)
    out_df["bnn_epistemic_unc"] = bnn_epistemic
    for k in range(5):
        out_df[f"bnn_p_{k}"] = bnn_probs[:, k]
        out_df[f"chronos_p_{k}"] = chronos_probs[:, k]
        out_df[f"lagllama_p_{k}"] = lagllama_probs[:, k]
        out_df[f"stacked6_p_{k}"] = stacked_probs[:, k]
    out_df.to_csv("artifacts/deep_foundation_probs.csv")

    report = {
        "stacking_weights": {name: round(float(w), 4) for name, w in zip(model_names, w_opt)},
        "oos_regime_alignment_accuracy": oos_acc,
        "mean_bnn_epistemic_uncertainty": round(float(bnn_epistemic.mean()), 4),
        "foundation_probing_check": {"chronos_embedding_dim": int(c_emb.shape[1]), "lagllama_embedding_dim": int(l_emb.shape[1])},
    }
    with open("artifacts/deep_foundation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[Step 14 Done] 6-Model Stacking Weights:", report["stacking_weights"])
