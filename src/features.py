import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class SectorGCN(nn.Module):
    """Lightweight 2-layer GCN encoding cross-asset correlation graphs (Section A7.3)."""
    def __init__(self, in_dim: int = 4, hidden: int = 16, out_dim: int = 4):
        super().__init__()
        torch.manual_seed(42)
        self.w1 = nn.Linear(in_dim, hidden)
        self.w2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
        norm_adj = adj / deg
        h = F.relu(norm_adj @ self.w1(x))
        h = norm_adj @ self.w2(h)
        return h.mean(dim=0)

def compute_tda_and_gcn_features(df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Rolling correlation manifold topology (H0/H1 persistence) + SectorGCN embeddings."""
    cols = ["NiftyReturn", "MidcapClose", "SmallcapClose", "IndiaVIX", "USDINR", "Gilt10Y"]
    sub = df[cols].copy()
    for c in ["MidcapClose", "SmallcapClose", "IndiaVIX", "USDINR", "Gilt10Y"]:
        sub[c] = sub[c].pct_change().fillna(0.0)
    sub["NiftyReturn"] = sub["NiftyReturn"].fillna(0.0)

    vals = sub.values
    n = len(df)
    tda_h0_mean = np.zeros(n)
    tda_h0_max = np.zeros(n)
    tda_h1_loop = np.zeros(n)
    gcn_embs = np.zeros((n, 4))

    gcn = SectorGCN(in_dim=4, hidden=16, out_dim=4)
    gcn.eval()

    with torch.no_grad():
        for t in range(window, n):
            win = vals[t - window : t]
            corr = np.corrcoef(win, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            dist = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, 4.0))

            upper = dist[np.triu_indices_from(dist, k=1)]
            sorted_d = np.sort(upper)
            tda_h0_mean[t] = sorted_d[: len(cols) - 1].mean()
            tda_h0_max[t] = sorted_d[: len(cols) - 1].max()
            tda_h1_loop[t] = float(np.mean(sorted_d[len(cols) - 1 :]) - tda_h0_mean[t])

            node_x = torch.tensor(
                np.column_stack([win.mean(0), win.std(0) + 1e-6, (win[-5:].mean(0) - win.mean(0)), win[-1]]),
                dtype=torch.float32,
            )
            adj = torch.tensor(np.maximum(corr, 0.0), dtype=torch.float32)
            gcn_embs[t] = gcn(node_x, adj).numpy()

    out = pd.DataFrame(index=df.index)
    out["tda_h0_mean"] = tda_h0_mean
    out["tda_h0_max"] = tda_h0_max
    out["tda_h1_loop"] = tda_h1_loop
    for i in range(4):
        out[f"gcn_emb_{i}"] = gcn_embs[:, i]
    return out

def engineer_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full 43 PIT feature matrix (Sections A4.5, A7, A24, C1, C2)."""
    f = pd.DataFrame(index=df.index)

    f["ret_1d"] = df["NiftyClose"].pct_change()
    f["ret_5d"] = df["NiftyClose"].pct_change(5)
    f["ret_21d"] = df["NiftyClose"].pct_change(21)
    f["ret_63d"] = df["NiftyClose"].pct_change(63)

    f["ma_50_200"] = df["NiftyClose"].rolling(50).mean() / df["NiftyClose"].rolling(200).mean() - 1.0
    f["above_200dma"] = (df["NiftyClose"] > df["NiftyClose"].rolling(200).mean()).astype(float)
    f["dist_50dma"] = df["NiftyClose"] / df["NiftyClose"].rolling(50).mean() - 1.0

    f["vol_21d"] = f["ret_1d"].rolling(21).std() * np.sqrt(252)
    f["vol_63d"] = f["ret_1d"].rolling(63).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol_21d"] / (f["vol_63d"] + 1e-9)
    f["vix_level"] = df["IndiaVIX"]
    f["vix_change_5d"] = df["IndiaVIX"].pct_change(5)
    f["vix_z"] = (df["IndiaVIX"] - df["IndiaVIX"].rolling(252).mean()) / (df["IndiaVIX"].rolling(252).std() + 1e-9)

    f["adv_dec_ratio"] = df["Advances"] / (df["Declines"] + 1.0)
    f["pct_above_50dma"] = df["PctAbove50DMA"]
    f["breadth_change_21d"] = df["PctAbove50DMA"].diff(21)
    f["new_highs_lows"] = df["NewHighs"] - df["NewLows"]

    f["mid_large_rel_21d"] = df["MidcapClose"].pct_change(21) - f["ret_21d"]
    f["small_large_rel_21d"] = df["SmallcapClose"].pct_change(21) - f["ret_21d"]
    f["small_large_rel_63d"] = df["SmallcapClose"].pct_change(63) - f["ret_63d"]
    mid_ratio = df["MidcapClose"] / (df["NiftyClose"] + 1e-9)
    f["midcap_val_z"] = (mid_ratio - mid_ratio.rolling(252).mean()) / (mid_ratio.rolling(252).std() + 1e-9)

    f["gilt_10y_change_21d"] = df["Gilt10Y"].diff(21)
    f["inr_change_21d"] = df["USDINR"].pct_change(21)
    f["usdinr_z_60d"] = (df["USDINR"] - df["USDINR"].rolling(60).mean()) / (df["USDINR"].rolling(60).std() + 1e-9)
    f["credit_spread"] = df["AAA10Y"] - df["Gilt10Y"]
    f["credit_spread_z"] = (f["credit_spread"] - f["credit_spread"].rolling(252).mean()) / (f["credit_spread"].rolling(252).std() + 1e-9)
    f["us10y_change_21d"] = df["US10Y"].diff(21)
    f["dxy_change_21d"] = df["DXY"].pct_change(21)
    f["brent_change_21d"] = df["Brent"].pct_change(21)

    f["fii_eq_5d"] = df["FII_Equity"].rolling(5).sum()
    f["dii_eq_5d"] = df["DII_Equity"].rolling(5).sum()
    f["fii_flow_z_60d"] = (df["FII_Equity"] - df["FII_Equity"].rolling(60).mean()) / (df["FII_Equity"].rolling(60).std() + 1e-9)
    f["sip_momentum"] = df["SIP_Monthly"].pct_change(63)
    f["flow_divergence"] = ((df["DII_Equity"] > 0) & (df["FII_Equity"] < 0)).astype(float)
    f["flow_balance"] = f["dii_eq_5d"] / (f["fii_eq_5d"].abs() + 1.0)

    f["crisis_signal_count"] = (
        (f["vix_z"] > 2.5).astype(int)
        + (f["pct_above_50dma"] < 25).astype(int)
        + (f["fii_flow_z_60d"] < -1.5).astype(int)
        + (f["usdinr_z_60d"] > 1.5).astype(int)
    ).astype(float)

    topo_gcn = compute_tda_and_gcn_features(df, window=63)
    f = pd.concat([f, topo_gcn], axis=1)

    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f

if __name__ == "__main__":
    df = pd.read_csv("data/indian_market_pit.csv", parse_dates=["Date"], index_col="Date")
    feat = engineer_regime_features(df)
    feat.to_csv("data/features_pit.csv")
    print(f"Engineered {feat.shape[1]} PIT features across {feat.shape[0]} sessions ({feat.index.min().date()} to {feat.index.max().date()})")
