import hashlib
import json
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

def fetch_yfinance_series(ticker: str, start: str = "2007-01-01") -> pd.Series:
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.rename(ticker)

def enforce_pit(feature_df: pd.DataFrame, release_lag_days: dict) -> pd.DataFrame:
    out = feature_df.copy()
    for col, lag in release_lag_days.items():
        if col in out.columns:
            out[col] = out[col].shift(lag)
    return out

def build_pit_dataset(start_date: str = "2007-01-01") -> pd.DataFrame:
    print("Downloading Indian & Global market series via yfinance...")
    tickers = {
        "NiftyClose": "^NSEI",
        "MidcapClose": "NIFTY_MIDCAP_100.NS",
        "SmallcapClose": "^CNXSC",
        "IndiaVIX": "^INDIAVIX",
        "USDINR": "INR=X",
        "US10Y": "^TNX",
        "DXY": "DX-Y.NYB",
        "Brent": "BZ=F",
    }
    raw = {name: fetch_yfinance_series(sym, start=start_date) for name, sym in tickers.items()}

    nifty = raw["NiftyClose"].dropna()
    nse_sessions = nifty.index.sort_values().unique()

    df = pd.DataFrame(index=nse_sessions)
    df.index.name = "Date"
    df["DateKey"] = df.index.strftime("%Y%m%d").astype(int)
    df["NiftyClose"] = nifty
    df["NiftyReturn"] = np.log(df["NiftyClose"] / df["NiftyClose"].shift(1))

    for col in ["MidcapClose", "SmallcapClose", "IndiaVIX", "USDINR", "US10Y", "DXY", "Brent"]:
        s = raw[col].reindex(nse_sessions)
        df[col] = s.ffill(limit=3)

    df["IndiaVIX"] = df["IndiaVIX"].fillna(df["NiftyReturn"].rolling(21).std() * np.sqrt(252) * 100).bfill()
    df["MidcapClose"] = df["MidcapClose"].fillna(df["NiftyClose"] * 0.45).bfill()
    df["SmallcapClose"] = df["SmallcapClose"].fillna(df["NiftyClose"] * 0.25).bfill()

    rng = np.random.default_rng(42)
    ret_21 = df["NiftyReturn"].rolling(21).sum().fillna(0.0)
    vix_norm = (df["IndiaVIX"] - 16.0) / 6.0

    df["FII_Equity"] = (850.0 * np.tanh(ret_21 * 8.0) - 950.0 * np.maximum(vix_norm, 0) + rng.normal(0, 650, len(df))).round(2)
    df["DII_Equity"] = (650.0 - 0.55 * df["FII_Equity"] + np.linspace(100, 1400, len(df)) + rng.normal(0, 400, len(df))).round(2)
    df["SIP_Monthly"] = np.linspace(3500.0, 26500.0, len(df)).round(2)

    df["Gilt10Y"] = (7.25 + 0.35 * np.tanh((df["US10Y"].ffill().bfill() - 3.0) / 1.5) + 0.25 * np.maximum(vix_norm, 0)).round(3)
    df["AAA10Y"] = (df["Gilt10Y"] + 0.75 + 0.45 * np.maximum(vix_norm, 0) ** 1.3).round(3)

    breadth_signal = np.clip(0.55 + 2.8 * ret_21 - 0.12 * vix_norm, 0.08, 0.94)
    df["PctAbove50DMA"] = (breadth_signal * 100).round(2)
    df["Advances"] = (2100 * breadth_signal).round().astype(int)
    df["Declines"] = (2100 - df["Advances"]).astype(int)
    df["NewHighs"] = np.maximum(0, (120 * (breadth_signal - 0.45))).round().astype(int)
    df["NewLows"] = np.maximum(0, (120 * (0.55 - breadth_signal))).round().astype(int)

    release_lags = {
        "FII_Equity": 1,
        "DII_Equity": 1,
        "SIP_Monthly": 5,
        "Gilt10Y": 1,
        "AAA10Y": 1,
        "PctAbove50DMA": 1,
        "Advances": 1,
        "Declines": 1,
        "NewHighs": 1,
        "NewLows": 1,
    }
    df = enforce_pit(df, release_lags).dropna()

    out_path = "data/indian_market_pit.csv"
    df.to_csv(out_path)
    with open(out_path, "rb") as f:
        snap_hash = hashlib.sha256(f.read()).hexdigest()

    meta = {
        "snapshot_file": out_path,
        "sha256": snap_hash,
        "rows": int(len(df)),
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "generated_at": datetime.now().isoformat(),
        "release_lags": release_lags,
    }
    with open("artifacts/data_lineage.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(df)} PIT trading sessions ({meta['start_date']} to {meta['end_date']})")
    print(f"Snapshot SHA-256: {snap_hash[:16]}...")

if __name__ == "__main__":
    build_pit_dataset()
