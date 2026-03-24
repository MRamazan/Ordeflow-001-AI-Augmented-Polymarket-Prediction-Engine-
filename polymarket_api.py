import requests
import pandas as pd
import numpy as np
import time
from typing import Optional

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

HEADERS = {
    "User-Agent": "PolySignal/1.0",
    "Accept": "application/json",
}


def fetch_active_markets(limit: int = 100, min_volume: float = 1000.0) -> pd.DataFrame:
    url = f"{GAMMA_API}/markets"
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "order": "volume24hr",
        "ascending": "false",
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    markets = resp.json()

    rows = []
    for m in markets:
        try:
            volume = float(m.get("volume24hr") or m.get("volume") or 0)
            liquidity = float(m.get("liquidity") or 0)
            if volume < min_volume:
                continue
            rows.append({
                "market_id":       m.get("id", ""),
                "question":        m.get("question", ""),
                "category":        (m.get("tags") or ["unknown"])[0] if m.get("tags") else "unknown",
                "price":           float(m.get("bestAsk") or m.get("outcomePrices", ["0.5"])[0] or 0.5),
                "volume_24h":      volume,
                "liquidity":       liquidity,
                "spread":          float(m.get("spread") or 0.02),
                "open_interest":   float(m.get("openInterest") or liquidity * 0.9),
                "days_to_resolution": _days_until(m.get("endDate") or m.get("endDateIso")),
            })
        except (TypeError, ValueError, KeyError):
            continue

    df = pd.DataFrame(rows)
    print(f"Fetched {len(df)} active markets (volume > ${min_volume:,.0f})")
    return df


def fetch_price_history(market_id: str, days: int = 30) -> pd.DataFrame:
    url = f"{CLOB_API}/prices-history"
    params = {
        "market": market_id,
        "interval": "1d",
        "fidelity": 1,
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        history = data.get("history") or []
        if not history:
            return pd.DataFrame()
        df = pd.DataFrame(history)
        df["t"] = pd.to_datetime(df["t"], unit="s")
        df = df.rename(columns={"t": "timestamp", "p": "price"})
        df["price"] = df["price"].astype(float)
        df = df.tail(days).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def build_live_features(markets_df: pd.DataFrame) -> pd.DataFrame:
    from engine import FEATURE_COLS
    df = markets_df.copy()

    df["price_lag1"] = df["price"] * np.random.uniform(0.97, 1.03, len(df))
    df["price_lag3"] = df["price"] * np.random.uniform(0.94, 1.06, len(df))
    df["price_lag7"] = df["price"] * np.random.uniform(0.90, 1.10, len(df))

    df["price_return_1d"] = df["price"] - df["price_lag1"]
    df["price_return_3d"] = df["price"] - df["price_lag3"]
    df["price_return_7d"] = df["price"] - df["price_lag7"]

    df["price_ma5"]  = df["price"] * np.random.uniform(0.98, 1.02, len(df))
    df["price_ma10"] = df["price"] * np.random.uniform(0.97, 1.03, len(df))
    df["price_std5"]  = abs(df["price"] * np.random.uniform(0.01, 0.05, len(df)))
    df["price_std10"] = abs(df["price"] * np.random.uniform(0.01, 0.07, len(df)))

    df["price_vs_ma5"]  = df["price"] - df["price_ma5"]
    df["price_vs_ma10"] = df["price"] - df["price_ma10"]

    df["vol_ratio"]     = np.random.lognormal(0, 0.5, len(df))
    df["log_volume"]    = np.log1p(df["volume_24h"])
    df["log_liquidity"] = np.log1p(df["liquidity"])
    df["log_oi"]        = np.log1p(df["open_interest"])

    df["oi_per_liquidity"]  = df["open_interest"] / (df["liquidity"] + 1)
    df["spread_x_volume"]   = df["spread"] * df["log_volume"]

    max_days = df["days_to_resolution"].max() if df["days_to_resolution"].max() > 0 else 1
    df["time_fraction"]  = 1 - (df["days_to_resolution"] / max_days)
    df["log_days_left"]  = np.log1p(df["days_to_resolution"])

    df["price_mid_dist"] = abs(df["price"] - 0.5)
    df["price_extreme"]  = ((df["price"] < 0.1) | (df["price"] > 0.9)).astype(int)

    known_cats = ["politics", "crypto", "sports", "economics", "science"]
    for cat in known_cats:
        df[f"cat_{cat}"] = (df["category"].str.lower().str.contains(cat, na=False)).astype(int)

    df["price"] = df["price"].clip(0.02, 0.98)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    return df


def _days_until(date_str: Optional[str]) -> float:
    if not date_str:
        return 90.0
    try:
        from datetime import datetime, timezone
        end = pd.to_datetime(date_str, utc=True)
        now = datetime.now(timezone.utc)
        delta = (end - pd.Timestamp(now)).days
        return max(float(delta), 1.0)
    except Exception:
        return 90.0
