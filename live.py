import numpy as np
import pandas as pd
import json
import sys
import time
from engine import (
    simulate_polymarket_data, build_features, FEATURE_COLS,
    EnsembleSignalEngine, walk_forward_backtest,
    generate_live_signals, get_feature_importance_df
)

SEED = 42
np.random.seed(SEED)


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def train_model() -> tuple[EnsembleSignalEngine, pd.DataFrame]:
    raw_df = simulate_polymarket_data(n_markets=80, n_snapshots=200)
    df_feat = build_features(raw_df)
    X = df_feat[FEATURE_COLS].values
    y = df_feat["outcome"].values
    engine = EnsembleSignalEngine()
    engine.fit(X, y)
    return engine, df_feat


def run_live():
    print_section("POLYSIGNAL — Live Mode")

    print("Attempting Polymarket API connection...")
    live_df = None
    try:
        from polymarket_api import fetch_active_markets, build_live_features
        markets_df = fetch_active_markets(limit=100, min_volume=500)
        if len(markets_df) > 0:
            live_df = build_live_features(markets_df)
            print(f"Live data loaded: {len(live_df)} markets")
        else:
            print("No markets returned from API.")
    except Exception as e:
        print(f"API unavailable ({type(e).__name__}). Running on simulation data.")

    print("\nTraining ensemble model on historical simulation...")
    t0 = time.time()
    engine, df_feat = train_model()
    print(f"Model trained in {time.time() - t0:.1f}s")

    if live_df is not None and len(live_df) > 0:
        print_section("Live Signals — Real Polymarket Markets")
        X_live = live_df[FEATURE_COLS].values
        probs = engine.predict_proba(X_live)[:, 1]
        live_df = live_df.copy()
        live_df["predicted_prob"] = probs
        live_df["edge"] = live_df["predicted_prob"] - live_df["price"]
        live_df["signal"] = np.where(
            live_df["edge"] > 0.06, "BUY",
            np.where(live_df["edge"] < -0.06, "SELL", "HOLD")
        )
        signals = live_df
    else:
        print_section("Live Signals — Simulation Mode")
        signals = generate_live_signals(engine, df_feat)

    buy = signals[signals["signal"] == "BUY"]
    sell = signals[signals["signal"] == "SELL"]
    hold = signals[signals["signal"] == "HOLD"]

    print(f"\nSignal summary:  BUY={len(buy)}  SELL={len(sell)}  HOLD={len(hold)}")

    if "question" in buy.columns:
        cols = ["question", "price", "predicted_prob", "edge", "signal"]
    else:
        cols = ["market_id", "category", "current_price", "predicted_prob", "edge", "signal"]

    print(f"\nTop 5 BUY opportunities:")
    print(buy.nlargest(5, "edge")[cols].to_string(index=False))

    print(f"\nTop 5 SELL opportunities:")
    print(sell.nsmallest(5, "edge")[cols].to_string(index=False))

    signals.to_csv("live_signals.csv", index=False)
    print("\nlive_signals.csv saved")
    print_section("DONE")


if __name__ == "__main__":
    run_live()
