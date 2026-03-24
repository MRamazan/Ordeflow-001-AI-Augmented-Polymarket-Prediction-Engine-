import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")


SEED = 42
np.random.seed(SEED)


def simulate_polymarket_data(n_markets: int = 80, n_snapshots: int = 200) -> pd.DataFrame:
    records = []
    for market_id in range(n_markets):
        true_prob = np.random.beta(2, 2)
        resolution_day = np.random.randint(60, 365)
        volume_base = np.random.lognormal(mean=8, sigma=2)
        liquidity_base = np.random.lognormal(mean=6, sigma=1.5)
        category = np.random.choice(["politics", "crypto", "sports", "economics", "science"])

        price = np.clip(true_prob + np.random.normal(0, 0.08), 0.02, 0.98)
        for day in range(n_snapshots):
            days_left = max(resolution_day - day, 1)
            time_decay = day / resolution_day
            noise = np.random.normal(0, 0.02 * (1 + time_decay))
            price = np.clip(price + noise, 0.02, 0.98)
            volume_spike = np.random.lognormal(0, 0.8)
            volume = volume_base * volume_spike * (1 + time_decay * 0.5)
            liquidity = liquidity_base * np.random.lognormal(0, 0.3)
            spread = np.clip(np.random.exponential(0.015) / np.sqrt(liquidity / 1000), 0.001, 0.1)
            open_interest = liquidity * np.random.uniform(0.8, 1.4)
            outcome = int(np.random.random() < true_prob)

            records.append({
                "market_id": market_id,
                "day": day,
                "category": category,
                "price": round(price, 4),
                "volume_24h": round(volume, 2),
                "liquidity": round(liquidity, 2),
                "spread": round(spread, 4),
                "open_interest": round(open_interest, 2),
                "days_to_resolution": days_left,
                "true_prob": round(true_prob, 4),
                "outcome": outcome,
            })

    return pd.DataFrame(records)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["market_id", "day"]).reset_index(drop=True)

    grp = df.groupby("market_id")

    df["price_lag1"] = grp["price"].shift(1)
    df["price_lag3"] = grp["price"].shift(3)
    df["price_lag7"] = grp["price"].shift(7)

    df["price_return_1d"] = df["price"] - df["price_lag1"]
    df["price_return_3d"] = df["price"] - df["price_lag3"]
    df["price_return_7d"] = df["price"] - df["price_lag7"]

    df["price_ma5"] = grp["price"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["price_ma10"] = grp["price"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df["price_std5"] = grp["price"].transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
    df["price_std10"] = grp["price"].transform(lambda x: x.rolling(10, min_periods=1).std().fillna(0))

    df["price_vs_ma5"] = df["price"] - df["price_ma5"]
    df["price_vs_ma10"] = df["price"] - df["price_ma10"]

    df["vol_ma5"] = grp["volume_24h"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["vol_ratio"] = df["volume_24h"] / (df["vol_ma5"] + 1)
    df["log_volume"] = np.log1p(df["volume_24h"])
    df["log_liquidity"] = np.log1p(df["liquidity"])
    df["log_oi"] = np.log1p(df["open_interest"])

    df["oi_per_liquidity"] = df["open_interest"] / (df["liquidity"] + 1)
    df["spread_x_volume"] = df["spread"] * df["log_volume"]

    df["time_fraction"] = 1 - (df["days_to_resolution"] / df["days_to_resolution"].max())
    df["log_days_left"] = np.log1p(df["days_to_resolution"])

    df["price_mid_dist"] = abs(df["price"] - 0.5)
    df["price_extreme"] = (df["price"] < 0.1) | (df["price"] > 0.9)
    df["price_extreme"] = df["price_extreme"].astype(int)

    df["mispricing"] = df["price"] - df["true_prob"]
    df["abs_mispricing"] = df["mispricing"].abs()

    for cat in ["politics", "crypto", "sports", "economics", "science"]:
        df[f"cat_{cat}"] = (df["category"] == cat).astype(int)

    df = df.dropna(subset=["price_lag7"]).reset_index(drop=True)
    return df


FEATURE_COLS = [
    "price", "price_lag1", "price_lag3", "price_lag7",
    "price_return_1d", "price_return_3d", "price_return_7d",
    "price_ma5", "price_ma10", "price_std5", "price_std10",
    "price_vs_ma5", "price_vs_ma10",
    "vol_ratio", "log_volume", "log_liquidity", "log_oi",
    "oi_per_liquidity", "spread_x_volume", "spread",
    "time_fraction", "log_days_left",
    "price_mid_dist", "price_extreme",
    "cat_politics", "cat_crypto", "cat_sports", "cat_economics", "cat_science",
]


class EnsembleSignalEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=SEED
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=10,
                random_state=SEED, n_jobs=-1
            ),
            "logistic_regression": CalibratedClassifierCV(
                LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
            ),
        }
        self.weights = {"gradient_boosting": 0.5, "random_forest": 0.35, "logistic_regression": 0.15}
        self.feature_importances_ = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        for name, model in self.models.items():
            model.fit(X_scaled, y)
        gb = self.models["gradient_boosting"]
        rf = self.models["random_forest"]
        self.feature_importances_ = (
            0.6 * gb.feature_importances_ + 0.4 * rf.feature_importances_
        )
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        blend = np.zeros((len(X), 2))
        for name, model in self.models.items():
            blend += self.weights[name] * model.predict_proba(X_scaled)
        return blend

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def walk_forward_backtest(df: pd.DataFrame, n_splits: int = 5) -> dict:
    df_feat = build_features(df)
    X = df_feat[FEATURE_COLS].values
    y = df_feat["outcome"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        engine = EnsembleSignalEngine()
        engine.fit(X_train, y_train)

        probs = engine.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        ll = log_loss(y_test, probs)

        prices_test = df_feat.iloc[test_idx]["price"].values
        signals = probs - prices_test
        strong_mask = abs(signals) > 0.05
        pnl = np.where(strong_mask, signals * (y_test - prices_test), 0)
        total_pnl = pnl.sum()
        n_trades = strong_mask.sum()
        win_rate = (pnl[strong_mask] > 0).mean() if n_trades > 0 else 0.0

        fold_metrics.append({
            "fold": fold + 1,
            "accuracy": acc,
            "roc_auc": auc,
            "log_loss": ll,
            "total_pnl": total_pnl,
            "n_trades": int(n_trades),
            "win_rate": win_rate,
        })

    results_df = pd.DataFrame(fold_metrics)

    pnl_series = results_df["total_pnl"].values
    returns = pnl_series / (abs(pnl_series).mean() + 1e-9)
    sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    max_dd = _max_drawdown(np.cumsum(pnl_series))

    return {
        "fold_metrics": results_df,
        "mean_accuracy": results_df["accuracy"].mean(),
        "mean_auc": results_df["roc_auc"].mean(),
        "mean_log_loss": results_df["log_loss"].mean(),
        "total_pnl": results_df["total_pnl"].sum(),
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "total_trades": results_df["n_trades"].sum(),
        "mean_win_rate": results_df["win_rate"].mean(),
    }


def _max_drawdown(cum_pnl: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    return float(drawdown.min())


def compute_kelly_fraction(win_prob: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 0.0
    b = avg_win / avg_loss
    kelly = (win_prob * b - (1 - win_prob)) / b
    return max(0.0, min(kelly * 0.5, 0.25))


def generate_live_signals(engine: EnsembleSignalEngine, df_feat: pd.DataFrame) -> pd.DataFrame:
    latest = df_feat.sort_values("day").groupby("market_id").tail(1).copy()
    X = latest[FEATURE_COLS].values
    probs = engine.predict_proba(X)[:, 1]
    latest = latest.copy()
    latest["predicted_prob"] = probs
    latest["current_price"] = latest["price"]
    latest["edge"] = latest["predicted_prob"] - latest["current_price"]
    latest["signal"] = np.where(
        latest["edge"] > 0.06, "BUY",
        np.where(latest["edge"] < -0.06, "SELL", "HOLD")
    )
    latest["kelly_fraction"] = latest.apply(
        lambda r: compute_kelly_fraction(
            r["predicted_prob"] if r["signal"] == "BUY" else 1 - r["predicted_prob"],
            abs(r["edge"]), 1 - abs(r["edge"])
        ), axis=1
    )
    return latest[["market_id", "category", "current_price", "predicted_prob",
                    "edge", "signal", "kelly_fraction", "days_to_resolution"]].reset_index(drop=True)


def get_feature_importance_df(engine: EnsembleSignalEngine) -> pd.DataFrame:
    imp_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": engine.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df
