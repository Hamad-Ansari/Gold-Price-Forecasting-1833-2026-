"""
preprocessing.py
────────────────
Data loading, cleaning, and feature engineering utilities
for the Gold Price Forecasting project.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
RANDOM_STATE = 42


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_gold_data(path: str = None) -> pd.DataFrame:
    """
    Load and preprocess the gold price dataset.

    Parameters
    ----------
    path : str, optional
        Path to the CSV file. Defaults to data/gold_advanced_features.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with DatetimeIndex at monthly frequency.
    """
    if path is None:
        path = DATA_DIR / "gold_advanced_features.csv"

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # Ensure monthly frequency, forward-fill any gaps
    df = df.resample("MS").ffill()

    print(f"Loaded: {df.shape[0]:,} rows | {df.index.min().date()} → {df.index.max().date()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    return df


def load_inflation_data(path: str = None) -> pd.DataFrame:
    """
    Load World Bank CPI inflation data and return US series.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['year', 'inflation_usa'].
    """
    if path is None:
        path = DATA_DIR / "worldbank_inflation_clean.csv"

    wb = pd.read_csv(path)
    usa = wb[wb["Country_Code"] == "USA"][["Year", "Inflation_CPI"]].copy()
    usa.columns = ["year", "inflation_usa"]
    usa = usa.dropna().reset_index(drop=True)
    return usa


# ── Feature Engineering ───────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, price_col: str = "price",
                     lags: list = None) -> pd.DataFrame:
    """Add lagged price features."""
    if lags is None:
        lags = [1, 2, 3, 6, 12]
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[price_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, price_col: str = "price",
                         windows: list = None) -> pd.DataFrame:
    """Add rolling mean and std features."""
    if windows is None:
        windows = [3, 6, 12]
    df = df.copy()
    for w in windows:
        df[f"roll_mean_{w}"] = df[price_col].rolling(w).mean()
        df[f"roll_std_{w}"]  = df[price_col].rolling(w).std()
    return df


def add_momentum_features(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """Add momentum and percentage change features."""
    df = df.copy()
    df["pct_change_1"]  = df[price_col].pct_change(1)
    df["pct_change_3"]  = df[price_col].pct_change(3)
    df["pct_change_12"] = df[price_col].pct_change(12)
    df["momentum_1"]    = df[price_col] - df[price_col].shift(1)
    df["momentum_3"]    = df[price_col] - df[price_col].shift(3)
    return df


def add_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Encode month as sin/cos to capture cyclical seasonality."""
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    return df


def add_ratio_features(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """Add price-to-rolling-mean ratio features."""
    df = df.copy()
    for w in [3, 6, 12]:
        roll = df[price_col].rolling(w).mean()
        df[f"price_to_roll{w}"] = df[price_col] / roll
    return df


def build_full_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in one call.

    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features, NaN rows dropped.
    """
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_momentum_features(df)
    df = add_cyclical_encoding(df)
    df = add_ratio_features(df)
    return df


# ── Train/Test Split ──────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame, cutoff: str = "2020-01-01"):
    """
    Time-based train/test split. No shuffling.

    Parameters
    ----------
    df      : Full dataframe
    cutoff  : ISO date string for split point

    Returns
    -------
    tuple : (train, test)
    """
    train = df[df.index < cutoff].copy()
    test  = df[df.index >= cutoff].copy()
    print(f"Train: {len(train):,} obs  |  Test: {len(test):,} obs  |  Cutoff: {cutoff}")
    return train, test


def make_sequences(data: np.ndarray, lookback: int):
    """
    Create (X, y) sequences for LSTM input.

    Parameters
    ----------
    data     : 1-D scaled array
    lookback : Number of time steps to look back

    Returns
    -------
    tuple : (X, y) as numpy arrays
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
