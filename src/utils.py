"""
utils.py
────────
General helper functions for the Gold Price Forecasting project.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings("ignore")


# ── Stationarity Tests ────────────────────────────────────────────────────────

def adf_test(series: pd.Series, label: str = "", verbose: bool = True) -> dict:
    """
    Run Augmented Dickey-Fuller test.

    Returns
    -------
    dict with keys: stat, p_value, is_stationary
    """
    result = adfuller(series.dropna())
    stat, p = result[0], result[1]
    is_stationary = p < 0.05

    if verbose:
        status = "STATIONARY" if is_stationary else "NON-STATIONARY"
        print(f"  ADF [{label:20s}]  stat={stat:8.4f}  p={p:.4f}  -> {status}")

    return {"stat": stat, "p_value": p, "is_stationary": is_stationary}


def kpss_test(series: pd.Series, label: str = "", verbose: bool = True) -> dict:
    """
    Run KPSS test.

    Returns
    -------
    dict with keys: stat, p_value, is_stationary
    """
    stat, p, _, _ = kpss(series.dropna(), regression="c", nlags="auto")
    is_stationary = p > 0.05

    if verbose:
        status = "STATIONARY" if is_stationary else "NON-STATIONARY"
        print(f"  KPSS [{label:20s}]  stat={stat:7.4f}  p~{p:.4f}  -> {status}")

    return {"stat": stat, "p_value": p, "is_stationary": is_stationary}


def full_stationarity_report(series: pd.Series, name: str = "Series") -> None:
    """Run ADF + KPSS on raw, log, and differenced series."""
    log_s    = np.log(series)
    diff_s   = series.diff().dropna()
    log_diff = log_s.diff().dropna()

    print(f"\n{'='*55}")
    print(f"  Stationarity Report — {name}")
    print(f"{'='*55}")
    print("  ADF Test:")
    adf_test(series,   f"Raw {name}")
    adf_test(log_s,    f"Log {name}")
    adf_test(diff_s,   f"Diff {name}")
    adf_test(log_diff, f"Log Diff {name}")
    print("  KPSS Test:")
    kpss_test(series,   f"Raw {name}")
    kpss_test(log_s,    f"Log {name}")
    kpss_test(diff_s,   f"Diff {name}")
    kpss_test(log_diff, f"Log Diff {name}")


# ── Financial Metrics ─────────────────────────────────────────────────────────

def rolling_sharpe(series: pd.Series, window: int = 12,
                    risk_free: float = 0.0) -> pd.Series:
    """
    Compute rolling annualized Sharpe ratio.

    Parameters
    ----------
    series     : Price series
    window     : Rolling window in months
    risk_free  : Monthly risk-free rate (default 0)
    """
    monthly_ret = series.pct_change()
    excess_ret  = monthly_ret - risk_free
    roll_mean   = excess_ret.rolling(window).mean()
    roll_std    = excess_ret.rolling(window).std()
    return (roll_mean / roll_std) * np.sqrt(12)


def max_drawdown(series: pd.Series) -> pd.Series:
    """Compute rolling max drawdown from all-time high."""
    cummax = series.cummax()
    return (series - cummax) / cummax * 100


def rolling_cagr(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute rolling CAGR over a given window (in months).

    Returns
    -------
    pd.Series of annualized CAGR in percent
    """
    return ((series / series.shift(window)) ** (12 / window) - 1) * 100


def annualized_volatility(series: pd.Series, window: int = 12) -> pd.Series:
    """Rolling annualized volatility (% p.a.)."""
    return series.pct_change().rolling(window).std() * np.sqrt(12) * 100


def performance_summary(series: pd.Series) -> pd.DataFrame:
    """
    Full performance summary table.

    Returns
    -------
    pd.DataFrame with one row per decade containing:
    start_price, end_price, total_return%, cagr%, max_drawdown%, avg_vol%
    """
    df = series.to_frame("price")
    df["decade"] = (df.index.year // 10) * 10

    rows = []
    for decade, grp in df.groupby("decade"):
        start   = grp["price"].iloc[0]
        end     = grp["price"].iloc[-1]
        n       = len(grp)
        tr      = (end / start - 1) * 100
        cagr    = ((end / start) ** (12 / n) - 1) * 100
        vol     = grp["price"].pct_change().std() * np.sqrt(12) * 100
        cummax  = grp["price"].cummax()
        mdd     = ((grp["price"] - cummax) / cummax * 100).min()
        rows.append({
            "Decade":       f"{decade}s",
            "Start Price":  round(start, 2),
            "End Price":    round(end, 2),
            "Total Return": round(tr, 1),
            "CAGR %":       round(cagr, 2),
            "Avg Vol %":    round(vol, 1),
            "Max DD %":     round(mdd, 1),
        })

    return pd.DataFrame(rows)


# ── Cycle Detection ───────────────────────────────────────────────────────────

def dominant_cycles(series: pd.Series, top_n: int = 6,
                     max_period: int = 200) -> pd.DataFrame:
    """
    Detect dominant cyclical periods via FFT.

    Parameters
    ----------
    series     : Log-differenced return series
    top_n      : Number of top cycles to return
    max_period : Maximum period to consider (months)

    Returns
    -------
    pd.DataFrame with columns ['period_months', 'period_years', 'power']
    """
    from scipy.fft import fft, fftfreq
    from scipy.signal import find_peaks

    values = series.dropna().values
    N  = len(values)
    yf = np.abs(fft(values))[:N // 2]
    xf = fftfreq(N, d=1)[:N // 2]

    periods = 1 / xf[1:]
    power   = yf[1:] ** 2

    # Filter to reasonable periods
    mask    = periods <= max_period
    periods = periods[mask]
    power   = power[mask]

    peaks, _ = find_peaks(power, height=np.percentile(power, 85), distance=3)
    top      = sorted(peaks, key=lambda i: power[i], reverse=True)[:top_n]

    return pd.DataFrame({
        "period_months": [round(periods[p], 1) for p in top],
        "period_years":  [round(periods[p] / 12, 1) for p in top],
        "power":         [round(power[p], 2) for p in top],
    }).sort_values("period_months").reset_index(drop=True)


# ── Anomaly Detection ─────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame,
                      features: list = None,
                      contamination: float = 0.04) -> pd.DataFrame:
    """
    Detect anomalous months using Isolation Forest.

    Returns
    -------
    pd.DataFrame with added columns ['is_anomaly', 'anomaly_score']
    """
    from sklearn.ensemble import IsolationForest

    if features is None:
        features = ["price", "pct_change_1", "volatility_3", "roll_mean_12"]

    df_out = df.copy()
    X      = df_out[features].dropna()

    iso         = IsolationForest(contamination=contamination,
                                   n_estimators=200, random_state=42)
    labels      = iso.fit_predict(X)
    scores      = iso.score_samples(X)

    df_out.loc[X.index, "is_anomaly"]    = labels == -1
    df_out.loc[X.index, "anomaly_score"] = scores

    n_anom = (labels == -1).sum()
    print(f"Detected {n_anom} anomalies ({n_anom / len(X) * 100:.1f}% of {len(X)} months)")
    return df_out
