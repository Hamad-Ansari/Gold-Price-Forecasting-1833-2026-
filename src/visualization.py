"""
visualization.py
────────────────
Reusable Plotly chart functions using the dark/neon theme.
All charts are fully interactive (hover, zoom, pan).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Theme Constants ────────────────────────────────────────────────────────────

DARK_BG  = "#0d0d0d"
GOLD     = "#FFD700"
CYAN     = "#00FFFF"
MAGENTA  = "#FF00FF"
RED      = "#FF4C4C"
PURPLE   = "#9B59B6"
GRID_COL = "#2a2a2a"

BASE_LAYOUT = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor="#111111",
    font=dict(color="#e0e0e0", family="Courier New"),
    xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL, showgrid=True),
    yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL, showgrid=True),
    margin=dict(l=60, r=40, t=80, b=60),
)


def _layout(**kwargs) -> dict:
    """Merge BASE_LAYOUT with extra kwargs."""
    return {**BASE_LAYOUT, **kwargs}


# ── Price Charts ───────────────────────────────────────────────────────────────

def plot_price_history(df: pd.DataFrame, price_col: str = "price",
                        rolling_windows: list = None,
                        title: str = "Gold Price History") -> go.Figure:
    """
    Full historical price chart with optional rolling mean overlays.

    Parameters
    ----------
    df              : DataFrame with DatetimeIndex and price column
    price_col       : Name of the price column
    rolling_windows : List of rolling window sizes (e.g. [30, 90, 180])
    title           : Chart title
    """
    if rolling_windows is None:
        rolling_windows = []

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.65, 0.35], vertical_spacing=0.05,
                         subplot_titles=[title, "Log-Scale"])

    fig.add_trace(go.Scatter(
        x=df.index, y=df[price_col], name="Price",
        line=dict(color=GOLD, width=1.2),
        fill="tozeroy", fillcolor="rgba(255,215,0,0.06)",
    ), row=1, col=1)

    colors = [CYAN, MAGENTA, RED, PURPLE]
    for w, col in zip(rolling_windows, colors):
        rm = df[price_col].rolling(w).mean()
        fig.add_trace(go.Scatter(
            x=rm.index, y=rm, name=f"{w}M MA",
            line=dict(color=col, width=1.5, dash="dot"),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=np.log(df[price_col]), name="Log Price",
        line=dict(color="#7FDBFF", width=1.3),
    ), row=2, col=1)

    fig.update_layout(**_layout(height=700, title=title,
                                 title_font=dict(color=GOLD, size=18),
                                 legend=dict(bgcolor="rgba(0,0,0,0.5)",
                                             bordercolor=GOLD, borderwidth=1)))
    return fig


def plot_actual_vs_predicted(actual: pd.Series, predicted: np.ndarray,
                              model_name: str = "Model",
                              train: pd.Series = None,
                              n_train_tail: int = 100) -> go.Figure:
    """Plot actual vs predicted with optional training tail."""
    fig = go.Figure()

    if train is not None:
        fig.add_trace(go.Scatter(
            x=train.index[-n_train_tail:], y=train.values[-n_train_tail:],
            name="Train", line=dict(color=GOLD, width=1.5, dash="dot"),
        ))

    fig.add_trace(go.Scatter(
        x=actual.index, y=actual.values,
        name="Actual", line=dict(color=CYAN, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=actual.index, y=predicted,
        name=f"{model_name} Forecast", line=dict(color=RED, width=2, dash="dash"),
    ))

    fig.update_layout(**_layout(height=500,
                                 title=f"{model_name} — Actual vs Predicted"))
    return fig


# ── Distribution Charts ────────────────────────────────────────────────────────

def plot_distribution(series: pd.Series, title: str = "Price Distribution") -> go.Figure:
    """Histogram of price distribution."""
    fig = go.Figure(go.Histogram(
        x=series, nbinsx=100, name="Distribution",
        marker_color=GOLD, opacity=0.75, histnorm="probability density",
    ))
    fig.update_layout(**_layout(height=450, title=title,
                                 xaxis_title="Price (USD/oz)",
                                 yaxis_title="Density"))
    return fig


def plot_era_boxplot(df: pd.DataFrame, price_col: str = "price") -> go.Figure:
    """Boxplot by historical era."""
    eras = {
        "1833-1900": df[df.index.year <= 1900][price_col],
        "1901-1971": df[(df.index.year > 1900) & (df.index.year <= 1971)][price_col],
        "1972-1999": df[(df.index.year > 1971) & (df.index.year <= 1999)][price_col],
        "2000-2015": df[(df.index.year > 1999) & (df.index.year <= 2015)][price_col],
        "2016-2026": df[df.index.year > 2015][price_col],
    }
    fig = go.Figure()
    for (era, vals), col in zip(eras.items(), [GOLD, CYAN, MAGENTA, RED, PURPLE]):
        fig.add_trace(go.Box(y=vals, name=era, marker_color=col, line_color=col))

    fig.update_layout(**_layout(height=500, title="Gold Price — Boxplot by Era"))
    return fig


# ── Risk / Volatility Charts ──────────────────────────────────────────────────

def plot_volatility_dashboard(df: pd.DataFrame, price_col: str = "price") -> go.Figure:
    """3-panel: price, rolling volatility, drawdown."""
    returns  = df[price_col].pct_change()
    vol_12   = returns.rolling(12).std() * np.sqrt(12) * 100
    vol_36   = returns.rolling(36).std() * np.sqrt(12) * 100
    cummax   = df[price_col].cummax()
    drawdown = (df[price_col] - cummax) / cummax * 100

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=["Price", "Ann. Volatility (%)", "Drawdown from ATH (%)"],
                         row_heights=[0.4, 0.3, 0.3], vertical_spacing=0.04)

    fig.add_trace(go.Scatter(x=df.index, y=df[price_col], name="Price",
        line=dict(color=GOLD, width=1.2), fill="tozeroy",
        fillcolor="rgba(255,215,0,0.05)"), row=1, col=1)

    fig.add_trace(go.Scatter(x=vol_12.index, y=vol_12, name="12M Vol",
        line=dict(color=PURPLE, width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=vol_36.index, y=vol_36, name="36M Vol",
        line=dict(color=CYAN, width=1.2, dash="dot")), row=2, col=1)

    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown",
        line=dict(color=RED, width=1.2), fill="tozeroy",
        fillcolor="rgba(255,76,76,0.15)"), row=3, col=1)

    fig.update_layout(**_layout(height=750, title="Volatility & Risk Dashboard"))
    return fig


# ── Forecast Chart ────────────────────────────────────────────────────────────

def plot_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame,
                   price_col: str = "price",
                   hist_years: int = 10,
                   title: str = "Gold Price Forecast") -> go.Figure:
    """
    Plot historical price + forecast with confidence interval ribbon.

    Parameters
    ----------
    df          : Historical DataFrame
    forecast_df : DataFrame with columns [ds, yhat, yhat_lower, yhat_upper]
    hist_years  : Number of historical years to show
    """
    cutoff = df.index.max() - pd.DateOffset(years=hist_years)
    hist   = df[df.index >= cutoff]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist[price_col],
        name="Historical", line=dict(color=GOLD, width=2),
        fill="tozeroy", fillcolor="rgba(255,215,0,0.06)",
    ))

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(forecast_df["ds"]), y=forecast_df["yhat"],
        name="Forecast", line=dict(color=RED, width=2.5),
    ))

    x_ci = (pd.to_datetime(forecast_df["ds"]).tolist() +
             pd.to_datetime(forecast_df["ds"]).tolist()[::-1])
    y_ci = (forecast_df["yhat_upper"].tolist() +
             forecast_df["yhat_lower"].tolist()[::-1])

    fig.add_trace(go.Scatter(
        x=x_ci, y=y_ci, fill="toself",
        fillcolor="rgba(255,76,76,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% CI",
    ))

    fig.update_layout(**_layout(height=600, title=title,
                                 xaxis_title="Date",
                                 yaxis_title="Gold Price (USD/oz)",
                                 legend=dict(bgcolor="rgba(0,0,0,0.6)",
                                             bordercolor=GOLD, borderwidth=1)))
    return fig


# ── Model Leaderboard ─────────────────────────────────────────────────────────

def plot_leaderboard(metrics_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparison of all models by MAE, RMSE, MAPE."""
    fig = make_subplots(rows=1, cols=3,
                         subplot_titles=["MAE", "RMSE", "MAPE (%)"])

    n = len(metrics_df)
    bar_colors = [GOLD if i == 0 else CYAN if i == 1 else "#555"
                  for i in range(n)]

    for col_i, metric in enumerate(["MAE", "RMSE", "MAPE"], 1):
        vals = metrics_df[metric]
        fig.add_trace(go.Bar(
            x=vals.index, y=vals.values,
            marker_color=bar_colors,
            text=[f"{v:.1f}" for v in vals.values],
            textposition="outside",
            name=metric,
        ), row=1, col=col_i)

    fig.update_layout(**_layout(height=520, showlegend=False,
                                 title="Model Leaderboard — All Models Compared",
                                 title_font=dict(color=GOLD, size=16)))
    return fig
