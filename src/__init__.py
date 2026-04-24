"""
Gold Price Forecasting — Source Package
"""
from .preprocessing import load_gold_data, load_inflation_data, build_full_feature_set, time_split
from .models import compute_metrics, metrics_table
from .utils import full_stationarity_report, performance_summary, detect_anomalies
from .visualization import plot_price_history, plot_actual_vs_predicted, plot_forecast, plot_leaderboard

__all__ = [
    "load_gold_data",
    "load_inflation_data",
    "build_full_feature_set",
    "time_split",
    "compute_metrics",
    "metrics_table",
    "full_stationarity_report",
    "performance_summary",
    "detect_anomalies",
    "plot_price_history",
    "plot_actual_vs_predicted",
    "plot_forecast",
    "plot_leaderboard",
]
