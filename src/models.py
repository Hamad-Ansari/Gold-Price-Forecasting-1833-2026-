"""
models.py
─────────
Model training, evaluation, and walk-forward validation utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, predicted: np.ndarray,
                    label: str = "", verbose: bool = True) -> dict:
    """
    Compute MAE, RMSE, and MAPE.

    Returns
    -------
    dict : {'MAE': float, 'RMSE': float, 'MAPE': float}
    """
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    result = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}

    if verbose and label:
        print(f"  [{label:22s}]  MAE={mae:8.2f}  RMSE={rmse:8.2f}  MAPE={mape:5.2f}%")

    return result


def metrics_table(metrics_store: dict) -> pd.DataFrame:
    """Convert a metrics_store dict to a sorted DataFrame."""
    df = pd.DataFrame(metrics_store).T.round(3)
    df.index.name = "Model"
    df = df.sort_values("MAPE")
    df["Rank"] = range(1, len(df) + 1)
    return df[["Rank", "MAE", "RMSE", "MAPE"]]


# ── ARIMA ─────────────────────────────────────────────────────────────────────

def fit_arima(train_series: pd.Series, order: tuple = (2, 1, 1)):
    """Fit ARIMA model. Returns fitted model."""
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(train_series, order=order)
    return model.fit()


def predict_arima(fitted_model, n_steps: int, index=None) -> pd.Series:
    """Generate ARIMA forecast."""
    fc = fitted_model.forecast(steps=n_steps)
    if index is not None:
        fc.index = index
    return fc


# ── Prophet ───────────────────────────────────────────────────────────────────

def build_prophet_df(series: pd.Series) -> pd.DataFrame:
    """Convert a price Series to Prophet-format DataFrame."""
    df = series.reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def fit_prophet(train_df: pd.DataFrame,
                changepoint_prior: float = 0.3,
                seasonality_prior: float = 10.0,
                interval_width: float = 0.95,
                regressors: list = None):
    """
    Fit a Prophet model with optional external regressors.

    Parameters
    ----------
    train_df         : Prophet-format DataFrame with columns [ds, y, ...]
    changepoint_prior: Flexibility of trend changepoints
    seasonality_prior: Seasonality strength
    interval_width   : Confidence interval width
    regressors       : List of extra regressor column names

    Returns
    -------
    Fitted Prophet model
    """
    from prophet import Prophet
    m = Prophet(
        changepoint_prior_scale=changepoint_prior,
        seasonality_prior_scale=seasonality_prior,
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=interval_width,
    )
    m.add_seasonality(name="quarterly", period=91.25, fourier_order=5)
    if regressors:
        for reg in regressors:
            m.add_regressor(reg, standardize=True)
    m.fit(train_df)
    return m


# ── XGBoost ───────────────────────────────────────────────────────────────────

def fit_xgboost(X_train, y_train, X_val=None, y_val=None,
                n_estimators: int = 500,
                max_depth: int = 5,
                learning_rate: float = 0.05,
                subsample: float = 0.8,
                colsample_bytree: float = 0.8) -> xgb.XGBRegressor:
    """Fit an XGBoost regressor."""
    eval_set = [(X_val, y_val)] if X_val is not None else None
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=eval_set,
              verbose=False)
    return model


# ── Random Forest ─────────────────────────────────────────────────────────────

def fit_random_forest(X_train, y_train,
                      n_estimators: int = 300,
                      max_depth: int = 8) -> RandomForestRegressor:
    """Fit a Random Forest regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── LSTM ──────────────────────────────────────────────────────────────────────

def build_lstm(lookback: int = 24, units: tuple = (128, 64),
               dropout: float = 0.2):
    """
    Build a 2-layer LSTM model.

    Parameters
    ----------
    lookback : Number of timesteps in input sequence
    units    : Tuple of (layer1_units, layer2_units)
    dropout  : Dropout rate after each LSTM layer

    Returns
    -------
    Compiled Keras model
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    tf.random.set_seed(42)

    model = Sequential([
        LSTM(units[0], return_sequences=True, input_shape=(lookback, 1)),
        Dropout(dropout),
        LSTM(units[1], return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber")
    return model


def fit_lstm(model, X_train, y_train, X_val, y_val,
             epochs: int = 150, batch_size: int = 32,
             patience: int = 20):
    """
    Train the LSTM model with early stopping.

    Returns
    -------
    tuple : (fitted model, training history)
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(patience=patience, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(patience=patience // 2, factor=0.5, verbose=0),
    ]
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=0,
    )
    return model, history


# ── Stacked Ensemble ──────────────────────────────────────────────────────────

def stacked_ensemble_oof(X_train, y_train, features: list,
                          n_folds: int = 5) -> tuple:
    """
    Generate out-of-fold (OOF) predictions for stacking.

    Returns
    -------
    tuple : (oof_xgb, oof_rf) — numpy arrays of OOF predictions
    """
    n = len(X_train)
    fold_size = n // n_folds
    oof_xgb = np.zeros(n)
    oof_rf  = np.zeros(n)

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end   = val_start + fold_size if fold < n_folds - 1 else n
        tr_idx  = list(range(0, val_start)) + list(range(val_end, n))
        val_idx = list(range(val_start, val_end))
        if len(tr_idx) < 50:
            continue

        Xtr = X_train.iloc[tr_idx][features]
        ytr = y_train.iloc[tr_idx]
        Xvl = X_train.iloc[val_idx][features]

        m_xgb = xgb.XGBRegressor(n_estimators=200, max_depth=4,
                                   learning_rate=0.05, random_state=42, verbosity=0)
        m_xgb.fit(Xtr, ytr)
        oof_xgb[val_idx] = m_xgb.predict(Xvl)

        m_rf = RandomForestRegressor(n_estimators=100, max_depth=6,
                                      random_state=42, n_jobs=-1)
        m_rf.fit(Xtr, ytr)
        oof_rf[val_idx] = m_rf.predict(Xvl)

    return oof_xgb, oof_rf


def fit_meta_learner(oof_xgb, oof_rf, y_train,
                      alpha: float = 10.0) -> Ridge:
    """Fit a Ridge meta-learner on OOF predictions."""
    meta_X = np.column_stack([oof_xgb, oof_rf])
    meta   = Ridge(alpha=alpha)
    meta.fit(meta_X, y_train)
    return meta


# ── Walk-Forward Validation ───────────────────────────────────────────────────

def walk_forward_xgb(df: pd.DataFrame, features: list, target: str = "price",
                      initial_train: int = 120, step: int = 12) -> list:
    """
    Walk-forward cross-validation using XGBoost with expanding window.

    Parameters
    ----------
    df            : DataFrame with features and target
    features      : List of feature column names
    target        : Target column name
    initial_train : Number of initial training observations
    step          : Number of steps to forecast per fold

    Returns
    -------
    list of dicts : [{'window_end', 'mae', 'mape', 'dates', 'actual', 'predicted'}, ...]
    """
    df_wf   = df[features + [target]].dropna().copy()
    indices = df_wf.index
    results = []

    for start in range(initial_train, len(df_wf) - step, step):
        X_tr = df_wf.iloc[:start][features]
        y_tr = df_wf.iloc[:start][target]
        X_te = df_wf.iloc[start:start + step][features]
        y_te = df_wf.iloc[start:start + step][target]

        m = xgb.XGBRegressor(n_estimators=200, max_depth=4,
                               learning_rate=0.05, random_state=42, verbosity=0)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)

        mae  = mean_absolute_error(y_te, preds)
        mape = np.mean(np.abs((y_te.values - preds) / y_te.values)) * 100

        results.append({
            "window_end": indices[start],
            "mae": mae,
            "mape": mape,
            "dates": indices[start:start + step],
            "actual": y_te.values,
            "predicted": preds,
        })

    return results
