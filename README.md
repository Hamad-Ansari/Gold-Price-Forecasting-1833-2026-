# 🪙 Gold Price Time Series Forecasting
### Advanced Analytics & Modeling | 1833–2026

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-brightgreen?style=for-the-badge&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-gold?style=for-the-badge)

**193 years · 2,300+ observations · 6 forecasting models · 20 analytical sections**

</div>

---

## 📌 Overview

A **Kaggle Gold Medal–level** end-to-end time series project covering the complete lifecycle of quantitative financial analysis — from raw data ingestion to production-ready forecasting — applied to gold price data spanning nearly two centuries.

This project demonstrates:
- Rigorous statistical analysis and stationarity testing
- Classical, ML, and deep learning forecasting approaches
- Proper time-series cross-validation (no data leakage)
- Professional-grade interactive visualizations
- Macro-economic feature integration (World Bank CPI data)

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. Nothing here constitutes financial advice.

---

## 🗂️ Repository Structure

```
gold-forecasting/
│
├── 📓 notebooks/
│   └── gold_forecasting_notebook.ipynb   # Full analysis (64 cells, pre-executed)
│
├── 📊 data/
│   ├── gold_advanced_features.csv        # Primary dataset with engineered features
│   ├── monthly.csv                       # Raw monthly gold prices
│   └── worldbank_inflation_clean.csv     # World Bank CPI inflation data
│
├── 🔧 src/
│   ├── preprocessing.py                  # Data loading & feature engineering
│   ├── models.py                         # Model training & evaluation utilities
│   ├── visualization.py                  # Reusable Plotly chart functions
│   └── utils.py                          # Helper functions & metrics
│
├── 📈 results/
│   └── model_metrics.csv                 # Final model comparison table
│
├── requirements.txt                      # Python dependencies
├── environment.yml                       # Conda environment
├── .gitignore                            # Git ignore rules
└── README.md                             # You are here
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | Historical gold price records + World Bank CPI |
| **Frequency** | Monthly |
| **Date Range** | January 1833 – February 2026 |
| **Observations** | 2,306 |
| **Primary Target** | Gold Price (USD/troy oz) |
| **Features** | 26 engineered (lags, rolling stats, momentum, cyclical encoding) |

### Key Features Engineered

| Feature | Description |
|---|---|
| `lag_1` … `lag_12` | Lagged price values (1–12 months) |
| `roll_mean_3/6/12` | Rolling mean (3, 6, 12 months) |
| `volatility_3` | 3-month rolling std deviation |
| `momentum_1/3` | Price momentum over 1 and 3 months |
| `pct_change_1/3` | Percentage price change |
| `price_to_roll3/12` | Price relative to rolling mean |
| `month_sin/cos` | Cyclical month encoding |

---

## 🔍 Analysis Pipeline

### Step 1 — Data Loading & Preprocessing
- CSV ingestion, datetime parsing, index alignment
- Missing value handling (forward fill + interpolation)
- Time continuity validation

### Step 2 — Exploratory Data Analysis
- Full historical price evolution (1833–2026) with log-scale view
- Rolling mean overlays (30M, 90M, 180M)
- Distribution analysis: histogram, KDE, boxplot by era
- Monthly seasonality heatmap by decade

### Step 3 — Time Series Decomposition
- Multiplicative STL decomposition (trend / seasonality / residuals)
- Economic interpretation of each component

### Step 4 — Stationarity Testing
- Augmented Dickey-Fuller (ADF) test
- KPSS test
- Applied to: raw price, log price, 1st difference, log 1st difference
- ACF / PACF plots for ARIMA order selection

### Step 5 — Correlation & Feature Importance
- Pearson correlation heatmap across all features
- Lag autocorrelation analysis (lags 1–24)
- Random Forest feature importance ranking

### Step 6 — Volatility & Risk Analysis
- Rolling annualized volatility (12M & 36M)
- Max drawdown from all-time high
- Volatility clustering visualization

### Step 7 — Forecasting Models

| Model | Type | Key Config |
|---|---|---|
| ARIMA(2,1,1) | Classical | AIC-selected orders |
| Prophet | Statistical ML | Multiplicative seasonality + quarterly component |
| Prophet + Inflation | Statistical ML | US CPI as external regressor |
| XGBoost | Gradient Boosting | 500 trees, depth=5, lr=0.05 |
| Random Forest | Ensemble | 300 trees, depth=8 |
| LSTM | Deep Learning | 128→64 units, Huber loss, dropout=0.2 |
| **Stacked Ensemble** | Meta-learning | XGB + RF → Ridge (OOF stacking) |

### Step 8 — Model Evaluation
- MAE, RMSE, MAPE across all models
- Time-based train/test split (cutoff: Jan 2020)
- Visual actual vs. predicted comparison

### Step 9 — Future Forecast
- 24-month forward forecast (2026–2027)
- 90% confidence intervals
- Month-by-month price range table

### Step 10 — Macro Correlation
- Gold vs. US CPI inflation (World Bank data, 1960–2024)
- Annual return scatter with OLS regression
- High-inflation year analysis (CPI > 5%)

### Bonus Sections
- **Rolling Sharpe Ratio** (12M & 36M) + rolling CAGR (5Y & 10Y)
- **Walk-Forward Cross-Validation** (expanding window, 12-month steps)
- **Fourier Spectral Analysis** — hidden price cycles via FFT
- **GARCH(1,1) Volatility Model** — conditional vol + persistence metrics
- **Stacked Ensemble** — OOF meta-learning
- **Isolation Forest** anomaly detection
- **Z-score Regime Classification** — Bull / Bear / Neutral
- **Interactive Time-Range Explorer** — range slider + period buttons
- **Decade-by-Decade Performance** — return, CAGR, vol, drawdown

---

## 📈 Key Findings

| Insight | Finding |
|---|---|
| **Long-term appreciation** | Gold: $19 → $5,020 (~265× in 193 years) |
| **Momentum** | Lag-1 autocorrelation ≈ 0.99 — strongest single predictor |
| **Volatility persistence** | GARCH alpha+beta > 0.97 — shocks last months |
| **Seasonality** | Best months: Jan–Feb, Aug–Sep. Weakest: Mar–Apr |
| **Inflation hedge** | Positive average return in high-CPI years, but high variance |
| **Hidden cycles** | Dominant cycles at ~12M, ~36M, ~84M (via FFT) |
| **Risk** | Max drawdown: −65% (1980 peak → 1982 trough) |
| **Post-2000 Sharpe** | Rolling 12M Sharpe > 0 for 65%+ of months since 2000 |

---

## 🚀 Getting Started

### Option 1 — Conda (Recommended)

```bash
git clone https://github.com/YOUR_USERNAME/gold-forecasting.git
cd gold-forecasting
conda env create -f environment.yml
conda activate gold-forecasting
jupyter notebook notebooks/gold_forecasting_notebook.ipynb
```

### Option 2 — pip

```bash
git clone https://github.com/YOUR_USERNAME/gold-forecasting.git
cd gold-forecasting
pip install -r requirements.txt
jupyter notebook notebooks/gold_forecasting_notebook.ipynb
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.24 | Numerical computing |
| `pandas` | ≥2.0 | Data manipulation |
| `matplotlib` | ≥3.7 | Static plots (decomposition, ACF) |
| `plotly` | ≥5.15 | Interactive visualizations |
| `statsmodels` | ≥0.14 | ARIMA, decomposition, stationarity tests |
| `prophet` | ≥1.1 | Trend + seasonality forecasting |
| `scikit-learn` | ≥1.3 | ML models, metrics, preprocessing |
| `xgboost` | ≥1.7 | Gradient boosting |
| `tensorflow` | ≥2.13 | LSTM deep learning |
| `arch` | ≥6.0 | GARCH volatility modeling |
| `scipy` | ≥1.11 | FFT spectral analysis |

---

## 📉 Model Results Summary

| Rank | Model | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 🥇 | Stacked Ensemble | — | — | Best |
| 🥈 | LSTM | — | — | 2nd |
| 🥉 | XGBoost | — | — | 3rd |
| 4 | Random Forest | — | — | — |
| 5 | Prophet + Inflation | — | — | — |
| 6 | Prophet | — | — | — |
| 7 | ARIMA(2,1,1) | — | — | Baseline |

> Exact metrics vary by run. See `results/model_metrics.csv` or execute the notebook for your dataset's results.

---

## 🎨 Visualization Style

All interactive charts use a **dark neon theme**:

| Element | Color |
|---|---|
| Gold price | `#FFD700` (gold) |
| Trend / MA lines | `#00FFFF` (cyan) |
| Forecasts | `#FF4C4C` (red) |
| Volatility | `#9B59B6` (purple) |
| Anomalies | `#FF4C4C` (red) |
| Background | `#0d0d0d` (near black) |

---

## 📁 Reproducing Results

All random seeds are fixed (`random_state=42`, `tf.random.set_seed(42)`).
Data splits are **strictly time-based** — no shuffling at any stage.
Walk-forward validation uses an expanding window to simulate real deployment conditions.

```python
# Train/test split
CUTOFF = '2020-01-01'
train = df[df.index < CUTOFF]
test  = df[df.index >= CUTOFF]
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repo
2. Create your branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

## 🌟 Acknowledgements

- Gold price data sourced from historical commodity records
- Macro data from the [World Bank Open Data](https://data.worldbank.org/)
- Inspired by the open-source time series and quantitative finance community

---

<div align="center">

**If this project helped you, please consider giving it a ⭐**

Made with 🪙 by a data scientist who spent way too long staring at gold charts.

</div>
