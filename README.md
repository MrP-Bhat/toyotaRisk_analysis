# Toyota Stock Risk Analysis

A machine-learning project that models and forecasts the financial risk of **Toyota Motor Corporation** stock using **GARCH(1,1)** volatility estimation and **XGBoost** regression. The project includes a Streamlit dashboard for interactive risk exploration.

---

## Overview

This project follows a three-stage pipeline:

1. **GARCH(1,1) Volatility Modelling** — Fits a GARCH(1,1) model to Toyota's historical log returns to estimate daily conditional volatility, and computes rolling volatility over multiple time windows (7-day, 30-day, 90-day, 250-day).
2. **XGBoost Risk Forecasting** — Trains an XGBoost regressor on engineered features (moving averages, momentum, lagged returns, lagged volatility) to predict future daily volatility, then runs a 365-day recursive forecast simulation to produce risk scores.
3. **Streamlit Dashboard** — An interactive web app to explore both the predicted risk scores (month-by-month with day-level drill-down) and the historical rolling volatility.

---

## Dataset

| Property | Value |
|----------|-------|
| Asset | Toyota Motor Corporation (TM) |
| Source | Yahoo Finance (historical OHLCV data) |
| Date Range | March 1980 – December 2024 |
| Records | 11,291 trading days |
| File | `data/toyotaData.csv` |

**Columns:** `Date`, `Adj Close`, `Close`, `High`, `Low`, `Open`, `Volume`

---

## Project Structure

```
Garch Risk Analysis/
├── app.ipynb                    # Full ML pipeline (EDA → GARCH → feature engineering → XGBoost → forecasting)
├── app.py                       # Streamlit dashboard application
├── xgboost_risk_model.joblib    # Trained XGBoost model (serialized with joblib)
├── predicted_risk_scores.csv    # 365-day forecast output (Dec 2024 – Dec 2025)
├── rolling_vol_7d.csv           # 7-day rolling GARCH volatility
├── rolling_vol_30d.csv          # 30-day rolling GARCH volatility
├── rolling_vol_90d.csv          # 90-day rolling GARCH volatility
├── rolling_vol_250d.csv         # 250-day rolling GARCH volatility
├── data/
│   ├── toyotaData.csv           # Raw Toyota stock data (1980–2024)
│   ├── df_with_volatility.csv   # Enriched dataset with all engineered features
│   └── rolling_vol_7d.csv       # Backup copy of 7-day rolling volatility
├── LICENSE                      # MIT License
└── .gitignore
```

---

## ML Pipeline (app.ipynb)

### Step 1 — Data Loading & Exploration
- Load `data/toyotaData.csv`, sort by date
- Inspect for missing values and basic statistics

### Step 2 — Log Returns
- Compute daily log returns: `log(Adj Close_t / Adj Close_{t-1})`

### Step 3 — GARCH(1,1) Model
- Fit a GARCH(1,1) model to the log return series using the `arch` library
- Extract the in-sample conditional volatility

### Step 4 — Rolling Volatility
- Compute GARCH-based rolling volatility using windows of **7, 30, 90, and 250 days**
- Each window's results are saved to a separate CSV for the dashboard

### Step 5 — Feature Engineering
Engineered features used for the XGBoost model:

| Feature | Description |
|---------|-------------|
| `MA_5` | 5-day moving average of Adj Close |
| `MA_10` | 10-day moving average of Adj Close |
| `Momentum` | Difference between consecutive log returns |
| `Lag1_Return` | Log return from 1 day ago |
| `Lag2_Return` | Log return from 2 days ago |
| `Lag1_Volatility` | GARCH volatility from 1 day ago |
| `RollingVolatility_7d` | 7-day rolling GARCH volatility |
| `TargetVolatility` | Target variable — next-day GARCH volatility |

### Step 6 — XGBoost Training & Evaluation
- Train an `XGBRegressor` (100 estimators, learning rate 0.1, max depth 3) on the engineered features
- Evaluate with MAE, RMSE, and R² score
- Visualize feature importance

### Step 7 — 365-Day Recursive Forecast
- Recursively simulate 365 future trading days
- At each step, predict next-day volatility, then simulate a new price point using the predicted volatility
- Convert raw volatility predictions to a 0–100 risk score percentage
- Save results to `predicted_risk_scores.csv`

### Step 8 — Model Export
- Serialize the trained XGBoost model to `xgboost_risk_model.joblib`

---

## Dashboard (app.py)

The Streamlit dashboard provides two main views:

### Risk Forecast View
- **Monthly navigation** — Browse predicted risk month-by-month with Prev/Next buttons
- **Day-level inspection** — Select a specific date within the month to see its risk score
- **Monthly summary metrics** — Average, highest, and lowest risk scores for the selected month
- **Forecast chart** — Line plot of daily predicted risk scores with a blue marker for the selected day

### Historical Risk Explorer
- **Multi-select rolling volatility** — Choose any combination of 7-day, 30-day, 90-day, and 250-day windows
- **Overlay chart** — Visualize how volatility has evolved over 44 years of Toyota stock history

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/MrP-Bhat/toyotaRisk_analysis.git
cd toyotaRisk_analysis

# Install dependencies
pip install streamlit pandas numpy matplotlib joblib python-dateutil
```

To re-run the full ML pipeline in the notebook, you will also need:

```bash
pip install arch xgboost scikit-learn tqdm
```

### Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

### Run the Notebook

Open `app.ipynb` in Jupyter Notebook or JupyterLab to step through the full pipeline:

```bash
jupyter notebook app.ipynb
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Interactive web dashboard |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `matplotlib` | Plotting and visualization |
| `arch` | GARCH(1,1) volatility modelling |
| `xgboost` | Gradient-boosted regression for volatility prediction |
| `scikit-learn` | Train/test split, evaluation metrics |
| `joblib` | Model serialization |
| `tqdm` | Progress bars for long-running loops |

---

## Output Files

| File | Description |
|------|-------------|
| `predicted_risk_scores.csv` | 365 rows — daily predicted volatility and risk score (%) from Dec 2024 to Dec 2025 |
| `rolling_vol_7d.csv` | GARCH-based 7-day rolling volatility (1980–2024) |
| `rolling_vol_30d.csv` | GARCH-based 30-day rolling volatility (1980–2024) |
| `rolling_vol_90d.csv` | GARCH-based 90-day rolling volatility (1980–2024) |
| `rolling_vol_250d.csv` | GARCH-based 250-day rolling volatility (1980–2024) |
| `xgboost_risk_model.joblib` | Serialized trained XGBoost model |
| `data/df_with_volatility.csv` | Full dataset with all engineered features and GARCH volatility |

---

## License

This project is licensed under the [MIT License](LICENSE).
