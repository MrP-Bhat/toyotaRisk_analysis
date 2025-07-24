import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Toyota Stock Risk Central",
    page_icon="🚗",
    layout="wide"
)

# --- DATA LOADING (Cached for performance) ---
@st.cache_data
def load_data():
    data = {}
    if os.path.exists('predicted_risk_scores.csv'):
        data['forecast'] = pd.read_csv('predicted_risk_scores.csv', parse_dates=['Date'])
    for w in [7, 30, 90, 250]:
        filename = f'rolling_vol_{w}d.csv'
        if os.path.exists(filename):
            data[f'hist_{w}d'] = pd.read_csv(filename, parse_dates=['Date'])
    return data

@st.cache_resource
def load_model():
    if os.path.exists('xgboost_risk_model.joblib'):
        model = joblib.load('xgboost_risk_model.joblib')
        return model
    return None

data_dict = load_data()
model = load_model()

# --- Initialize Session State for Monthly Navigation ---
if 'view_date' not in st.session_state:
    if 'forecast' in data_dict:
        st.session_state.view_date = data_dict['forecast']['Date'].min()
    else:
        st.session_state.view_date = datetime.now()

# --- SIDEBAR ---
st.sidebar.title("Control Panel ")
st.sidebar.header("Check Future Risk")

# --- Monthly Navigator ---
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col1:
    if st.button("< Prev", use_container_width=True):
        st.session_state.view_date -= relativedelta(months=1)
with col2:
    st.markdown(f"<h4 style='text-align: center;'>{st.session_state.view_date.strftime('%B %Y')}</h4>", unsafe_allow_html=True)
with col3:
    if st.button("Next >", use_container_width=True):
        st.session_state.view_date += relativedelta(months=1)

# --- NEW: Specific Date Selector (within the selected month) ---
start_of_month = st.session_state.view_date.replace(day=1)
end_of_month = start_of_month + relativedelta(months=1) - relativedelta(days=1)

if 'forecast' in data_dict:
    # Get all valid dates within the current month from the forecast data
    dates_in_month = data_dict['forecast'][
        (data_dict['forecast']['Date'] >= start_of_month) & 
        (data_dict['forecast']['Date'] <= end_of_month)
    ]['Date']

    # Create the dropdown for day selection
    selected_day = st.sidebar.selectbox(
        "Select a specific day to inspect:",
        options=dates_in_month,
        format_func=lambda date: date.strftime("%A, %d %B %Y") # e.g., "Wednesday, 23 July 2025"
    )

st.sidebar.header("Explore Historical Risk")
historical_options = st.sidebar.multiselect(
    "Choose historical volatility to display:",
    options=['7-Day (Weekly)', '30-Day (Monthly)', '90-Day (Quarterly)', '250-Day (Annual)'],
    default=['30-Day (Monthly)', '250-Day (Annual)']
)

# --- MAIN DASHBOARD ---
st.title(" Toyota Stock Risk Central")

# --- Section A: The Forecast ---
st.header(f"Risk Analysis for {start_of_month.strftime('%B %Y')}")

if 'forecast' in data_dict and not dates_in_month.empty:
    # Filter forecast data to the selected month
    forecast_df_filtered = data_dict['forecast'].set_index('Date').loc[dates_in_month]
    
    # --- NEW: Metric Card for the SPECIFIC selected day ---
    if selected_day:
        risk_score_day = forecast_df_filtered.loc[selected_day, 'RiskScore_Percent']
        st.metric(
            label=f"Risk Score for {selected_day.strftime('%d/%m/%Y')}",
            value=f"{risk_score_day:.1f}%"
        )
    
    st.markdown("---") # Visual separator

    # Monthly overview metrics
    avg_risk = forecast_df_filtered['RiskScore_Percent'].mean()
    max_risk = forecast_df_filtered['RiskScore_Percent'].max()
    min_risk = forecast_df_filtered['RiskScore_Percent'].min()

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Average Monthly Risk", f"{avg_risk:.1f}%")
    m_col2.metric("Highest Risk Day in Month", f"{max_risk:.1f}%")
    m_col3.metric("Lowest Risk Day in Month", f"{min_risk:.1f}%")
    
    # Monthly plot
    fig_forecast, ax_forecast = plt.subplots(figsize=(12, 5))
    ax_forecast.plot(forecast_df_filtered.index, forecast_df_filtered['RiskScore_Percent'], label='Predicted Risk Score (%)', color='red', marker='.', linestyle='-')
    
    # Highlight the specific day on the plot
    if selected_day:
        ax_forecast.axvline(selected_day, color='blue', linestyle='--', lw=2, label=f'Selected Day')

    ax_forecast.set_title(f"Predicted Risk for {start_of_month.strftime('%B %Y')}")
    ax_forecast.set_ylabel("Risk Score (0-100)")
    ax_forecast.set_ylim(0, max(100, forecast_df_filtered['RiskScore_Percent'].max() * 1.1))
    ax_forecast.grid(True, linestyle='--', alpha=0.6)
    ax_forecast.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    ax_forecast.legend()
    st.pyplot(fig_forecast)
else:
    st.warning("`predicted_risk_scores.csv` not found or contains no data for the selected month.")

# --- The rest of your dashboard remains the same ---
st.header("Historical Risk Explorer")
if historical_options:
    fig_hist, ax_hist = plt.subplots(figsize=(12, 5))
    mapper = {'7-Day (Weekly)': 'hist_7d', '30-Day (Monthly)': 'hist_30d', '90-Day (Quarterly)': 'hist_90d', '250-Day (Annual)': 'hist_250d'}
    for option in historical_options:
        key = mapper[option]
        if key in data_dict:
            df_to_plot = data_dict[key]
            ax_hist.plot(df_to_plot['Date'], df_to_plot[f'RollingVolatility_{key.split("_")[1]}'], label=option)
    ax_hist.set_title("Historical Rolling Volatility")
    ax_hist.set_ylabel("GARCH Forecasted Volatility")
    ax_hist.set_xlabel("Date")
    ax_hist.grid(True, linestyle='--', alpha=0.6)
    ax_hist.legend()
    st.pyplot(fig_hist)

