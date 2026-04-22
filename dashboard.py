# Retail_Analytics_Streamlit_Dashboard.py
# Streamlit app that loads the "Online Retail (1).xlsx" dataset, performs cleaning, EDA,
# RFM + k-means clustering, trains five classifiers to predict cluster membership,
# compares them, runs a Prophet forecast for daily sales, and shows price-sensitivity and CLV.
#
# Usage: install required packages and run:
#   pip install -r requirements.txt
#   streamlit run Retail_Analytics_Streamlit_Dashboard.py
#
# Minimal requirements.txt (example):
# pandas
# numpy
# matplotlib
# seaborn
# scikit-learn
# xgboost
# catboost
# prophet
# streamlit
# openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from io import BytesIO
import base64

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Prophet import (wrapped)
try:
    from prophet import Prophet
except Exception:
    Prophet = None

st.set_page_config(layout="wide", page_title="Retail Analytics Multi-Dashboard")

# ---------------------- Helper functions ----------------------
@st.cache_data
def load_data(uploaded_file=None, filepath="Online Retail (1).xlsx"):
    # Read Excel; allow user upload to override local file
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            df = pd.read_excel(filepath, engine='openpyxl')
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    # Ensure datetime
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    return df

@st.cache_data
def clean_data(df):
    df = df.copy()

    # Fill Description
    if 'Description' in df.columns:
        df['Description'] = df['Description'].fillna('Unknown Description')

    # Drop rows without CustomerID
    if 'CustomerID' in df.columns:
        df = df.dropna(subset=['CustomerID'])
        # Ensure CustomerID as int or string
        try:
            df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
        except Exception:
            df['CustomerID'] = df['CustomerID'].astype(str)

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Convert InvoiceNo and StockCode to string if present
    for col in ['InvoiceNo', 'StockCode']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Ensure numeric columns
    for col in ['Quantity', 'UnitPrice']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # TotalSales
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        df['TotalSales'] = df['Quantity'] * df['UnitPrice']

    # Remove negative or zero invoices if that is desired (keep returns? provide toggle later)
    return df

@st.cache_data
def prepare_monthly_sales(df):
    df = df.copy()
    df = df[df['InvoiceDate'].notna()]
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
    monthly = df.groupby('InvoiceMonth')['TotalSales'].sum().reset_index()
    monthly = monthly.set_index('InvoiceMonth')
    return monthly

@st.cache_data
def top_countries(df, n=10):
    return df.groupby('Country')['TotalSales'].sum().sort_values(ascending=False).head(n)

@st.cache_data
def top_products(df, n=10):
    return df.groupby('Description')['TotalSales'].sum().sort_values(ascending=False).head(n)

# RFM and clustering
@st.cache_data
def compute_rfm(df):
    latest_purchase = df['InvoiceDate'].max()
    current_date = latest_purchase + timedelta(days=1)

    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (current_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalSales', 'sum')
    ).reset_index()

    rfm = rfm[rfm['Monetary'] > 0]
    rfm[['Recency', 'Frequency', 'Monetary']] = rfm[['Recency', 'Frequency', 'Monetary']].fillna(0)
    return rfm

@st.cache_data
def scale_rfm(rfm):
    rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=rfm_log.columns, index=rfm.index)
    return rfm_scaled_df, scaler

@st.cache_data
def run_kmeans(rfm_scaled_df, k=3):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled_df)
    return labels, km

# Modeling and evaluation

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'CatBoost': CatBoostClassifier(verbose=0, random_seed=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, preds)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

            # feature importances: try multiple attributes
            if hasattr(model, 'feature_importances_'):
                feat_imp = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feat_imp = np.mean(np.abs(model.coef_), axis=0)
            elif hasattr(model, 'get_feature_importance'):
                try:
                    feat_imp = model.get_feature_importance()
                except Exception:
                    feat_imp = np.zeros(X_train.shape[1])
            else:
                feat_imp = np.zeros(X_train.shape[1])

            results[name] = {
                'model': model,
                'report': report,
                'confusion_matrix': cm,
                'accuracy': acc,
                'f1_weighted': f1,
                'feature_importances': feat_imp
            }
        except Exception as e:
            st.warning(f"Model {name} failed: {e}")

    return results

# Utility to download dataframe
def get_table_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# ---------------------- Streamlit layout ----------------------
st.title("ANALYSIS ON E-COMMERCE - MARKET DASHBOARD")

# Sidebar: data upload & options
st.sidebar.header("Data & Settings")
uploaded_file = st.sidebar.file_uploader("Upload an Excel file (Optional)", type=['xlsx', 'xls'])
local_path = st.sidebar.text_input("Local path (used if no upload)", value="Online Retail (1).xlsx")
use_local = st.sidebar.checkbox("Use local path if upload is empty", value=True)

# RFM clustering settings
k_clusters = st.sidebar.slider("K for K-Means (Clusters)", min_value=2, max_value=8, value=3)
train_test_split_ratio = st.sidebar.slider("Test set fraction", min_value=0.1, max_value=0.5, value=0.2)

# Forecast settings
forecast_periods = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=365, value=30)

# Load data
if uploaded_file is not None:
    df = load_data(uploaded_file)
elif use_local:
    df = load_data(filepath=local_path)
else:
    st.info("Please upload a file or enable using a local path.")
    df = None

if df is None:
    st.stop()

# Show raw data toggle
if st.sidebar.checkbox("Show raw data (first 100 rows)"):
    st.subheader("Raw data sample")
    st.dataframe(df.head(100))

# Clean data
df_clean = clean_data(df)

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Revenue", f"{df_clean['TotalSales'].sum():,.2f}")
with col2:
    st.metric("Unique Customers", df_clean['CustomerID'].nunique())
with col3:
    st.metric("Total Orders", df_clean['InvoiceNo'].nunique())
with col4:
    st.metric("Date Range", f"{df_clean['InvoiceDate'].min().date()} to {df_clean['InvoiceDate'].max().date()}")

# Tabs for organization
tabs = st.tabs(["EDA", "RFM & Clustering", "Modeling & Comparison", "Forecasting", "Price Sensitivity & CLV", "Download & Notes"])

# ------------- EDA Tab ---------------
with tabs[0]:
    st.header("Exploratory Data Analysis")
    st.subheader("Monthly Sales")
    monthly_sales = prepare_monthly_sales(df_clean)
    st.line_chart(monthly_sales['TotalSales'])

    st.subheader("Top 10 Countries by Total Sales")
    top10_c = top_countries(df_clean, 10)
    st.bar_chart(top10_c)
    st.dataframe(top10_c.reset_index())

    st.subheader("Top 10 Products by Total Sales")
    top10_p = top_products(df_clean, 10)
    st.bar_chart(top10_p)
    st.dataframe(top10_p.reset_index())

# ------------- RFM & Clustering Tab ---------------
with tabs[1]:
    st.header("RFM Analysis & Customer Segmentation")
    rfm = compute_rfm(df_clean)
    st.subheader("RFM sample")
    st.dataframe(rfm.head())

    rfm_scaled_df, scaler = scale_rfm(rfm)

    st.subheader("K-Means clustering")
    labels, km_model = run_kmeans(rfm_scaled_df, k=k_clusters)
    rfm['Cluster'] = labels

    st.write(rfm['Cluster'].value_counts().sort_index())

    st.subheader("Cluster profiles (mean RFM)")
    st.dataframe(rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2))

    # PCA visualization
    pca = PCA(n_components=2)
    rfm_pca = pca.fit_transform(rfm_scaled_df)
    rfm_pca_df = pd.DataFrame(rfm_pca, columns=['PC1', 'PC2'])
    rfm_pca_df['Cluster'] = rfm['Cluster'].values

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=rfm_pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=60, ax=ax)
    ax.set_title('Customer Segments (PCA)')
    st.pyplot(fig)

# ------------- Modeling & Comparison Tab ---------------
with tabs[2]:
    st.header("Train classifiers to predict cluster membership and compare")

    # Prepare features & target
    X = rfm_scaled_df
    y = rfm['Cluster']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_ratio, random_state=42)
    st.write(f"Training size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Train and evaluate
    with st.spinner("Training models... this may take a moment"):
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Show comparison table
    compare_df = []
    for name, res in results.items():
        compare_df.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'Weighted F1': res['f1_weighted']
        })
    compare_df = pd.DataFrame(compare_df).sort_values('Weighted F1', ascending=False)

    st.subheader("Model comparison (sorted by Weighted F1)")
    st.dataframe(compare_df.set_index('Model').style.format({ 'Accuracy': '{:.3f}', 'Weighted F1':'{:.3f}'}))

    best_model_name = compare_df.iloc[0]['Model'] if not compare_df.empty else None
    if best_model_name:
        st.success(f"Best model by Weighted F1: {best_model_name}")

    # Detailed view per model
    st.subheader("Model details & confusion matrices")
    for name, res in results.items():
        st.markdown(f"### {name}")
        st.write(f"Accuracy: {res['accuracy']:.3f} — Weighted F1: {res['f1_weighted']:.3f}")
        st.write(pd.DataFrame(res['report']).transpose())

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"Confusion Matrix — {name}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        # Feature importances
        fi = res['feature_importances']
        if fi is not None and fi.sum() != 0:
            fi_series = pd.Series(fi, index=X.columns).sort_values(ascending=False)
            st.bar_chart(fi_series)

# ------------- Forecasting Tab ---------------
with tabs[3]:
    st.header("Daily Sales Forecasting using Prophet")
    if Prophet is None:
        st.warning("The 'prophet' package is not available in the environment. To enable forecasting, install prophet (pip install prophet).")
    else:
        daily_sales = df_clean.groupby(pd.Grouper(key='InvoiceDate', freq='D'))['TotalSales'].sum().reset_index()
        daily_sales = daily_sales.rename(columns={'InvoiceDate': 'ds', 'TotalSales': 'y'})
        daily_sales = daily_sales.dropna(subset=['ds'])

        st.subheader("Daily sales sample")
        st.dataframe(daily_sales.head())

        # Fit Prophet
        m = Prophet(interval_width=0.95)
        m.fit(daily_sales)
        future = m.make_future_dataframe(periods=int(forecast_periods), freq='D')
        forecast = m.predict(future)

        st.subheader("Forecast (next {} days)".format(forecast_periods))
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

        fig1 = m.plot(forecast)
        st.pyplot(fig1)

        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)

# ------------- Price Sensitivity & CLV Tab ---------------
with tabs[4]:
    st.header("Price Sensitivity & CLV")

    # Price sensitivity scatter (log-log)
    filtered_df = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    price_quantity = filtered_df.groupby(['Description', 'UnitPrice'])['Quantity'].sum().reset_index()

    st.subheader("Price vs Quantity (log-log scatter)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(price_quantity['UnitPrice'], price_quantity['Quantity'], alpha=0.4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Unit Price')
    ax.set_ylabel('Total Quantity')
    ax.set_title('Price Sensitivity')
    st.pyplot(fig)

    # CLV proxy
    clv = rfm[['CustomerID', 'Monetary']].copy()
    clv = clv.rename(columns={'Monetary': 'CLV'}).reset_index(drop=True)
    st.subheader("Customer Lifetime Value (CLV) proxy — Monetary")
    st.dataframe(clv.describe())

# ------------- Download & Notes Tab ---------------
with tabs[5]:
    st.header("Download cleaned data and artifacts")
    st.markdown(get_table_download_link(df_clean.reset_index(drop=True), filename="cleaned_retail_data.csv"), unsafe_allow_html=True)

    st.subheader("Notes & What I fixed")
    st.markdown("""
    - Ensured InvoiceDate is parsed as datetime.
    - Filled missing Descriptions and dropped rows without CustomerID.
    - Calculated TotalSales and removed duplicate rows.
    - Kept daily sales as datetime for Prophet (avoid converting to date-only prematurely).
    - Applied log1p + StandardScaler to RFM before clustering and modeling.
    - Wrapped Prophet import to avoid crashing the app if not available.
    - Provided model training with error handling to continue even if a specific model fails.

    How the best model is chosen in the dashboard:
    - Models are compared using weighted F1 score (weighted average across classes).
    - The dashboard declares the model with the highest Weighted F1 as the best.
    """)

    st.markdown("### How to run")
    st.code("pip install -r requirements.txt\nstreamlit run Retail_Analytics_Streamlit_Dashboard.py")

    st.markdown("### Limitations & next steps")
    st.markdown("""
    - You may want to tune models with hyperparameter search (GridSearchCV or Optuna).
    - Consider using time-aware CV for forecasting and customer lifetime predictions.
    - Feature engineering: include recency buckets, day-of-week, product categories, promotion flags, etc.
    - For large datasets, replace in-memory operations with chunking or a database back-end.
    """)

# End of app

