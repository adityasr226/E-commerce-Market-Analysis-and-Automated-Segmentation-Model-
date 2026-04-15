

import streamlit as st
st.set_page_config(layout='wide', page_title='E-Commerce Market Analysis', initial_sidebar_state='expanded')

# --- Imports ---
import os
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import base64

# ML & preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# Sentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

# Linear regression for elasticity
from sklearn.linear_model import LinearRegression

# --- Utility functions ---
@st.cache_data
def read_excel_uploaded(uploaded_file):
    # Read excel or csv uploaded file
    try:
        if uploaded_file.name.lower().endswith('.xlsx') or uploaded_file.name.lower().endswith('.xls'):
            return pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            return pd.read_csv(uploaded_file)
    except Exception as e:
        raise

@st.cache_data
def load_data(uploaded_file=None, server_path='/mnt/data/Online Retail (1).xlsx'):
    """Load dataset from uploaded file or server path if it exists.
    Prefer uploaded_file. If none provided, check server_path exists and load it.
    """
    if uploaded_file is not None:
        try:
            df = read_excel_uploaded(uploaded_file)
            return df
        except Exception as e:
            raise RuntimeError(f'Failed to read uploaded file: {e}')
    # No upload — check local server path
    if server_path and os.path.exists(server_path):
        try:
            df = pd.read_excel(server_path, engine='openpyxl')
            return df
        except Exception as e:
            raise RuntimeError(f'Failed to read server file: {e}')
    # Nothing available
    return None

@st.cache_data
def clean_retail(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    if 'InvoiceNo' in df.columns:
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C', na=False)]
    if 'Quantity' in df.columns:
        df = df[df['Quantity'] > 0]
    if 'UnitPrice' in df.columns and 'Quantity' in df.columns:
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    # Drop rows with missing key fields
    key_cols = [c for c in ['InvoiceNo','StockCode','Description','Quantity','InvoiceDate'] if c in df.columns]
    if key_cols:
        df = df.dropna(subset=key_cols)
    return df

# RFM
@st.cache_data
def compute_rfm(df: pd.DataFrame, snapshot_date: datetime=None) -> pd.DataFrame:
    if 'InvoiceDate' not in df.columns:
        raise ValueError('InvoiceDate column required for RFM')
    if snapshot_date is None:
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    grouped = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    grouped.columns = ['CustomerID','Recency','Frequency','Monetary']
    # rank-based qcut
    grouped['R_score'] = pd.qcut(grouped['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(int)
    grouped['F_score'] = pd.qcut(grouped['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    grouped['M_score'] = pd.qcut(grouped['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    grouped['RFM_Segment'] = grouped['R_score'].map(str) + grouped['F_score'].map(str) + grouped['M_score'].map(str)
    grouped['RFM_Score'] = grouped[['R_score','F_score','M_score']].sum(axis=1)
    return grouped

# KMeans clustering on RFM
@st.cache_data
def rfm_kmeans(rfm_df: pd.DataFrame, n_clusters=4):
    features = rfm_df[['Recency','Frequency','Monetary']].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(Xs)
    rfm_df = rfm_df.copy()
    rfm_df['cluster'] = labels
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['Recency','Frequency','Monetary'])
    return rfm_df, centroids

# PCA for 2D visualization
@st.cache_data
def rfm_pca_plot(rfm_df: pd.DataFrame):
    features = rfm_df[['Recency','Frequency','Monetary']].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    comp = pca.fit_transform(Xs)
    out = rfm_df.copy()
    out['pca1'] = comp[:,0]
    out['pca2'] = comp[:,1]
    return out, pca

# Elasticity
@st.cache_data
def price_elasticity(df: pd.DataFrame, product_col='StockCode', price_col='UnitPrice', qty_col='Quantity', min_obs=8):
    out = []
    for prod, g in df.groupby(product_col):
        g2 = g[(g[price_col] > 0) & (g[qty_col] > 0)]
        if len(g2) < min_obs:
            continue
        try:
            X = np.log(g2[[price_col]].values)
            y = np.log(g2[qty_col].values)
            if np.isfinite(X).all() and np.isfinite(y).all():
                lr = LinearRegression().fit(X, y)
                elasticity = float(lr.coef_[0])
                out.append((prod, elasticity, len(g2), g2[price_col].mean(), g2[qty_col].sum()))
        except Exception:
            continue
    return pd.DataFrame(out, columns=[product_col,'elasticity','n_obs','avg_price','total_qty']).sort_values('n_obs', ascending=False)

# Prophet forecast
@st.cache_data
def prophet_forecast(df_sales: pd.DataFrame, product_code, periods=90, freq='D'):
    if not PROPHET_AVAILABLE:
        raise ImportError('Prophet not available')
    g = df_sales[df_sales['StockCode'] == product_code].copy()
    series = g.groupby('InvoiceDate')['Quantity'].sum().reset_index()
    series = series.rename(columns={'InvoiceDate':'ds','Quantity':'y'})
    if len(series) < 10:
        raise ValueError('Not enough data to forecast')
    m = Prophet(daily_seasonality=True)
    m.fit(series)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return forecast, m

# AutoML comparison
@st.cache_data
def compare_models(X, y, cv=5):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostClassifier(verbose=0, random_state=42)

    scoring = {'accuracy': 'accuracy', 'f1': 'f1'}
    results = []
    for name, model in models.items():
        try:
            cvres = cross_validate(model, Xs, y, cv=skf, scoring=scoring, return_train_score=False)
            acc = float(np.mean(cvres['test_accuracy']))
            f1 = float(np.mean(cvres['test_f1']))
            results.append({'model': name, 'accuracy': acc, 'f1': f1})
        except Exception as e:
            results.append({'model': name, 'accuracy': np.nan, 'f1': np.nan, 'error': str(e)})
    return pd.DataFrame(results).sort_values('accuracy', ascending=False).reset_index(drop=True)

# Helper download link
def get_download_link(df: pd.DataFrame, filename='data.csv'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"<a href='data:file/csv;base64,{b64}' download='{filename}'>Download {filename}</a>"

# --- Streamlit App UI ---
st.title('E-Commerce Market Analysis — Full Multi-Dashboard')

with st.sidebar:
    st.header('Data input & Controls')
    uploaded = st.file_uploader('Upload transactional file (.xlsx or .csv)', type=['xlsx','xls','csv'])
    use_server = st.checkbox('Use server file if available (/mnt/data/Online Retail (1).xlsx)', value=False)
    st.markdown('---')
    page = st.selectbox('Choose dashboard', ['Home','EDA','RFM (KMeans)','Forecasting','Price Sensitivity','AutoML','About'])
    st.markdown('---')
    st.write('Optional libs:')
    st.write({'xgboost': XGBOOST_AVAILABLE, 'catboost': CATBOOST_AVAILABLE, 'prophet': PROPHET_AVAILABLE, 'vader': VADER_AVAILABLE})

# Load data
server_path = '/mnt/data/Online Retail (1).xlsx' if use_server else None
try:
    df_raw = load_data(uploaded_file=uploaded, server_path=server_path)
except Exception as e:
    st.error(f'Failed loading data: {e}')
    st.stop()

if df_raw is None:
    st.info('Please upload a dataset or enable server path (if the file exists on the server).')
    st.stop()

st.success(f'Data loaded — rows: {df_raw.shape[0]}, cols: {df_raw.shape[1]}')

# Clean
with st.spinner('Cleaning data...'):
    df = clean_retail(df_raw)

# HOME
if page == 'Home':
    st.header('Overview & Quick Stats')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Transactions', len(df))
    c2.metric('Unique Customers', int(df['CustomerID'].nunique()) if 'CustomerID' in df.columns else 'N/A')
    c3.metric('Unique Products', int(df['StockCode'].nunique()) if 'StockCode' in df.columns else 'N/A')
    c4.metric('Total Revenue', f"{df['TotalPrice'].sum():.2f}" if 'TotalPrice' in df.columns else 'N/A')
    st.markdown('---')
    st.subheader('Snapshot of cleaned data')
    st.dataframe(df.head(200))

# EDA
if page == 'EDA':
    st.header('Exploratory Data Analysis')
    if 'InvoiceDate' in df.columns and 'TotalPrice' in df.columns:
        sales_ts = df.set_index('InvoiceDate').resample('D')['TotalPrice'].sum().fillna(0)
        fig = px.line(sales_ts.reset_index(), x='InvoiceDate', y='TotalPrice', title='Daily Revenue')
        st.plotly_chart(fig, use_container_width=True)
    # Top products
    if 'Description' in df.columns:
        top_prod = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        st.subheader('Top 10 products by quantity')
        fig2 = px.bar(top_prod.reset_index(), x='Quantity', y='Description', orientation='h')
        st.plotly_chart(fig2, use_container_width=True)
    # Top countries
    if 'Country' in df.columns:
        top_country = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
        st.subheader('Top 10 countries by revenue')
        st.table(top_country)
    # Distributions
    if 'UnitPrice' in df.columns:
        st.subheader('UnitPrice distribution')
        st.plotly_chart(px.histogram(df, x='UnitPrice', nbins=50), use_container_width=True)
    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        st.subheader('Correlation matrix')
        fig_corr = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig_corr, use_container_width=True)

# RFM + KMeans
if page == 'RFM (KMeans)':
    st.header('RFM Analysis & K-Means Clustering')
    if 'CustomerID' not in df.columns:
        st.error('CustomerID required for RFM analysis')
    else:
        rfm = compute_rfm(df)
        st.subheader('RFM head')
        st.dataframe(rfm.head(200))
        n_clusters = st.slider('K (clusters)', min_value=2, max_value=8, value=4)
        rfm_clustered, centroids = rfm_kmeans(rfm, n_clusters=n_clusters)
        st.subheader('Cluster summary (mean RFM)')
        st.dataframe(rfm_clustered.groupby('cluster')[['Recency','Frequency','Monetary']].mean().round(2))
        # PCA scatter
        rfm_pca_df, pca = rfm_pca_plot(rfm_clustered)
        fig = px.scatter(rfm_pca_df, x='pca1', y='pca2', color='cluster', hover_data=['CustomerID','RFM_Segment','RFM_Score'])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(get_download_link(rfm_clustered, 'rfm_clusters.csv'), unsafe_allow_html=True)

# Forecasting
if page == 'Forecasting':
    st.header('Demand Forecasting (Prophet)')
    if not PROPHET_AVAILABLE:
        st.warning('Prophet not installed. Install prophet to enable forecasting.')
    elif 'StockCode' not in df.columns:
        st.error('StockCode required for forecasting')
    else:
        prod_list = df['StockCode'].unique().tolist()[:500]
        selected = st.selectbox('Select product to forecast', prod_list)
        periods = st.number_input('Forecast horizon (days)', min_value=7, max_value=365, value=90)
        if st.button('Run Forecast'):
            try:
                fc, model = prophet_forecast(df, selected, periods=int(periods))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fc['ds'], y=fc['yhat'], name='yhat'))
                fig.add_trace(go.Scatter(x=fc['ds'], y=fc['yhat_lower'], name='lower', line=dict(dash='dash')))
                fig.add_trace(go.Scatter(x=fc['ds'], y=fc['yhat_upper'], name='upper', line=dict(dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
                st.subheader('Forecast components')
                comp = model.plot_components(fc)
                st.pyplot(comp)
                st.dataframe(fc[['ds','yhat','yhat_lower','yhat_upper']].tail(200))
            except Exception as e:
                st.error(f'Forecast failed: {e}')

# Price Sensitivity
if page == 'Price Sensitivity':
    st.header('Price Sensitivity & Elasticity')
    if not all(c in df.columns for c in ['StockCode','UnitPrice','Quantity']):
        st.error('Required columns missing: StockCode, UnitPrice, Quantity')
    else:
        with st.spinner('Computing elasticity...'):
            elast = price_elasticity(df)
        st.subheader('Elasticity samples')
        st.dataframe(elast.head(200))
        st.markdown(get_download_link(elast, 'price_elasticity.csv'), unsafe_allow_html=True)
        # Scatter average price vs total quantity
        agg = df.groupby('StockCode').agg(avg_price=('UnitPrice','mean'), total_qty=('Quantity','sum')).reset_index()
        fig = px.scatter(agg, x='avg_price', y='total_qty', hover_data=['StockCode'], log_x=True, log_y=True, title='Avg Price vs Total Qty (log-log)')
        st.plotly_chart(fig, use_container_width=True)

# AutoML
if page == 'AutoML':
    st.header('AutoML: Compare models for high-value customer prediction')
    if 'CustomerID' not in df.columns:
        st.error('CustomerID required for AutoML')
    else:
        rfm = compute_rfm(df)
        data = df.groupby('CustomerID').agg(total_qty=('Quantity','sum'), total_revenue=('TotalPrice','sum')).reset_index()
        data = rfm.merge(data, on='CustomerID', how='left').fillna(0)
        # label: high-value threshold adjustable
        threshold = st.slider('RFM score threshold for high-value label', min_value=int(data['RFM_Score'].min()), max_value=int(data['RFM_Score'].max()), value=12)
        data['label'] = (data['RFM_Score'] >= threshold).astype(int)
        st.write('Label distribution')
        st.bar_chart(data['label'].value_counts())
        X = data[['total_qty','total_revenue']]
        y = data['label']
        if st.button('Run model comparison (5-fold CV)'):
            with st.spinner('Training models...'):
                res = compare_models(X, y, cv=5)
                st.success('Comparison done')
                st.dataframe(res)
                st.markdown(get_download_link(res, 'model_comparison.csv'), unsafe_allow_html=True)
                best = res.dropna(subset=['accuracy']).iloc[0]
                st.write(f"Best model: {best['model']} (accuracy={best['accuracy']:.3f}, f1={best['f1']:.3f})")
                # train best model on full data
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                model_name = best['model']
                if model_name == 'LogisticRegression':
                    final = LogisticRegression(max_iter=1000)
                elif model_name == 'RandomForest':
                    final = RandomForestClassifier(n_estimators=200, random_state=42)
                elif model_name == 'GradientBoosting':
                    final = GradientBoostingClassifier(n_estimators=200, random_state=42)
                elif model_name == 'KNN':
                    final = KNeighborsClassifier(n_neighbors=5)
                elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                    final = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                elif model_name == 'CatBoost' and CATBOOST_AVAILABLE:
                    final = CatBoostClassifier(verbose=0, random_state=42)
                else:
                    st.error('Best model not available to retrain locally')
                    final = None
                if final is not None:
                    final.fit(Xs, y.values)
                    preds = final.predict(Xs)
                    st.subheader('Classification report (trained on full data)')
                    st.text(classification_report(y, preds))
                    cm = confusion_matrix(y, preds)
                    fig = px.imshow(cm, text_auto=True, labels=dict(x='pred', y='true'))
                    st.plotly_chart(fig)

# About
if page == 'About':
    st.header('About this App')
    st.markdown('- Multi-dashboard Streamlit app for E-commerce market analysis')
    st.markdown('- Modules: EDA, RFM (KMeans), Forecasting (Prophet), Price Elasticity, AutoML')
    st.markdown('- Optional libs used if installed: xgboost, catboost, prophet, vaderSentiment')
    st.markdown('- Extend with SHAP explainability, hyperparameter search, model export as needed')

st.caption('')
