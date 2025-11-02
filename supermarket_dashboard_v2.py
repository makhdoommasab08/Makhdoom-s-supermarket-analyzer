
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

st.set_page_config(page_title="Makhdoom's Supermarket Analyzer", layout="wide")
st.title("🧾 Makhdoom's Supermarket Analyzer")
st.markdown("Analyze your supermarket customers, segment them, train a classifier, and forecast revenue — interactive Streamlit dashboard.")

# Load Data: uploader or default data file included in repo
uploaded_file = st.file_uploader("Upload your supermarket CSV (optional)", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        st.stop()
else:
    default_path = os.path.join("data", "supermarket_customers.csv")
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        st.warning("No dataset found. Please upload a CSV or place your CSV in the data/ folder.")
        st.stop()

st.subheader("📊 Dataset preview")
st.dataframe(df.head(10), use_container_width=True)
st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# Basic checks for required columns
required_cols = ['Age','Monthly_Income','Average_Visits','Items_Shopped','Shopping_Amount','Customer_Type']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.warning(f"The following expected columns are missing from your CSV: {missing}. You can still proceed but some features may not work.")

# Metrics
col1, col2, col3 = st.columns(3)
if 'Monthly_Income' in df.columns:
    col1.metric("Avg Monthly Income", f"Rs {df['Monthly_Income'].mean():,.0f}")
else:
    col1.metric("Avg Monthly Income", "N/A")
if 'Shopping_Amount' in df.columns:
    col2.metric("Avg Shopping Amount", f"Rs {df['Shopping_Amount'].mean():,.0f}")
else:
    col2.metric("Avg Shopping Amount", "N/A")
if 'Customer_Type' in df.columns:
    col3.metric("Returning %", f"{(df['Customer_Type'].eq('Returning').mean() * 100):.1f}%")
else:
    col3.metric("Returning %", "N/A")

# Pie chart
if 'Customer_Type' in df.columns:
    fig = px.pie(df, names='Customer_Type', title='Customer Type Distribution', hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

# KMeans clustering
st.subheader("🧠 Customer Segmentation (KMeans)")
if all(c in df.columns for c in ['Monthly_Income','Shopping_Amount']):
    X = df[['Monthly_Income','Shopping_Amount']].dropna()
    num_clusters = st.slider("Number of clusters", 2, 8, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    fig2 = px.scatter(df, x='Monthly_Income', y='Shopping_Amount', color='Cluster',
                      hover_data=['Age','Average_Visits'], title="Customer Clusters")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Need columns 'Monthly_Income' and 'Shopping_Amount' for clustering.")

# Decision Tree Classification
st.subheader("🌳 Decision Tree — Predict Returning Customers")
features = ['Age','Monthly_Income','Average_Visits','Items_Shopped','Shopping_Amount']
if all(f in df.columns for f in features) and 'Customer_Type' in df.columns:
    X = df[features].fillna(0)
    y = df['Customer_Type'].map({'Returning':1,'Non-Repeating':0})
    if y.isnull().any():
        st.warning("Customer_Type contains values other than 'Returning' and 'Non-Repeating'. Some rows will be ignored.")
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", f"{acc*100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    fig3, ax = plt.subplots(figsize=(12,6))
    plot_tree(model, feature_names=features, class_names=['Non-Repeating','Returning'], filled=True)
    st.pyplot(fig3)
    joblib.dump(model, "app/makhdoom_decision_tree.pkl")
else:
    st.info("Not enough columns to train Decision Tree. Required: " + ", ".join(features) + ", Customer_Type")

# Manual prediction UI
st.subheader("🔍 Predict customer type (manual input)")
age = st.slider("Age", 18, 80, 30)
income = st.number_input("Monthly Income (Rs)", min_value=1000, max_value=1000000, value=50000)
visits = st.slider("Average Visits per Month", 0, 60, 5)
items = st.slider("Items Shopped (avg)", 0, 500, 20)
amount = st.number_input("Shopping Amount (Rs)", min_value=0, max_value=1000000, value=8000)

if st.button("Predict (using trained Decision Tree)"):
    if os.path.exists("app/makhdoom_decision_tree.pkl"):
        clf = joblib.load("app/makhdoom_decision_tree.pkl")
        pred = clf.predict([[age, income, visits, items, amount]])
        st.success("Prediction: Returning Customer ✅" if pred[0]==1 else "Prediction: Non-Repeating Customer ❌")
    else:
        st.error("Trained model not found. Train model first by ensuring dataset has required columns.")

# Revenue forecast
st.subheader("📈 Simple 12-Month Revenue Projection")
if 'Shopping_Amount' in df.columns and 'Average_Visits' in df.columns:
    months = np.arange(1,13)
    baseline = df['Shopping_Amount'].sum() * (df['Average_Visits'].mean()/max(df['Average_Visits'].max(),1))
    trend = baseline * (1 + 0.03 * np.sin(2 * np.pi * months / 12))
    proj_df = pd.DataFrame({'Month': months, 'Projected_Revenue': trend})
    fig4 = px.line(proj_df, x='Month', y='Projected_Revenue', markers=True, title='Projected Revenue (12 months)')
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Need 'Shopping_Amount' and 'Average_Visits' for revenue projection.")
