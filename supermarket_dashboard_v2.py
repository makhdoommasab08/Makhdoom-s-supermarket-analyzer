import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Makhdoom's Supermarket Analyzer", layout="wide")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/supermarket_customers.csv")
    df = df.head(1000)  # limit to 1000 entries
    return df

df = load_data()
st.title("🛒 Makhdoom's Supermarket Analyzer")
st.write("Explore, filter, and analyze your supermarket customers interactively!")

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.header("🔍 Data Filters")

min_visits, max_visits = st.sidebar.slider("Average Visits", 
                                           int(df['Average_Visits'].min()), 
                                           int(df['Average_Visits'].max()),
                                           (int(df['Average_Visits'].min()), int(df['Average_Visits'].max())))

min_amount, max_amount = st.sidebar.slider("Shopping Amount", 
                                           int(df['Shopping_Amount'].min()), 
                                           int(df['Shopping_Amount'].max()),
                                           (int(df['Shopping_Amount'].min()), int(df['Shopping_Amount'].max())))

customer_type = st.sidebar.multiselect("Customer Type", df['Customer_Type'].unique(), df['Customer_Type'].unique())

df_filtered = df[
    (df['Average_Visits'].between(min_visits, max_visits)) &
    (df['Shopping_Amount'].between(min_amount, max_amount)) &
    (df['Customer_Type'].isin(customer_type))
]

st.write(f"### Showing {len(df_filtered)} filtered entries")

# ----------------------------
# DATA PREVIEW
# ----------------------------
st.dataframe(df_filtered)

# ----------------------------
# K-MEANS CLUSTERING
# ----------------------------
st.subheader("📊 K-Means Customer Segmentation")

X = df_filtered[['Average_Visits', 'Items_Shopped', 'Shopping_Amount']]
kmeans = KMeans(n_clusters=3, random_state=42)
df_filtered['Cluster'] = kmeans.fit_predict(X)

fig_kmeans = px.scatter(df_filtered, 
                        x='Items_Shopped', 
                        y='Shopping_Amount', 
                        color='Cluster', 
                        size='Average_Visits',
                        title="Customer Clusters (K-Means)")
st.plotly_chart(fig_kmeans, use_container_width=True)

# ----------------------------
# DECISION TREE CLASSIFICATION
# ----------------------------
st.subheader("🌳 Decision Tree Classifier")

X = df_filtered[['Average_Visits', 'Items_Shopped', 'Shopping_Amount']]
y = df_filtered['Customer_Type'].apply(lambda x: 1 if x == 'Returning' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Model trained successfully! ✅ Accuracy: {acc*100:.2f}%")

# Save model
joblib.dump(model, "models/makhdoom_decision_tree_v3.pkl")

# ----------------------------
# BUSINESS INSIGHTS
# ----------------------------
st.subheader("📈 Business Insights")

col1, col2 = st.columns(2)

with col1:
    fig_amount = px.histogram(df_filtered, x="Shopping_Amount", nbins=20, title="Distribution of Shopping Amounts")
    st.plotly_chart(fig_amount, use_container_width=True)

with col2:
    fig_visits = px.pie(df_filtered, names="Customer_Type", title="Returning vs Non-Repeating Customers")
    st.plotly_chart(fig_visits, use_container_width=True)

st.caption("Developed by Makhdoom Masab")
