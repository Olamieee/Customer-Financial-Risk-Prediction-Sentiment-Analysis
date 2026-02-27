import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Customer Financial Risk Dashboard",
    page_icon="💰",
    layout="wide"
)

# Title
st.title("💰 Customer Financial Risk Prediction Dashboard")
st.markdown("**African Financial Markets - Customer Segmentation Analysis**")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('finance_clustered2_complete.csv')
        return df
    except:
        st.error("⚠️ Please ensure finance_clustered2_complete.csv is in the same directory")
        return None

df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.header("🔍 Filters")
    
    clusters = sorted(df['cluster_final'].unique())
    selected_cluster = st.sidebar.selectbox(
        "Select Cluster",
        ["All"] + [f"Cluster {c}" for c in clusters]
    )
    
    income_levels = ["All"] + list(df['Income_Level'].unique())
    selected_income = st.sidebar.selectbox("Income Level", income_levels)
    
    channels = ["All"] + list(df['Transaction_Channel'].unique())
    selected_channel = st.sidebar.selectbox("Transaction Channel", channels)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_cluster != "All":
        cluster_num = int(selected_cluster.split()[-1])
        filtered_df = filtered_df[filtered_df['cluster_final'] == cluster_num]
    if selected_income != "All":
        filtered_df = filtered_df[filtered_df['Income_Level'] == selected_income]
    if selected_channel != "All":
        filtered_df = filtered_df[filtered_df['Transaction_Channel'] == selected_channel]
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Customers", f"{len(filtered_df):,}")
    with col2:
        st.metric("Avg Expenditure", f"₦{filtered_df['Monthly_Expenditure'].mean():,.0f}")
    with col3:
        st.metric("Avg Credit Score", f"{filtered_df['Credit_Score'].mean():.0f}")
    with col4:
        st.metric("Avg Sentiment", f"{filtered_df['sentiment_score'].mean():.2f}")
    with col5:
        st.metric("Active Loans", f"{(filtered_df['Loan_Status']=='Active Loan').sum():,}")
    
    st.markdown("---")
    
    # Cluster Distribution
    st.header("🎯 Cluster Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        cluster_counts = df['cluster_final'].value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Customers'},
            title='Customer Distribution by Cluster'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cluster_pct = (cluster_counts / len(df) * 100).round(1)
        fig = px.pie(
            values=cluster_pct.values,
            names=[f'Cluster {i}' for i in cluster_pct.index],
            title='Cluster Percentage Distribution',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Financial Metrics
    st.header("💵 Financial Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        avg_exp = df.groupby('cluster_final')['Monthly_Expenditure'].mean()
        fig = px.bar(
            x=avg_exp.index,
            y=avg_exp.values,
            labels={'x': 'Cluster', 'y': 'Average Expenditure (₦)'},
            title='Average Monthly Expenditure by Cluster'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        avg_credit = df.groupby('cluster_final')['Credit_Score'].mean()
        fig = px.bar(
            x=avg_credit.index,
            y=avg_credit.values,
            labels={'x': 'Cluster', 'y': 'Average Credit Score'},
            title='Average Credit Score by Cluster'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment Analysis
    st.header("💭 Sentiment Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_counts = filtered_df['sentiment_label'].value_counts()
        colors = {'Positive': '#2ca02c', 'Neutral': '#7f7f7f', 'Negative': '#d62728'}
        fig = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            labels={'x': 'Sentiment', 'y': 'Count'},
            title='Sentiment Distribution',
            color=sentiment_counts.index,
            color_discrete_map=colors
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        sent_cluster = pd.crosstab(df['cluster_final'], df['sentiment_label'], normalize='index') * 100
        fig = px.bar(
            sent_cluster,
            barmode='stack',
            labels={'value': 'Percentage (%)', 'cluster_final': 'Cluster'},
            title='Sentiment by Cluster (%)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Channel Analysis
    st.header("📱 Transaction Channels")
    col1, col2 = st.columns(2)
    
    with col1:
        channel_counts = filtered_df['Transaction_Channel'].value_counts()
        fig = px.bar(
            x=channel_counts.index,
            y=channel_counts.values,
            labels={'x': 'Channel', 'y': 'Count'},
            title='Channel Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        channel_cluster = pd.crosstab(df['cluster_final'], df['Transaction_Channel'])
        fig = px.bar(
            channel_cluster,
            barmode='group',
            labels={'value': 'Count', 'cluster_final': 'Cluster'},
            title='Channels by Cluster'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster Profiles
    st.header("👥 Cluster Profiles")
    for cluster_id in sorted(df['cluster_final'].unique()):
        with st.expander(f"📋 Cluster {cluster_id} Details"):
            cluster_data = df[df['cluster_final'] == cluster_id]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Size", f"{len(cluster_data):,} ({len(cluster_data)/len(df)*100:.1f}%)")
                st.metric("Avg Expenditure", f"₦{cluster_data['Monthly_Expenditure'].mean():,.0f}")
                st.metric("Avg Credit", f"{cluster_data['Credit_Score'].mean():.0f}")
            
            with col2:
                st.write("**Demographics:**")
                st.write(f"• Income: {cluster_data['Income_Level'].mode()[0]}")
                st.write(f"• Saving: {cluster_data['Saving_Behavior'].mode()[0]}")
                st.write(f"• Loan Status: {cluster_data['Loan_Status'].mode()[0]}")
            
            with col3:
                st.write("**Behavior:**")
                st.write(f"• Channel: {cluster_data['Transaction_Channel'].mode()[0]}")
                st.write(f"• Spending: {cluster_data['Spending_Category'].mode()[0]}")
                st.write(f"• Sentiment: {cluster_data['sentiment_score'].mean():.2f}")
    
    # Data Table
    st.header("📄 Customer Data")
    display_cols = ['Customer_ID', 'Monthly_Expenditure', 'Income_Level', 
                   'Credit_Score', 'Loan_Status', 'Transaction_Channel',
                   'sentiment_label', 'topic_name', 'cluster_final', 'cluster_name']
    
    st.dataframe(filtered_df[display_cols].head(100), use_container_width=True)
    
    # Download
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name="filtered_customers.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.info("""
    **Model Information**  
    Algorithm: K-Means | Scaler: MinMaxScaler | Silhouette: 0.111 | Clusters: 4 | Features: 37
    """)