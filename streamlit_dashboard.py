import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="Customer Financial Risk Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stMetric {background-color: white; padding: 10px; border-radius: 5px;}
    h1 {color: #1f77b4;}
    h2 {color: #2ca02c;}
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("💰 Customer Financial Risk Prediction Dashboard")
st.markdown("**Unsupervised ML + NLP Analysis | MinMaxScaler + K-Means Clustering**")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('finance_clustered_complete.csv')
        return df
    except:
        st.error("Please ensure finance_clustered_complete.csv is in the same directory")
        return None

df = load_data()

if df is not None:
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Cluster filter
    clusters = sorted(df['cluster_final'].unique())
    selected_cluster = st.sidebar.selectbox(
        "Select Cluster",
        ["All"] + [f"Cluster {c}" for c in clusters]
    )
    
    # Income filter
    income_levels = ["All"] + list(df['Income_Level'].unique())
    selected_income = st.sidebar.selectbox("Income Level", income_levels)
    
    # Channel filter
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
    st.header("📊 Key Metrics")
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
    
    # Cluster Overview
    st.header("🎯 Cluster Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster sizes
        cluster_counts = df['cluster_final'].value_counts().sort_index()
        fig_cluster = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Number of Customers'},
            title='Customers per Cluster',
            color=cluster_counts.values,
            color_continuous_scale='viridis'
        )
        fig_cluster.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_cluster, use_container_width=True)
    
    with col2:
        # Cluster percentages
        cluster_pct = (cluster_counts / len(df) * 100).round(1)
        fig_pie = px.pie(
            values=cluster_pct.values,
            names=[f'Cluster {i}' for i in cluster_pct.index],
            title='Cluster Distribution (%)',
            hole=0.4
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Financial Metrics by Cluster
    st.header("💵 Financial Metrics by Cluster")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average expenditure by cluster
        avg_exp = df.groupby('cluster_final')['Monthly_Expenditure'].mean()
        fig_exp = px.bar(
            x=avg_exp.index,
            y=avg_exp.values,
            labels={'x': 'Cluster', 'y': 'Avg Monthly Expenditure (₦)'},
            title='Average Monthly Expenditure by Cluster',
            color=avg_exp.values,
            color_continuous_scale='Blues'
        )
        fig_exp.update_layout(showlegend=False)
        st.plotly_chart(fig_exp, use_container_width=True)
    
    with col2:
        # Average credit score by cluster
        avg_credit = df.groupby('cluster_final')['Credit_Score'].mean()
        fig_credit = px.bar(
            x=avg_credit.index,
            y=avg_credit.values,
            labels={'x': 'Cluster', 'y': 'Avg Credit Score'},
            title='Average Credit Score by Cluster',
            color=avg_credit.values,
            color_continuous_scale='Greens'
        )
        fig_credit.update_layout(showlegend=False)
        st.plotly_chart(fig_credit, use_container_width=True)
    
    st.markdown("---")
    
    # Sentiment Analysis
    st.header("💭 Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        sentiment_counts = filtered_df['sentiment_label'].value_counts()
        colors = {'Positive': '#2ca02c', 'Neutral': '#7f7f7f', 'Negative': '#d62728'}
        fig_sent = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            labels={'x': 'Sentiment', 'y': 'Count'},
            title='Sentiment Distribution',
            color=sentiment_counts.index,
            color_discrete_map=colors
        )
        st.plotly_chart(fig_sent, use_container_width=True)
    
    with col2:
        # Sentiment by cluster
        sent_cluster = pd.crosstab(df['cluster_final'], df['sentiment_label'], normalize='index') * 100
        fig_sent_cluster = px.bar(
            sent_cluster,
            barmode='stack',
            labels={'value': 'Percentage (%)', 'cluster_final': 'Cluster'},
            title='Sentiment Distribution by Cluster (%)',
            color_discrete_map=colors
        )
        st.plotly_chart(fig_sent_cluster, use_container_width=True)
    
    st.markdown("---")
    
    # Topic Analysis
    st.header("🏷️ Topic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Topic distribution
        topic_counts = filtered_df['topic_name'].value_counts()
        fig_topic = px.pie(
            values=topic_counts.values,
            names=topic_counts.index,
            title='Topic Distribution',
            hole=0.3
        )
        st.plotly_chart(fig_topic, use_container_width=True)
    
    with col2:
        # Topics by cluster
        topic_cluster = pd.crosstab(df['cluster_final'], df['topic_name'])
        fig_topic_cluster = px.bar(
            topic_cluster,
            barmode='group',
            labels={'value': 'Count', 'cluster_final': 'Cluster'},
            title='Topics by Cluster'
        )
        st.plotly_chart(fig_topic_cluster, use_container_width=True)
    
    st.markdown("---")
    
    # Cluster Profiles
    st.header("👥 Detailed Cluster Profiles")
    
    for cluster_id in sorted(df['cluster_final'].unique()):
        with st.expander(f"📋 Cluster {cluster_id} Profile"):
            cluster_data = df[df['cluster_final'] == cluster_id]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Size", f"{len(cluster_data):,} ({len(cluster_data)/len(df)*100:.1f}%)")
                st.metric("Avg Expenditure", f"₦{cluster_data['Monthly_Expenditure'].mean():,.0f}")
                st.metric("Avg Credit Score", f"{cluster_data['Credit_Score'].mean():.0f}")
            
            with col2:
                st.write("**Demographics:**")
                st.write(f"• Income: {cluster_data['Income_Level'].mode()[0]}")
                st.write(f"• Saving: {cluster_data['Saving_Behavior'].mode()[0]}")
                st.write(f"• Loan Status: {cluster_data['Loan_Status'].mode()[0]}")
            
            with col3:
                st.write("**Behavior:**")
                st.write(f"• Channel: {cluster_data['Transaction_Channel'].mode()[0]}")
                st.write(f"• Top Spending: {cluster_data['Spending_Category'].mode()[0]}")
                st.write(f"• Sentiment: {cluster_data['sentiment_score'].mean():.2f}")
    
    st.markdown("---")
    
    # Transaction Channels
    st.header("📱 Transaction Channel Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Channel distribution
        channel_counts = filtered_df['Transaction_Channel'].value_counts()
        fig_channel = px.bar(
            x=channel_counts.index,
            y=channel_counts.values,
            labels={'x': 'Channel', 'y': 'Count'},
            title='Transaction Channel Distribution',
            color=channel_counts.values,
            color_continuous_scale='Teal'
        )
        st.plotly_chart(fig_channel, use_container_width=True)
    
    with col2:
        # Channel by cluster
        channel_cluster = pd.crosstab(df['cluster_final'], df['Transaction_Channel'])
        fig_channel_cluster = px.bar(
            channel_cluster,
            barmode='group',
            labels={'value': 'Count', 'cluster_final': 'Cluster'},
            title='Transaction Channels by Cluster'
        )
        st.plotly_chart(fig_channel_cluster, use_container_width=True)
    
    st.markdown("---")
    
    # Data Table
    st.header("📄 Customer Data")
    
    # Select columns to display
    display_cols = ['Customer_ID', 'Monthly_Expenditure', 'Income_Level', 
                   'Credit_Score', 'Loan_Status', 'Transaction_Channel',
                   'sentiment_label', 'topic_name', 'cluster_final']
    
    st.dataframe(
        filtered_df[display_cols].head(100),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name="filtered_customers.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Footer
    st.markdown("### 📊 Model Information")
    st.info("""
    **Algorithm:** K-Means Clustering  
    **Scaler:** MinMaxScaler  
    **Silhouette Score:** 0.111  
    **Optimal Clusters:** 4  
    **Features:** 27 (Hybrid Encoding)  
    **Dataset:** 5,200 customers
    """)
    
    st.success("✅ Dashboard loaded successfully! Use filters in the sidebar to explore different segments.")

else:
    st.error("Failed to load data. Please check that finance_clustered_complete.csv is available.")