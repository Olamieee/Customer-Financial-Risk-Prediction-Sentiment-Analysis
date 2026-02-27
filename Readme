# Customer Financial Risk Prediction & Sentiment Analysis

> Unsupervised machine learning solution for customer segmentation in African financial markets

---

## Overview

This project analyzes 5,200 customers across Nigeria, Ghana, Kenya, South Africa, Uganda, and Tanzania to identify distinct behavioral segments using unsupervised clustering and natural language processing.

**Key Features:**
- 4 clustering algorithms compared (K-Means winner)
- Real-time sentiment analysis with VADER
- Topic modeling with LDA
- Interactive Streamlit dashboard
- Production REST API with FastAPI
- 37 engineered features (27 structured + 10 text PCA)

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install pandas numpy scikit-learn spacy vaderSentiment streamlit fastapi uvicorn plotly joblib matplotlib seaborn

# Download Spacy model
python -m spacy download en_core_web_lg

```

### Run Dashboard

```bash
streamlit run streamlit_dashboard.py
```

Access at: http://localhost:8501

### Run API

```bash
python fastapi_app.py
```

Access at: http://localhost:8000  
Docs at: http://localhost:8000/docs

---

## Dataset

**Size:** 5,200 customers  
**Countries:** Nigeria, Ghana, Kenya, South Africa, Uganda, Tanzania  
**Features:** 14 original + NLP-derived features

**Key Features:**
- Monthly Expenditure (₦)
- Credit Score (300-850)
- Income Level (Low to High)
- Transaction Channel (USSD, Mobile App, Web, Branch)
- Customer Feedback (text)
- Complaint Type
- Loan Status
- Saving Behavior

---

## Technical Approach

### 1. Data Preprocessing
- Missing value handling (Complaint_Type)
- Spacy text preprocessing with lemmatization
- MinMaxScaler for feature normalization
- Hybrid encoding (ordinal + label + one-hot)

### 2. NLP Processing
- **Sentiment Analysis:** VADER on raw feedback
- **Topic Modeling:** LDA with 4 topics (CountVectorizer)
- **Text Embeddings:** TF-IDF (20 features) with PCA (10 components)

### 3. Clustering
Compared 4 algorithms:

| Algorithm | Silhouette Score | Status |
|-----------|-----------------|--------|
| K-Means | 0.111 | WINNER |
| Hierarchical | 0.033 | Good |
| GMM | 0.099 | Good |
| DBSCAN | N/A | Outlier detection |

**Key Finding:** MinMaxScaler outperformed StandardScaler by 54%

---

## Results

### Customer Segments

| Cluster | Name | Size | Sentiment | Strategy |
|---------|------|------|-----------|----------|
| 0 | Mobile App Users | 24.8% | 0.07 | Premium features |
| 1 | Traditional USSD Users | 30.3% | 0.07 | Digital migration |
| 2 | Engaged Mobile Users | 24.7% | 0.09 | Brand ambassadors |
| 3 | Standard Mobile Users | 20.2% | 0.08 | Savings products |

### Key Insights
- 70% use mobile channels (digital adoption)
- 30% use USSD (migration opportunity)
- All segments Lower-Middle income
- Similar spending patterns (₦150-152k monthly)

---

## API Usage

### Single Customer Prediction

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "Customer_Feedback": "Great mobile app experience",
    "Monthly_Expenditure": 155000,
    "Credit_Score": 680,
    "Loan_Amount": 50000,
    "Income_Level": "Lower-Middle",
    "Saving_Behavior": "Average",
    "Loan_Status": "Active Loan",
    "Time_With_Bank": "2-5 Years",
    "Transaction_Channel": "Mobile App",
    "Location": "Nigeria",
    "Spending_Category": "Essential",
    "Complaint_Type": "No_Complaint"
}

response = requests.post(url, json=data)
print(response.json())
```

**Response:**
```json
{
  "cluster": {
    "id": 2,
    "name": "Engaged Mobile Users",
    "description": "Highly satisfied mobile customers",
    "segment_size": "24.7%"
  },
  "sentiment": {
    "score": 0.836,
    "label": "Positive"
  },
  "topic": {
    "id": 3,
    "name": "General_Feedback",
    "confidence": 0.645
  },
  "recommendations": [
    "Leverage as brand ambassadors",
    "Offer referral bonuses"
  ]
}
```

### Batch Prediction

```python
url = "http://localhost:8000/predict/batch"
data = {
    "customers": [customer1, customer2, customer3]
}

response = requests.post(url, json=data)
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/predict` | POST | Single customer prediction |
| `/predict/batch` | POST | Batch prediction |
| `/clusters` | GET | All cluster information |
| `/cluster/{id}` | GET | Specific cluster details |
| `/health` | GET | Health check |

---

## Project Structure

```
├── Customer_Financial_Risk.ipynb    # Main analysis
├── streamlit_dashboard.py             # Interactive dashboard
├── fastapi_app.py                     # REST API
│
├── README.md                          # This file
├── requirements.txt               # Libraries
│
├── finance_customer_behavior_dataset.csv  # Original data
├── finance_clustered2_complete.csv        # Results
├── cluster_summary2.csv                   # Summary stats
│
├── scaler.pkl                         # MinMaxScaler
├── kmeans_model.pkl                   # K-Means model
├── lda_model.pkl                      # LDA model
├── vectorizer.pkl                     # CountVectorizer
│
└── charts2/                           # Visualizations (7 charts)
```

---

## Business Recommendations

### Mobile App Users (24.8%)
- Offer premium mobile features
- App-exclusive rewards
- Cross-sell investment products

### Traditional USSD Users (30.3%)
- Digital migration incentives
- USSD-based savings products
- Simplified app onboarding

### Engaged Mobile Users (24.7%)
- Referral reward programs
- Brand ambassador opportunities
- Premium credit products

### Standard Mobile Users (20.2%)
- Savings and investment products
- Spending analytics features
- Insurance cross-selling

---

## Technologies

**Core:**
- Python 3.8+
- Scikit-learn
- Pandas & NumPy
- Spacy
- VADER

**Visualization:**
- Matplotlib
- Seaborn
- Plotly

**Deployment:**
- Streamlit
- FastAPI
- Uvicorn
- Joblib

---

## Model Performance

**Final Model:** K-Means with 4 clusters  
**Silhouette Score:** 0.111  
**Features:** 37 (27 structured + 10 text PCA)  
**Scaler:** MinMaxScaler

**Scaler Comparison:**
- MinMaxScaler: 0.111 silhouette
- StandardScaler: 0.077 silhouette
- Improvement: 54%

---

## Files Included

**Analysis:**
- Customer_Financial_Risk_2.ipynb
- finance_clustered2_complete.csv
- cluster_summary2.csv

**Deployments:**
- streamlit_dashboard.py
- fastapi_app.py

**Documentation:**
- README.md
- requirements.txt

**Visualizations (7 charts):**
- elbow_method.png
- silhouette_scores_by_algorithm.png
- davies_bouldin_scores.png
- sentiment_analysis.png
- cluster_distribution.png
- cluster_comparison.png
- pca_clusters.png

---

## Acknowledgments

Thanks to Poshem Technology Institute for this comprehensive data science challenge.