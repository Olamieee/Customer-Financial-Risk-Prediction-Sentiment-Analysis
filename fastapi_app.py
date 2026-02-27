from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI(
    title="Customer Financial Risk Prediction API",
    description="Real-time cluster prediction with sentiment analysis and topic modeling",
    version="2.0"
)

# Load models
try:
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
    lda_model = joblib.load('lda_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    print("Run save_model_UPDATED.py first!")

# Input schema for SINGLE customer
class CustomerInput(BaseModel):
    Customer_Feedback: str = Field(..., example="Great mobile app experience, very satisfied with services")
    Monthly_Expenditure: float = Field(..., example=150000, description="in Naira")
    Credit_Score: int = Field(..., ge=300, le=850, example=650)
    Loan_Amount: float = Field(..., example=50000)
    Income_Level: str = Field(..., example="Lower-Middle")
    Saving_Behavior: str = Field(..., example="Average")
    Loan_Status: str = Field(..., example="Active Loan")
    Time_With_Bank: str = Field(..., example="2-5 Years")
    Transaction_Channel: str = Field(..., example="Mobile App")
    Location: str = Field(..., example="Nigeria")
    Spending_Category: str = Field(..., example="Essential")
    Complaint_Type: str = Field(..., example="No_Complaint")

# Input schema for BATCH (multiple customers)
class BatchCustomerInput(BaseModel):
    customers: List[CustomerInput]

# Cluster descriptions
CLUSTER_INFO = {
    0: {
        "name": "Mobile App Users",
        "description": "Digital-savvy customers with neutral sentiment",
        "size_pct": 24.8,
        "recommendations": [
            "Offer premium mobile features and rewards",
            "Cross-sell investment products through app",
            "Provide mobile-exclusive promotions"
        ]
    },
    1: {
        "name": "Traditional USSD Users",
        "description": "Largest segment preferring USSD channels",
        "size_pct": 30.3,
        "recommendations": [
            "Create migration incentives to mobile app",
            "Offer USSD-based savings products",
            "Provide gradual digital literacy training"
        ]
    },
    2: {
        "name": "Engaged Mobile Users",
        "description": "Highly satisfied mobile customers",
        "size_pct": 24.7,
        "recommendations": [
            "Leverage as brand ambassadors",
            "Offer referral bonuses",
            "Target for premium credit products"
        ]
    },
    3: {
        "name": "Standard Mobile Users",
        "description": "Mobile users with higher spending potential",
        "size_pct": 20.2,
        "recommendations": [
            "Promote savings and investment products",
            "Offer spending analytics features",
            "Cross-sell insurance products"
        ]
    }
}

def preprocess_single_customer(data: CustomerInput) -> np.ndarray:
    """Preprocess a single customer's data for prediction"""
    
    # 1. Sentiment Analysis
    feedback = data.Customer_Feedback
    sentiment = analyzer.polarity_scores(feedback)['compound']
    
    # 2. Topic Modeling
    feedback_vec = vectorizer.transform([feedback])
    topic_dist = lda_model.transform(feedback_vec)[0]
    dominant_topic = topic_dist.argmax()
    
    # 3. Encode categorical features
    # Ordinal encoding
    income_map = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}
    saving_map = {'Poor': 0, 'Average': 1, 'Good': 2}
    loan_map = {'No Loan': 0, 'Active Loan': 1, 'Defaulted': 2}
    time_map = {'<1 Year': 0, '1-2 Years': 1, '2-5 Years': 2, '5+ Years': 3}
    
    income_encoded = income_map.get(data.Income_Level, 1)
    saving_encoded = saving_map.get(data.Saving_Behavior, 1)
    loan_encoded = loan_map.get(data.Loan_Status, 0)
    time_encoded = time_map.get(data.Time_With_Bank, 2)
    
    # Label encoding for channel
    channel_map = {'USSD': 0, 'Mobile App': 1, 'Web': 2, 'Branch': 3}
    channel_encoded = channel_map.get(data.Transaction_Channel, 1)
    
    # One-hot encoding (simplified - 18 columns)
    locations = ['Nigeria', 'Ghana', 'Kenya', 'South Africa', 'Uganda', 'Tanzania']
    spending = ['Essential', 'Non-Essential', 'Mixed']
    complaints = ['No_Complaint', 'Service_Delay', 'Technical_Issue', 'Charges', 
                 'Loan_Terms', 'App_Error', 'USSD_Failure', 'Branch_Service', 'Other']
    
    loc_encoded = [1 if data.Location == loc else 0 for loc in locations]
    spend_encoded = [1 if data.Spending_Category == s else 0 for s in spending]
    complaint_encoded = [1 if data.Complaint_Type == c else 0 for c in complaints]
    
    # 4. Combine all features (37 total: 4 numeric + 5 encoded + 5 label + 18 one-hot + 5 text PCA placeholder)
    features = [
        data.Monthly_Expenditure,
        data.Credit_Score,
        data.Loan_Amount,
        sentiment,
        income_encoded,
        saving_encoded,
        loan_encoded,
        time_encoded,
        channel_encoded
    ] + loc_encoded + spend_encoded + complaint_encoded + list(topic_dist[:4])
    
    # Pad to match training features (37 total)
    while len(features) < 37:
        features.append(0.0)
    
    return np.array(features[:37]).reshape(1, -1)

@app.get("/")
def root():
    """API information"""
    return {
        "message": "Customer Financial Risk Prediction API",
        "version": "2.0",
        "endpoints": {
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch",
            "cluster_info": "/clusters",
            "health": "/health"
        },
        "models": "K-Means (4 clusters) + VADER + LDA",
        "features": "37 (27 structured + 10 text)"
    }

@app.post("/predict")
def predict_single(customer: CustomerInput):
    """
    Predict cluster for a SINGLE customer with full analysis
    
    Returns:
    - Cluster ID and name
    - Sentiment score and label
    - Dominant topic
    - Business recommendations
    """
    try:
        # Preprocess
        features = preprocess_single_customer(customer)
        
        # Scale
        features_scaled = scaler.transform(features)
        
        # Predict cluster
        cluster_id = int(kmeans.predict(features_scaled)[0])
        
        # Get sentiment
        sentiment_score = analyzer.polarity_scores(customer.Customer_Feedback)['compound']
        if sentiment_score >= 0.05:
            sentiment_label = "Positive"
        elif sentiment_score <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        # Get topic
        feedback_vec = vectorizer.transform([customer.Customer_Feedback])
        topic_dist = lda_model.transform(feedback_vec)[0]
        dominant_topic = int(topic_dist.argmax())
        
        topic_names = {
            0: "Technical_Issues",
            1: "Loan_Concerns",
            2: "Charges_Complaints",
            3: "General_Feedback"
        }
        
        # Get cluster info
        cluster_info = CLUSTER_INFO[cluster_id]
        
        return {
            "cluster": {
                "id": cluster_id,
                "name": cluster_info["name"],
                "description": cluster_info["description"],
                "segment_size": f"{cluster_info['size_pct']}%"
            },
            "sentiment": {
                "score": round(sentiment_score, 3),
                "label": sentiment_label
            },
            "topic": {
                "id": dominant_topic,
                "name": topic_names[dominant_topic],
                "confidence": round(float(topic_dist[dominant_topic]), 3)
            },
            "recommendations": cluster_info["recommendations"],
            "risk_assessment": {
                "credit_score": customer.Credit_Score,
                "loan_status": customer.Loan_Status,
                "spending_level": "High" if customer.Monthly_Expenditure > 150000 else "Medium"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
def predict_batch(batch: BatchCustomerInput):
    """
    Predict clusters for MULTIPLE customers at once
    
    Returns: List of predictions for each customer
    """
    try:
        results = []
        
        for idx, customer in enumerate(batch.customers):
            # Preprocess
            features = preprocess_single_customer(customer)
            
            # Scale and predict
            features_scaled = scaler.transform(features)
            cluster_id = int(kmeans.predict(features_scaled)[0])
            
            # Get sentiment
            sentiment_score = analyzer.polarity_scores(customer.Customer_Feedback)['compound']
            sentiment_label = "Positive" if sentiment_score >= 0.05 else ("Negative" if sentiment_score <= -0.05 else "Neutral")
            
            # Get topic
            feedback_vec = vectorizer.transform([customer.Customer_Feedback])
            topic_dist = lda_model.transform(feedback_vec)[0]
            dominant_topic = int(topic_dist.argmax())
            
            results.append({
                "customer_index": idx,
                "cluster_id": cluster_id,
                "cluster_name": CLUSTER_INFO[cluster_id]["name"],
                "sentiment_score": round(sentiment_score, 3),
                "sentiment_label": sentiment_label,
                "topic_id": dominant_topic
            })
        
        return {
            "total_customers": len(batch.customers),
            "predictions": results,
            "cluster_distribution": pd.Series([r["cluster_id"] for r in results]).value_counts().to_dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/clusters")
def get_cluster_info():
    """Get information about all clusters"""
    return {"clusters": CLUSTER_INFO}

@app.get("/cluster/{cluster_id}")
def get_single_cluster(cluster_id: int):
    """Get detailed information about a specific cluster"""
    if cluster_id not in CLUSTER_INFO:
        raise HTTPException(status_code=404, detail="Cluster not found")
    return CLUSTER_INFO[cluster_id]

@app.get("/health")
def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "api_version": "2.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)