from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List
import uvicorn

app = FastAPI(
    title="Customer Financial Risk API",
    description="Predict customer cluster and risk profile using K-Means clustering",
    version="1.0.0"
)

# Load model and scaler
try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('kmeans_model.pkl')
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    scaler, model = None, None

# Cluster interpretations
CLUSTER_PROFILES = {
    0: {
        "name": "Mobile App Users",
        "description": "Lower-Middle income, Average savings, Mobile App preference",
        "characteristics": {
            "income": "Lower-Middle",
            "saving": "Average",
            "channel": "Mobile App",
            "expenditure": "₦151,995",
            "sentiment": "Neutral (0.07)"
        },
        "recommendations": [
            "Target with in-app promotions",
            "Upsell premium mobile features",
            "Personalized push notifications"
        ],
        "risk_level": "Low"
    },
    1: {
        "name": "Traditional USSD Users",
        "description": "Lower-Middle income, Average savings, USSD preference",
        "characteristics": {
            "income": "Lower-Middle",
            "saving": "Average",
            "channel": "USSD",
            "expenditure": "₦151,146",
            "sentiment": "Neutral (0.07)"
        },
        "recommendations": [
            "Educate on mobile app benefits",
            "Offer smartphone upgrade incentives",
            "Simplified USSD menu"
        ],
        "risk_level": "Low"
    },
    2: {
        "name": "Engaged Mobile Users",
        "description": "Lower-Middle income, Average savings, Active mobile users",
        "characteristics": {
            "income": "Lower-Middle",
            "saving": "Average",
            "channel": "Mobile App",
            "expenditure": "₦150,655",
            "sentiment": "Positive (0.09)"
        },
        "recommendations": [
            "Beta test new features",
            "Referral program ambassadors",
            "Loyalty rewards"
        ],
        "risk_level": "Very Low"
    },
    3: {
        "name": "Standard Mobile Users",
        "description": "Lower-Middle income, Average savings, Stable mobile users",
        "characteristics": {
            "income": "Lower-Middle",
            "saving": "Average",
            "channel": "Mobile App",
            "expenditure": "₦152,180",
            "sentiment": "Neutral-Positive (0.08)"
        },
        "recommendations": [
            "Target for savings products",
            "Investment options",
            "Financial planning tools"
        ],
        "risk_level": "Low"
    }
}

# Input model
class CustomerInput(BaseModel):
    Monthly_Expenditure: float
    Credit_Score: float
    Loan_Amount: float
    sentiment_score: float
    Income_encoded: int  # 0=Low, 1=Lower-Middle, 2=Middle, 3=Upper-Middle, 4=High
    Saving_encoded: int  # 0=Poor, 1=Average, 2=Good
    Loan_encoded: int  # 0=No Loan, 1=Active Loan, 2=Default Risk
    Time_encoded: int  # 0-3
    Channel_encoded: int  # 0-3
    # One-hot encoded features (add based on your actual features)
    Location_Abuja: int = 0
    Location_Enugu: int = 0
    Location_Ibadan: int = 0
    Location_Kaduna: int = 0
    Location_Kano: int = 0
    Location_Lagos: int = 0
    Location_Port_Harcourt: int = 0
    Spending_Category_Education: int = 0
    Spending_Category_Entertainment: int = 0
    Spending_Category_Groceries: int = 0
    Spending_Category_Health: int = 0
    Spending_Category_Online_Shopping: int = 0
    Spending_Category_Rent: int = 0
    Spending_Category_Savings_Deposit: int = 0
    Spending_Category_Transport: int = 0
    Spending_Category_Utilities: int = 0
    Complaint_Type_General_Feedback: int = 0
    Complaint_Type_Loan_Issue: int = 0
    Complaint_Type_No_Complaint: int = 0
    Complaint_Type_Technical_Issue: int = 0

    class Config:
        schema_extra = {
            "example": {
                "Monthly_Expenditure": 150000,
                "Credit_Score": 620,
                "Loan_Amount": 0,
                "sentiment_score": 0.07,
                "Income_encoded": 1,
                "Saving_encoded": 1,
                "Loan_encoded": 0,
                "Time_encoded": 2,
                "Channel_encoded": 0,
                "Location_Lagos": 1,
                "Spending_Category_Groceries": 1,
                "Complaint_Type_No_Complaint": 1
            }
        }

@app.get("/")
def root():
    return {
        "message": "Customer Financial Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict customer cluster",
            "/clusters": "GET - Get all cluster profiles",
            "/cluster/{cluster_id}": "GET - Get specific cluster profile",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    model_status = "loaded" if model is not None else "not loaded"
    scaler_status = "loaded" if scaler is not None else "not loaded"
    
    return {
        "status": "healthy" if (model and scaler) else "unhealthy",
        "model": model_status,
        "scaler": scaler_status
    }

@app.post("/predict")
def predict_cluster(customer: CustomerInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare input
        input_dict = customer.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Scale features
        X_scaled = scaler.transform(input_df)
        
        # Predict
        cluster = int(model.predict(X_scaled)[0])
        
        # Get cluster profile
        profile = CLUSTER_PROFILES.get(cluster, {})
        
        return {
            "cluster_id": cluster,
            "cluster_name": profile.get("name", f"Cluster {cluster}"),
            "description": profile.get("description", ""),
            "characteristics": profile.get("characteristics", {}),
            "recommendations": profile.get("recommendations", []),
            "risk_level": profile.get("risk_level", "Unknown"),
            "input_data": {
                "expenditure": customer.Monthly_Expenditure,
                "credit_score": customer.Credit_Score,
                "loan_amount": customer.Loan_Amount,
                "sentiment": customer.sentiment_score
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/clusters")
def get_all_clusters():
    return {
        "total_clusters": len(CLUSTER_PROFILES),
        "clusters": CLUSTER_PROFILES
    }

@app.get("/cluster/{cluster_id}")
def get_cluster_profile(cluster_id: int):
    if cluster_id not in CLUSTER_PROFILES:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
    
    return {
        "cluster_id": cluster_id,
        **CLUSTER_PROFILES[cluster_id]
    }

@app.post("/batch_predict")
def batch_predict(customers: List[CustomerInput]):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    for customer in customers:
        try:
            input_dict = customer.dict()
            input_df = pd.DataFrame([input_dict])
            X_scaled = scaler.transform(input_df)
            cluster = int(model.predict(X_scaled)[0])
            
            results.append({
                "cluster_id": cluster,
                "cluster_name": CLUSTER_PROFILES.get(cluster, {}).get("name", f"Cluster {cluster}"),
                "risk_level": CLUSTER_PROFILES.get(cluster, {}).get("risk_level", "Unknown")
            })
        except Exception as e:
            results.append({"error": str(e)})
    
    return {"predictions": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)