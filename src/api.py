"""
FastAPI Backend for BTC Price Prediction
Serves the trained XGBoost model via REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
from feature_engineering import FeatureEngineer
from data_fetcher import BinanceDataFetcher

# Initialize FastAPI app
app = FastAPI(
    title="BTC Price Prediction API",
    description="XGBoost model for predicting BTC price direction (up/down) in next 24 hours",
    version="1.0.0"
)

# Global variables for model and feature engineer
model_data = None
feature_engineer = None
data_fetcher = None


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for prediction with OHLCV data"""
    timestamp: str = Field(..., description="Timestamp in ISO format")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    close: float = Field(..., description="Closing price")
    volume: float = Field(..., description="Trading volume")
    trades: Optional[int] = Field(None, description="Number of trades")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-07T12:00:00",
                "open": 45000.0,
                "high": 45500.0,
                "low": 44800.0,
                "close": 45200.0,
                "volume": 1234.56,
                "trades": 5000
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    prediction: int = Field(..., description="Predicted direction: 0=DOWN, 1=UP")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability_down: float = Field(..., description="Probability of price going down")
    probability_up: float = Field(..., description="Probability of price going up")
    confidence: float = Field(..., description="Confidence level (max probability)")
    timestamp: str = Field(..., description="Timestamp of prediction")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_training_date: Optional[str]
    features_count: int


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    training_date: str
    features_count: int
    hyperparameters: Dict
    feature_columns: List[str]


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model_data, feature_engineer, data_fetcher
    
    model_path = 'models/btc_xgboost_model.joblib'
    
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found at {model_path}")
        print("Please train the model first by running: python src/model_training.py")
        return
    
    try:
        model_data = joblib.load(model_path)
        feature_engineer = FeatureEngineer()
        data_fetcher = BinanceDataFetcher()
        print("Model loaded successfully!")
        print(f"Training date: {model_data.get('training_date', 'Unknown')}")
        print(f"Number of features: {len(model_data['feature_columns'])}")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "BTC Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_latest": "/predict/latest",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    model_loaded = model_data is not None
    
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "model_training_date": model_data.get('training_date') if model_loaded else None,
        "features_count": len(model_data['feature_columns']) if model_loaded else 0
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model information"""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost Classifier",
        "training_date": model_data.get('training_date', 'Unknown'),
        "features_count": len(model_data['feature_columns']),
        "hyperparameters": model_data.get('best_params', {}),
        "feature_columns": model_data['feature_columns']
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make prediction based on provided OHLCV data
    
    Note: This endpoint requires historical data context to compute technical indicators.
    For a single data point, you need to provide sufficient historical context.
    Use /predict/latest endpoint for real-time predictions using live data.
    """
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # This is a simplified version - in production, you'd need historical context
        # to compute technical indicators properly
        raise HTTPException(
            status_code=400, 
            detail="This endpoint requires historical context. Use /predict/latest for real-time predictions."
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/predict/latest", response_model=PredictionResponse, tags=["Prediction"])
async def predict_latest():
    """
    Make prediction using latest BTC data from Binance
    
    This endpoint:
    1. Fetches recent historical data from Binance
    2. Computes technical indicators
    3. Makes prediction for next 24 hours
    """
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Fetch recent data (need enough history for technical indicators)
        # Fetch last 500 hours to compute all indicators properly
        end_date = datetime.now()
        from datetime import timedelta
        start_date = end_date - timedelta(hours=500)
        
        print(f"Fetching data from {start_date} to {end_date}")
        df = data_fetcher.fetch_data(start_date, end_date)
        
        if len(df) < 200:
            raise HTTPException(
                status_code=500, 
                detail="Insufficient historical data to compute indicators"
            )
        
        # Create features
        df_features = feature_engineer.create_features(df)
        
        # Get the latest row (most recent data)
        latest_row = df_features.iloc[-1:][model_data['feature_columns']]
        
        # Handle any missing values (fill with 0 or mean)
        latest_row = latest_row.fillna(0)
        
        # Make prediction
        prediction = model_data['model'].predict(latest_row)[0]
        probabilities = model_data['model'].predict_proba(latest_row)[0]
        
        prob_down = float(probabilities[0])
        prob_up = float(probabilities[1])
        confidence = float(max(probabilities))
        
        prediction_label = "UP" if prediction == 1 else "DOWN"
        
        return {
            "prediction": int(prediction),
            "prediction_label": prediction_label,
            "probability_down": prob_down,
            "probability_up": prob_up,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Prediction"])
async def predict_batch(requests: List[PredictionRequest]):
    """
    Make predictions for multiple data points
    
    Note: Requires historical context for each prediction
    """
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    raise HTTPException(
        status_code=501, 
        detail="Batch prediction not implemented. Use /predict/latest for real-time predictions."
    )


@app.get("/features", tags=["Model"])
async def get_features():
    """Get list of features used by the model"""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "features_count": len(model_data['feature_columns']),
        "features": model_data['feature_columns']
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


