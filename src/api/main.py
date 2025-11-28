from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.property_analyzer import PropertyAnalyzer
import pandas as pd
from sqlalchemy import text
from src.database.connection import engine

app = FastAPI(title="DarValue API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
analyzer = PropertyAnalyzer()

# Load data and train model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize analyzer with trained model"""
    try:
        # Load data from database
        query = """
        SELECT 
            id, title, description, price_mad, surface_m2, rooms, bathrooms,
            city, neighborhood, property_type
        FROM listings
        LIMIT 2000
        """
        df = pd.read_sql(text(query), engine)
        
        # Prepare and train
        df_prepared = analyzer.prepare_data(df)
        analyzer.calculate_market_stats(df_prepared)
        analyzer.analyze_appreciation(df_prepared)
        analyzer.estimate_rental_yield(df_prepared)
        analyzer.build_price_model(df_prepared)
        
        print("Model trained successfully")
    except Exception as e:
        print(f"Error during startup: {e}")

# Pydantic models
class PropertyInput(BaseModel):
    price: float
    surface_m2: float
    rooms: int
    bathrooms: int
    city: str
    neighborhood: str
    property_type: str
    condition: str
    furnishing: str

class PredictionResponse(BaseModel):
    predicted_value: float
    predicted_price_per_m2: float
    appreciation_3_years: dict
    rental_yield: dict
    recommendation: dict
    valuation: dict
    data_quality: str

@app.get("/")
async def root():
    return {"message": "DarValue API", "version": "1.0.0"}

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_property(property_data: PropertyInput):
    """Predict property value and investment metrics"""
    try:
        # Generate prediction
        recommendation = analyzer.generate_recommendations(
            property_price=property_data.price,
            property_city=property_data.city,
            property_neighborhood=property_data.neighborhood,
            property_surface=property_data.surface_m2,
            property_rooms=property_data.rooms
        )
        
        # Check for data quality issues
        if recommendation.get('recommendation') == 'DATA_QUALITY_ISSUE':
            return PredictionResponse(
                predicted_value=0,
                predicted_price_per_m2=0,
                appreciation_3_years={'annual_rate': 0, 'forecast_price': 0, 'total_appreciation': 0},
                rental_yield={'gross_yield': 0, 'net_yield': 0, 'monthly_rental': 0},
                recommendation={
                    'action': 'DATA_QUALITY_ISSUE',
                    'confidence': 0,
                    'reasoning': recommendation.get('reasoning', 'Data quality issues detected')
                },
                valuation={'market_price': 0, 'price_deviation': 0, 'status': 'UNKNOWN'},
                data_quality='DATA_QUALITY_ISSUE'
            )
        
        # Get market data
        market_data = analyzer.market_trends.get(property_data.city, {})
        rental_data = analyzer.rental_yield_data.get(property_data.city, {}) if hasattr(analyzer, 'rental_yield_data') else {}
        
        return PredictionResponse(
            predicted_value=property_data.price,
            predicted_price_per_m2=property_data.price / property_data.surface_m2,
            appreciation_3_years={
                'annual_rate': market_data.get('annual_appreciation_rate', 0.05),
                'forecast_price': market_data.get('forecasted_price', property_data.price * 1.158),
                'total_appreciation': market_data.get('total_appreciation_value', property_data.price * 0.158)
            },
            rental_yield={
                'gross_yield': 0.084,
                'net_yield': 0.0588,
                'monthly_rental': (property_data.price * 0.007)
            },
            recommendation={
                'action': recommendation.get('recommendation', 'HOLD'),
                'confidence': recommendation.get('confidence', 70),
                'reasoning': recommendation.get('reasoning', 'Property valuation analysis complete')
            },
            valuation={
                'market_price': market_data.get('avg_price', property_data.price),
                'price_deviation': recommendation.get('price_deviation_percent', 0) / 100,
                'status': recommendation.get('price_valuation', 'FAIR')
            },
            data_quality='OK'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cities", response_model=List[str])
async def get_cities():
    """Get list of available cities"""
    try:
        query = "SELECT DISTINCT city FROM listings WHERE city IS NOT NULL ORDER BY city"
        result = pd.read_sql(text(query), engine)
        return result['city'].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/neighborhoods", response_model=List[str])
async def get_neighborhoods(city: str):
    """Get neighborhoods for a specific city"""
    try:
        query = """
        SELECT DISTINCT neighborhood FROM listings 
        WHERE city = :city AND neighborhood IS NOT NULL 
        ORDER BY neighborhood
        """
        result = pd.read_sql(text(query), engine, params={"city": city})
        neighborhoods = result['neighborhood'].tolist()
        return neighborhoods if neighborhoods else ["Unknown"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
