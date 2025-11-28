# DarValue.ai Platform Setup Guide

## Prerequisites
- Node.js 18+ 
- Python 3.13
- PostgreSQL (with data already loaded)

## Frontend Setup (Next.js 14 + Tailwind CSS)

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Backend Setup (FastAPI)

```bash
# Install dependencies
pip install -r src/api/requirements.txt

# Run the API server
python src/api/main.py
```

The API will be available at `http://localhost:8000`
API docs: `http://localhost:8000/docs`

## Platform Features

### Input Form
Users can enter:
- Price (MAD)
- Surface (m²)
- Number of rooms
- Number of bathrooms
- City (6 available: Casablanca, Rabat, Agadir, Tangier, Marrakech, Fes)
- Neighborhood (dynamic based on city)
- Property type (apartment, villa, house, land, commercial)
- Condition (New, Renovated, Standard, Old)
- Furnishing (Furnished, Unfurnished, Unknown)

### Prediction Output (5 Metrics)

1. **Predicted Property Value**
   - ML model prediction with 93% accuracy (R²=0.9310)
   - Price per m² analysis

2. **Market Valuation**
   - Neighborhood-based price comparison
   - Deviation percentage vs market

3. **3-Year Appreciation Forecast**
   - Annual appreciation rate (3-5% based on market volatility)
   - Forecasted price after 3 years
   - Total gain in MAD

4. **Rental Yield Analysis**
   - Gross yield: 8.4% per annum
   - Net yield: 5.88% (after 30% cost deduction)
   - Estimated monthly rental income

5. **Investment Recommendation**
   - Action: BUY, HOLD, or SELL
   - Confidence score (0-95%)
   - Reasoning with market analysis

### Data Quality
- Removes 306 bad records from dataset
- Flags suspicious data with `DATA_QUALITY_ISSUE` status
- 1,433 clean listings used for training

## API Endpoints

### POST /api/predict
```json
{
  "price": 2500000,
  "surface_m2": 120,
  "rooms": 3,
  "bathrooms": 2,
  "city": "Casablanca",
  "neighborhood": "Downtown",
  "property_type": "apartment",
  "condition": "Standard",
  "furnishing": "Unfurnished"
}
```

### GET /api/cities
Returns list of available cities

### GET /api/neighborhoods?city=Casablanca
Returns neighborhoods for specified city

## Technology Stack

**Frontend:**
- Next.js 14
- TypeScript
- Tailwind CSS
- Axios (API client)

**Backend:**
- FastAPI
- Pydantic (validation)
- SQLAlchemy (database)
- Scikit-learn (ML model)
- Pandas (data processing)

**Database:**
- PostgreSQL
- 1,608 Moroccan property listings

## Model Performance

- **Algorithm:** GradientBoostingRegressor
- **Training samples:** 1,442 (after cleaning)
- **Test R²:** 0.9310 (93.1% accuracy)
- **Test MAE:** 239,575 MAD
- **Features:** 12 engineered features
- **Top feature:** price_per_room (65.29%)

## Known Limitations

- Market data limited to 6 Moroccan cities
- Current market snapshot (no time-series data)
- Appreciation rate: Fixed 3-5% annually
- Rental yield: Standardized 5.88% net
- Currency: All values in MAD (Moroccan Dirham)

## Next Steps for Production

1. Deploy FastAPI to AWS/GCP/Azure
2. Deploy Next.js frontend
3. Add authentication & API rate limiting
4. Set up monitoring & alerting
5. Configure monthly model retraining
6. Add more cities to coverage
