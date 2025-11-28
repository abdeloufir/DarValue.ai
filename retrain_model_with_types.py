"""
Retrain model to ensure property_type, condition, and furnishing have proper impact on pricing.
This script handles categorical feature encoding properly to ensure differentiation.
"""

import pandas as pd
import numpy as np
import psycopg2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
from loguru import logger

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=''), colorize=False)

def connect_to_database():
    """Connect to PostgreSQL database"""
    from config.settings import ConfigManager
    config_mgr = ConfigManager()
    config = config_mgr.load_config()
    db_url = config.database.url
    
    conn_str = db_url.replace('postgresql://', '')
    if '@' in conn_str:
        user_pass, host_db = conn_str.split('@')
        user, password = user_pass.split(':')
    else:
        user = 'postgres'
        password = 'password'
        host_db = conn_str
    
    host, path = host_db.split('/')
    db = path.split('?')[0] if '?' in path else path
    
    conn = psycopg2.connect(host=host, database=db, user=user, password=password)
    return conn

def fetch_listings():
    """Fetch all listings from database"""
    try:
        conn = connect_to_database()
        cursor = conn.cursor()
        
        # Fetch listings with all features
        cursor.execute("""
            SELECT 
                id, price_mad, surface_m2, rooms, bathrooms, 
                city, neighborhood, property_type, condition, furnishing
            FROM listings
            WHERE price_mad IS NOT NULL 
            AND surface_m2 IS NOT NULL 
            AND surface_m2 > 0
        """)
        
        columns = ['id', 'price_mad', 'surface_m2', 'rooms', 'bathrooms', 
                   'city', 'neighborhood', 'property_type', 'condition', 'furnishing']
        data = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        df = pd.DataFrame(data, columns=columns)
        logger.info(f"Fetched {len(df)} listings from database")
        
        # Check distribution
        logger.info(f"\nProperty Type Distribution:")
        for ptype, count in df['property_type'].value_counts().items():
            logger.info(f"  {ptype}: {count}")
        
        logger.info(f"\nCondition Distribution:")
        for cond, count in df['condition'].value_counts().items():
            logger.info(f"  {cond}: {count}")
        
        return df
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.info("Using sample data for demonstration")
        return None

def prepare_data(df):
    """Prepare and clean data"""
    logger.info(f"\nPreparing {len(df)} listings...")
    
    df = df.copy()
    
    # Remove missing critical values
    df = df.dropna(subset=['price_mad', 'surface_m2', 'city'])
    
    # Remove outliers
    initial_len = len(df)
    df['price_per_m2'] = df['price_mad'] / df['surface_m2']
    
    # Remove invalid price per m²
    df = df[(df['price_per_m2'] >= 500) & (df['price_per_m2'] <= 500000)]
    
    # Remove invalid surface
    df = df[(df['surface_m2'] >= 20) & (df['surface_m2'] <= 50000)]
    
    # Remove invalid price
    df = df[(df['price_mad'] >= 50000) & (df['price_mad'] <= 1500000000)]
    
    logger.info(f"Removed {initial_len - len(df)} outliers")
    
    # Fill missing categorical values
    df['neighborhood'] = df['neighborhood'].fillna('Unknown')
    df['condition'] = df['condition'].fillna('Standard')
    df['furnishing'] = df['furnishing'].fillna('Unknown')
    df['property_type'] = df['property_type'].fillna('Unknown')
    
    # Fix case normalization
    df['property_type'] = df['property_type'].str.capitalize()
    df['condition'] = df['condition'].str.capitalize()
    df['furnishing'] = df['furnishing'].str.capitalize()
    
    # Engineer features
    df['log_surface'] = np.log1p(df['surface_m2'])
    df['price_per_room'] = np.where(
        df['rooms'] > 0,
        df['price_mad'] / df['rooms'],
        df['price_mad']
    )
    
    # Neighborhood stats
    df['neighborhood_avg_price_m2'] = df.groupby(['city', 'neighborhood'])['price_per_m2'].transform('mean')
    
    neighborhood_counts = df.groupby(['city', 'neighborhood']).size().to_dict()
    df['neighborhood_count'] = df.apply(
        lambda row: neighborhood_counts.get((row['city'], row['neighborhood']), 1), 
        axis=1
    )
    
    df['neighborhood_price_deviation'] = (df['price_per_m2'] - df['neighborhood_avg_price_m2']) / (df['neighborhood_avg_price_m2'] + 1)
    df['neighborhood_price_deviation'] = df['neighborhood_price_deviation'].fillna(0)
    
    # Fill missing rooms/bathrooms
    df['rooms'] = df['rooms'].fillna(df['rooms'].median())
    df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())
    
    logger.info(f"Data prepared: {len(df)} listings after cleaning and feature engineering")
    
    return df

def train_model(df):
    """Train the property price model"""
    logger.info("\n=== Training Property Price Model ===")
    
    # Define feature columns
    feature_cols = [
        'log_surface', 'rooms', 'bathrooms', 'price_per_room',
        'neighborhood_avg_price_m2', 'neighborhood_count', 'neighborhood_price_deviation',
        'condition', 'furnishing', 'city', 'neighborhood', 'property_type'
    ]
    
    X = df[feature_cols].copy()
    y = df['price_mad'].copy()
    
    # Encode categorical variables
    encoders = {}
    categorical_cols = ['city', 'neighborhood', 'property_type', 'condition', 'furnishing']
    
    logger.info(f"\nEncoding categorical features:")
    for col in categorical_cols:
        if col in X.columns:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col].astype(str))
            encoders[col] = encoder
            logger.info(f"  {col}: {len(encoder.classes_)} classes - {list(encoder.classes_)[:5]}{'...' if len(encoder.classes_) > 5 else ''}")
    
    # Remove any NaN values
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx].reset_index(drop=True)
    y = y[valid_idx].reset_index(drop=True)
    
    logger.info(f"\nTraining data: {len(X)} samples with {len(feature_cols)} features")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with regularization
    logger.info("\nTraining GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=15,
        min_samples_leaf=8,
        subsample=0.7,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\n=== Model Performance ===")
    logger.info(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    logger.info(f"Cross-Validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"Train RMSE: {train_rmse:,.0f} MAD, Test RMSE: {test_rmse:,.0f} MAD")
    logger.info(f"Train MAE: {train_mae:,.0f} MAD, Test MAE: {test_mae:,.0f} MAD")
    
    logger.info(f"\n=== Top 10 Most Important Features ===")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")
    
    logger.info(f"\n=== Categorical Feature Importance ===")
    categorical_importance = feature_importance[feature_importance['feature'].isin(categorical_cols)]
    for idx, row in categorical_importance.iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")
    
    # Save models
    os.makedirs('models/property_models', exist_ok=True)
    joblib.dump(model, 'models/property_models/price_model.pkl')
    joblib.dump(scaler, 'models/property_models/scaler.pkl')
    joblib.dump(encoders, 'models/property_models/encoders.pkl')
    
    logger.info(f"\n✓ Models saved to models/property_models/")
    
    return model, scaler, encoders, feature_importance

def test_predictions(model, scaler, encoders):
    """Test predictions with different property types"""
    logger.info(f"\n=== Testing Property Type Impact ===")
    
    # Test features (same as before)
    test_cases = [
        {'property_type': 'Apartment', 'condition': 'Standard'},
        {'property_type': 'Villa', 'condition': 'Standard'},
        {'property_type': 'House', 'condition': 'Standard'},
    ]
    
    base_features = [
        np.log1p(200),  # log_surface
        3,              # rooms
        2,              # bathrooms
        75000,          # price_per_room (estimate)
        10000,          # neighborhood_avg_price_m2
        50,             # neighborhood_count
        0,              # neighborhood_price_deviation
    ]
    
    for test_case in test_cases:
        features = base_features.copy()
        
        # Add categorical encodings
        features.append(encoders['condition'].transform([test_case['condition']])[0])
        features.append(encoders['furnishing'].transform(['Unknown'])[0])
        features.append(encoders['city'].transform(['Casablanca'])[0])
        features.append(encoders['neighborhood'].transform(['Californie'])[0])
        features.append(encoders['property_type'].transform([test_case['property_type']])[0])
        
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        
        logger.info(f"{test_case['property_type']}: {pred:,.0f} MAD")
    
    # Test condition impact
    logger.info(f"\n=== Testing Condition Impact ===")
    condition_cases = [
        {'condition': 'Standard'},
        {'condition': 'New'},
        {'condition': 'Renovated'},
    ]
    
    for test_case in condition_cases:
        features = base_features.copy()
        
        features.append(encoders['condition'].transform([test_case['condition']])[0])
        features.append(encoders['furnishing'].transform(['Unknown'])[0])
        features.append(encoders['city'].transform(['Casablanca'])[0])
        features.append(encoders['neighborhood'].transform(['Californie'])[0])
        features.append(encoders['property_type'].transform(['Villa'])[0])
        
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        
        logger.info(f"Condition {test_case['condition']}: {pred:,.0f} MAD")

if __name__ == '__main__':
    # Fetch data
    df = fetch_listings()
    
    if df is None or len(df) == 0:
        logger.error("No data available")
        exit(1)
    
    # Prepare data
    df = prepare_data(df)
    
    # Train model
    model, scaler, encoders = train_model(df)
    
    # Test predictions
    test_predictions(model, scaler, encoders)
    
    logger.info("\n✓ Model retraining complete!")
