"""
SQLite Database Setup and Migration
Converts the application to use SQLite instead of PostgreSQL for immediate availability
"""

import sqlite3
import os
import json
import pandas as pd
from pathlib import Path

DB_PATH = 'darvalue.db'

def create_database():
    """Create SQLite database with schema"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('PRAGMA foreign_keys = ON')
    cursor = conn.cursor()
    
    # Create listings table with all required columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            price_mad REAL NOT NULL,
            surface_m2 REAL NOT NULL,
            rooms INTEGER,
            bathrooms INTEGER,
            city TEXT,
            neighborhood TEXT,
            property_type TEXT,
            condition TEXT,
            furnishing TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_city ON listings(city)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_neighborhood ON listings(neighborhood)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_property_type ON listings(property_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_condition ON listings(condition)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_furnishing ON listings(furnishing)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_price ON listings(price_mad)')
    
    conn.commit()
    conn.close()
    print(f"Database created: {DB_PATH}")

def insert_sample_data():
    """Insert realistic sample data for testing"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # First, check if data already exists
    cursor.execute('SELECT COUNT(*) FROM listings')
    count = cursor.fetchone()[0]
    
    if count > 100:
        print(f"Database already has {count} listings, skipping sample data")
        conn.close()
        return
    
    import numpy as np
    np.random.seed(42)
    
    # Generate realistic property data
    n_samples = 1200
    
    cities = ['Casablanca', 'Rabat', 'Marrakech', 'Tangier', 'Fes', 'Agadir']
    neighborhoods = {
        'Casablanca': ['Californie', 'Anfa', 'Gauthier', 'Downtown', 'Ain Diab', 'Maârif'],
        'Rabat': ['Ryad', 'Zaouia', 'Takadoum', 'Souissi', 'Medina'],
        'Marrakech': ['Medina', 'Gueliz', 'Hivernage', 'Palmeraie'],
        'Tangier': ['Ville Nouvelle', 'Medina', 'Mountain'],
        'Fes': ['Medina', 'Ville Nouvelle', 'Zaouia'],
        'Agadir': ['Talborjt', 'Kasbah']
    }
    property_types = ['Apartment', 'Villa', 'House', 'Commercial', 'Land']
    conditions = ['New', 'Renovated', 'Standard', 'Old']
    furnishing_opts = ['Furnished', 'Unknown']
    
    data = []
    for i in range(n_samples):
        city = np.random.choice(cities)
        neighborhood = np.random.choice(neighborhoods.get(city, ['Unknown']))
        prop_type = np.random.choice(property_types)
        condition = np.random.choice(conditions)
        furnishing = np.random.choice(furnishing_opts)
        
        # Generate realistic prices based on type
        base_price = np.random.uniform(300000, 2000000)
        
        # Adjust by property type
        if prop_type == 'Villa':
            base_price *= 1.5
        elif prop_type == 'House':
            base_price *= 1.2
        elif prop_type == 'Commercial':
            base_price *= 0.8
        elif prop_type == 'Land':
            base_price *= 0.6
        
        # Adjust by condition
        if condition == 'New':
            base_price *= 1.25
        elif condition == 'Renovated':
            base_price *= 1.1
        elif condition == 'Old':
            base_price *= 0.85
        
        # Adjust by city
        city_multiplier = {
            'Casablanca': 1.3,
            'Rabat': 1.2,
            'Marrakech': 1.1,
            'Tangier': 0.9,
            'Fes': 0.85,
            'Agadir': 0.95
        }
        base_price *= city_multiplier.get(city, 1.0)
        
        surface = np.random.uniform(40, 300) if prop_type != 'Land' else np.random.uniform(200, 5000)
        rooms = np.random.randint(1, 6) if prop_type in ['Apartment', 'House', 'Villa'] else None
        bathrooms = np.random.randint(1, 4) if prop_type in ['Apartment', 'House', 'Villa'] else None
        
        title = f"{prop_type} - {surface:.0f}m² - {neighborhood}, {city}"
        
        data.append((
            title,
            f"Luxurious {prop_type.lower()} in {neighborhood}",
            int(base_price),
            surface,
            rooms,
            bathrooms,
            city,
            neighborhood,
            prop_type,
            condition,
            furnishing
        ))
    
    cursor.executemany('''
        INSERT INTO listings 
        (title, description, price_mad, surface_m2, rooms, bathrooms, city, neighborhood, 
         property_type, condition, furnishing)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data)
    
    conn.commit()
    count = cursor.execute('SELECT COUNT(*) FROM listings').fetchone()[0]
    conn.close()
    print(f"Inserted {n_samples} sample properties. Total: {count}")

def verify_data():
    """Verify data was inserted correctly"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check counts
    total = cursor.execute('SELECT COUNT(*) FROM listings').fetchone()[0]
    property_types = cursor.execute(
        'SELECT property_type, COUNT(*) as count FROM listings GROUP BY property_type'
    ).fetchall()
    conditions = cursor.execute(
        'SELECT condition, COUNT(*) as count FROM listings GROUP BY condition'
    ).fetchall()
    cities = cursor.execute(
        'SELECT city, COUNT(*) as count FROM listings GROUP BY city ORDER BY count DESC'
    ).fetchall()
    
    print(f"\nDatabase Verification:")
    print(f"  Total listings: {total}")
    print(f"\n  Property Types:")
    for ptype, count in property_types:
        print(f"    {ptype}: {count}")
    print(f"\n  Conditions:")
    for cond, count in conditions:
        print(f"    {cond}: {count}")
    print(f"\n  Top Cities:")
    for city, count in cities[:3]:
        print(f"    {city}: {count}")
    
    # Check sample data
    sample = cursor.execute('''
        SELECT id, title, price_mad, surface_m2, property_type, condition, city
        FROM listings LIMIT 5
    ''').fetchall()
    
    print(f"\n  Sample Records:")
    for row in sample:
        print(f"    ID {row[0]}: {row[1][:40]}... | {row[2]:,} MAD | {row[4]} | {row[6]}")
    
    conn.close()

if __name__ == '__main__':
    print("=== SQLite Database Setup ===\n")
    create_database()
    insert_sample_data()
    verify_data()
    print(f"\n✓ Database ready at: {DB_PATH}")
