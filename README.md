# DarValue.ai

An AI-powered real estate valuation platform tailored for the Moroccan housing market.

## Overview

DarValue.ai predicts accurate property prices in major Moroccan cities (Casablanca, Rabat, Marrakech, Tangier, Fes, Agadir) using machine learning and computer vision.

## Features

- **Data Collection**: Automated scraping from Mubawab.ma
- **Geospatial Enrichment**: Integration with OpenStreetMap, Google Maps API, and Moroccan government data
- **Machine Learning**: Advanced models for price prediction based on features and images
- **Computer Vision**: Automated image analysis for property characteristics
- **Web Interface**: Public web platform for property valuation
- **API**: RESTful API for third-party integrations

## Project Structure

```
DarValue.ai/
├── src/
│   ├── data_collection/    # Web scraping and data ingestion
│   │   ├── scrapers/      # Site-specific scrapers
│   │   └── enrichment/    # Geospatial and feature enrichment
│   ├── database/          # Database models and migrations
│   ├── models/            # ML models and training scripts
│   └── web_app/           # Web interface and API
├── config/                # Configuration files
├── data/                  # Raw and processed data
├── logs/                  # Application logs
└── tests/                 # Test suites
```

## Technologies

- **Backend**: Python, FastAPI, PostgreSQL
- **ML/AI**: scikit-learn, TensorFlow/PyTorch, OpenCV
- **Web Scraping**: Scrapy, BeautifulSoup, Selenium
- **Geospatial**: PostGIS, OSMnx, Google Maps API
- **Cloud**: AWS S3/Google Cloud Storage
- **Frontend**: React.js, Tailwind CSS
- **Deployment**: Docker, AWS/GCP

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment variables
4. Run database migrations
5. Start data collection pipeline
6. Launch web application

## Development Phases

1. **Data Collection** - Web scraping and geospatial enrichment
2. **Data Processing** - Cleaning and feature engineering
3. **Model Development** - ML and computer vision models
4. **Web Application** - Frontend and API development
5. **Deployment** - Production hosting and scaling

## License

MIT License