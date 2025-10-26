"""
Configuration module for SoloVision parking detection system.
Loads environment variables and provides application settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Database Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'parking_db')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    
    # Flask Configuration
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))
    
    # Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'yolov8n.pt')
    
    # CSV Fallback Configuration
    USE_CSV_FALLBACK = os.getenv('USE_CSV_FALLBACK', 'true').lower() == 'true'
    CSV_DATA_PATH = os.getenv('CSV_DATA_PATH', 'data/vehicle_detections.csv')
    
    # Upload Configuration
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

    # Polygon/Geometry Configuration
    POLYGON_REF_WIDTH = int(os.getenv('POLYGON_REF_WIDTH', '1280'))
    POLYGON_REF_HEIGHT = int(os.getenv('POLYGON_REF_HEIGHT', '720'))
    POLYGON_IOU_THRESHOLD = float(os.getenv('POLYGON_IOU_THRESHOLD', '0.15'))
    USE_POLYGON_FILE = os.getenv('USE_POLYGON_FILE', 'false').lower() == 'true'

    # Stationary vehicle filtering
    STATIONARY_FRAMES = int(os.getenv('STATIONARY_FRAMES', '2'))
    STATIONARY_PIXELS = int(os.getenv('STATIONARY_PIXELS', '12'))

