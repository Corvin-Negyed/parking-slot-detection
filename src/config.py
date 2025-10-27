import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # --- Database Configuration ---
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'parking_db')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_LOG_INTERVAL = int(os.getenv('DB_LOG_INTERVAL', '30')) # Saniye cinsinden loglama aralığı

    # --- Flask Web Application Configuration ---
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))

    # --- AI Model Configuration ---
    MODEL_PATH = os.getenv('MODEL_PATH', 'Models/yolov8n.pt') # Model yolu düzeltildi

    # --- Data Storage Fallback ---
    USE_CSV_FALLBACK = os.getenv('USE_CSV_FALLBACK', 'true').lower() == 'true'
    CSV_DATA_PATH = os.getenv('CSV_DATA_PATH', 'data/vehicle_detections.csv')

    # --- Video Upload Configuration ---
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

    # --- V8 PERFECT Detector Configuration ---
    # Stationary detection
    STATIONARY_FRAMES = int(os.getenv('STATIONARY_FRAMES', '3')) # Fast detection
    STATIONARY_PIXELS = int(os.getenv('STATIONARY_PIXELS', '30')) # Very tolerant
    
    # Detection interval
    DETECTION_INTERVAL_SECONDS = float(os.getenv('DETECTION_INTERVAL_SECONDS', '2.0')) # Frequent checks
    
    # Performance - Balanced for quality
    RESIZE_WIDTH = int(os.getenv('RESIZE_WIDTH', '960')) # Higher res for better detection
    
    # Frame skip - Balanced
    PROCESS_EVERY_N_FRAMES = int(os.getenv('PROCESS_EVERY_N_FRAMES', '2')) # Process more frames