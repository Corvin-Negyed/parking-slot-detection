#!/usr/bin/env python3

import sys

def test_imports():
    print("Testing imports...")
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✓ python-dotenv imported successfully")
    except ImportError as e:
        print(f"✗ python-dotenv import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics (YOLOv8) imported successfully")
    except ImportError as e:
        print(f"✗ Ultralytics import failed: {e}")
        return False
    
    print("\nAll imports successful!")
    return True


def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from src.config import Config
        print("✓ Config loaded successfully")
        print(f"  Model path: {Config.MODEL_PATH}")
        print(f"  CSV path: {Config.CSV_DATA_PATH}")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_database():
    """Test database connection"""
    print("\nTesting database...")
    
    try:
        from src.database import DatabaseManager
        db = DatabaseManager()
        storage_type = "PostgreSQL" if db.use_postgres else "CSV"
        print(f"✓ Database initialized successfully (using {storage_type})")
        db.close()
        return True
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("SoloVision - System Test")
    print("=" * 50)
    
    success = True
    success = test_imports() and success
    success = test_config() and success
    success = test_database() and success
    
    print("\n" + "=" * 50)
    if success:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed! ✗")
        sys.exit(1)

