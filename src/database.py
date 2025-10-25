"""
Database management module for SoloVision.
Handles PostgreSQL connection with CSV fallback for data persistence.
"""

import psycopg2
import csv
import os
from datetime import datetime
from src.config import Config


class DatabaseManager:
    def __init__(self):
        self.use_postgres = False
        self.connection = None
        self._setup_database()
    
    def _setup_database(self):
        """Try to connect to PostgreSQL, fallback to CSV if unavailable"""
        try:
            self.connection = psycopg2.connect(
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                database=Config.DB_NAME,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD
            )
            self.use_postgres = True
            self._init_postgres_tables()
            print("Successfully connected to PostgreSQL database")
        except Exception as e:
            print(f"PostgreSQL connection failed: {e}")
            print("Falling back to CSV storage")
            self.use_postgres = False
            self._init_csv_storage()
    
    def _init_postgres_tables(self):
        """Initialize PostgreSQL tables"""
        if not self.connection:
            return
        
        cursor = self.connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parking_events (
                id SERIAL PRIMARY KEY,
                spot_id INTEGER NOT NULL,
                status VARCHAR(20) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.connection.commit()
        cursor.close()
    
    def _init_csv_storage(self):
        """Initialize CSV storage"""
        csv_path = Config.CSV_DATA_PATH
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'spot_id', 'status', 'timestamp'])
    
    def save_event(self, spot_id, status):
        """Save parking event to database or CSV"""
        if self.use_postgres:
            self._save_to_postgres(spot_id, status)
        else:
            self._save_to_csv(spot_id, status)
    
    def _save_to_postgres(self, spot_id, status):
        """Save event to PostgreSQL"""
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO parking_events (spot_id, status) VALUES (%s, %s)",
                (spot_id, status)
            )
            self.connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error saving to PostgreSQL: {e}")
    
    def _save_to_csv(self, spot_id, status):
        """Save event to CSV file"""
        csv_path = Config.CSV_DATA_PATH
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            # Get next ID
            next_id = 1
            if os.path.exists(csv_path):
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    rows = list(reader)
                    if rows:
                        next_id = int(rows[-1][0]) + 1
            
            # Append new row
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([next_id, spot_id, status, timestamp])
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    
    def get_recent_events(self, limit=100):
        """Get recent parking events"""
        if self.use_postgres:
            return self._get_from_postgres(limit)
        else:
            return self._get_from_csv(limit)
    
    def _get_from_postgres(self, limit):
        """Get events from PostgreSQL"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT * FROM parking_events ORDER BY timestamp DESC LIMIT %s",
                (limit,)
            )
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            print(f"Error reading from PostgreSQL: {e}")
            return []
    
    def _get_from_csv(self, limit):
        """Get events from CSV file"""
        csv_path = Config.CSV_DATA_PATH
        
        if not os.path.exists(csv_path):
            return []
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                rows = list(reader)
                return rows[-limit:] if len(rows) > limit else rows
        except Exception as e:
            print(f"Error reading from CSV: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


# Convenience functions
def get_connection():
    """Get database connection (for backward compatibility)"""
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD
        )
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None


def init_database():
    """Initialize database (for backward compatibility)"""
    db = DatabaseManager()
    db.close()
