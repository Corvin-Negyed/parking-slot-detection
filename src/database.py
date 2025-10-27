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
            pass
        except Exception as e:
            pass
            self.use_postgres = False
            self._init_csv_storage()
    
    def _init_postgres_tables(self):
        if not self.connection:
            return
        
        cursor = self.connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_detections (
                id SERIAL PRIMARY KEY,
                total_spots INTEGER DEFAULT 0,
                occupied_spots INTEGER NOT NULL,
                available_spots INTEGER DEFAULT 0,
                occupancy_rate FLOAT DEFAULT 0.0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.connection.commit()
        cursor.close()
    
    def _init_csv_storage(self):
        csv_path = Config.CSV_DATA_PATH
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'total_spots', 'occupied_spots', 'available_spots', 'occupancy_rate', 'timestamp'])
    
    def save_detection(self, stats):
        if self.use_postgres:
            self._save_to_postgres(stats)
        else:
            self._save_to_csv(stats)
    
    def _save_to_postgres(self, stats):
        if not self.connection:
            return
        
        try:
            occupancy_rate = (stats['occupied'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
            
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO vehicle_detections (total_spots, occupied_spots, available_spots, occupancy_rate) VALUES (%s, %s, %s, %s)",
                (stats['total'], stats['occupied'], stats['available'], occupancy_rate)
            )
            self.connection.commit()
            cursor.close()
        except Exception as e:
            pass
    
    def _save_to_csv(self, stats):
        csv_path = Config.CSV_DATA_PATH
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            occupancy_rate = (stats['occupied'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
            
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
                writer.writerow([next_id, stats['total'], stats['occupied'], stats['available'], occupancy_rate, timestamp])
        except Exception as e:
            pass
    
    def get_recent_detections(self, limit=100):
        if self.use_postgres:
            return self._get_from_postgres(limit)
        else:
            return self._get_from_csv(limit)
    
    def _get_from_postgres(self, limit):
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT * FROM vehicle_detections ORDER BY timestamp DESC LIMIT %s",
                (limit,)
            )
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            pass
            return []
    
    def _get_from_csv(self, limit):
        csv_path = Config.CSV_DATA_PATH
        
        if not os.path.exists(csv_path):
            return []
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                rows = list(reader)
                # Return in reverse order (most recent first)
                return list(reversed(rows[-limit:])) if len(rows) > limit else list(reversed(rows))
        except Exception as e:
            pass
            return []
    
    def clear_all_data(self):
        if self.use_postgres:
            self._clear_postgres()
        else:
            self._clear_csv()
    
    def _clear_postgres(self):
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM vehicle_detections")
            self.connection.commit()
            cursor.close()
            pass
        except Exception as e:
            pass
    
    def _clear_csv(self):
        csv_path = Config.CSV_DATA_PATH
        
        try:
            # Recreate with just header
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'total_spots', 'occupied_spots', 'available_spots', 'occupancy_rate', 'timestamp'])
            pass
        except Exception as e:
            pass
    
    def close(self):
        if self.connection:
            self.connection.close()


# Convenience functions
def get_connection():
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
        pass
        return None


def init_database():
    db = DatabaseManager()
    db.close()
