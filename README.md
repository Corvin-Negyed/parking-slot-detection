<div align="center">
  <img src="static/logo.png" alt="SoloVision Logo" width="400"/>

  ### Smart Parking Management System
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
  [![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
  [![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

---

## Developers
- Umran Er (Neptun: BE59IT)
- Cem Akan (Neptun: JKVNGM)

## Project Overview

This project implements a smart parking management system using computer vision technology. The system processes real-time camera feeds to detect parking space occupancy and provides both live monitoring and historical analytics.

The main objective is to track parking spot status over time by logging when spaces become occupied or available. This historical data enables valuable insights such as identifying peak parking hours and usage patterns.

## Features

- Real-time parking space detection using YOLOv8
- Visual feedback with color-coded parking spots (red for occupied, green for available)
- Web-based dashboard for live monitoring
- Video upload via drag-and-drop or file selection
- Support for live camera stream URLs
- Cloud database integration with PostgreSQL
- CSV fallback storage when database is unavailable
- Historical data analytics
- Modern and responsive user interface

## Hardware Requirements

- Personal computer for development and testing
- Pre-recorded videos or live camera streams for testing

## Software Technologies

### Core Technologies

- Python: Main programming language
- OpenCV: Image processing and video handling
- YOLOv8: Object detection model
- NumPy: Data manipulation

### Cloud and Web Technologies

- Flask: Web framework
- PostgreSQL: Cloud database for historical data
- HTML/CSS/JavaScript: Frontend interface

## Installation

1. Clone the repository:
```
git clone <repository-url>
cd parking-slot-detection
```

2. Create and activate virtual environment:

**On Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```
Edit the `.env` file with your database credentials and configuration.

5. Initialize the database (optional if using PostgreSQL):
```bash
python -c "from src.database import init_database; init_database()"
```

## Usage

1. Start the application:
```
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload a video file:
   - Drag and drop a video file into the upload zone, or
   - Click "Browse Files" to select a video from your computer

4. Alternatively, connect to a live camera stream:
   - Enter the camera stream URL in the input field
   - Click "Connect to Stream"

5. View the results:
   - Watch real-time detection with color-coded parking spots
   - Monitor live statistics (total spots, occupied, available)
   - Access historical data and analytics

## Database Configuration

The system supports two storage modes:

1. PostgreSQL (Primary): Configure database credentials in `.env` file
2. CSV Fallback (Secondary): Automatically activates if PostgreSQL is unavailable

Data is stored with timestamp information for historical analysis.

## Project Structure

```
parking-slot-detection/
├── app.py                 # Flask application entry point
├── src/
│   ├── config.py         # Configuration management
│   ├── database.py       # Database operations
│   ├── detector.py       # YOLOv8 detection logic
│   └── video_processor.py # Video processing pipeline
├── static/
│   ├── app.js            # Frontend JavaScript
│   └── style.css         # Styling
├── templates/
│   └── index.html        # Main web interface
├── Models/               # YOLOv8 trained models
├── uploads/              # Uploaded video storage
└── data/                 # CSV data storage (fallback)
```

## Contributing

This is an academic project developed as part of coursework.

## License

This project is for educational purposes.
