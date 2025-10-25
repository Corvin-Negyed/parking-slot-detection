@echo off
REM SoloVision - Quick run script for Windows

echo Starting SoloVision Parking Detection System...
echo ==========================================

REM Check if virtual environment exists
if not exist ".venv" (
    echo Virtual environment not found. Creating...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Check if .env exists
if not exist ".env" (
    echo Creating .env file from example...
    copy env.example .env
    echo Please edit .env file with your configuration!
)

REM Run the application
echo Starting Flask application...
python app.py

pause

