"""
SoloVision - Smart Parking Management System
Main Flask application for web interface and video processing.
"""

import os
import time
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from src.config import Config
from src.video_processor import VideoProcessor
from src.database import DatabaseManager
from src.analytics import VehicleAnalytics

app = Flask(__name__)
CORS(app)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Create upload folder if it doesn't exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Global video processor instance
current_processor = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video file upload"""
    global current_processor
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Stop current processor if exists
        if current_processor:
            current_processor.stop()
            current_processor.close_video()
        
        # Create new video processor
        current_processor = VideoProcessor(filepath)
        current_processor.open_video()
        
        return jsonify({
            'status': 'success',
            'message': 'Video uploaded successfully',
            'filename': filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/connect_stream', methods=['POST'])
def connect_stream():
    """Connect to live camera stream"""
    global current_processor
    
    data = request.get_json()
    stream_url = data.get('url')
    
    if not stream_url:
        return jsonify({'error': 'No stream URL provided'}), 400
    
    try:
        # Stop current processor if exists
        if current_processor:
            current_processor.stop()
            current_processor.close_video()
        
        # Create new video processor for stream
        current_processor = VideoProcessor(stream_url)
        current_processor.open_video()
        
        return jsonify({
            'status': 'success',
            'message': 'Connected to stream successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/video_feed')
def video_feed():
    """Stream video frames with detection"""
    global current_processor
    
    if not current_processor:
        return jsonify({'error': 'No video source available'}), 400
    
    def generate():
        """Generate video frames"""
        try:
            for frame_bytes, stats in current_processor.process_video_stream():
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in video feed: {e}")
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def get_stats():
    """Get current vehicle detection statistics"""
    global current_processor
    
    if not current_processor:
        return jsonify({'error': 'No video source available'}), 400
    
    # Return simple vehicle count
    return jsonify({
        'total': 0,
        'occupied': 0,  # Updated from latest detection
        'available': 0
    })


@app.route('/history')
def get_history():
    """Get vehicle detection history data"""
    db = DatabaseManager()
    
    try:
        detections = db.get_recent_detections(limit=100)
        
        # Format detections for response
        formatted_detections = []
        for detection in detections:
            if db.use_postgres:
                formatted_detections.append({
                    'id': detection[0],
                    'vehicle_count': detection[1],
                    'timestamp': str(detection[2])
                })
            else:
                # CSV format
                formatted_detections.append({
                    'id': detection[0],
                    'vehicle_count': detection[1],
                    'timestamp': detection[2]
                })
        
        db.close()
        
        return jsonify({
            'status': 'success',
            'detections': formatted_detections,
            'storage': 'postgresql' if db.use_postgres else 'csv'
        })
    
    except Exception as e:
        db.close()
        return jsonify({'error': str(e)}), 500


@app.route('/analytics')
def get_analytics():
    """Get comprehensive analytics from detection history"""
    db = DatabaseManager()
    
    try:
        # Get all detections
        detections = db.get_recent_detections(limit=1000)
        
        if not detections:
            return jsonify({
                'status': 'success',
                'message': 'No data available for analysis',
                'data': {}
            })
        
        # Generate analytics
        analytics = VehicleAnalytics(detections)
        report = analytics.get_comprehensive_report()
        
        db.close()
        
        return jsonify({
            'status': 'success',
            'data': report
        })
    
    except Exception as e:
        db.close()
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/hourly')
def get_hourly_analytics():
    """Get hourly distribution analytics"""
    db = DatabaseManager()
    
    try:
        detections = db.get_recent_detections(limit=1000)
        
        if not detections:
            return jsonify({'status': 'success', 'data': {}})
        
        analytics = VehicleAnalytics(detections)
        hourly = analytics.get_hourly_distribution()
        
        db.close()
        
        return jsonify({
            'status': 'success',
            'data': hourly
        })
    
    except Exception as e:
        db.close()
        return jsonify({'error': str(e)}), 500


@app.route('/stop')
def stop_processing():
    """Stop current video processing"""
    global current_processor
    
    if current_processor:
        current_processor.stop()
        current_processor.close_video()
        current_processor = None
    
    return jsonify({'status': 'success', 'message': 'Processing stopped'})


if __name__ == '__main__':
    # Initialize database on startup
    db = DatabaseManager()
    db.close()
    
    print("Starting SoloVision Parking Detection System...")
    print(f"Server running on http://localhost:{Config.FLASK_PORT}")
    
    app.run(
        debug=(Config.FLASK_ENV == 'development'),
        port=Config.FLASK_PORT,
        host='0.0.0.0'
    )
