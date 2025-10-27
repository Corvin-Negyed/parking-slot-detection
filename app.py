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

# Configure upload folder and timeouts
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

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
    """Handle video file upload with error handling"""
    global current_processor
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File save failed'}), 500
        
        if current_processor:
            try:
                current_processor.stop()
                current_processor.close_video()
            except:
                pass
            current_processor = None
        
        current_processor = VideoProcessor(filepath)
        
        try:
            success = current_processor.open_video()
            if not success:
                return jsonify({'error': 'Cannot open video'}), 500
        except Exception as e:
            return jsonify({'error': f'Video open failed: {str(e)}'}), 500
        
        return jsonify({
            'status': 'success',
            'message': 'Video ready',
            'filename': filename
        })
    
    except Exception as e:
        if current_processor:
            try:
                current_processor.close_video()
            except:
                pass
            current_processor = None
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
        return Response(status=404)
    
    def generate():
        """Generate video frames with error handling"""
        try:
            for frame_bytes, stats in current_processor.process_video_stream():
                if frame_bytes:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except GeneratorExit:
            pass
        except Exception:
            pass
    
    response = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/stats')
def get_stats():
    """Get current parking statistics with detailed stage information"""
    global current_processor
    
    if not current_processor:
        return jsonify({'error': 'No video source available'}), 400
    
    try:
        # Get basic stats
        stats = current_processor.current_stats.copy()
        
        # Add detailed detector information
        detector = current_processor.detector
        stats['stage'] = detector.stage
        stats['is_initial_phase'] = detector.is_initial_phase
        stats['orientation'] = detector.orientation
        stats['detected_vehicles'] = len(detector.current_stationary_boxes)
        stats['grid_established'] = detector.grid_established
        stats['grid_slots'] = len(detector.parking_grid) if detector.parking_grid else 0
        
        # Stage description
        if detector.is_initial_phase:
            stats['stage_description'] = 'Stage 1: Learning - Detecting vehicles'
        elif not detector.grid_established:
            stats['stage_description'] = 'Stage 2: Building parking grid'
        else:
            stats['stage_description'] = f'Stage 3: Monitoring ({detector.orientation})'
        
        return jsonify(stats)
    
    except Exception as e:
        print(f"Error getting stats: {e}")
        # Return basic stats on error
        return jsonify(current_processor.current_stats)


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
                # PostgreSQL: (id, total, occupied, available, rate, timestamp)
                formatted_detections.append({
                    'id': detection[0],
                    'total_spots': detection[1],
                    'occupied': detection[2],
                    'available': detection[3],
                    'occupancy_rate': float(detection[4]),
                    'timestamp': str(detection[5])
                })
            else:
                # CSV format: same structure
                formatted_detections.append({
                    'id': detection[0],
                    'total_spots': detection[1],
                    'occupied': detection[2],
                    'available': detection[3],
                    'occupancy_rate': float(detection[4]),
                    'timestamp': detection[5]
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


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear all historical data"""
    db = DatabaseManager()
    
    try:
        db.clear_all_data()
        storage_type = 'PostgreSQL' if db.use_postgres else 'CSV'
        db.close()
        
        return jsonify({
            'status': 'success',
            'message': f'{storage_type} history cleared successfully'
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