# Quick Start Guide

## Get Started in 5 Minutes

### 1. Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy environment file
cp env.example .env

# (Optional) Edit .env if you want to use PostgreSQL
```

### 3. Run

```bash
python app.py
```

### 4. Open Browser

Navigate to: http://localhost:5000

### 5. Upload Video

- Drag and drop a video file, or
- Click "Browse Files" to select a video, or
- Enter a camera stream URL

## Features

### Video Upload
- Supports: MP4, AVI, MOV, MKV
- Drag and drop or file browser
- Automatic processing after upload

### Live Camera Streams
- Enter RTSP or HTTP stream URL
- Connect to public IP cameras
- Real-time detection

### Detection Results
- **Green spots**: Available parking spaces
- **Red spots**: Occupied parking spaces
- Live statistics: Total, Occupied, Available, Occupancy Rate

### Data Storage
- Automatic fallback to CSV if PostgreSQL unavailable
- Historical data logging
- Event tracking with timestamps

## Example Videos

You can test with any parking lot video. The system will:
1. Detect vehicles automatically
2. Generate default parking grid
3. Show occupancy in real-time
4. Log data for analytics

## Tips

- For better performance, use videos with clear parking lot view
- The system works best with overhead/aerial views
- First run might be slower (YOLOv8 model loading)
- Statistics update every 2 seconds

## Troubleshooting

**Video not processing?**
- Check file format is supported
- Ensure model file exists: `Models/yolov8m mAp 48/weights/best.pt`

**No detections showing?**
- Model might still be loading (first run)
- Check browser console for errors

**Can't connect to stream?**
- Verify stream URL is accessible
- Check network/firewall settings

## Support

For detailed installation guide, see INSTALL.md
For project documentation, see README.md

