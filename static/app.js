/**
 * SoloVision - Smart Parking Management System
 * Frontend JavaScript for video upload and real-time detection
 * Handles drag-drop, video streaming, and statistics display
 */

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const cameraUrl = document.getElementById('cameraUrl');
const connectBtn = document.getElementById('connectBtn');
const uploadSection = document.getElementById('uploadSection');
const videoSection = document.getElementById('videoSection');
const historySection = document.getElementById('historySection');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const streamStatus = document.getElementById('streamStatus');
const videoFeed = document.getElementById('videoFeed');
const stopBtn = document.getElementById('stopBtn');
const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

// Stats elements
const totalSpots = document.getElementById('totalSpots');
const occupiedSpots = document.getElementById('occupiedSpots');
const availableSpots = document.getElementById('availableSpots');
const occupancyRate = document.getElementById('occupancyRate');

// Browse button click handler
browseBtn.addEventListener('click', () => {
    fileInput.click();
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        handleFiles(files);
    }
});

// Drag and drop handlers
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight drop zone when dragging over it
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
        dropZone.classList.add('dragover');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
        dropZone.classList.remove('dragover');
    }, false);
});

// Handle drop
dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}, false);

// Handle file upload
function handleFiles(files) {
    if (files.length === 0) return;
    
        const file = files[0];
    
    // Check if file is a video
    if (!file.type.startsWith('video/')) {
        showStatus('Please select a valid video file', 'error');
        return;
    }
    
    uploadVideo(file);
}

// Upload video to server
function uploadVideo(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Show progress
    uploadProgress.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Uploading...';
    
    // Create XMLHttpRequest for upload progress
    const xhr = new XMLHttpRequest();
    
    // Track upload progress
    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            progressFill.style.width = percentComplete + '%';
            progressText.textContent = `Uploading: ${Math.round(percentComplete)}%`;
        }
    });
    
    // Handle upload completion
    xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
            const response = JSON.parse(xhr.responseText);
            progressText.textContent = 'Upload complete! Starting detection...';
            setTimeout(() => {
                uploadProgress.style.display = 'none';
                startVideoFeed();
            }, 1000);
        } else {
            const response = JSON.parse(xhr.responseText);
            showStatus('Upload failed: ' + (response.error || 'Unknown error'), 'error');
            uploadProgress.style.display = 'none';
        }
    });
    
    // Handle errors
    xhr.addEventListener('error', () => {
        showStatus('Upload failed. Please try again.', 'error');
        uploadProgress.style.display = 'none';
    });
    
    // Send request
    xhr.open('POST', '/upload');
    xhr.send(formData);
}

// Connect to camera stream
connectBtn.addEventListener('click', () => {
    const url = cameraUrl.value.trim();
    
    if (!url) {
        showStreamStatus('Please enter a stream URL', 'error');
        return;
    }
    
    showStreamStatus('Connecting to stream...', 'success');
    
    fetch('/connect_stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showStreamStatus('Connected successfully!', 'success');
            setTimeout(() => {
                startVideoFeed();
            }, 1000);
        } else {
            showStreamStatus('Connection failed: ' + data.error, 'error');
        }
    })
    .catch(error => {
        showStreamStatus('Connection failed: ' + error.message, 'error');
    });
});

// Start video feed display
function startVideoFeed() {
    uploadSection.style.display = 'none';
    videoSection.style.display = 'block';
    historySection.style.display = 'block';
    
    // Set video feed source
    videoFeed.src = '/video_feed?t=' + new Date().getTime();
    
    // Start stats update interval
    startStatsUpdate();
    
    // Load history
    loadHistory();
}

// Stop video processing
stopBtn.addEventListener('click', () => {
    fetch('/stop')
    .then(response => response.json())
    .then(data => {
        videoFeed.src = '';
        videoSection.style.display = 'none';
        historySection.style.display = 'none';
        uploadSection.style.display = 'block';
        
        // Reset stats
        totalSpots.textContent = '0';
        occupiedSpots.textContent = '0';
        availableSpots.textContent = '0';
        occupancyRate.textContent = '0%';
        
        // Clear intervals
        if (window.statsInterval) {
            clearInterval(window.statsInterval);
        }
    })
    .catch(error => {
        console.error('Error stopping video:', error);
    });
});

// Update statistics
function startStatsUpdate() {
    // Clear existing interval if any
    if (window.statsInterval) {
        clearInterval(window.statsInterval);
    }
    
    // Update stats every 2 seconds
    window.statsInterval = setInterval(() => {
        fetch('/stats')
        .then(response => response.json())
        .then(data => {
            if (data.error) return;
            
            totalSpots.textContent = data.total || 0;
            occupiedSpots.textContent = data.occupied || 0;
            availableSpots.textContent = data.available || 0;
            
            // Calculate occupancy rate
            const rate = data.total > 0 ? ((data.occupied / data.total) * 100).toFixed(1) : 0;
            occupancyRate.textContent = rate + '%';
        })
        .catch(error => {
            console.error('Error fetching stats:', error);
        });
    }, 2000);
}

// Load parking history
function loadHistory() {
    fetch('/history')
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            displayHistory(data.events, data.storage);
        }
    })
    .catch(error => {
        console.error('Error loading history:', error);
    });
}

// Display history in table
function displayHistory(events, storage) {
    const storageInfo = document.getElementById('storageInfo');
    const tableBody = document.getElementById('historyTableBody');
    
    // Show storage type
    storageInfo.textContent = `Storage: ${storage === 'postgresql' ? 'PostgreSQL Database' : 'Local CSV File'}`;
    
    // Clear table
    tableBody.innerHTML = '';
    
    if (events.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="3" class="no-data">No data available</td></tr>';
        return;
    }
    
    // Add rows
    events.reverse().forEach(event => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${event.id}</td>
            <td>${event.status}</td>
            <td>${event.timestamp}</td>
        `;
        tableBody.appendChild(row);
    });
}

// Refresh history button
refreshHistoryBtn.addEventListener('click', () => {
    loadHistory();
});

// Show status message
function showStatus(message, type) {
    // Create or update status element in drop zone
    let statusEl = dropZone.querySelector('.status-message');
    if (!statusEl) {
        statusEl = document.createElement('div');
        statusEl.className = 'status-message';
        dropZone.appendChild(statusEl);
    }
    
    statusEl.textContent = message;
    statusEl.className = 'status-message ' + type;
    
    // Remove after 5 seconds
    setTimeout(() => {
        statusEl.remove();
    }, 5000);
}

// Show stream status message
function showStreamStatus(message, type) {
    streamStatus.textContent = message;
    streamStatus.className = 'status-message ' + type;
    streamStatus.style.display = 'block';
    
    // Remove after 5 seconds
    setTimeout(() => {
        streamStatus.style.display = 'none';
    }, 5000);
}

// Initialize
console.log('SoloVision Parking Detection System initialized');
