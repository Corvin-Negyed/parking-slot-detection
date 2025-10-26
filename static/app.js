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
const analyticsSection = document.getElementById('analyticsSection');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const streamStatus = document.getElementById('streamStatus');
const videoFeed = document.getElementById('videoFeed');
const stopBtn = document.getElementById('stopBtn');
const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const refreshAnalyticsBtn = document.getElementById('refreshAnalyticsBtn');

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
    analyticsSection.style.display = 'block';
    
    // Set video feed source
    videoFeed.src = '/video_feed?t=' + new Date().getTime();
    
    // Start stats update interval
    startStatsUpdate();
    
    // Load history and analytics
    loadHistory();
    loadAnalytics();
}

// Stop video processing
stopBtn.addEventListener('click', () => {
    fetch('/stop')
    .then(response => response.json())
    .then(data => {
        videoFeed.src = '';
        videoSection.style.display = 'none';
        historySection.style.display = 'none';
        analyticsSection.style.display = 'none';
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
            
            // Calculate and display occupancy rate
            const rate = data.total > 0 ? ((data.occupied / data.total) * 100).toFixed(1) : 0;
            occupancyRate.textContent = rate + '%';
        })
        .catch(error => {
            console.error('Error fetching stats:', error);
        });
    }, 2000);
}

// Load vehicle detection history
function loadHistory() {
    fetch('/history')
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            displayHistory(data.detections, data.storage);
        }
    })
    .catch(error => {
        console.error('Error loading history:', error);
    });
}

// Display history in table
function displayHistory(detections, storage) {
    const storageInfo = document.getElementById('storageInfo');
    const tableBody = document.getElementById('historyTableBody');
    
    // Show storage type
    storageInfo.textContent = `Storage: ${storage === 'postgresql' ? 'PostgreSQL Database' : 'Local CSV File'}`;
    
    // Clear table
    tableBody.innerHTML = '';
    
    if (detections.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="6" class="no-data">No data available</td></tr>';
        return;
    }
    
    // Add rows (already sorted by timestamp DESC from backend)
    detections.forEach(detection => {
        const row = document.createElement('tr');
        const rate = detection.occupancy_rate ? detection.occupancy_rate.toFixed(1) : '0.0';
        row.innerHTML = `
            <td>${detection.id}</td>
            <td>${detection.total_spots || 0}</td>
            <td>${detection.occupied || 0}</td>
            <td>${detection.available || 0}</td>
            <td>${rate}%</td>
            <td>${detection.timestamp}</td>
        `;
        tableBody.appendChild(row);
    });
}

// Refresh history button
refreshHistoryBtn.addEventListener('click', () => {
    loadHistory();
});

// Clear history button
clearHistoryBtn.addEventListener('click', () => {
    if (confirm('Clear all history data?')) {
        fetch('/clear_history', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                loadHistory();
                loadAnalytics();
                
                // Clear analytics display
                document.getElementById('overallStats').innerHTML = 'No data available';
                document.getElementById('trendAnalysis').innerHTML = 'No data available';
                document.getElementById('peakHours').innerHTML = 'No data available';
                document.getElementById('dailySummary').innerHTML = 'No data available';
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});

// Load analytics
function loadAnalytics() {
    fetch('/analytics')
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success' && data.data) {
            displayAnalytics(data.data);
        }
    })
    .catch(error => {
        console.error('Error loading analytics:', error);
    });
}

// Display analytics
function displayAnalytics(analytics) {
    // Overall Stats
    if (analytics.overview) {
        const stats = analytics.overview.overall_stats;
        document.getElementById('overallStats').innerHTML = `
            <p><strong>Total Records:</strong> ${analytics.overview.total_records}</p>
            <p><strong>Average Vehicles:</strong> ${stats.average}</p>
            <p><strong>Maximum:</strong> ${stats.max}</p>
            <p><strong>Minimum:</strong> ${stats.min}</p>
            <p><strong>Time Span:</strong> ${analytics.overview.time_span.start} to ${analytics.overview.time_span.end}</p>
        `;
    }
    
    // Trend Analysis
    if (analytics.trend) {
        const trend = analytics.trend;
        const trendClass = trend.trend === 'increasing' ? 'trend-up' : trend.trend === 'decreasing' ? 'trend-down' : 'trend-stable';
        document.getElementById('trendAnalysis').innerHTML = `
            <p class="${trendClass}"><strong>Trend:</strong> ${trend.trend.toUpperCase()}</p>
            <p><strong>Change:</strong> ${trend.change_percent}%</p>
            <p><strong>First Half Avg:</strong> ${trend.first_half_avg}</p>
            <p><strong>Second Half Avg:</strong> ${trend.second_half_avg}</p>
        `;
    }
    
    // Peak Hours with recommendations
    if (analytics.peak_hours) {
        let peakHTML = '<div class="peak-recommendation"><p><strong>Traffic Analysis:</strong></p><ul>';
        analytics.peak_hours.forEach((peak, index) => {
            const timeRange = `${String(peak.hour).padStart(2, '0')}:00 - ${String(peak.hour + 1).padStart(2, '0')}:00`;
            if (index === 0) {
                peakHTML += `<li>🔴 <strong>Peak Hour:</strong> ${timeRange} (Avg: ${peak.average} vehicles) - Avoid this time</li>`;
            } else if (index === analytics.peak_hours.length - 1) {
                peakHTML += `<li>🟢 <strong>Best Time:</strong> ${timeRange} (Avg: ${peak.average} vehicles) - Recommended!</li>`;
            } else {
                peakHTML += `<li>🟡 ${timeRange}: ${peak.average} vehicles</li>`;
            }
        });
        peakHTML += '</ul></div>';
        
        peakHTML += '<table class="analytics-table"><thead><tr><th>Time</th><th>Average Vehicles</th><th>Maximum</th><th>Data Points</th></tr></thead><tbody>';
        analytics.peak_hours.forEach(peak => {
            const timeRange = `${String(peak.hour).padStart(2, '0')}:00 - ${String(peak.hour + 1).padStart(2, '0')}:00`;
            peakHTML += `
                <tr>
                    <td><strong>${timeRange}</strong></td>
                    <td>${peak.average}</td>
                    <td>${peak.max}</td>
                    <td>${peak.samples}</td>
                </tr>
            `;
        });
        peakHTML += '</tbody></table>';
        document.getElementById('peakHours').innerHTML = peakHTML;
    }
    
    // Daily Summary
    if (analytics.daily_summary) {
        let dailyHTML = '<table class="analytics-table"><thead><tr><th>Date</th><th>Average</th><th>Max</th><th>Min</th><th>Samples</th></tr></thead><tbody>';
        analytics.daily_summary.forEach(day => {
            dailyHTML += `
                <tr>
                    <td>${day.date}</td>
                    <td>${day.average}</td>
                    <td>${day.max}</td>
                    <td>${day.min}</td>
                    <td>${day.total_samples}</td>
                </tr>
            `;
        });
        dailyHTML += '</tbody></table>';
        document.getElementById('dailySummary').innerHTML = dailyHTML;
    }
    
    // Draw hourly chart
    drawHourlyChart(analytics.peak_hours || []);
}

// Draw hourly distribution chart (simple bar chart with canvas)
function drawHourlyChart(peakHours) {
    const canvas = document.getElementById('hourlyChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = 300;
    
    if (peakHours.length === 0) {
        ctx.fillText('No data available', canvas.width / 2, canvas.height / 2);
        return;
    }
    
    // Simple bar chart
    const padding = 40;
    const barWidth = (canvas.width - 2 * padding) / 24;
    const maxValue = Math.max(...peakHours.map(h => h.average));
    
    // Draw axes
    ctx.strokeStyle = '#666';
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();
    
    // Draw bars for all 24 hours
    for (let hour = 0; hour < 24; hour++) {
        const hourData = peakHours.find(h => h.hour === hour) || { average: 0 };
        const barHeight = (hourData.average / maxValue) * (canvas.height - 2 * padding);
        const x = padding + hour * barWidth;
        const y = canvas.height - padding - barHeight;
        
        ctx.fillStyle = '#667eea';
        ctx.fillRect(x, y, barWidth - 2, barHeight);
        
        // Draw hour labels (every 3 hours)
        if (hour % 3 === 0) {
            ctx.fillStyle = '#333';
            ctx.font = '10px Arial';
            ctx.fillText(hour + 'h', x, canvas.height - padding + 15);
        }
    }
}

// Refresh analytics button
refreshAnalyticsBtn.addEventListener('click', () => {
    loadAnalytics();
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
