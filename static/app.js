// Get DOM elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');

// Browse button click handler
browseBtn.addEventListener('click', () => {
    fileInput.click();
});

