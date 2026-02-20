// Configuration
const API_BASE_URL = 'http://localhost:5500';
const API_ENDPOINT = `${API_BASE_URL}/api/v1/tryon/process`;

// State
let userImage = null;
let selectedCloth = null;
let selectedSize = null;
let clothData = [];
let currentViewMode = 'compare';
let cameraStream = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const userImageInput = document.getElementById('userImageInput');
const userPreview = document.getElementById('userPreview');
const userImageEl = document.getElementById('userImage');
const removeUserImageBtn = document.getElementById('removeUserImage');

const cameraBtn = document.getElementById('cameraBtn');
const cameraContainer = document.getElementById('cameraContainer');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const captureBtn = document.getElementById('captureBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');

const clothGrid = document.getElementById('clothGrid');
const customClothBtn = document.getElementById('customClothBtn');
const customClothInput = document.getElementById('customClothInput');

const tryOnBtn = document.getElementById('tryOnBtn');
const resultSection = document.getElementById('resultSection');
const resultCanvas = document.getElementById('resultCanvas');
const loadingOverlay = document.getElementById('loadingOverlay');
const toastContainer = document.getElementById('toastContainer');

const downloadBtn = document.getElementById('downloadBtn');
const shareBtn = document.getElementById('shareBtn');
const tryAgainBtn = document.getElementById('tryAgainBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeUpload();
    initializeCamera();
    initializeClothSelection();
    initializeViewModes();
    initializeActions();
    loadClothData();
});

// Upload Functionality
function initializeUpload() {
    // Click to upload
    uploadArea.addEventListener('click', () => {
        if (!userImage) {
            userImageInput.click();
        }
    });

    userImageInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Remove image
    removeUserImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearUserImage();
    });
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file
    if (!file.type.startsWith('image/')) {
        showToast('Please select an image file', 'error');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showToast('File size must be less than 10MB', 'error');
        return;
    }

    // Read and display
    const reader = new FileReader();
    reader.onload = (e) => {
        userImage = file;
        userImageEl.src = e.target.result;
        userPreview.style.display = 'block';
        uploadArea.querySelector('.upload-content').style.display = 'none';
        checkReadyState();
        showToast('Photo uploaded successfully', 'success');
    };
    reader.readAsDataURL(file);
}

function clearUserImage() {
    userImage = null;
    userImageEl.src = '';
    userPreview.style.display = 'none';
    uploadArea.querySelector('.upload-content').style.display = 'block';
    userImageInput.value = '';
    checkReadyState();
}

// Camera Functionality
function initializeCamera() {
    cameraBtn.addEventListener('click', toggleCamera);
    captureBtn.addEventListener('click', capturePhoto);
    closeCameraBtn.addEventListener('click', closeCamera);
}

async function toggleCamera() {
    if (cameraStream) {
        closeCamera();
    } else {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'user' } 
            });
            cameraVideo.srcObject = cameraStream;
            cameraContainer.style.display = 'block';
            showToast('Camera activated', 'success');
        } catch (error) {
            showToast('Camera access denied', 'error');
            console.error('Camera error:', error);
        }
    }
}

function capturePhoto() {
    const context = cameraCanvas.getContext('2d');
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    context.drawImage(cameraVideo, 0, 0);
    
    cameraCanvas.toBlob((blob) => {
        const file = new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' });
        handleFile(file);
        closeCamera();
    }, 'image/jpeg', 0.9);
}

function closeCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
        cameraContainer.style.display = 'none';
    }
}

// Cloth Selection
function initializeClothSelection() {
    // Category filters
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            filterClothes(btn.dataset.category);
        });
    });

    // Custom cloth upload
    customClothBtn.addEventListener('click', () => customClothInput.click());
    customClothInput.addEventListener('change', handleCustomCloth);
}

async function loadClothData() {
    try {
        const response = await fetch('/assets/clothes/clothes.json');
        if (!response.ok) {
            throw new Error('Failed to load clothes data');
        }
        clothData = await response.json();
        console.log('Loaded clothes:', clothData.length);
        renderClothes(clothData);
    } catch (error) {
        console.error('Error loading clothes:', error);
        showToast('Failed to load clothing items', 'error');
        // Fallback to empty array
        clothData = [];
        renderClothes(clothData);
    }
}

function renderClothes(clothes) {
    clothGrid.innerHTML = '';
    
    clothes.forEach(cloth => {
        const item = document.createElement('div');
        item.className = 'cloth-item';
        item.dataset.id = cloth.id;
        
        const genderBadge = cloth.gender ? `<span class="cloth-gender">${cloth.gender}</span>` : '';
        
        // Handle both regular images and base64 data URLs
        const imageSrc = cloth.image.startsWith('data:') ? cloth.image : `/assets/clothes/${cloth.image}`;
        
        item.innerHTML = `
            <img src="${imageSrc}" alt="${cloth.name}" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22250%22 height=%22250%22%3E%3Crect fill=%22%23f0f0f0%22 width=%22250%22 height=%22250%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22 fill=%22%23999%22 font-size=%2220%22%3E${cloth.name}%3C/text%3E%3C/svg%3E'">
            <div class="cloth-info">
                <div class="cloth-name">${cloth.name}</div>
                <div class="cloth-description">${cloth.description}</div>
                ${genderBadge}
            </div>
        `;
        
        item.addEventListener('click', () => selectCloth(cloth, item));
        clothGrid.appendChild(item);
    });
}

function filterClothes(category) {
    const filtered = category === 'all' 
        ? clothData 
        : clothData.filter(c => c.category === category);
    renderClothes(filtered);
}

function selectCloth(cloth, element) {
    // Remove previous selection
    document.querySelectorAll('.cloth-item').forEach(item => {
        item.classList.remove('selected');
        const badge = item.querySelector('.selected-badge');
        if (badge) badge.remove();
    });
    
    // Select new cloth
    element.classList.add('selected');
    
    // Add selected badge
    const badge = document.createElement('div');
    badge.className = 'selected-badge';
    badge.textContent = '✓ Selected';
    element.appendChild(badge);
    
    // Store cloth with full image path
    const clothWithFullPath = {
        ...cloth,
        image: `/assets/clothes/${cloth.image}`
    };
    
    // Show size modal if sizes available
    if (cloth.sizes) {
        showSizeModal(clothWithFullPath);
    } else {
        selectedCloth = clothWithFullPath;
        selectedSize = null;
        checkReadyState();
        showToast(`Selected: ${cloth.name}`, 'info');
    }
}

function handleCustomCloth(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const customCloth = {
                id: 'custom',
                name: 'Custom Cloth',
                category: 'custom',
                image: e.target.result,
                file: file
            };
            
            // Add to grid
            clothData.unshift(customCloth);
            renderClothes(clothData);
            
            // Auto-select
            const firstItem = clothGrid.querySelector('.cloth-item');
            selectCloth(customCloth, firstItem);
        };
        reader.readAsDataURL(file);
    }
}

// Try-On Processing
function checkReadyState() {
    tryOnBtn.disabled = !(userImage && selectedCloth);
}

tryOnBtn.addEventListener('click', processTryOn);

async function processTryOn() {
    if (!userImage || !selectedCloth) return;
    
    showLoading(true);
    
    try {
        // Check if API mode is enabled
        const useAPI = document.getElementById('useAPIToggle')?.checked || false;
        
        // Prepare form data
        const formData = new FormData();
        formData.append('user_image', userImage);
        
        // Get cloth image
        if (selectedCloth.file) {
            formData.append('cloth_image', selectedCloth.file);
        } else {
            // Fetch cloth image from URL
            const response = await fetch(selectedCloth.image);
            
            // Check if fetch was successful
            if (!response.ok) {
                throw new Error(`Failed to load clothing image: ${selectedCloth.name}. Please ensure the image file exists.`);
            }
            
            const clothBlob = await response.blob();
            
            // Verify it's actually an image
            if (!clothBlob.type.startsWith('image/')) {
                throw new Error(`Invalid image file for: ${selectedCloth.name}. Got ${clothBlob.type} instead of image.`);
            }
            
            // Get proper extension from URL
            const ext = selectedCloth.image.split('.').pop().split('?')[0] || 'png';
            formData.append('cloth_image', clothBlob, `cloth.${ext}`);
        }
        
        // Add use_api as form field
        formData.append('use_api', useAPI.toString());
        
        // Call API
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Display result
        displayResult(result);
        showToast('Try-on complete!', 'success');
        
    } catch (error) {
        console.error('Try-on error:', error);
        showToast('Try-on failed. Please try again.', 'error');
    } finally {
        showLoading(false);
    }
}

function displayResult(result) {
    // Store original and result images for comparison
    window.originalImage = userImageEl.src;
    window.resultImageUrl = `${API_BASE_URL}${result.image_url}`;
    
    // Load result image
    const resultImage = new Image();
    resultImage.onload = () => {
        const ctx = resultCanvas.getContext('2d');
        
        // Default view mode: side-by-side comparison
        if (currentViewMode === 'compare' || currentViewMode === 'side-by-side') {
            // Create side-by-side comparison
            const originalImg = new Image();
            originalImg.onload = () => {
                const width = Math.max(originalImg.width, resultImage.width);
                const height = Math.max(originalImg.height, resultImage.height);
                
                // Set canvas to show both images side by side
                resultCanvas.width = width * 2 + 40; // 40px gap
                resultCanvas.height = height + 60; // 60px for labels
                
                // Clear canvas
                ctx.fillStyle = '#1a1a2e';
                ctx.fillRect(0, 0, resultCanvas.width, resultCanvas.height);
                
                // Draw labels
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 20px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('BEFORE', width / 2, 30);
                ctx.fillText('AFTER', width * 1.5 + 40, 30);
                
                // Draw original image (left)
                const origX = (width - originalImg.width) / 2;
                ctx.drawImage(originalImg, origX, 50, originalImg.width, originalImg.height);
                
                // Draw result image (right)
                const resultX = width + 40 + (width - resultImage.width) / 2;
                ctx.drawImage(resultImage, resultX, 50, resultImage.width, resultImage.height);
                
                // Add divider line
                ctx.strokeStyle = '#e91e63';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(width + 20, 50);
                ctx.lineTo(width + 20, height + 50);
                ctx.stroke();
            };
            originalImg.src = window.originalImage;
        } else {
            // Result only view
            resultCanvas.width = resultImage.width;
            resultCanvas.height = resultImage.height;
            ctx.drawImage(resultImage, 0, 0);
        }
        
        resultSection.style.display = 'block';
        resultSection.scrollIntoView({ behavior: 'smooth' });
    };
    resultImage.src = window.resultImageUrl;
}

// View Modes
function initializeViewModes() {
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentViewMode = btn.dataset.mode;
            updateViewMode();
        });
    });
}

function updateViewMode() {
    // Re-render with new view mode if result exists
    if (window.resultImageUrl) {
        const resultImage = new Image();
        resultImage.onload = () => {
            const ctx = resultCanvas.getContext('2d');
            
            if (currentViewMode === 'compare' || currentViewMode === 'side-by-side') {
                // Redraw side-by-side
                const originalImg = new Image();
                originalImg.onload = () => {
                    const width = Math.max(originalImg.width, resultImage.width);
                    const height = Math.max(originalImg.height, resultImage.height);
                    
                    resultCanvas.width = width * 2 + 40;
                    resultCanvas.height = height + 60;
                    
                    ctx.fillStyle = '#1a1a2e';
                    ctx.fillRect(0, 0, resultCanvas.width, resultCanvas.height);
                    
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 20px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText('BEFORE', width / 2, 30);
                    ctx.fillText('AFTER', width * 1.5 + 40, 30);
                    
                    const origX = (width - originalImg.width) / 2;
                    ctx.drawImage(originalImg, origX, 50, originalImg.width, originalImg.height);
                    
                    const resultX = width + 40 + (width - resultImage.width) / 2;
                    ctx.drawImage(resultImage, resultX, 50, resultImage.width, resultImage.height);
                    
                    ctx.strokeStyle = '#e91e63';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.moveTo(width + 20, 50);
                    ctx.lineTo(width + 20, height + 50);
                    ctx.stroke();
                };
                originalImg.src = window.originalImage;
            } else {
                // Result only
                resultCanvas.width = resultImage.width;
                resultCanvas.height = resultImage.height;
                ctx.drawImage(resultImage, 0, 0);
            }
        };
        resultImage.src = window.resultImageUrl;
    }
}

// Actions
function initializeActions() {
    downloadBtn.addEventListener('click', downloadResult);
    shareBtn.addEventListener('click', shareResult);
    tryAgainBtn.addEventListener('click', resetApp);
}

function downloadResult() {
    const link = document.createElement('a');
    link.download = 'tryon-result.jpg';
    link.href = resultCanvas.toDataURL('image/jpeg', 0.9);
    link.click();
    showToast('Image downloaded', 'success');
}

async function shareResult() {
    try {
        const blob = await new Promise(resolve => resultCanvas.toBlob(resolve, 'image/jpeg', 0.9));
        const file = new File([blob], 'tryon-result.jpg', { type: 'image/jpeg' });
        
        if (navigator.share) {
            await navigator.share({
                title: 'My Virtual Try-On',
                text: 'Check out my virtual try-on result!',
                files: [file]
            });
            showToast('Shared successfully', 'success');
        } else {
            showToast('Sharing not supported on this device', 'warning');
        }
    } catch (error) {
        console.error('Share error:', error);
    }
}

function resetApp() {
    clearUserImage();
    selectedCloth = null;
    document.querySelectorAll('.cloth-item').forEach(item => {
        item.classList.remove('selected');
    });
    resultSection.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// UI Helpers
function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span>${getToastIcon(type)}</span>
        <span>${message}</span>
    `;
    
    toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideInRight 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function getToastIcon(type) {
    const icons = {
        success: '✓',
        error: '✕',
        warning: '⚠',
        info: 'ℹ'
    };
    return icons[type] || icons.info;
}

// Made with Bob

// Size Modal Functions
function showSizeModal(cloth) {
    const modal = document.getElementById('sizeModal');
    const sizeChart = document.getElementById('sizeChart');
    const sizeButtons = document.getElementById('sizeButtons');
    const confirmBtn = document.getElementById('confirmSize');
    
    // Clear previous content
    sizeChart.innerHTML = '';
    sizeButtons.innerHTML = '';
    
    // Populate size buttons
    Object.keys(cloth.sizes).forEach(size => {
        const btn = document.createElement('button');
        btn.className = 'size-btn';
        btn.textContent = size;
        btn.dataset.size = size;
        btn.addEventListener('click', () => selectSize(size, cloth));
        sizeButtons.appendChild(btn);
    });
    
    // Show modal
    modal.style.display = 'flex';
    
    // Close modal handlers
    document.getElementById('closeSizeModal').onclick = () => {
        modal.style.display = 'none';
    };
    
    modal.onclick = (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    };
    
    // Confirm size
    confirmBtn.onclick = () => {
        if (selectedSize) {
            selectedCloth = cloth;
            modal.style.display = 'none';
            checkReadyState();
            showToast(`Selected: ${cloth.name} - Size ${selectedSize}`, 'success');
        } else {
            showToast('Please select a size', 'error');
        }
    };
}

function selectSize(size, cloth) {
    selectedSize = size;
    
    // Update button states
    document.querySelectorAll('.size-btn').forEach(btn => {
        btn.classList.remove('selected');
    });
    event.target.classList.add('selected');
    
    // Display size chart
    const sizeChart = document.getElementById('sizeChart');
    const measurements = cloth.sizes[size];
    
    sizeChart.innerHTML = '<h4 style="margin-bottom: 1rem; color: var(--dark);">Size ' + size + ' Measurements (inches)</h4>';
    
    Object.entries(measurements).forEach(([key, value]) => {
        const row = document.createElement('div');
        row.className = 'size-chart-row';
        row.innerHTML = `
            <span class="size-chart-label">${key.charAt(0).toUpperCase() + key.slice(1)}:</span>
            <span class="size-chart-value">${value}"</span>
        `;
        sizeChart.appendChild(row);
    });
}
