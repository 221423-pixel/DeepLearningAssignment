document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('dropZone');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('previewContainer');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    const removeBtn = document.getElementById('removeBtn');
    const predictBtn = document.getElementById('predictBtn');
    const resultCard = document.getElementById('resultCard');
    const actionName = document.getElementById('actionName');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const detailsToggle = document.getElementById('detailsToggle');
    const detailsContent = document.getElementById('detailsContent');
    const toggleIcon = document.getElementById('toggleIcon');
    const probabilityList = document.getElementById('probabilityList');

    // Drag and Drop Logic
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('drag-over');
    }

    function unhighlight(e) {
        dropZone.classList.remove('drag-over');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // Click to upload
    dropZone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
            imageInput.click();
        }
    });

    imageInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (!file.type.startsWith('image/')) {
                alert('Please upload an image file.');
                return;
            }

            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove('hidden');
                uploadPlaceholder.classList.add('hidden');
                predictBtn.disabled = false;
                resultCard.classList.add('hidden'); // Hide previous results
            }
            reader.readAsDataURL(file);
        }
    }

    // Remove Image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        imageInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        uploadPlaceholder.classList.remove('hidden');
        predictBtn.disabled = true;
        resultCard.classList.add('hidden');
    });

    // Prediction Logic
    predictBtn.addEventListener('click', async () => {
        const file = imageInput.files[0];
        if (!file) return;

        // UI Updates
        loadingOverlay.classList.remove('hidden');
        resultCard.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            
            // Display Results
            actionName.textContent = data.action;
            const confidencePercent = (data.confidence * 100).toFixed(1);
            confidenceValue.textContent = confidencePercent + '%';
            
            // Animate progress bar
            setTimeout(() => {
                confidenceBar.style.width = confidencePercent + '%';
            }, 100);
            
            // Populate details
            probabilityList.innerHTML = '';
            const sortedProbs = Object.entries(data.all_predictions)
                .sort(([,a], [,b]) => b - a);
                
            sortedProbs.forEach(([action, prob]) => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <span class="prob-label">${action}</span>
                    <span class="prob-val">${(prob * 100).toFixed(2)}%</span>
                `;
                probabilityList.appendChild(li);
            });

            resultCard.classList.remove('hidden');

            // Scroll to results on mobile
            if (window.innerWidth < 768) {
                resultCard.scrollIntoView({ behavior: 'smooth' });
            }

        } catch (error) {
            alert('Error predicting action: ' + error.message);
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    });

    // Accordion Logic
    detailsToggle.addEventListener('click', () => {
        const isExpanded = detailsContent.style.maxHeight;
        
        if (isExpanded) {
            detailsContent.style.maxHeight = null;
            toggleIcon.classList.remove('fa-chevron-up');
            toggleIcon.classList.add('fa-chevron-down');
        } else {
            detailsContent.style.maxHeight = detailsContent.scrollHeight + "px";
            toggleIcon.classList.remove('fa-chevron-down');
            toggleIcon.classList.add('fa-chevron-up');
        }
    });
});
