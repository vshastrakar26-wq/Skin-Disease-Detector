// Get necessary elements once
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const previewImg = document.getElementById('preview');

// --- 1. Basic Click/File Selection (Improved) ---

// Clicking the drop area triggers the hidden file input
dropArea.addEventListener('click', () => {
    fileInput.click();
});

// Handling file selection change
fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        previewImage(file);
    }
});

// --- 2. Drag and Drop Functionality ---

// Prevent default drag behavior (necessary to allow drop)
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false); // For safety
});

// Highlight drop area when dragging file over it
['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => {
        dropArea.classList.add('highlight');
    }, false);
});

// Remove highlight when drag leaves
['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => {
        dropArea.classList.remove('highlight');
    }, false);
});

// Handle the dropped files
dropArea.addEventListener('drop', (event) => {
    // Get the file list from the drop event
    const dt = event.dataTransfer;
    const file = dt.files[0];

    if (file) {
        handleFile(file);
    }
}, false);


// --- 3. Paste Functionality (Kept and Integrated) ---

document.addEventListener('paste', (event) => {
    const items = (event.clipboardData || event.originalEvent.clipboardData).items;
    for (const item of items) {
        // Check if the pasted item is a file (image)
        if (item.kind === 'file') {
            const file = item.getAsFile();
            if (file) {
                // Ensure it is an image type before processing
                if (file.type.startsWith('image/')) {
                    handleFile(file);
                }
            }
            break;
        }
    }
});

// --- Core Helper Functions ---

/** Prevents default behavior (like opening the file in the browser) */
function preventDefaults(event) {
    event.preventDefault();
    event.stopPropagation();
}

/** Previews the image and sets the file in the hidden input for submission. */
function handleFile(file) {
    previewImage(file);

    // CRITICAL: Set the file to the input field so the form can submit it
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;
}

/** Reads the file and updates the image preview element. */
function previewImage(file) {
    const reader = new FileReader();
    reader.onload = (event) => {
        previewImg.src = event.target.result;
        previewImg.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// NOTE: You should add this CSS class to your <style> block for highlighting:
// .highlight {
//     background-color: #E0F7FA !important;
//     border-color: #005bb5 !important;
//     box-shadow: 0 0 15px rgba(25, 118, 210, 0.5);
// }