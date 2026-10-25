/**
 * Zero-Day GraphIDS Interactive Utilities
 */

document.addEventListener('DOMContentLoaded', function() {
    // Enable Bootstrap tooltips if bootstrap is loaded
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(el) {
            return new bootstrap.Tooltip(el);
        });
    }

    // Auto-scroll console output to bottom
    const consoleBox = document.querySelector('.console-output');
    if (consoleBox) {
        consoleBox.scrollTop = consoleBox.scrollHeight;
    }
});
