"""
Validators for uploaded network dataset files.
"""

import os
from django.core.exceptions import ValidationError

ALLOWED_EXTENSIONS = ['.csv']
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

def validate_dataset_file(file):
    """
    Validate file extension and size to ensure secure file handling.
    """
    ext = os.path.splitext(file.name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(f"Unsupported file format '{ext}'. Only standard CSV files (.csv) are accepted.")

    if file.size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        raise ValidationError(f"File size exceeds maximum threshold of {max_mb} MB.")

    return file
