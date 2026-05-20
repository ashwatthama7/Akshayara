import os
import uuid
from pathlib import Path


def generate_unique_filename(original_filename):
    """
    Generate unique filename to avoid collisions
    
    Args:
        original_filename: Original uploaded filename
    
    Returns:
        Unique filename with UUID
    """
    ext = Path(original_filename).suffix
    unique_id = str(uuid.uuid4())[:8]
    return f"{unique_id}{ext}"


def cleanup_temp_files(file_paths):
    """
    Clean up temporary files
    
    Args:
        file_paths: List of file paths to delete
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not delete {file_path}: {e}")


def ensure_directory_exists(directory):
    """
    Ensure directory exists, create if not
    
    Args:
        directory: Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)