# backend/app/smart_preprocessing.py

import cv2
import numpy as np
from pathlib import Path


def detect_image_type(image):
    """
    Detect if image is already preprocessed or needs preprocessing
    
    Args:
        image: numpy array (grayscale image)
    
    Returns:
        str: 'preprocessed', 'natural', or 'needs_preprocessing'
    """
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate statistics
    unique_values = len(np.unique(gray))
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    # Check if binary (only 2 values: 0 and 255 or close to it)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Find peaks in histogram
    peaks = []
    for i in range(256):
        if hist[i] > gray.size * 0.01:  # Peak if >1% of pixels
            peaks.append(i)
    
    # Decision logic
    if unique_values <= 10:
        # Very few unique values = likely already binary/preprocessed
        return 'preprocessed'
    
    elif len(peaks) <= 2 and std_val < 50:
        # Two distinct peaks (black & white) with low variance = binary
        return 'preprocessed'
    
    elif mean_val > 200 and std_val < 30:
        # Very bright with low variance = likely overprocessed/washed out
        return 'preprocessed'
    
    elif std_val > 50 and unique_values > 50:
        # High variance, many values = natural/unprocessed image
        return 'natural'
    
    else:
        # Default: needs preprocessing
        return 'needs_preprocessing'


def smart_preprocess_image(image_path, target_height=100, crop_height=64,
                           block_size=35, C=6, denoise_strength=25,
                           force_preprocess=False):
    """
    Smart preprocessing that adapts to input image type
    
    Args:
        image_path: Path to input image
        target_height: Height to resize before cropping
        crop_height: Final height after crop
        block_size: Adaptive threshold block size
        C: Adaptive threshold constant
        denoise_strength: Denoising strength
        force_preprocess: If True, always preprocess (ignore detection)
    
    Returns:
        Preprocessed image (numpy array)
    """
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image from {image_path}")
    
    h, w = image.shape[:2]
    
    # Step 1: Resize to target height maintaining aspect ratio
    new_width = int(w * (target_height / h))
    image = cv2.resize(image, (new_width, target_height))
    
    # Step 2: Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Step 3: Detect image type
    if not force_preprocess:
        image_type = detect_image_type(gray)
        print(f"📊 Detected image type: {image_type}")
    else:
        image_type = 'needs_preprocessing'
        print(f"⚙️ Force preprocessing enabled")
    
    # Step 4: Apply appropriate preprocessing based on type
    if image_type == 'preprocessed':
        print("✓ Image already preprocessed, applying minimal processing")
        processed = apply_minimal_preprocessing(gray, crop_height, target_height)
    
    elif image_type == 'natural':
        print("✓ Natural image detected, applying full preprocessing")
        processed = apply_full_preprocessing(
            gray, crop_height, target_height, 
            block_size, C, denoise_strength
        )
    
    else:  # needs_preprocessing
        print("✓ Applying standard preprocessing")
        processed = apply_full_preprocessing(
            gray, crop_height, target_height,
            block_size, C, denoise_strength
        )
    
    return processed


def apply_minimal_preprocessing(gray, crop_height, target_height):
    """
    Minimal preprocessing for already preprocessed images
    Just resize and crop, no thresholding or denoising
    """
    
    # Ensure binary (in case it's close but not exactly 0/255)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Check if inverted (black background, white text)
    mean_val = np.mean(binary)
    if mean_val < 127:  # More black than white = inverted
        print("  ↻ Inverting image (black background detected)")
        binary = cv2.bitwise_not(binary)
    
    # Crop vertically (center crop)
    if crop_height < target_height:
        center_y = target_height // 2
        y1 = max(0, center_y - crop_height // 2)
        y2 = min(target_height, center_y + crop_height // 2)
        cropped = binary[y1:y2, :]
    else:
        cropped = binary
    
    return cropped


def apply_full_preprocessing(gray, crop_height, target_height,
                             block_size, C, denoise_strength):
    """
    Full preprocessing pipeline for natural/raw images
    """
    
    # Check if image needs inversion first
    mean_val = np.mean(gray)
    if mean_val < 127:  # Dark image = might have white text on black
        print("  ↻ Inverting grayscale (dark background detected)")
        gray = cv2.bitwise_not(gray)
    
    # Denoise (preserves edges while removing noise)
    denoised = cv2.fastNlMeansDenoising(gray, None, denoise_strength, 4, 25)
    
    # Crop vertically (center crop)
    if crop_height < target_height:
        center_y = target_height // 2
        y1 = max(0, center_y - crop_height // 2)
        y2 = min(target_height, center_y + crop_height // 2)
        cropped = denoised[y1:y2, :]
    else:
        cropped = denoised
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        cropped, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, block_size, C
    )
    
    # Light morphology
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return cleaned


def preprocess_with_fallback(image_path, target_height=100, crop_height=64):
    """
    Preprocess with fallback strategies if smart detection fails
    
    Tries 3 strategies:
    1. Smart preprocessing (auto-detect)
    2. Force full preprocessing
    3. Force minimal preprocessing
    
    Returns the one that produces best contrast
    """
    
    results = []
    
    # Strategy 1: Smart (auto-detect)
    try:
        img1 = smart_preprocess_image(image_path, target_height, crop_height, 
                                      force_preprocess=False)
        contrast1 = calculate_contrast(img1)
        results.append(('smart', img1, contrast1))
        print(f"  Smart preprocessing: contrast = {contrast1:.2f}")
    except Exception as e:
        print(f"  Smart preprocessing failed: {e}")
    
    # Strategy 2: Force full preprocessing
    try:
        img2 = smart_preprocess_image(image_path, target_height, crop_height,
                                      force_preprocess=True)
        contrast2 = calculate_contrast(img2)
        results.append(('full', img2, contrast2))
        print(f"  Full preprocessing: contrast = {contrast2:.2f}")
    except Exception as e:
        print(f"  Full preprocessing failed: {e}")
    
    # Strategy 3: Minimal (assume already preprocessed)
    try:
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        new_width = int(w * (target_height / h))
        image = cv2.resize(image, (new_width, target_height))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        img3 = apply_minimal_preprocessing(gray, crop_height, target_height)
        contrast3 = calculate_contrast(img3)
        results.append(('minimal', img3, contrast3))
        print(f"  Minimal preprocessing: contrast = {contrast3:.2f}")
    except Exception as e:
        print(f"  Minimal preprocessing failed: {e}")
    
    # Pick best result (highest contrast)
    if not results:
        raise ValueError("All preprocessing strategies failed!")
    
    best_strategy, best_image, best_contrast = max(results, key=lambda x: x[2])
    print(f"✓ Selected strategy: {best_strategy} (contrast: {best_contrast:.2f})")
    
    return best_image


def calculate_contrast(image):
    """
    Calculate image contrast (higher = better for OCR)
    
    Args:
        image: grayscale numpy array
    
    Returns:
        float: contrast score
    """
    # Standard deviation of pixel values
    # Higher std = more contrast between text and background
    return np.std(image)


def analyze_image_quality(image):
    """
    Analyze image quality for debugging
    
    Args:
        image: grayscale numpy array
    
    Returns:
        dict: Quality metrics
    """
    
    unique_values = len(np.unique(image))
    mean_val = np.mean(image)
    std_val = np.std(image)
    
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # Find histogram peaks
    peaks = []
    for i in range(256):
        if hist[i] > image.size * 0.01:
            peaks.append((i, int(hist[i][0])))
    
    # Text-to-background ratio
    dark_pixels = np.sum(image < 127)
    light_pixels = np.sum(image >= 127)
    text_bg_ratio = dark_pixels / (light_pixels + 1e-6)
    
    return {
        'unique_values': unique_values,
        'mean': float(mean_val),
        'std': float(std_val),
        'contrast': float(std_val),
        'peaks': peaks,
        'text_bg_ratio': float(text_bg_ratio),
        'is_binary': unique_values <= 10,
        'is_inverted': mean_val < 127
    }




def preprocess_for_production(image_path, output_path=None, 
                               save_debug=False, debug_dir=None):
    """
    Production-ready preprocessing with all safeguards
    
    Args:
        image_path: Input image path
        output_path: Where to save preprocessed image (optional)
        save_debug: If True, save intermediate steps
        debug_dir: Directory to save debug images
    
    Returns:
        preprocessed image (numpy array)
    """
    
    print(f"\n{'='*70}")
    print(f"SMART PREPROCESSING: {Path(image_path).name}")
    print(f"{'='*70}")
    
    # Step 1: Load and analyze
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    print(f"📏 Original size: {image.shape[1]}×{image.shape[0]} pixels")
    
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Analyze quality
    quality = analyze_image_quality(gray)
    print(f"  Image analysis:")
    print(f"   Unique values: {quality['unique_values']}")
    print(f"   Mean brightness: {quality['mean']:.1f}/255")
    print(f"   Contrast (std): {quality['contrast']:.1f}")
    print(f"   Binary: {quality['is_binary']}")
    print(f"   Inverted: {quality['is_inverted']}")
    print("Yeha samma chaldai xa")
    # Step 2: Preprocess with fallback
    try:
        processed = preprocess_with_fallback(image_path)
    except Exception as e:
        print(f"OOPS!! All preprocessing failed: {e}")
        raise
    
    # Step 3: Validate output
    output_quality = analyze_image_quality(processed)
    print(f"\n✓ Preprocessing complete")
    print(f"Output size: {processed.shape[1]}×{processed.shape[0]} pixels")
    print(f"Output contrast: {output_quality['contrast']:.1f}")
    
    # Warning if contrast is too low
    if output_quality['contrast'] < 30:
        print(f"Warning: Low contrast ({output_quality['contrast']:.1f}), OCR might struggle")
    
    # Step 4: Save output
    if output_path:
        cv2.imwrite(output_path, processed)
        print(f" Saved to: {output_path}")
    
    # Step 5: Save debug images
    if save_debug and debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(exist_ok=True, parents=True)
        
        base_name = Path(image_path).stem
        
        # Save originalS
        cv2.imwrite(str(debug_dir / f"{base_name}_1_original.png"), gray)
        
        # Save final
        cv2.imwrite(str(debug_dir / f"{base_name}_2_preprocessed.png"), processed)
        
        print(f"Debug images saved to: {debug_dir}")
    
    print(f"{'='*70}\n")
    
    return processed


# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_preprocessing_on_samples(image_paths):
    """
    Test preprocessing on multiple sample images
    
    Args:
        image_paths: List of image paths to test
    """
    
    print("\n" + "="*70)
    print("TESTING SMART PREPROCESSING")
    print("="*70 + "\n")
    
    results = []
    
    for img_path in image_paths:
        try:
            print(f"\nTesting: {img_path}")
            processed = preprocess_for_production(img_path, save_debug=False)
            
            quality = analyze_image_quality(processed)
            results.append({
                'image': Path(img_path).name,
                'success': True,
                'contrast': quality['contrast'],
                'binary': quality['is_binary']
            })
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                'image': Path(img_path).name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['image']}")
        if result['success']:
            print(f"   Contrast: {result['contrast']:.1f}")
    
    success_rate = sum(1 for r in results if r['success']) / len(results) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    test_images = [
        "test_images/natural.png",        # Natural photo
        "test_images/preprocessed.png",   # Already preprocessed
        "test_images/scanned.jpg",        # Scanned document
    ]
    
    test_preprocessing_on_samples(test_images)