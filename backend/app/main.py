# backend/app/main.py (UPDATED with smart preprocessing)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from pathlib import Path
import tempfile

from .predictor import OCRPredictor
from .smart_preprocessing import preprocess_for_production, analyze_image_quality
from .utils import generate_unique_filename, cleanup_temp_files, ensure_directory_exists


# Initialize FastAPI app
app = FastAPI(
    title="Akshayara OCR API",
    description="Nepali Handwritten Text Recognition API with Smart Preprocessing",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
DEBUG_DIR = BASE_DIR / "debug"  # For debugging preprocessing

# Ensure directories exist
ensure_directory_exists(UPLOADS_DIR)
ensure_directory_exists(DEBUG_DIR)

# Initialize OCR Predictor
MODEL_PATH = MODELS_DIR / "best_model.pth"
CHARSET_PATH = MODELS_DIR / "charset.txt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not CHARSET_PATH.exists():
    raise FileNotFoundError(f"Charset not found: {CHARSET_PATH}")

print("\n" + "="*70)
print("🚀 STARTING AKSHAYARA OCR API v2.0 (Smart Preprocessing)")
print("="*70)

predictor = OCRPredictor(
    model_path=str(MODEL_PATH),
    charset_path=str(CHARSET_PATH)
)

print("="*70)
print("✅ API READY WITH SMART PREPROCESSING")
print("="*70 + "\n")


@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "message": "Akshayara OCR API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Smart preprocessing (auto-detects image type)",
            "Handles preprocessed images",
            "Handles natural photos",
            "Automatic contrast optimization"
        ],
        "endpoints": {
            "predict": "/api/predict (POST)",
            "predict_debug": "/api/predict-debug (POST)",
            "analyze": "/api/analyze-image (POST)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(predictor.device),
        "smart_preprocessing": True
    }


@app.post("/api/predict")
async def predict_text(file: UploadFile = File(...)):
    """
    OCR Prediction endpoint with SMART PREPROCESSING
    
    Automatically detects if image is:
    - Already preprocessed (applies minimal processing)
    - Natural/raw image (applies full preprocessing)
    - Needs optimization (tries multiple strategies)
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON with predicted text, confidence, and preprocessing info
    """
    
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Validate file size (max 5MB)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 5MB limit"
        )
    
    temp_files = []
    
    try:
        # Save uploaded file
        unique_filename = generate_unique_filename(file.filename)
        upload_path = UPLOADS_DIR / f"upload_{unique_filename}"
        
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        temp_files.append(upload_path)
        
        print(f"📥 Received image: {file.filename}")
        print(f"   Size: {file_size / 1024:.2f} KB")
        
        # SMART PREPROCESSING (new!)
        preprocessed_path = UPLOADS_DIR / f"preprocessed_{unique_filename}"
        temp_files.append(preprocessed_path)
        
        print(f"🔧 Smart preprocessing...")
        
        # Preprocess with automatic type detection
        preprocessed_image = preprocess_for_production(
            str(upload_path),
            output_path=str(preprocessed_path),
            save_debug=False  # Set to True for debugging
        )
        
        print(f"✓ Image preprocessed successfully")
        
        # Predict text
        print(f"🔮 Running OCR prediction...")
        predicted_text, confidence = predictor.predict(str(preprocessed_path))
        
        print(f"✓ Prediction complete")
        print(f"   Text: {predicted_text}")
        print(f"   Confidence: {confidence*100:.2f}%\n")
        
        # Return result
        return JSONResponse(content={
            "success": True,
            "text": predicted_text,
            "confidence": float(confidence),
            "preprocessing": "smart",
            "message": "Text extracted successfully with smart preprocessing"
        })
    
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    finally:
        # Clean up temporary files
        cleanup_temp_files(temp_files)


@app.post("/api/predict-debug")
async def predict_text_debug(file: UploadFile = File(...)):
    """
    OCR Prediction with DEBUG mode
    
    Returns additional debugging information:
    - Image quality analysis
    - Preprocessing strategy used
    - Intermediate images saved
    
    Useful for troubleshooting preprocessing issues
    """
    
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    temp_files = []
    
    try:
        # Save uploaded file
        unique_filename = generate_unique_filename(file.filename)
        upload_path = UPLOADS_DIR / f"upload_{unique_filename}"
        
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        temp_files.append(upload_path)
        
        print(f"📥 DEBUG MODE: {file.filename}")
        
        # Analyze original image
        import cv2
        original = cv2.imread(str(upload_path), cv2.IMREAD_GRAYSCALE)
        original_quality = analyze_image_quality(original)
        
        print(f"📊 Original image analysis:")
        print(f"   Type: {original_quality}")
        
        # Preprocess with debug mode
        preprocessed_path = UPLOADS_DIR / f"preprocessed_{unique_filename}"
        temp_files.append(preprocessed_path)
        
        preprocessed_image = preprocess_for_production(
            str(upload_path),
            output_path=str(preprocessed_path),
            save_debug=True,
            debug_dir=str(DEBUG_DIR)
        )
        
        # Analyze preprocessed image
        preprocessed_quality = analyze_image_quality(preprocessed_image)
        
        # Predict
        predicted_text, confidence = predictor.predict(str(preprocessed_path))
        
        print(f"✓ Debug prediction complete\n")
        
        # Return detailed debug info
        return JSONResponse(content={
            "success": True,
            "text": predicted_text,
            "confidence": float(confidence),
            "debug_info": {
                "original_analysis": {
                    "unique_values": original_quality['unique_values'],
                    "mean_brightness": original_quality['mean'],
                    "contrast": original_quality['contrast'],
                    "is_binary": original_quality['is_binary'],
                    "is_inverted": original_quality['is_inverted']
                },
                "preprocessed_analysis": {
                    "contrast": preprocessed_quality['contrast'],
                    "is_binary": preprocessed_quality['is_binary']
                },
                "debug_images_saved": True,
                "debug_directory": str(DEBUG_DIR)
            },
            "message": "Debug prediction complete, check debug directory for images"
        })
    
    except Exception as e:
        print(f"❌ Debug prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Don't cleanup in debug mode so files can be inspected
        pass


@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze image quality without running OCR
    
    Useful for:
    - Checking if image is suitable for OCR
    - Understanding preprocessing strategy that will be used
    - Debugging image issues
    
    Returns detailed image analysis
    """
    
    temp_files = []
    
    try:
        # Save uploaded file
        unique_filename = generate_unique_filename(file.filename)
        upload_path = UPLOADS_DIR / f"upload_{unique_filename}"
        
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        temp_files.append(upload_path)
        
        # Analyze image
        import cv2
        image = cv2.imread(str(upload_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError("Cannot read image")
        
        quality = analyze_image_quality(image)
        
        # Determine recommended strategy
        from .smart_preprocessing import detect_image_type
        image_type = detect_image_type(image)
        
        # Provide recommendations
        recommendations = []
        
        if quality['contrast'] < 30:
            recommendations.append("Low contrast - consider adjusting brightness/contrast before upload")
        
        if quality['is_inverted']:
            recommendations.append("Image appears inverted (white text on black) - will be auto-corrected")
        
        if quality['is_binary']:
            recommendations.append("Image already preprocessed - minimal processing will be applied")
        
        if quality['unique_values'] < 5:
            recommendations.append("Very low color depth - might be over-processed")
        
        return JSONResponse(content={
            "success": True,
            "analysis": {
                "detected_type": image_type,
                "dimensions": {
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0])
                },
                "quality": {
                    "unique_values": quality['unique_values'],
                    "mean_brightness": round(quality['mean'], 2),
                    "contrast": round(quality['contrast'], 2),
                    "is_binary": quality['is_binary'],
                    "is_inverted": quality['is_inverted'],
                    "text_background_ratio": round(quality['text_bg_ratio'], 2)
                },
                "recommendations": recommendations,
                "suitable_for_ocr": quality['contrast'] > 20
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        cleanup_temp_files(temp_files)


@app.get("/api/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "CRNN (CNN + BiLSTM)",
        "input_height": 64,
        "charset_size": len(predictor.converter.charset),
        "total_classes": predictor.converter.num_classes,
        "device": str(predictor.device),
        "parameters": predictor.model.get_total_parameters(),
        "preprocessing": "smart (auto-adaptive)"
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "message": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Internal server error"}
    )