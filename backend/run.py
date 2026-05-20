import uvicorn
import sys
from pathlib import Path

if __name__ == "__main__":
    # backend ko path mileko xa ki xaina check garne
    backend_path = Path(__file__).resolve().parent
    if str(backend_path) not in sys.path:
        sys.path.append(str(backend_path))

    print("\n🚀 Starting Akshayara OCR Backend Server...")
    print("#" * 50)
    print("API will be available at: http://localhost:3001")
    print("#" * 50 + "\n")

    # yo fast api run garaune code ho
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=3001,
        reload=True,
        log_level="info"
    )
