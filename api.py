"""Oskar Cephalo v2 - Production REST API (FastAPI).

Run server:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from PIL import Image

from oskar_cephalo import OskarCephaloDetector

app = FastAPI(
    title="Oskar Cephalo v2 API",
    description="High-Precision AI Cephalometric Landmark Detection (29 Points) & CVM Stage Classification",
    version="2.0.0"
)

# Enable CORS for web apps (React, Vue, Angular, Mobile)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = OskarCephaloDetector()


@app.get("/")
def root():
    return {
        "service": "Oskar Cephalo v2 API",
        "status": "online",
        "endpoints": {
            "predict_json": "POST /predict",
            "predict_overlay_image": "POST /predict/overlay",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "oskar_cephalo.onnx"}


@app.post("/predict")
async def predict_landmarks_json(
    file: UploadFile = File(...),
    point_size: int = Query(6, ge=2, le=20),
    show_labels: bool = Query(True),
    draw_lines: bool = Query(True),
    include_base64_image: bool = Query(False)
):
    """
    Upload cephalometric radiograph (PNG, JPG, DICOM converted).
    Returns JSON with 29 landmark coordinates and CVM stage classification.
    """
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    results = detector.predict(
        image_input=pil_image,
        point_size=point_size,
        show_labels=show_labels,
        draw_lines=draw_lines
    )

    response_data = {
        "success": True,
        "image_size": {
            "width": results["image_width"],
            "height": results["image_height"]
        },
        "cvm_stage": results["cvm_stage"],
        "landmarks_count": len(results["landmarks"]),
        "landmarks": results["landmarks"]
    }

    if include_base64_image:
        rendered_pil = Image.fromarray(results["rendered_image"])
        buffered = io.BytesIO()
        rendered_pil.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        response_data["rendered_image_base64"] = f"data:image/jpeg;base64,{img_str}"

    return JSONResponse(content=response_data)


@app.post("/predict/overlay")
async def predict_overlay_image(
    file: UploadFile = File(...),
    point_size: int = Query(6, ge=2, le=20),
    show_labels: bool = Query(True),
    draw_lines: bool = Query(True)
):
    """
    Upload cephalometric radiograph and directly receive the annotated JPEG image.
    """
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    results = detector.predict(
        image_input=pil_image,
        point_size=point_size,
        show_labels=show_labels,
        draw_lines=draw_lines
    )

    rendered_pil = Image.fromarray(results["rendered_image"])
    buffered = io.BytesIO()
    rendered_pil.save(buffered, format="JPEG", quality=92)
    return Response(content=buffered.getvalue(), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
