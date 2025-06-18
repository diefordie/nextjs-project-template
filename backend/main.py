from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import pickle
import os
from skimage import morphology, measure, feature
import json

app = FastAPI()

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model (model.pkl should be uploaded manually to backend folder)
model_path = "backend/model.pkl"
model = None
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    img_bytes = buffer.tobytes()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str

def extract_features(json_data, image_directory):
    # Placeholder for user provided feature extraction code
    # For now, simulate feature extraction with dummy values
    features = {
        "Mean Area": 100.0,
        "Mean Perimeter": 50.0,
        "Mean Eccentricity": 0.5,
        "Mean Roundness": 0.7,
        "Mean Contrast": 0.3,
        "Mean Homogeneity": 0.4,
        "Mean Correlation": 0.6,
        "Mean Intensity": 120.0
    }
    return features

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg"]:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Save original image temporarily
    cv2.imwrite("backend/temp_original.png", img)
    return {"message": "Image uploaded successfully"}

@app.post("/process-image")
async def process_image():
    # Load original image
    img = cv2.imread("backend/temp_original.png")
    if img is None:
        return JSONResponse(status_code=400, content={"error": "No image uploaded"})
    results = {}

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results["grayscale"] = image_to_base64(gray)

    # Otsu Thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results["otsu_thresholding"] = image_to_base64(otsu)

    # Morphological Opening
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=2)
    results["morphological_opening"] = image_to_base64(opening)

    # Watershed Segmentation
    # Compute sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Compute sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    # Compute unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    img_watershed = img.copy()
    markers = cv2.watershed(img_watershed, markers)
    img_watershed[markers == -1] = [255,0,0]  # mark boundaries in red
    results["watershed_segmentation"] = image_to_base64(img_watershed)

    # Feature extraction (simulate with dummy json_data and image_directory)
    features = extract_features({}, "backend")

    # Prediction
    prediction = None
    if model is not None:
        # For demonstration, flatten grayscale image and predict
        flat_img = gray.flatten().reshape(1, -1)
        try:
            pred = model.predict(flat_img)
            prediction = "infected" if pred[0] == 1 else "non-infected"
        except Exception as e:
            prediction = f"Prediction error: {str(e)}"
    else:
        prediction = "Model not loaded"

    return {
        "processed_images": results,
        "features": features,
        "prediction": prediction
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
