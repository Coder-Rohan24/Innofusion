from fastapi import APIRouter, UploadFile, File
import io
import numpy as np
import cv2  # OpenCV for image processing
from models import predict_disease  # Import function

router = APIRouter()

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read image bytes
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)

        # Decode image using OpenCV
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Read as BGR image

        if img is None:
            return {"error": "Invalid image file. Please upload a valid JPG, PNG, or BMP image."}

        # Convert BGR to RGB if needed (TensorFlow models expect RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Call the prediction function
        result = predict_disease(img)

        return result

    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}
