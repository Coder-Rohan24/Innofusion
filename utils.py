import cv2
import numpy as np
from PIL import Image
import io

def preprocess_image(image_file):
    """
    Preprocesses an uploaded image for model prediction.
    - Reads the image.
    - Converts to RGB (if needed).
    - Resizes to match model input size (assumed 224x224).
    - Normalizes pixel values.
    
    Args:
        image_file: Uploaded image file (FastAPI UploadFile)
    
    Returns:
        Numpy array ready for model inference.
    """
    # Read image bytes
    image = Image.open(io.BytesIO(image_file)).convert("RGB")
    
    # Resize image to (224, 224) - Modify this if your model uses a different size
    image = image.resize((224, 224))
    
    # Convert to NumPy array
    image_array = np.array(image) / 255.0  # Normalize pixel values
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array
