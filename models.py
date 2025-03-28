from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Initialize FastAPI
app = FastAPI()

# Load the Keras Model
model_path = "plant_disease_model.h5"  # Ensure this is the correct path
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define disease class labels
labels = {
    0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy',
    4: 'Blueberry___healthy', 5: 'Cherry_(including_sour)___Powdery_mildew', 6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 8: 'Corn_(maize)___Common_rust_', 
    9: 'Corn_(maize)___Northern_Leaf_Blight', 10: 'Corn_(maize)___healthy', 11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)', 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)', 16: 'Peach___Bacterial_spot', 17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot', 19: 'Pepper,_bell___healthy', 20: 'Potato___Early_blight',
    21: 'Potato___Late_blight', 22: 'Potato___healthy', 23: 'Raspberry___healthy', 24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew', 26: 'Strawberry___Leaf_scorch', 27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot', 29: 'Tomato___Early_blight', 30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold', 32: 'Tomato___Septoria_leaf_spot', 33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot', 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

# Image preprocessing function
def preprocess_image(file: UploadFile):
    contents = file.file.read()
    img = load_img(io.BytesIO(contents), target_size=(224, 224))  # Use Keras image loader
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    return img_array

def predict_disease(file: UploadFile):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        img_tensor = preprocess_image(file)  # Preprocess image
        predictions = model.predict(img_tensor)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        # return {
        #     "predicted_disease": labels[predicted_class],
        #     "confidence": confidence
        # }
       
        return labels[predicted_class]
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# Prediction function
# def predict_disease(img: Image.Image):
#     """Predict plant disease from an image."""
#     if model is None:
#         return {"error": "Model not loaded"}

#     try:
#         img_tensor = preprocess_image(img)  # Preprocess image
#         predictions = model.predict(img_tensor)  # Get model predictions
#         predicted_class = np.argmax(predictions, axis=1)[0]  # Get class with highest probability
#         confidence = float(np.max(predictions))  # Extract confidence score

#         return {
#             "predicted_disease": labels[predicted_class],
#             "confidence": confidence
#         }
#     except Exception as e:
#         return {"error": f"Prediction failed: {str(e)}"}

