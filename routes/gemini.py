import google.generativeai as genai
from fastapi import APIRouter, HTTPException
import os
router = APIRouter()
API_KEY = os.getenv("GEMINI_API_KEY")
# Configure Gemini API
genai.configure(api_key=API_KEY)

@router.get("/")
def get_treatment(disease: str):
    try:
        prompt = f"Suggest an organic treatment for {disease} in crops in a professional and concise manner. If the disease name contains the word 'healthy,' first state: 'Your plant is healthy.' Then, provide essential precautions for maintaining its health by saying below are the precautions just without adding unnecessary introductions or explanations or saying understood."
        
        # Use gemini-2.0-flash model
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        # return {"treatment": response.text}
        return response.text 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
