from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from routes.predict import predict_disease
from routes.gemini import get_treatment
import os
from dotenv import load_dotenv
import uvicorn
# Load environment variables
load_dotenv()

app = FastAPI(title="Plant Disease Detection and Treatment API", version="0.1")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/service")
def service(request: Request):
    return templates.TemplateResponse("service.html", {"request": request})
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    disease_name = predict_disease(file)
    treatment = get_treatment(disease_name)
    if "healthy" in disease_name.lower():
        disease_name="No Disease"
            
    formatted_treatment = treatment.replace("\n", "<br>")
    return templates.TemplateResponse("result.html", {"request": request, "disease": disease_name, "treatment": formatted_treatment})

if __name__ == "__main__":
    workers = multiprocessing.cpu_count() * 2 + 1  # Dynamically set workers
    port = int(os.getenv("PORT", "10000"))  # Ensure PORT is set correctly

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        workers=workers  # Use multiple workers for performance
    )