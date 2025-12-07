from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import pickle

app = FastAPI()

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load model
MODEL_PATH = os.path.join(BASE_DIR, "RF_heart_disease_model.pkl")
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

# Static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("heart.html", {"request": request, "result": None})


@app.post("/predictPage", response_class=HTMLResponse)
async def predict_page(
    request: Request,
    Age: str = Form(...),
    Sex: str = Form(...),
    ChestPainType: str = Form(...),
    RestingBP: str = Form(...),
    Cholesterol: str = Form(...),
    FastingBS: str = Form(...),
    RestingECG: str = Form(...),
    MaxHR: str = Form(...),
    ExerciseAngina: str = Form(...),
    Oldpeak: str = Form(...),
    ST_Slope: str = Form(...)
):
    try:
        # Prepare input
        input_data = [[
            float(Age), float(Sex), float(ChestPainType), float(RestingBP), float(Cholesterol),
            float(FastingBS), float(RestingECG), float(MaxHR), float(ExerciseAngina),
            float(Oldpeak), float(ST_Slope)
        ]]

        # Predict
        prediction = model.predict(input_data)
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    except Exception as e:
        result = f"Error: {str(e)}"

    return templates.TemplateResponse("heart.html", {"request": request, "result": result})
