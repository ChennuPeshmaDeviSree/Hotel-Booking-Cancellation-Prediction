
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Hotel Booking Cancellation API")

# Load trained Logistic Regression pipeline
model = joblib.load("hotel_bookings_pipeline.pkl")

# Input structure
class BookingInput(BaseModel):
    data: dict

# Home endpoint
@app.get("/")
def home():
    return {"message": "Hotel Booking API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(input_data: BookingInput):

    # Convert input into DataFrame
    input_df = pd.DataFrame([input_data.data])

    # Predict booking cancellation
    prediction = int(model.predict(input_df)[0])

    # Calculate cancellation probability
    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    return {
        "prediction": prediction,
        "probability": probability
    }
