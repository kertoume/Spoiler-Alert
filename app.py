
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Spoiler Alert API")

# Load our trained AI model
model = joblib.load("spoiler_model.pkl")

# Define what users can send
class Review(BaseModel):
    text: str

# Simple health check
@app.get("/")
def home():
    return {"message": "Spoiler Alert API is running!"}

# Predict whether a review contains spoilers
@app.post("/predict")
def predict(review: Review):
    prediction = model.predict([review.text])[0]

    probability = model.predict_proba(
        [review.text]
    )[0]

    spoiler_index = list(model.classes_).index(True)

    return {
        "review": review.text,
        "is_spoiler": bool(prediction),
        "spoiler_probability": round(
            float(probability[spoiler_index]), 4
        )
    }
