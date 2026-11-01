
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Spoiler Alert API - Version 2")

# Load our new model and threshold
package = joblib.load("spoiler_model_v2.pkl")

model = package["model"]
threshold = package["threshold"]

# Define what users can send
class Review(BaseModel):
    text: str

# Check whether the API is running
@app.get("/")
def home():
    return {
        "message": "Spoiler Alert API Version 2 is running!",
        "threshold": threshold
    }

# Predict whether a review contains spoilers
@app.post("/predict")
def predict(review: Review):

    # Get probability of the review containing spoilers
    probabilities = model.predict_proba([review.text])[0]

    spoiler_index = list(model.classes_).index(True)

    spoiler_probability = float(probabilities[spoiler_index])

    # Apply our new threshold
    prediction = spoiler_probability >= threshold

    return {
        "review": review.text,
        "is_spoiler": bool(prediction),
        "spoiler_probability": round(spoiler_probability, 4),
        "threshold": threshold
    }
