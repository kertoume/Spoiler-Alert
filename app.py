
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging

# Create the API
app = FastAPI(title="Spoiler Alert API")

# Load the trained machine learning model
model = joblib.load("spoiler_model.pkl")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spoiler_alert")

# Monitoring counters
total_requests = 0
spoiler_predictions = 0
non_spoiler_predictions = 0
prediction_errors = 0

class Review(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Spoiler Alert API is running!"}

@app.post("/predict")
def predict(review: Review):
    global total_requests
    global spoiler_predictions
    global non_spoiler_predictions
    global prediction_errors

    total_requests += 1

    try:
        prediction = bool(model.predict([review.text])[0])
        probabilities = model.predict_proba([review.text])[0]
        spoiler_index = list(model.classes_).index(True)
        probability = float(probabilities[spoiler_index])

        if prediction:
            spoiler_predictions += 1
        else:
            non_spoiler_predictions += 1

        logger.info(
            "Prediction completed: is_spoiler=%s",
            prediction
        )

        return {
            "review": review.text,
            "is_spoiler": prediction,
            "spoiler_probability": round(probability, 4)
        }

    except Exception:
        prediction_errors += 1
        logger.exception("Prediction failed")
        raise

@app.get("/stats")
def stats():
    return {
        "total_requests": total_requests,
        "spoiler_predictions": spoiler_predictions,
        "non_spoiler_predictions": non_spoiler_predictions,
        "prediction_errors": prediction_errors
    }

