# 🎬 Spoiler Alert — Movie Review Spoiler Detection

## Project Overview

Spoiler Alert is a machine learning project that predicts whether an IMDb movie review contains spoilers.

The goal is to help users identify potentially spoiler-containing reviews before reading them.

This project demonstrates the five main stages of a Machine Learning Operations (MLOps) workflow: Setup, Data, Model, Deployment, and Monitoring.

## 1. Setup

**Technologies used:**
- Python
- Google Colab
- Pandas
- Scikit-learn
- FastAPI
- Docker
- GitHub and GitHub Actions

## 2. Dataset

**Dataset:** IMDb Spoiler Dataset

**Source:** https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset

The dataset contains IMDb movie reviews and labels indicating whether each review contains spoilers.

For the initial model, we prepared a balanced sample of 2,000 reviews:
- 1,000 spoiler reviews
- 1,000 non-spoiler reviews

The dataset was processed in Google Colab. The original dataset is not included in this repository because it can be downloaded from Kaggle.

## 3. Model Training

We trained a text classification model using:

- TF-IDF Vectorizer for converting review text into numerical features.
- Logistic Regression for predicting spoiler or non-spoiler labels.

The dataset was divided into:
- 75% training data (1,500 reviews)
- 25% testing data (500 reviews)

**Preliminary test accuracy: 70.8%**

The trained model was saved as `spoiler_model.pkl`.

The model training and evaluation code can be found in `Spoiler_Alert_Project.ipynb`.

## 4. Deployment

The trained model was integrated into a FastAPI application.

**API endpoints:**
- `GET /` — Check whether the API is running.
- `POST /predict` — Predict whether a movie review contains spoilers.
- `GET /stats` — View basic monitoring statistics.

Docker was used to package the application and its dependencies into a container.

### Run the application using Docker

```bash
docker build -t spoiler-alert .
docker run -d --name spoiler-api -p 8000:8000 spoiler-alert
```

After starting the container, open:

http://localhost:8000/docs

This displays the interactive FastAPI documentation, where users can test predictions.

The application has been tested locally using Docker. It is not currently hosted as a public online service.

## 5. Continuous Integration (CI)

GitHub Actions automatically runs checks when changes are pushed to the repository.

The CI workflow:
1. Checks out the repository.
2. Sets up Python.
3. Installs the required dependencies.
4. Checks that the API loads.
5. Builds the Docker image.

The workflow is stored in `.github/workflows/ci.yml`.

## 6. Monitoring

Basic monitoring was added to the FastAPI application.

The `/stats` endpoint tracks:
- Total prediction requests
- Spoiler predictions
- Non-spoiler predictions
- Prediction-processing errors

Logging is also used to record prediction events and errors.

The monitoring counters are stored in memory and reset when the application restarts. They do not measure model accuracy or long-term performance.

## 7. Project Files

- `Spoiler_Alert_Project.ipynb` — Dataset preparation, model training, and evaluation
- `app.py` — FastAPI application and monitoring
- `spoiler_model.pkl` — Trained machine learning model
- `requirements.txt` — Python dependencies
- `Dockerfile` — Docker container configuration
- `.github/workflows/ci.yml` — GitHub Actions workflow

## 8. Limitations and Future Improvements

- The initial model was trained on a relatively small sample of reviews.
- The sampling and evaluation approach should be improved to reduce possible data leakage.
- More representative training and testing data could improve the reliability of results.
- Monitoring counters currently reset when the server restarts.
- Future versions could include a user-friendly website, persistent monitoring, and improved model evaluation.

## Conclusion

Spoiler Alert demonstrates an end-to-end MLOps workflow, from dataset preparation and machine learning model training to local Docker deployment, automated CI checks, and basic application monitoring.
