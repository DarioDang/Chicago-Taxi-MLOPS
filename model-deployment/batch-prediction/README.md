# Batch Prediction: Chicago Taxi Duration Estimator

This module is responsible for **loading a registered ML model** and its associated **preprocessing pipeline** (specifically, a `DictVectorizer`) from the MLflow tracking server based on the **"Production"** stage. It then uses the model to perform batch predictions on input data.

This setup is ideal for **offline inference**, where predictions are generated in bulk (e.g. for a daily scheduled job or historical reprocessing task).

---

## 📦 Project Structure

```bash
batch-prediction/
├── predict.py # Main script: loads model & vectorizer, applies predictions
├── test.py # Simple test or entry point (optional use)
└── README.md # Project documentation
```

---

## 🚀 How It Works

1. Connects to the **MLflow Tracking Server** (local or remote)
2. Loads the **latest model version** from the **Production stage** via the model registry:
   - e.g. `models:/taxi_duration_predictor/Production`.
3. Automatically resolves the associated `run_id` and **downloads** the `dict_vectorizer.bin` from the run's `preprocessing` artifact directory.
4. Applies feature engineering + transformation.
5. Makes predictions and outputs them.


## 🔧 Prerequisites

- Python 3.8+
- MLflow Tracking Server running at `http://127.0.0.1:5000`.
- Model named `randomforest-reg-v2` is registered and promoted to `"Production"` stage.
- Artifacts are stored locally or on a remote backend accessible to MLflow.

## 🗃️ Notes on Artifact Structure
This script expects:

The model to be saved under: artifacts/model/

The vectorizer (dict_vectorizer.bin) to be saved under: artifacts/preprocessing/

Both of these are downloaded using MLflow's artifact resolution logic.

## 🗃️ Notes on Artifact Structure

You may adapt test.py or build a batch input loader to feed data into predict.py.

## 👤 Author
Dario Dang




