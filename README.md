# 🚕 Chicago Taxi Trip Duration Prediction – MLOps Pipeline

This project builds a complete end-to-end **MLOps workflow** using real-world **Chicago Taxi Trips** data. It covers everything from data ingestion and exploratory data analysis to experiment tracking, model deployment, and monitoring using modern tools like **MLflow**, **Mage**, **AWS S3**, **Docker**, and **Evidently**.

> ✅ Built by **Dario Dang**  
> 🛠 Technologies: Python · MLflow · Mage · AWS · Docker · Evidently · Grafana

## 🗂️ Project Structure

```bash
├── ingest-data/ # Data ingestion and EDA
├── experiment-tracking/ # MLflow experiment tracking and model registry
├── workflow-orchestration/ # Mage pipeline orchestration
├── aws-model-deployment/ # Flask API to serve predictions from S3/MLflow
├── model-monitoring/ # Drift detection and model quality monitoring
├── best-practice/ # Unit & integration tests, Makefile, linting
└── README.md # This file
```

---

## 📊 Dataset
The project uses the **Chicago Taxi Trips** dataset from the City of Chicago’s open data portal:

📥 [Download the data](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)

- Contains millions of trip records since 2013.
- Used features include:
  - Trip start timestamp
  - Pickup/dropoff community area
  - Fare, trip miles, trip total
- Target variable: **Trip Duration**

You can download monthly Parquet files using the automated script in `ingest-data/ingest_data.py`.

---

## 🔁 Workflow Overview

1. **Data Ingestion & EDA**  
   Download and explore the data, perform feature engineering, and visualize patterns.

2. **Experiment Tracking**  
   Train, tune, and evaluate models (RandomForest, XGBoost, SVR, etc.) using MLflow and Hyperopt.

3. **Pipeline Orchestration**  
   Mage dynamically tunes and trains models. The best model is registered and pushed to S3.

4. **Model Deployment**  
   Flask app downloads the production model from MLflow/S3 and serves real-time predictions via REST API.

5. **Monitoring**  
   Track drift and quality metrics using Evidently + Grafana dashboards, stored in PostgreSQL.

---

## 🚀 Key Technologies

| Area              | Tools Used                                         |
|-------------------|----------------------------------------------------|
| Orchestration     | [Mage](https://github.com/mage-ai/mage-ai)         |
| Tracking & Registry | [MLflow](https://mlflow.org/)                    |
| Deployment        | Flask, Docker, AWS S3                              |
| Monitoring        | Evidently, Grafana, PostgreSQL                     |
| Testing & Quality | Pytest, Ruff, Black, Makefile                      |

---

## 📦 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/DarioDang/Chicago-Taxi-MLOPS.git
cd Chicago-Taxi-MLOPS
```

### 2. Launch Environment

```bash
pipenv shell
pipenv install --dev
```

## 🔧 Main Components

### 📥 1. [Ingestion + EDA](https://github.com/DarioDang/Chicago-Taxi-MLOPS/tree/main/ingest-data)
Download monthly Parquet files

Generate EDA plots (duration distribution, feature importances)

### 🧪 2. [Experiment Tracking](https://github.com/DarioDang/Chicago-Taxi-MLOPS/tree/main/experiment-tracking)
Train and tune multiple regressors

Log runs, metrics, and models to MLflow

Register and promote best model to production

### ⚙️ 3. [Workflow Orchestration](https://github.com/DarioDang/Chicago-Taxi-MLOPS/tree/main/workflow-orchestration)
Mage DAG trains models, registers them, and stores best one in S3

Modular blocks: preprocessing, tuning, training, tracking

### 🛰 4. [Model Deployment](https://github.com/DarioDang/Chicago-Taxi-MLOPS/tree/main/model-deployment)
Flask API that:

Downloads production model from MLflow

Preprocesses input JSON

Returns predicted trip duration

### 📈 5. [Model Monitoring](https://github.com/DarioDang/Chicago-Taxi-MLOPS/tree/main/model-monitoring)
Detect drift in predictions using Evidently

Store inference data in PostgreSQL

Visualize metrics in Grafana

### 6. 🧪 [Testing & Best Practices](https://github.com/DarioDang/Chicago-Taxi-MLOPS/tree/main/best-practice)
Unit and integration tests using Pytest

Auto-linting with Ruff

Auto-formatting with Black

Makefile for automation


| Component                                                                                                     | Description                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 📥 [Ingestion + EDA](https://github.com/DarioDang/Chicago-Taxi-MLOPS/tree/main/ingest-data)                   | - Download monthly Parquet files <br> - Generate EDA plots (duration distribution, feature importances)                                  |
| 🧪 [Experiment Tracking](https://github.com/DarioDang/Chicago-Taxi-MLOPS/tree/main/experiment-tracking)       | - Train and tune multiple regressors <br> - Log runs, metrics, and models to MLflow <br> - Register and promote best model to production |
| ⚙️ [Workflow Orchestration](https://github.com/DarioDang/Chicago-Taxi-MLOPS/tree/main/workflow-orchestration) | - Mage DAG trains models, registers them, and stores best one in S3 <br> - Modular blocks: preprocessing, tuning, training, tracking     |
| 🛰 [Model Deployment](https://github.com/DarioDang/Chicago-Taxi-MLOPS/tree/main/model-deployment)         | - Flask API to serve predictions <br> - Downloads production model from MLflow <br> - Preprocesses input JSON and returns duration       |
| 📈 [Model Monitoring](https://github.com/DarioDang/Chicago-Taxi-MLOPS/tree/main/model-monitoring)             | - Detect drift in predictions using Evidently <br> - Store inference data in PostgreSQL <br> - Visualize metrics in Grafana              |
| 🧪 [Testing & Best Practices](https://github.com/DarioDang/Chicago-Taxi-MLOPS/tree/main/best-practice)        | - Unit and integration tests with Pytest <br> - Auto-linting with Ruff <br> - Auto-formatting with Black <br> - Makefile for automation  |


## 👤 Author
Dario Dang

MLOps Engineer | DataOps Practitioner | ML Enthusiast