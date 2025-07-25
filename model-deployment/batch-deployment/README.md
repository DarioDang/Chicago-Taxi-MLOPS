# Chicago Taxi Batch Deployment 🚖
This repository contains the batch deployment pipeline for predicting taxi trip durations in Chicago using a trained machine learning model. The system is designed for realistic data generation, scheduled inference, and database storage using tools like Prefect, Flask, MLflow, and PostgreSQL.

---

## 📁 Project Structure
```bash
batch-deployment/
├── docker-compose.yaml # Docker configuration for required services
├── predict.py # Flask app with Prefect flow to predict and log trip durations
├── README.md # Project documentation
```

---

## ⚙️ Components

### 🧠 ML Model
- **Model:** Random Forest Regressor
- **Versioning:** Managed with MLflow
- **Stage:** `Production`
- **Tracking URI:** `http://127.0.0.1:5000`

### 🎯 Functionality
- Generates **synthetic but realistic taxi rides** biased toward actual Chicago pickup/drop-off areas and peak hours.
- Predicts trip duration using a deployed model stored in the MLflow Model Registry.
- Logs each prediction (with metadata) into a PostgreSQL database.
- **UUID-based trip ID** is generated for each record.
- Runs every 5 minutes as a **Prefect 2.0 flow** (can be scheduled through Prefect Cloud/Server).
- Offers a simple **Flask endpoint** for status check.

---

## 🛠️ Technologies Used

- **Flask**: Lightweight web server
- **MLflow**: Model tracking, registry, and artifact management
- **PostgreSQL**: Storage for predictions
- **APScheduler**: Time-based job scheduling
- **pandas**, **uuid**, **psycopg**, **scikit-learn**, **pickle**

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/chicago-taxi-batch-deployment.git
cd batch-deployment
```

### 2. Start the MLFLOW UI
Since the tracking server was stored at workflow-orchestration folder which using **mage** we need to start the mlflow tracking server from this folder to get the current model.

```bash
cd CHICAGO-TAXI-MLOPS/workflow-orchestration/
mlflow ui --backend-store-uri sqlite:///mlflow.db
```


### 3. Start MLflow and PostgreSQL via Docker

Navigate back to the model ```deployment/batch-deployment``` to run: 

```bash
docker-compose up
```

### 4. Run Flask App with Prefect Flow
```bash
python predict.py
```

- The app starts a local Flask server at http://localhost:9696.
- Schedules a background job using APScheduler to run every 5 minutes

## 🗃️ Database Output
The batch prediction results are saved into a PostgreSQL table: ride_predictions. The schema includes:

```trip_id```: unique UUID

```pickup_community_area```: integer

```dropoff_community_area```: integer

```trip_start_timestamp```: timestamp

```fare```: float

```trip_total```: float

```trip_miles```: float

```predicted_duration```: float

```created_at```: timestamp of insertion


## 👤 Author
Dario Dang



