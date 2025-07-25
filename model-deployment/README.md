# AWS Model Deployment: Chicago Taxi Duration Predictor

This service hosts a machine learning model for predicting taxi ride durations in Chicago. The model has been trained and orchestrated via [Mage](https://github.com/mage-ai/mage-ai), promoted to production, and saved in an S3 bucket. This repository handles loading the production model and preprocessing artifacts, and exposes a REST API for real-time predictions.

---

## 🔧 Project Structure
```bash
aws-model-deployment/
├── dict_vectorizer.bin # Cached vectorizer (optional, downloaded at runtime)
├── Dockerfile # Containerization setup for deployment
├── Pipfile / Pipfile.lock # Python dependencies managed via pipenv
├── predict.py # Main Flask app to serve predictions
├── test.py # Simple test client for the prediction API
```

---

## 🚀 How It Works

- The model and its associated preprocessing vectorizer (`dict_vectorizer.bin`) are stored in an **S3 bucket**: ```s3://dario-mlflow-models-storage/<RUN_ID>/artifacts/```

- `predict.py` downloads the model and vectorizer at runtime using `mlflow` and `boto3`.

- The Flask web server exposes a `/predict` endpoint, which:
    1. Parses the input ride data
    2. Applies preprocessing
    3. Transforms features using the downloaded vectorizer
    4. Returns the predicted duration

---

## 🐳 Docker Deployment

### 1. **Build the Docker image**
From the project root:

```bash
docker build -t duration-predictor:v1 -f model-deployment/aws-model-deployment/Dockerfile .
```

### 2. **Run the container with AWS credentials**
Option A: using environment variables

```bash
docker run -p 9696:9696 \
  -e AWS_ACCESS_KEY_ID=your_access_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret_key \
  -e AWS_DEFAULT_REGION=your_region \
  duration-predictor:v1
```

Option B: mounting your AWS config
```bash
docker run -p 9696:9696 \
  -v ~/.aws:/root/.aws \
  duration-predictor:v1
```

#### 📬 Testing Request
Run this on terminal:

```bash
python test.py
```

#### Expected response:

```json
{
  "duration": ...,
  "model_id": "..."
}
```


## 👤 Author
Dario Dang