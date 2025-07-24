# 🚕 Chicago Taxi Data Ingestion and EDA

This part focuses on **downloading**, **preprocessing**, and **exploring** the Chicago Taxi Trips dataset. It includes a Python script for automated data ingestion and a Jupyter notebook for performing exploratory data analysis (EDA) and training a basic machine learning model.

---

## 📁 Folder Structure
```
ingest-data/
├── ingest_data.py       # Script to download monthly data and save as Parquet
├── EDA.ipynb            # Jupyter notebook for data preprocessing, visualization, and modeling
├── ../Dataset/          # Output directory where downloaded Parquet files are saved
```

## 🚀 Getting Started

### 1. Environment Setup

Make sure you have Python 3.x and the required packages installed. You can install dependencies using:
```bash
pipenv shell 
```

### 2. Download Monthly Data
The ingest_data.py script fetches taxi trip data for a specific month and saves it as a Parquet file under the Dataset/ folder.
✅ Usage:

    ```bash
    python ingest_data.py --year [enter year]  --month [enter month]
    ```

--year: Year of the data (e.g., 2023)

--month: Month of the data (1–12)

The script will save the file as: ../Dataset/chicago_taxi_[year]_[month].parquet

### 3. Exploratory Data Analysis 

The EDA.ipynb notebook includes:

✅ Timestamp parsing and duration feature engineering

📉 Data cleaning and filtering for valid trips

🔍 Feature extraction:

Trip distance

Time of day

Day of week

Weekend indicator

🤖 Model training using RandomForestRegressor

🔢 Feature importance analysis

📈 Visualizations:

KDE plot comparing trip durations (missing vs known dropoff area)

Bar plot for average trip duration by day of week


### 4. Output

Feature Importances
Bar plot showing top 20 features influencing trip duration.

Duration Distribution
KDE plot comparing trip durations between trips with missing and known dropoff areas.

Daily Averages
Bar chart showing how average trip durations vary across weekdays.

👤 Author
Developed by Dario Dang.