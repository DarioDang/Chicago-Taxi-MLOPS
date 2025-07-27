# Chicago Taxi MLOps Best Practice
This folder contains best practices for testing and maintaining a machine learning project using the Chicago Taxi dataset. It includes well-structured **unit tests**, **integration tests**, and development automation through a **Makefile**. The goal is to ensure high code quality and reproducibility when building and maintaining a Flask-based ML prediction service.

---

## 🚀 Features

- 🔁 Reproducible environment with `pipenv`
- 🧪 Unit and integration tests with `pytest`
- 🧼 Code linting with `pylint` or `ruff`
- 🧹 Auto-formatting with `black`
- 🛠 Task automation via `Makefile`
- ⚙️ ML model and preprocessing artifacts loaded from S3 using MLflow


## 📁 Project Structure
```bash
Best-Practices/
├── Code/
│ ├── aws_predict.py # Main Flask app and prediction logic
│ ├── init.py
│ └── tests/
│ ├── unit/ # Unit tests for feature functions
│ │ └── test_unit.py
│ └── integration/ # Integration tests for API
│ └── test_integration.py
├── Pipfile / Pipfile.lock # Dependency management via Pipenv
├── Makefile # Dev commands for test, lint, run, etc.
├── README.md
```

## 🧪 Running Tests

🧰 Setup Instructions

1. Install Pipenv

2. Install dependencies:
```bash
make install
```

3. From project root (Best-Practices/), run:

| Command                 | What it does                                 |
| ----------------------- | -------------------------------------------- |
| `make install`          | Install all dev dependencies via Pipenv      |
| `make format`           | Auto-format code using Black                 |
| `make lint`             | Lint your code using Ruff                    |
| `make unit-test`        | Run only unit tests                          |
| `make integration-test` | Run only integration tests                   |
| `make test`             | Run all tests                                |
| `make run`              | Start the Flask app in development mode      |
| `make clean`            | Remove caches, compiled files, and temp data |


## 👤 Author

Dario Dang

ML Engineer | MLOps Enthusiast | DataOps Practitioner
