# Makefile for managing testing, linting, formatting, and running the Flask app

# Makefile for managing testing, linting, formatting, and running the Flask app

# Python and Pipenv commands
PIPENV_RUN=pipenv run
PYTHON=${PIPENV_RUN} python
PYTEST=${PIPENV_RUN} pytest

# Paths
SRC_DIR=Code
UNIT_TEST_DIR=tests/unit
INTEGRATION_TEST_DIR=tests/integration

# Default task
.DEFAULT_GOAL := help

## help: Show this help message
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

## install: Install dependencies
install:
	pipenv install --dev

## format: Format code with black
format:
	$(PIPENV_RUN) black $(SRC_DIR)

## lint: Lint code with pylint
lint:
	$(PIPENV_RUN) pylint $(SRC_DIR)

## test: Run all tests
test: unit-test integration-test

## unit-test: Run unit tests only
unit-test:
	$(PYTEST) $(UNIT_TEST_DIR)

## integration-test: Run integration tests only
integration-test:
	$(PYTEST) $(INTEGRATION_TEST_DIR)

## run: Run the Flask app (for development)
run:
	FLASK_APP=Code/aws_predict.py FLASK_ENV=development $(PYTHON) -m flask run --host=0.0.0.0 --port=9696

## clean: Remove __pycache__, .pytest_cache, and other temporary files
clean:
	find . -type d -name '__pycache__' -exec rm -r {} +;
	find . -type d -name '.pytest_cache' -exec rm -r {} +;
	rm -rf mlruns dict_vectorizer.bin


