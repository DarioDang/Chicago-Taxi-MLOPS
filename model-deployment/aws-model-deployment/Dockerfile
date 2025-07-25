FROM python:3.9.12

# Install pipenv and update pip
RUN pip install -U pip
RUN pip install pipenv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install all dependencies (system-wide)
RUN pipenv install --system --deploy

# Install boto3 explicitly for S3 access
RUN pip install boto3

COPY ["predict.py", "./"]
COPY ["test.py", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

# command to run the application
# docker build -t duration-predictor:v1 .
# docker run -p 9696:9696  -v ~/.aws:/root/.aws duration-predictor:v1

