#Base Image
FROM python:3.11-slim

# Install Python
RUN apt update && \  
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY breast_cancer_classification/ breast_cancer_classification/
COPY data/ data/