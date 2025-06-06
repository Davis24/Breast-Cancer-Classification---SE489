#Base Image
FROM python:3.11-slim

# Install Python
RUN apt update && \  
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


# Upgrade pip
RUN pip install --upgrade pip

# Copy all required files to the docker image.
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY breast_cancer_classification/ breast_cancer_classification/
COPY models/ models/
COPY data/ data/

# Run the installation, Run takes about 1.1 minute, 1.4GB
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir


# Need this directory to exist for logging
RUN mkdir -p /logs

# This command currently takes _way_ to long, 
#RUN pip install . --no-deps --no-cache-dir 

# Run Training Script
ENTRYPOINT ["python", "-u", "breast_cancer_classification/modeling/predict.py"]

