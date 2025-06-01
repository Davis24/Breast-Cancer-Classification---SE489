import os.path
from pathlib import Path
import sys
import time
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich import print as rprint
from rich.panel import Panel

import logging

from loguru import logger
import pandas as pd
import typer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from breast_cancer_classification.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import os
print("Current working directory:", os.getcwd())
# Set up logging with RichHandler for console and FileHandler for file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = Path("logs") / f"dataset_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True, markup=True),
        logging.FileHandler(log_file, mode="w", encoding="utf-8")
    ]
)
log = logging.getLogger("dataset")

def load_data(data_filepath: Path, debug: bool = False) -> pd.DataFrame:
    """Loads the .csv file into a dataframe.

    Args:
        data_filepath: The file path for the csv.
        debug: The option to turn on printing for dataframe information.

    Returns:
        The new dataframe.
    """
    start_time = time.time()
    log.info(f"Loading data from: {data_filepath}")

    try:
        data = pd.read_csv(data_filepath)
        load_time = time.time() - start_time
        
        log.info(f"Data loaded successfully in {load_time:.2f} seconds")
        log.info(f"Dataset dimensions: {data.shape[0]} rows, {data.shape[1]} columns")
        
        if 'diagnosis' in data.columns:
            class_counts = data['diagnosis'].value_counts()
            log.info(f"Class distribution: {dict(class_counts)}")
        
        type_counts = data.dtypes.value_counts()
        log.info(f"Column types: {dict(type_counts)}")
        
        missing = data.isnull().sum().sum()
        if missing > 0:
            log.warning(f"Found {missing} missing values in the dataset")
        else:
            log.info("No missing values found in the dataset")

        if debug:
            print("Data Head")
            print(data.head())
            print("Data Info")
            data.info()
    except Exception as e:
        log.error(f"Error loading data: {str(e)}")
        raise
    return data

def preprocess_data(data: pd.DataFrame):
    """Preprocess the data in the dataframe.

    Args:
        data: The dataframe containing the data.

    Returns:
        None.
    """
    log.info("Starting data preprocessing")
    start_time = time.time()

    data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})
    load_time = time.time() - start_time
    
    log.info(f"Data preprocessing completed in {load_time:.2f} seconds")

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    log.info("Starting data preprocessing pipeline")
    start_time = time.time()
    log.info(f"Loading data from: {input_path}")
    data = load_data(input_path)
    preprocess_data(data)
    data.to_csv(output_path, index=False)
    load_time = time.time() - start_time
    log.info(f"Data preprocessing completed in {load_time:.2f} seconds")
    logger.success(f"Finished preprocessing the data, output: {output_path}")

if __name__ == "__main__":
    app()