import os.path
from pathlib import Path
import sys
import time

# Rich imports for enhanced logging
from rich.console import Console
from rich import print as rprint
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.traceback import install
import logging

from loguru import logger
import pandas as pd
import typer

sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from breast_cancer_classification.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
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
    log.info(f"Loading data from: [blue]{data_filepath}[/blue]")

    try:
        data = pd.read_csv(data_filepath)
        load_time = time.time() - start_time
        
        # Log basic information about the loaded data
        log.info(f"Data loaded successfully in [green]{load_time:.2f}[/green] seconds")
        log.info(f"Dataset dimensions: [yellow]{data.shape[0]}[/yellow] rows, [yellow]{data.shape[1]}[/yellow] columns")
        
        # Calculate and log class distribution if 'diagnosis' column exists
        if 'diagnosis' in data.columns:
            class_counts = data['diagnosis'].value_counts()
            log.info(f"Class distribution: {dict(class_counts)}")
        
        # Log column types summary
        type_counts = data.dtypes.value_counts()
        log.info(f"Column types: {dict(type_counts)}")
        
        # Check for missing values
        missing = data.isnull().sum().sum()
        if missing > 0:
            log.warning(f"Found [red]{missing}[/red] missing values in the dataset")
        else:
            log.info("[green]No missing values[/green] found in the dataset")

        # Verify the data is loaded correctly
        if debug:
            print("Data Head")
            print(data.head())  # Display the first few rows of the dataset
            print("Data Info")
            data.info()  # Display information about the dataset
    except Exception as e:
        log.error(f"[bold red]Error loading data:[/bold red] {str(e)}")
        raise
    return data


# Pass by Object Reference >> Dataframes are fine to pass and not return
def preprocess_data(data: pd.DataFrame):
    """Preprocess the data in the dataframe.

    Args:
        data: The dataframe containing the data.

    Returns:
        None.

    """
    log.info("Starting data preprocessing")
    start_time = time.time()

    # Convert 'diagnosis' column to numeric (e.g., 'B' -> 0, 'M' -> 1)
    data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})
    load_time = time.time() - start_time
    
    # Log basic information about the loaded data
    log.info(f"Data loaded successfully in [green]{load_time:.2f}[/green] seconds")

# Typer used to allow us to test methods specifically if we direct call it
app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    
    
    log.info("[bold green]Starting data preprocessing pipeline[/bold green]")
    start_time = time.time()
    logger.info(f"Loading data from: {input_path}")
    data = load_data(input_path)
    preprocess_data(data)
    data.to_csv(output_path, index=False)
    load_time = time.time() - start_time
    log.info(f"Data preprocessing completed in [green]{load_time:.2f}[/green] seconds")
    logger.success(f"Finished preprocessing the data, output: {output_path}")


if __name__ == "__main__":
    app()
