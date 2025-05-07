from pathlib import Path
import pandas as pd

from loguru import logger
import typer

from breast_cancer_classification.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

def load_data(data_filepath: Path, debug: bool = False) -> pd.DataFrame:
    """Loads the .csv file into a dataframe.

    Args:
        data_filepath: The file path for the csv.
        debug: The option to turn on printing for dataframe information.

    Returns:
        The new dataframe.
    
    """
    
    data = pd.read_csv(data_filepath)

    # Verify the data is loaded correctly
    if(debug):
        print("Data Head")
        print(data.head())  # Display the first few rows of the dataset
        print("Data Info")
        data.info()  # Display information about the dataset

    return data

#Pass by Object Reference >> Dataframes are fine to pass and not return
def preprocess_data(data: pd.DataFrame):
    """Preprocess the data in the dataframe.

    Args:
        data: The dataframe containing the data.
        
    Returns:
        None.
    
    """
    # Convert 'diagnosis' column to numeric (e.g., 'B' -> 0, 'M' -> 1)
    data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

#Typer used to allow us to test methods specifically if we direct call it
app = typer.Typer()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    logger.info(f"Loading data from: {input_path}")
    data = load_data(input_path)
    preprocess_data(data)
    data.to_csv(output_path, index=False)
    logger.success(f"Finished preprocessing the data, output: {output_path}")


if __name__ == "__main__":
    app()