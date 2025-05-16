import os.path
from pathlib import Path
import pickle
import sys

from loguru import logger
import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import typer

## Required otherwise it will error out in the makefile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..')) )

from breast_cancer_classification.config import MODELS_DIR, PROCESSED_DATA_DIR
from breast_cancer_classification.dataset import load_data

app = typer.Typer()


def create_test_train_split(data: pd.DataFrame, debug: bool = False):
    """Create the test train split for the data.

    Args:
        data: The dataframe containing the data.
        debug: The option to turn on printing information about the training and test shapes.

    Returns:
        The X,y train and test splits.

    """
    X = data.drop(["diagnosis", "id"], axis=1)
    y = data["diagnosis"]
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    if debug:
        print(f"X_train shape:{X_train.shape}")
        print(f"X_test shape:{X_test.shape}")
        print(f"y_train shape:{y_train.shape}")
        print(f"y_test shape:{y_test.shape}")

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """Scales the X train and X_Test data to have a mean of 0 and standard deviation of 1.

    Args:
        X_train : The X training dataset.
        X_test: The X test dataset.

    Returns:
        The X_train and X_test data scaled.

    """
    # Scale features to ensure each has a mean of 0 and stdv of 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def create_lr_model(max_iterv: int = 100) -> LogisticRegression:
    """Creates the logistic regression model.

    Args:
        max_iter : The max number of iterations before converging on the best solution.

    Returns:
        The logistic regression model.

    """
    # Initialize logistic regression model
    lr_model = LogisticRegression(
        max_iter=max_iterv
    )  # default is 100, we were stopping at `100` iterations before converging on best solution
    return lr_model


def fit_lr_model(lr_model: LogisticRegression, X_train, y_train):
    """Fits the logistic regression model based on the X_train and y_train data.

    Args:
        lr_model : The logisitic regression model.
        X_train : The X training dataset.
        y_train : The y training dataset.

    Returns:
        None

    """

    lr_model.fit(X_train, y_train)


def save_trained_model(lr_model: LogisticRegression, file_path: Path):
    """Save the trained model to the /models directory.

    Args:
        lr_model : The logisitic regression model.
        file_name : The name of the output file.

    Returns:
        None.


    """
    with open(file_path, "wb") as file:
        pickle.dump(lr_model, file)


@app.command()
def main(
    # ---- File paths ----
    processed_data_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    logger.info(f"Loading data from: {processed_data_path}")
    data = load_data(processed_data_path)

    logger.info("Creating & Training logistic regression (LR) model...")
    X_train, X_test, y_train, y_test = create_test_train_split(data)
    lr_model = create_lr_model()
    fit_lr_model(lr_model, X_train, y_train)

    # Save the Test_Data for testing in the predict section
    # test_data = pd.concat([X_test, y_test], axis=1)
    logger.info(f"Saving labels and features for LF model: {labels_path}, {features_path}")
    y_test.to_csv(labels_path, index=False)
    X_test.to_csv(features_path, index=False)

    logger.info(f"Saving LR model: {model_path}")
    save_trained_model(lr_model, model_path)
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
