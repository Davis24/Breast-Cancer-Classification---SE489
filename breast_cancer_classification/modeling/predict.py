import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from breast_cancer_classification.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def evaluate_lr_model(lr_model: LogisticRegression, X_test, y_test):
    """Evalutes the accuracy of logistic regression model and prints the results.

    Args:
        lr_model : The logisitic regression model.
        X_train : The X training dataset.
        y_train : The y training dataset.
        
    Returns:
        The y_predictions, accuracy, confusion matrix, and the classification report.
    
    
    """
    # Make predictions
    y_pred = lr_model.predict(X_test) #maybe split predictions out 
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n {conf_matrix}")
    print(f"Classification Report:\n {class_report}")

    return y_pred, accuracy, conf_matrix, class_report

def generate_feature_importance(lr_model:LogisticRegression, original_df: pd.DataFrame) -> pd.DataFrame:
    """Generates the feature importance report and outputs it.

    Args:
        lr_model : The logisitic regression model.
        original_df: The original dataframe with all the columns.
        
    Returns:
        The important features dataframe.
    
    
    """
    # Extract feature importance
    # Extract feature names from the dataset
    feature_names = original_df.columns
    # Get the coefficients
    coefficients = lr_model.coef_[0]  # shape (n_features,)

    # Pair each coefficient with its feature name
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)  # for easier sorting by strength
    })

    # Sort features by importance (absolute value of coefficient)
    feature_importance = feature_importance.sort_values(by='abs_coefficient', ascending=False)

    print("Feature Importance Data:")
    print(feature_importance)

    return feature_importance

def load_lr_model(filepath: str):
    """Load an already existing model from a pickle file.

    Args:
        file_path : The location of the logistic regression pickle file.
        
    Returns:
        Loaded model from file.
    
    
    """
    # Load the model   
    with open(filepath, 'rb') as file:
        lr_model = pickle.load(file)

    return lr_model

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    processed_data_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    #predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    lr_model= load_lr_model(model_path)
    X_test = pd.read_csv(features_path)
    y_test = pd.read_csv(labels_path)

    print(f"X_test Shape{X_test.shape}")
    print(f"y_test Shape{y_test.shape}")

    data = pd.read_csv(processed_data_path)
    X = data.drop(['diagnosis', 'id'], axis=1)

    evaluate_lr_model(lr_model, X_test, y_test)
    generate_feature_importance(lr_model, X)

    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()