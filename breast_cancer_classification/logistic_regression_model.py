# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


def load_data(data_filepath: str, debug: bool = False) -> pd.DataFrame:
    """

    
    """
    
    data = pd.read_csv(data_filepath)

    # Verify the data is loaded correctly
    if(debug):
        print("Data Head")
        print(data.head())  # Display the first few rows of the dataset
        print("Data Info")
        print(data.info())  # Display information about the dataset

    return data


#Pass by Object Reference >> Dataframes are fine to pass and not return
def preprocess_data(data: pd.DataFrame):
    """
    
    """
    # Convert 'diagnosis' column to numeric (e.g., 'B' -> 0, 'M' -> 1)
    data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})


def create_test_train_split(data: pd.DataFrame, debug: bool = False):
    """
    
    """
    X = data.drop(['diagnosis', 'id'], axis=1)
    y = data['diagnosis']
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    if(debug):
        print(f"X_train shape:{X_train.shape}")
        print(f"X_test shape:{ X_test.shape}")
        print(f"y_train shape:{y_train.shape}")
        print(f"y_test shape:{y_test.shape}")

    return X_train, X_test, y_train, y_test
