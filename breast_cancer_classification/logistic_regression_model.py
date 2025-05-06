# Import necessary libraries
import numpy as np
import pandas as pd
import pickle
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

def scale_data(X_train, X_test):
    """
    
    """
    # Scale features to ensure each has a mean of 0 and stdv of 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def create_lr_model(max_iter: int = 100):
    """
    
    """
    # Initialize logistic regression model
    lr_model = LogisticRegression(max_iter) #default is 100, we were stopping at `100` iterations before converging on best solution
    return lr_model
   

def fit_lr_model(lr_model: LogisticRegression, X_train, y_train):
    """
    
    """
    
    lr_model.fit(X_train, y_train)
    

def evaluate_lr_model(lr_model: LogisticRegression, X_train, X_test, y_train, y_test):
    """
    
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

def generate_feature_importance(lr_model:LogisticRegression, X):
    """
    
    """
    # Extract feature importance
    # Extract feature names from the dataset
    feature_names = X.columns
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


def save_trained_model(lr_model:LogisticRegression, file_name: str):
    """
    
    """
    with open(f"../models/", 'wb') as file:
        pickle.dump(lr_model, file)


def load_lr_model(lr_model: LogisticRegression, filepath: str):
    # Load the model   
    with open(filepath, 'rb') as file:
        lr_model = pickle.load(file)