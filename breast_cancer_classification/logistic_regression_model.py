# Import necessary libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


def load_data(data_filepath: str, debug: bool = False) -> pd.DataFrame:
    """Loads the .csv file into a dataframe.

    Args:
        data_filepath: The csv file path.
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
        print(data.info())  # Display information about the dataset

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


def create_test_train_split(data: pd.DataFrame, debug: bool = False):
    """Create the test train split for the data.

    Args:
        data: The dataframe containing the data.
        debug: The option to turn on printing information about the training and test shapes.
        
    Returns:
        The X,y train and test splits.
    
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


def create_lr_model(max_iter: int = 100) -> LogisticRegression:
    """Creates the logistic regression model.

    Args:
        max_iter : The max number of iterations before converging on the best solution.
        
    Returns:
        The logistic regression model.
    
    """
    # Initialize logistic regression model
    lr_model = LogisticRegression(max_iter) #default is 100, we were stopping at `100` iterations before converging on best solution
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


def save_trained_model(lr_model:LogisticRegression, file_name: str):
    """Save the trained model to the /models directory.

    Args:
        lr_model : The logisitic regression model.
        file_name : The name of the output file.
        
    Returns:
        None.
    
    
    """
    with open(f"../models/{file_name}", 'wb') as file:
        pickle.dump(lr_model, file)


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