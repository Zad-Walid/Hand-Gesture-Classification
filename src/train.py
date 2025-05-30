
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report , ConfusionMatrixDisplay , confusion_matrix
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn


def load_data(path):
    '''
    Load the dataset from the specified path.
    Args:
        path (str): Path to the dataset file.
    Returns:
        pd.DataFrame: DataFrame containing the dataset.
    '''
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    """
    Preprocess the data by encoding categorical variables and splitting into features and target.
    
    Args:
        df (pd.DataFrame): DataFrame with features and target variable.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with encoded labels.
        LabelEncoder: Fitted LabelEncoder for the target variable.

    """
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    return df , le


def normalize_landmarks(df):
    """"
    Normalize the landmarks in the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame with landmarks.
    Returns:
        pd.DataFrame: DataFrame with normalized landmarks.
    """
    for i in range(df.shape[0]):
        landmarks = np.array(df.iloc[i, :-1]).reshape(21, 3)
        wrist = landmarks[0, :].copy()
        landmarks -= wrist
        mid_finger_tip = landmarks[12, :].copy()
        distance = np.linalg.norm(mid_finger_tip)
        if distance > 0:
            landmarks /= distance
        df.iloc[i, :-1] = landmarks.flatten()
    
    return df


def split_data(df):
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        df (pd.DataFrame): DataFrame with features and target variable.
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: Split datasets.
    """
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_log_model(model , params , X_train, y_train , X_val, y_val, model_name , output_dir):
    """
    Train a logistic regression model with hyperparameter tuning.
    
    Args:
        model: The machine learning model to train.
        params (dict): Hyperparameters for the model.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        
    Returns:
        model: Trained model.
    """
    with mlflow.start_run(run_name=model_name):
        grid = GridSearchCV(model, params, cv=3)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        prec = precision_score(y_val, y_pred, average='weighted')
        mlflow.set_tag("model_name", model_name)
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.sklearn.log_model(best_model, f"{model_name}_model")
        print(f"{model_name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")

        conf_matrix = confusion_matrix(y_val, y_pred ,labels=best_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=best_model.classes_)
        disp.plot() 
        fig_path = f'{output_dir}/mat_{model_name}.png'
        disp.figure_.savefig(fig_path)
        mlflow.log_artifact(fig_path)
        mlflow.log_artifact("dataset\hand_landmarks_data.csv")
        
        return best_model
    
def main():
    os.environ["LOGNAME"] = "Zad"
    output_dir = "output"
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    ### Set the experiment name
    mlflow.set_experiment("_Hand_Gesture_Classification_")

    ### Load the data
    df = load_data("dataset\hand_landmarks_data.csv")
    df, le = preprocess_data(df)
    df = normalize_landmarks(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}, Test set size: {X_test.shape[0]}")

    models = {
        "RandomForest": (RandomForestClassifier(random_state=42), {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        }),
        "SVM": (SVC(random_state=42), {
            'kernel': ['rbf'],
            'C': [10, 100],
            'gamma': [0.1, 1]
        }),
        "XGBoost": (XGBClassifier(random_state=42), {
            'n_estimators': [100, 500],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2]
        })
    }

    for name, (model, params) in models.items():
        train_log_model(model, params, X_train, y_train, X_val, y_val, name , output_dir)

if __name__ == "__main__":
    main()


