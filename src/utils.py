import os
import sys
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Save a Python object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

from sklearn.metrics import r2_score

def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluates multiple models and returns a dictionary containing their R2 scores.

    Args:
        X_train (array): Training features.
        y_train (array): Training target.
        X_test (array): Testing features.
        y_test (array): Testing target.
        models (dict): A dictionary of models to evaluate.

    Returns:
        dict: A dictionary with model names as keys and R2 scores as values.
    """
    try:
        model_report = {}
        
        for model_name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)
            
            # Predict on the test data
            y_test_pred = model.predict(X_test)
            
            # Calculate R2 score
            r2 = r2_score(y_test, y_test_pred)
            
            # Save the R2 score in the dictionary
            model_report[model_name] = r2
        
        return model_report

    except Exception as e:
        raise CustomException(e, sys)
