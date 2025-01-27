import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            
            # Splitting the input data into features and targets
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All columns except the last
                train_array[:, -1],   # The last column
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Defining the models to be evaluated
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            logging.info("Evaluating models")
            # Evaluate models using the utility function
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

            # Selecting the best model based on R2 score
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            # Check if the best model's score meets the threshold
            if model_report[best_model_name] < 0.6:
                raise CustomException("No model found with sufficient performance.")

            logging.info(f"Best model: {best_model_name} with R2 score: {model_report[best_model_name]}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Making predictions and calculating the R2 score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

