import warnings
import os
import pandas as pd
import numpy as np
import joblib
from typing import Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")

PREDICTOR_FILE_NAME = 'predictor.joblib'


class Classifier:
    """A wrapper class for the Random Forest binary classifier.

        This class provides a consistent interface that can be used with other
        classifier models.
    """

    model_name = 'knn_binary_classifier'

    def __init__(self,
                 n_neighbors: Optional[int] = 5,
                 weights: Optional[str] = 'uniform',
                 p: Optional[int] = 2,
                 leaf_size: Optional[int] = 5
                 ):
        """Construct a new Random Forest binary classifier.

        Args:
            n_neighbors (int, optional): The number of neighbors to be used by the classifier.
                Defaults to 5.
            weights (str, optional): Weight function used in predictions. Possible values:
                ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
                ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors
                of a query point will have a greater influence than neighbors which are further away.
                Defaults to 'uniform'.

        """

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.leaf_size = leaf_size
        self._is_trained = False
        self.model = self.build_model()

    def build_model(self) -> KNeighborsClassifier:
        """Build a new KNN binary classifier."""
        return KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                    weights=self.weights,
                                    p=self.p,
                                    leaf_size=self.leaf_size)

    def fit(self, train_input: pd.DataFrame, train_target: pd.DataFrame):
        """Fit the  KNN binary classifier to the training data.

        Args:
            train_input: The features of the training data.
            train_target: The labels of the training data.
        """

        self.model.fit(train_input, train_target)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        return self.model.predict(inputs)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the KNN binary classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The accuracy of the KNN binary classifier.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the KNN binary classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """

        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the KNN binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded KNN binary classifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    @classmethod
    def train_predictor_model(cls, train_inputs: pd.DataFrame, train_targets: pd.Series,
                              hyperparameters: dict) -> "Classifier":
        """
        Instantiate and train the predictor model.

        Args:
            train_inputs (pd.DataFrame): The training data inputs.
            train_targets (pd.Series): The training data labels.
            hyperparameters (dict): Hyperparameters for the classifier.

        Returns:
            'Classifier': The classifier model
        """
        classifier = Classifier(**hyperparameters)
        classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
        return classifier

    @classmethod
    def predict_with_model(cls, classifier: "Classifier", data: pd.DataFrame, return_probs=False
                           ) -> np.ndarray:
        """
        Predict class probabilities for the given data.

        Args:
            classifier (Classifier): The classifier model.
            data (pd.DataFrame): The input data.
            return_probs (bool): Whether to return class probabilities or labels.
                Defaults to True.

        Returns:
            np.ndarray: The predicted classes or class probabilities.
        """
        if return_probs:
            return classifier.predict_proba(data)
        return classifier.predict(data)

    @classmethod
    def save_predictor_model(cls, model: "Classifier", predictor_dir_path: str) -> None:

        """
        Save the classifier model to disk.

        Args:
            model (Classifier): The classifier model to save.
            predictor_dir_path (str): Dir path to which to save the model.
        """
        if not os.path.exists(predictor_dir_path):
            os.makedirs(predictor_dir_path)
        model.save(predictor_dir_path)

    @classmethod
    def load_predictor_model(cls, predictor_dir_path: str) -> "Classifier":
        """
        Load the classifier model from disk.

        Args:
            predictor_dir_path (str): Dir path where model is saved.

        Returns:
            Classifier: A new instance of the loaded classifier model.
        """
        return Classifier.load(predictor_dir_path)

    @classmethod
    def evaluate_predictor_model(cls,
                                 model: "Classifier", x_test: pd.DataFrame, y_test: pd.Series
                                 ) -> float:
        """
        Evaluate the classifier model and return the accuracy.

        Args:
            model (Classifier): The classifier model.
            x_test (pd.DataFrame): The features of the test data.
            y_test (pd.Series): The labels of the test data.

        Returns:
            float: The accuracy of the classifier model.
        """
        return model.evaluate(x_test, y_test)
