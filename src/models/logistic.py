# type: ignore
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression as LogisticRegressionClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils import compute_class_weight
from src.logger import Logger
from src.models import BaseModel, ModelDataset
import json
import pickle
import gc

logging = Logger()
logger = logging.get_logger()


class LogisticRegression(BaseModel):
    def __init__(
        self,
        max_iter: int | None = None,
        C: float | None = None,
        solver: str = "lbfgs",
        random_state: int = 42,
        n_jobs: int = -1,
        *args, **kwargs):
        """
        Initialize Logistic Regression model with fixed parameters.
        
        Args:
            max_iter: Maximum number of iterations
            C: Inverse of regularization strength (smaller values = stronger regularization)
            solver: Algorithm to use in optimization problem
            random_state: Random seed for reproducibility
            n_jobs: Number of CPU cores to use (-1 = all cores)
        """
        super().__init__()
        self.max_iter = max_iter
        self.C = C
        self.solver = solver
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.imputer = SimpleImputer(strategy='median')
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        logger.info(
            f"Initialized LogisticRegression with max_iter={max_iter}, C={C}, solver={solver}"
        )

    def train(self, data: ModelDataset) -> None:
        """
        Train logistic regression model on data.
        
        Args:
            data: ModelDataset with train/test features and targets
        """
        logger.info("Training Logistic Regression model...")
        
        # Remove rows with missing target values
        train_mask = data.train_target.notna()
        train_data_clean = data.train[train_mask]
        train_target_clean = data.train_target[train_mask]
        
        # Impute missing values
        self.imputer = SimpleImputer(strategy='median')
        train_data_imputed = self.imputer.fit_transform(train_data_clean)
        train_data_scaled = self.scaler.fit_transform(train_data_imputed)
        train_target_encoded = self.encoder.fit_transform(train_target_clean)
        
        logger.debug(f"Train data shape: {train_data_scaled.shape}")
        logger.debug(f"Train target shape: {train_target_encoded.shape}")
        
        classes = np.unique(train_target_encoded)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_target_encoded)
        class_weights = dict(zip(classes, weights))
        # Train model
        self.model = LogisticRegressionClassifier(
            max_iter=self.max_iter,
            C=self.C,
            solver=self.solver,
            random_state=self.random_state,
            #class_weight=class_weights,
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        logger.info(f"Starting training...")
        
        self.model.fit(train_data_scaled, train_target_encoded)
        
        logger.info("Training complete!")

    def predict(self, input_data) -> list:
        """
        Predict labels for input data.
        
        Args:
            input_data: DataFrame with same features as training data
            
        Returns:
            List of predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        if self.imputer is None:
            raise ValueError("Imputer not fitted. Call train() first or load a trained model.")
        
        # Impute missing values
        data_imputed = self.imputer.transform(input_data)
        data_scaled = self.scaler.transform(data_imputed)
        # Make predictions
        predictions_encoded = self.model.predict(data_scaled)
        predictions = self.encoder.inverse_transform(predictions_encoded)
        
        return predictions

    def evaluate(self, data: ModelDataset) -> dict:
        """
        Evaluate model on test data.
        
        Args:
            data: ModelDataset with test features and targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        if self.imputer is None:
            raise ValueError("Imputer not fitted. Call train() first or load a trained model.")
        
        # Remove rows with missing target values
        test_mask = data.test_target.notna()
        test_data_clean = data.test[test_mask]
        test_target_clean = data.test_target[test_mask]
        
        # Impute missing values
        test_data_imputed = self.imputer.transform(test_data_clean)
        test_data_scaled = self.scaler.transform(test_data_imputed)
        
        logger.debug(f"Test data shape: {data.test.shape}")
        logger.debug(f"Test target shape: {data.test_target.shape}")
        
        
        # Make predictions
        predictions_encoded = self.model.predict(test_data_scaled)
        predictions = self.encoder.inverse_transform(predictions_encoded)
        
        # Calculate metrics
        accuracy = accuracy_score(test_target_clean, predictions)
        f1_macro = f1_score(test_target_clean, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(test_target_clean, predictions, average='weighted', zero_division=0)
        report = classification_report(test_target_clean, predictions, output_dict=True)
        cm = confusion_matrix(test_target_clean, predictions)
        
        
        results = {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1 (macro): {f1_macro:.4f}")
        logger.info(f"Test F1 (weighted): {f1_weighted:.4f}")
        
        return results

    def save(self, file_path: str = "logistic_regression_model.pkl") -> None:
        """
        Save trained model, imputer, and label encoder to disk.
        
        Args:
            file_path: Path where to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            "model": self.model,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "encoder": self.encoder,
            "hyperparameters": {
                "max_iter": self.max_iter,
                "C": self.C,
                "solver": self.solver,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs
            }
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {file_path}")

    def load(self, file_path: str = "logistic_regression_model.pkl") -> None:
        """
        Load a trained model from disk.
        
        Args:
            file_path: Path to the saved model
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore model and preprocessors
        self.model = model_data["model"]
        self.imputer = model_data["imputer"]
        self.scaler = model_data["scaler"]
        self.encoder = model_data["encoder"]
        
        # Restore hyperparameters
        hparams = model_data["hyperparameters"]
        self.max_iter = hparams["max_iter"]
        self.C = hparams["C"]
        self.solver = hparams["solver"]
        self.random_state = hparams["random_state"]
        self.n_jobs = hparams["n_jobs"]
        
        logger.info(f"Model loaded from {file_path}")