# type: ignore
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from src.logger import Logger
from src.models import BaseModel, ModelDataset
import pickle
import xgboost as xgb
import gc

logging = Logger()
logger = logging.get_logger()


class XGBoost(BaseModel):
    def __init__(
        self,
        n_estimators: int | None = None,
        subsample: float | None = None,
        colsample_bytree: float | None = None,
        tree_method: str = "hist",
        eval_metric: str = "mlogloss",
        learning_rate: float | None = 0.0001,
        max_depth: int | None = None,
        min_child_weight: int | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
        *args, **kwargs):
        """
        Initialize XGBoost model with fixed parameters.
        
        Args:
            n_estimators: Number of boosting rounds
            subsample: Fraction of samples used for fitting each tree
            colsample_bytree: Fraction of features used for fitting each tree
            learning_rate: Learning rate (eta)
            max_depth: Maximum depth of trees
            min_child_weight: Minimum sum of instance weight in a child
            random_state: Random seed for reproducibility
            n_jobs: Number of CPU cores to use (-1 = all cores)
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.tree_method = tree_method
        self.eval_metric = eval_metric
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.imputer = SimpleImputer(strategy='median')
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        logger.info(
            f"Initialized XGBoost with n_estimators={n_estimators}, "
            f"learning_rate={learning_rate}, max_depth={max_depth}"
        )

    def train(self, data: ModelDataset) -> None:
        """
        Train XGBoost model on data.
        
        Args:
            data: ModelDataset with train/test features and targets
        """
        logger.info("Training XGBoost model...")
        
        # Remove rows with missing target values
        train_mask = data.train_target.notna() # type: ignore
        train_data_clean = data.train[train_mask]
        train_target_clean = data.train_target[train_mask]

        # Impute missing values
        train_data_imputed = self.imputer.fit_transform(train_data_clean)
        train_data_scaled = self.scaler.fit_transform(train_data_imputed)
        train_target_encoded = self.encoder.fit_transform(train_target_clean)
        
        logger.debug(f"Train data shape: {train_data_scaled.shape}")
        logger.debug(f"Train target shape: {train_target_encoded.shape}") # type: ignore
        logger.debug(f"Classes: {self.encoder.classes_}")
        
        # Train model
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method=self.tree_method,
            eval_metric=self.eval_metric,
            verbose=2,
            #class_weight='balanced'
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
            List of predicted labels (decoded)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        if self.imputer is None:
            raise ValueError("Imputer not fitted. Call train() first or load a trained model.")
        
        if self.encoder is None:
            raise ValueError("Label encoder not fitted. Call train() first or load a trained model.")
        
        # Impute missing values
        data_imputed = self.imputer.transform(input_data) # type: ignore
        data_scaled = self.scaler.transform(data_imputed)
        # Make predictions (encoded)
        predictions_encoded = self.model.predict(data_scaled)
        
        # Decode labels
        predictions = self.encoder.inverse_transform(predictions_encoded)
        
        return predictions.tolist()

    def evaluate(self, data: ModelDataset) -> dict: # type: ignore
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
        
        if self.encoder is None:
            raise ValueError("Label encoder not fitted. Call train() first or load a trained model.")
        
        # Remove rows with missing target values
        test_mask = data.test_target.notna() # type: ignore
        test_data_clean = data.test[test_mask]
        test_target_clean = data.test_target[test_mask]
        
        # Impute missing values
        test_data_imputed = self.imputer.transform(test_data_clean)
        test_data_scaled = self.scaler.transform(test_data_imputed)
        
        logger.debug(f"Test data shape: {test_data_scaled.shape}")
        logger.debug(f"Test target shape: {test_target_clean.shape}")
        # Make predictions
        predictions_encoded = self.model.predict(test_data_scaled)
        
        # Decode for reporting
        predictions_decoded = self.encoder.inverse_transform(predictions_encoded)
        
        # Calculate metrics
        accuracy = accuracy_score(test_target_clean, predictions_decoded)
        f1 = f1_score(test_target_clean, predictions_decoded, average='macro', zero_division=0)
        f1_weighted = f1_score(test_target_clean, predictions_decoded, average='weighted', zero_division=0)
        report = classification_report(test_target_clean, predictions_decoded, output_dict=True)
        cm = confusion_matrix(test_target_clean, predictions_decoded)
        
        # Get feature importances
        feature_importances = self.model.feature_importances_.tolist()
        
        results = {
            "accuracy": float(accuracy),
            "f1_macro": float(f1),
            "f1_weighted": float(f1_weighted),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "feature_importances": feature_importances
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1 (macro): {f1:.4f}")
        logger.info(f"Test F1 (weighted): {f1_weighted:.4f}")
        logger.debug(f"Feature Importances: {feature_importances}")
        return results

    def save(self, file_path: str = "xgboost_model.pkl") -> None:
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
            "encoder": self.encoder,
            "scaler": self.scaler,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "min_child_weight": self.min_child_weight,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs
            }
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {file_path}")

    def load(self, file_path: str = "xgboost_model.pkl") -> None:
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
        self.encoder = model_data["encoder"]
        self.scaler = model_data["scaler"]
        # Restore hyperparameters
        hparams = model_data["hyperparameters"]
        self.n_estimators = hparams["n_estimators"]
        self.subsample = hparams["subsample"]
        self.colsample_bytree = hparams["colsample_bytree"]
        self.learning_rate = hparams["learning_rate"]
        self.max_depth = hparams["max_depth"]
        self.min_child_weight = hparams["min_child_weight"]
        self.random_state = hparams["random_state"]
        self.n_jobs = hparams["n_jobs"]
        
        logger.info(f"Model loaded from {file_path}")