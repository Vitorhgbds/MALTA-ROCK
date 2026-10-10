# type: ignore
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from src.logger import Logger
from src.models import BaseModel, ModelDataset
import pickle
import gc

logging = Logger()
logger = logging.get_logger()


class DecisionTree(BaseModel):
    def __init__(
        self,
        min_samples_split: int | None = None,
        min_samples_leaf: int | None = None,
        max_depth: int | None = None,
        random_state: int = 42,
        *args, **kwargs):
        """
        Initialize Decision Tree model with fixed parameters.
        
        Args:
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_depth: Maximum depth of the tree (None = unlimited)
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.imputer = SimpleImputer(strategy='median')
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.label_encoder_mapping = None
        logger.info(
            f"Initialized DecisionTree with min_samples_split={min_samples_split}, "
            f"min_samples_leaf={min_samples_leaf}, max_depth={max_depth}"
        )

    def train(self, data: ModelDataset) -> None:
        """
        Train Decision Tree model on data.
        
        Args:
            data: ModelDataset with train/test features and targets
        """
        logger.info("Training Decision Tree model...")
        
        # Remove rows with missing target values
        train_mask = data.train_target.notna() # type: ignore
        train_data_clean = data.train[train_mask]
        train_target_clean = data.train_target[train_mask]
        
        # Impute missing values
        train_data_imputed = self.imputer.fit_transform(train_data_clean)
        train_data_scaled = self.scaler.fit_transform(train_data_imputed)
        train_target_encoded = self.encoder.fit_transform(train_target_clean)
        
        
        logger.debug(f"Train data shape: {train_data_imputed.shape}")
        logger.debug(f"Train target shape: {train_target_encoded.shape}") # type: ignore
        
        # Train model
        self.model = DecisionTreeClassifier(
            min_samples_split=self.min_samples_split, # type: ignore
            min_samples_leaf=self.min_samples_leaf, # type: ignore
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight='balanced'
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
        data_imputed = self.imputer.transform(input_data) # type: ignore
        data_scaled = self.scaler.transform(data_imputed)
        # Make predictions
        predictions_encoded = self.model.predict(data_scaled)
        predictions = self.encoder.inverse_transform(predictions_encoded)
        
        return predictions # type: ignore

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
        predictions = self.encoder.inverse_transform(predictions_encoded)
        
        
        # Calculate metrics
        accuracy = accuracy_score(test_target_clean, predictions)
        f1 = f1_score(test_target_clean, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(test_target_clean, predictions, average='weighted', zero_division=0)
        report = classification_report(test_target_clean, predictions, output_dict=True)
        cm = confusion_matrix(test_target_clean, predictions)
        
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
        logger.info(f"Classification Report:\n{classification_report(test_target_clean, predictions)}")
        logger.debug(f"Feature Importances: {feature_importances}")
        return results

    def save(self, file_path: str = "decision_tree_model.pkl") -> None:
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
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_depth": self.max_depth,
                "random_state": self.random_state
            }
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {file_path}")

    def load(self, file_path: str = "decision_tree_model.pkl") -> None:
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
        self.min_samples_split = hparams["min_samples_split"]
        self.min_samples_leaf = hparams["min_samples_leaf"]
        self.max_depth = hparams["max_depth"]
        self.random_state = hparams["random_state"]
        
        logger.info(f"Model loaded from {file_path}")