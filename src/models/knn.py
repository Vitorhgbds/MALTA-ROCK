# type: ignore
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from src.logger import Logger
from src.models import BaseModel, ModelDataset
import pickle
import gc

logging = Logger()
logger = logging.get_logger()


class KNN(BaseModel):
    def __init__(
        self,
        n_neighbors: int | None = None,
        weights: str | None = None,
        p: int | None = None,
        algorithm: str | None = None,
        metric: str = "minkowski",
        leaf_size: int = 30,
        n_jobs: int = -1,
        *args, **kwargs):
        """
        Initialize KNN model with fixed parameters.
        
        Args:
            n_neighbors: Number of neighbors to consider
            weights: Weight function ('uniform' or 'distance')
            p: Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean)
            algorithm: Algorithm to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree')
            metric: Distance metric to use
            leaf_size: Leaf size for tree-based algorithms
            n_jobs: Number of CPU cores to use (-1 = all cores)
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.algorithm = algorithm
        self.metric = metric
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.model = None
        self.imputer = SimpleImputer(strategy='median')
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        logger.info(
            f"Initialized KNN with n_neighbors={n_neighbors}, weights={weights}, p={p}, "
            f"algorithm={algorithm}, metric={metric}, leaf_size={leaf_size}, n_jobs={n_jobs}"
        )

    def train(self, data: ModelDataset) -> None:
        """
        Train KNN model on data.
        
        Args:
            data: ModelDataset with train/test features and targets
        """
        logger.info("Training KNN model...")
        
        # Remove rows with missing target values
        train_mask = data.train_target.notna()
        train_data_clean = data.train[train_mask]
        train_target_clean = data.train_target[train_mask]
        
        # Impute missing values (KNN requires no missing values)
        train_data_imputed = self.imputer.fit_transform(train_data_clean)
        train_data_scaled = self.scaler.fit_transform(train_data_imputed)
        train_target_encoded = self.encoder.fit_transform(train_target_clean)
        
        logger.debug(f"Train data shape: {train_data_imputed.shape}")
        logger.debug(f"Train target shape: {train_target_encoded.shape}")
        
        # Train model
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            p=self.p,
            algorithm=self.algorithm,
            metric=self.metric,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs
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
        
        # Remove rows with missing target values
        test_mask = data.test_target.notna()
        test_data_clean = data.test[test_mask]
        test_target_clean = data.test_target[test_mask]
        
        logger.debug(f"Test data shape: {data.test.shape}")
        logger.debug(f"Test target shape: {data.test_target.shape}")
        
        # Impute missing values
        test_data_imputed = self.imputer.transform(test_data_clean)
        test_data_scaled = self.scaler.transform(test_data_imputed)
        
        # Make predictions
        predictions_encoded = self.model.predict(test_data_scaled)
        predictions = self.encoder.inverse_transform(predictions_encoded)
        
        # Calculate metrics
        accuracy = accuracy_score(test_target_clean, predictions)
        f1 = f1_score(test_target_clean, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(test_target_clean, predictions, average='weighted', zero_division=0)
        report = classification_report(test_target_clean, predictions, output_dict=True)
        cm = confusion_matrix(test_target_clean, predictions)
        
        results = {
            "accuracy": float(accuracy),
            "f1_macro": float(f1),
            "f1_weighted": float(f1_weighted),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1 (macro): {f1:.4f}")
        logger.info(f"Classification Report:\n{classification_report(test_target_clean, predictions)}")
        logger.info(f"Test F1 (weighted): {f1_weighted:.4f}")
        return results

    def save(self, file_path: str = "knn_model.pkl") -> None:
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
                "n_neighbors": self.n_neighbors,
                "weights": self.weights,
                "p": self.p,
                "algorithm": self.algorithm,
                "metric": self.metric,
                "leaf_size": self.leaf_size,
                "n_jobs": self.n_jobs
            }
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {file_path}")

    def load(self, file_path: str = "knn_model.pkl") -> None:
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
        self.label_encoder_mapping = model_data["label_encoder_mapping"]
        
        # Restore hyperparameters
        hparams = model_data["hyperparameters"]
        self.n_neighbors = hparams["n_neighbors"]
        self.weights = hparams["weights"]
        self.p = hparams["p"]
        self.leaf_size = hparams["leaf_size"]
        self.algorithm = hparams["algorithm"]
        self.metric = hparams["metric"]
        self.n_jobs = hparams["n_jobs"]
        
        logger.info(f"Model loaded from {file_path}")