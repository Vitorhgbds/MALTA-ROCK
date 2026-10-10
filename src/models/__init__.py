# type: ignore
from abc import ABC, abstractmethod
from typing import NamedTuple, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

ModelDataset = NamedTuple('ModelDataset', [('train', list), ('train_target', list), ('test', list), ('test_target', list)])

class BaseModel(ABC):
    def __init__(self):
        self.model = None
    
    @abstractmethod
    def train(self, data: ModelDataset) -> None:
        pass
    
    @abstractmethod
    def predict(self, input_data: list) -> list:
        pass
    
    @abstractmethod
    def evaluate(self, test_data: ModelDataset) -> dict:
        pass
    
    @abstractmethod
    def save(self, file_path: str) -> None:
        pass

class FeatureProcessingPolicy(ABC):
    def __init__(
        self, 
        labels: list[str] | None = None,
        total_wells: int | None = None, 
        apply_scaling: bool = True,
        apply_presence: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
        *args, **kwargs):
        self.labels = labels
        self.features = [
            "depth", "GR", "SP", "DT", "CALI", "ILD", "RHOB",
            "NPHI", "MSFL", "DRHO", "SFLU" 
        ]
        self.total_wells = total_wells
        self.apply_scaling = apply_scaling
        self.apply_presence = apply_presence
        self.test_size = test_size
        self.random_state = random_state
    
    @property
    def __select_features(self) -> str:
        """
        Constructs the SQL SELECT clause for fetching features.
        If self.apply_scaling is True, it applies value range filters to each feature.
        Otherwise, it selects all features without filtering.
        """
        return """SELECT  
            Features.well,
            Features.depth,
            CASE WHEN Features.GR BETWEEN 0 AND 200 THEN Features.GR ELSE NULL END AS GR,
            CASE WHEN Features.SP BETWEEN -100 AND 100 THEN Features.SP ELSE NULL END AS SP,
            CASE WHEN Features.DT BETWEEN 40 AND 140 THEN Features.DT ELSE NULL END AS DT,
            CASE WHEN Features.CALI BETWEEN 6 AND 16 THEN Features.CALI ELSE NULL END AS CALI,
            CASE WHEN Features.ILD BETWEEN 0.2 AND 2000 THEN Features.ILD ELSE NULL END AS ILD,
            CASE WHEN Features.RHOB BETWEEN 1.95 AND 2.95 THEN Features.RHOB ELSE NULL END AS RHOB,
            CASE WHEN Features.NPHI BETWEEN -15 AND 45 THEN Features.NPHI ELSE NULL END AS NPHI,
            CASE WHEN Features.MSFL BETWEEN 0.2 AND 2000 THEN Features.MSFL ELSE NULL END AS MSFL,
            CASE WHEN Features.DRHO BETWEEN -0.3 AND 0.3 THEN Features.DRHO ELSE NULL END AS DRHO,
            CASE WHEN Features.SFLU BETWEEN 0.2 AND 2000 THEN Features.SFLU ELSE NULL END AS SFLU,
            Features.rock
        """ if self.apply_scaling else "SELECT *"
        
    @property
    def __label_filter(self) -> str:
        """
        Constructs the SQL WHERE clause for filtering based on specified labels.
        If self.labels is provided, it filters the 'rock' column to include only those labels (case-insensitive).
        If no labels are specified, it ensures that the 'rock' column is not NULL.
        """
        return f"""WHERE
            lower(Features.rock) IN ({', '.join(f"'{label.lower()}'" for label in self.labels)})
        """ if self.labels else "WHERE Features.rock IS NOT NULL"
        
    @property    
    def sql_features(self) -> str:
        limit = f"LIMIT {self.total_wells}" if self.total_wells else ""
        return f"""
            WITH wells AS (
                SELECT DISTINCT well FROM Features
                {self.__label_filter if self.labels else ""}
                {limit}
            )
            {self.__select_features}
            FROM Features 
            JOIN wells W USING (well) 
            {self.__label_filter}
            ORDER BY Features.well, Features.depth ASC;
            """
            
    def build_presence(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
        """
        Adds presence indicator columns for each feature in the dataframe.
        - Updates self.features to include the new presence columns.
        
        Args:
            df (pd.DataFrame): The input dataframe containing feature columns.
        Returns:
            pd.DataFrame: The dataframe with added presence indicator columns.
        """
        features = self.features.copy()
        features.remove("depth")  # Exclude 'depth' from presence indicators
        new_features = []
        for f in features:
            presence_col = f"{f}_present"
            df[presence_col] = df[f].notna().astype(int)
            new_features.append(presence_col)
        
        return df, new_features
    
    def split_train_test(self, df: pd.DataFrame) -> ModelDataset:
        
        # scikit-learn indexes the input with NumPy arrays; use a plain list
        # so PyArrow-backed pandas arrays do not receive those indices directly.
        unique_wells = df['well'].dropna().unique().tolist()
        # Split wells into train and test (80-20 split)
        train_wells, test_wells = train_test_split(unique_wells, test_size=self.test_size, random_state=self.random_state)

        # Create train and test datasets based on well splits
        train_data = df[df['well'].isin(train_wells)].sort_values(['well', 'depth'])
        test_data = df[df['well'].isin(test_wells)].sort_values(['well', 'depth'])

        train_target = train_data['rock']
        test_target = test_data['rock']
        return ModelDataset(
            train=train_data[self.features],
            train_target=train_target,
            test=test_data[self.features],
            test_target=test_target
        )
    
    @abstractmethod
    def fetch(self) -> ModelDataset:
        pass


class ModelBuilderEngine(ABC):
    def __init__(self, policy: FeatureProcessingPolicy, model: BaseModel) -> None:
        self.policy: FeatureProcessingPolicy = policy
        self.model: BaseModel = model

    @abstractmethod
    def build(self) -> BaseModel:
        pass
