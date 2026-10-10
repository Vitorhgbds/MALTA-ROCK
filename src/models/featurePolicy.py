# type: ignore
from pathlib import Path
import sqlite3
from typing import Tuple
from src.logger import Logger
from src.models import ModelDataset, FeatureProcessingPolicy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


logger = Logger().get_logger()

class DefaultFeatures(FeatureProcessingPolicy):
    
    def __init__(self, apply_rolling_stats: bool = True, window: int = 5, *args, **kwargs) -> None:
        self.apply_rolling_stats = apply_rolling_stats
        self.window = window
        super().__init__(*args, **kwargs)
    
    def _add_sequential_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
        """Add lagged features to capture sequential dependencies"""
        features = self.features.copy()
        features.remove("depth")  # Exclude 'depth' from sequential features
        new_features = []
        
        df = df.sort_values(['well', 'depth'])  # Ensure sorted by well and depth
        for f in features:
            df[f'{f}_rolling_mean'] = df.groupby('well')[f].transform(
                lambda x: x.rolling(window=self.window, min_periods=1).mean()
            )
            df[f'{f}_rolling_std'] = df.groupby('well')[f].transform(
                lambda x: x.rolling(window=self.window, min_periods=1).std()
            )
            new_features.extend([f'{f}_rolling_mean', f'{f}_rolling_std'])
        
        return df, new_features
    
    
    def fetch(self) -> ModelDataset:
        database_path = Path(__file__).parent.parent / "database" / "features.db"
        
        logger.info(f"Fetching features from database at {database_path}")
        conn = sqlite3.connect(database_path)
        
        logger.debug("Connected to the database successfully.")
        logger.debug("Executing SQL query to fetch features.")
        logger.debug(f"SQL Query: \n{self.sql_features}")
        df_features = pd.read_sql_query(self.sql_features, conn)
        
        logger.debug(f"Fetched {len(df_features)} rows from the database.")
        logger.debug(f"Total wells fetched: {len(df_features['well'].unique())}")
        features_columns = [
                'depth',
                'GR', 
                'SP', 
                'DT', 
                'CALI', 
                'ILD', 
                'RHOB', 
                'NPHI', 
                'MSFL',
                'DRHO', 
                'SFLU']
        valid_rocks = [
            'FOLHELHO', 'ARENITO', 'SILTITO', 'CALCILUTITO', 'AREIA', 'CALCARENITO',
            'ARGILA', 'CALCARIO', 'CONGLOMERADO', 'ARGILITO', 'ANIDRITA', 'MARGA',
            'GRANITO', 'BASALTO', 'DOLOMITO'
        ]

        all_features = self.features.copy()
        
        if self.apply_presence:
            logger.info("Applying presence indicators to features.")
            df_features, new_features = self.build_presence(df_features)
            all_features.extend(new_features)
            logger.info("Done.")
        
        if self.apply_rolling_stats:
            logger.info("Applying rolling statistics to features.")
            df_features, new_features = self._add_sequential_features(df_features)
            all_features.extend(new_features)
            logger.info("Done.")
        
        logger.debug("Completed data cleaning and preprocessing steps.")
        logger.debug(f"Dataframe shape after cleaning: {df_features.shape}")
        logger.debug(f"Dataframe sample after cleaning: {df_features.head()}")
        logger.debug(f"Total features used: {len(all_features)}")
        logger.debug(f"Features: {all_features}")
        self.features = all_features
        
        return self.split_train_test(df_features)