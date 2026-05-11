"""
SQLite-based Feature Store.

Minimal feature store for SME credit risk.
Production would use Feast or cloud-native equivalent.
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime
from typing import Dict, Optional
import os


class FeatureStore:
    """Minimal feature store with point-in-time correctness."""
    
    def __init__(self, db_path: str = "data/processed/feature_vectors.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialise SQLite schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_vectors (
                company_id TEXT,
                feature_date TEXT,
                features_json TEXT,
                created_at TEXT,
                PRIMARY KEY (company_id, feature_date)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_company_date 
            ON feature_vectors (company_id, feature_date)
        ''')
        
        conn.commit()
        conn.close()
    
    def get_feature_vector(self, company_id: str, as_of_date: pd.Timestamp) -> pd.Series:
        """Retrieve pre-materialised features for an entity as of a specific date."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM feature_vectors
            WHERE company_id = ? AND feature_date <= ?
            ORDER BY feature_date DESC
            LIMIT 1
        """
        
        df = pd.read_sql(query, conn, params=(company_id, as_of_date.strftime('%Y-%m-%d')))
        conn.close()
        
        if len(df) == 0:
            return pd.Series(dtype=float)
        
        features = json.loads(df.iloc[0]['features_json'])
        return pd.Series(features)
    
    def materialise_features(self, company_id: str, feature_date: pd.Timestamp,
                             features: Dict):
        """Write computed feature vector to store."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert numpy types to Python types for JSON serialization
        serialisable = {}
        for k, v in features.items():
            if isinstance(v, (np.integer, np.floating)):
                v = float(v) if np.isfinite(v) else None
            elif isinstance(v, np.ndarray):
                v = v.tolist()
            serialisable[k] = v
        
        cursor.execute('''
            INSERT OR REPLACE INTO feature_vectors 
            (company_id, feature_date, features_json, created_at)
            VALUES (?, ?, ?, ?)
        ''', (
            company_id,
            feature_date.strftime('%Y-%m-%d'),
            json.dumps(serialisable),
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_all_vectors(self) -> pd.DataFrame:
        """Retrieve all feature vectors as a DataFrame."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM feature_vectors", conn)
        conn.close()
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Expand JSON features
        features_list = []
        for _, row in df.iterrows():
            features = json.loads(row['features_json'])
            features['company_id'] = row['company_id']
            features['feature_date'] = row['feature_date']
            features_list.append(features)
        
        return pd.DataFrame(features_list)
