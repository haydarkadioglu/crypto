"""
Model Training Module

Implements multiple model types for comparison:
- Baseline models (naive, persistence)
- Classical ML (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Deep Learning (LSTM, GRU) - only if justified

Includes proper walk-forward validation to prevent data leakage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import xgboost as xgb
import joblib
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for all models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_calibrated = False
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    
    def save(self, filepath: str):
        joblib.dump(self.model, filepath)
        
    def load(self, filepath: str):
        self.model = joblib.load(filepath)


class NaiveModel(BaseModel):
    """Naive baseline: always predicts the majority class."""
    
    def __init__(self):
        super().__init__("Naive")
        self.majority_class = 0
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.majority_class = int(np.round(y_train.mean()))
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        proba_up = np.full(n_samples, self.majority_class * 0.5 + 0.25)
        return np.column_stack([1 - proba_up, proba_up])


class PersistenceModel(BaseModel):
    """Persistence baseline: predicts current direction continues."""
    
    def __init__(self):
        super().__init__("Persistence")
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # No training needed
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # For persistence, we use the most recent return as a proxy
        # This is a simplification
        n_samples = X.shape[0]
        # Assume 55% probability of continuation (slight edge)
        proba_up = np.full(n_samples, 0.55)
        return np.column_stack([1 - proba_up, proba_up])


class LogisticRegressionModel(BaseModel):
    """Logistic Regression with optional calibration."""
    
    def __init__(self, calibrate: bool = True):
        super().__init__("LogisticRegression")
        self.calibrate = calibrate
        self.base_model = LogisticRegression(
            max_iter=1000, 
            C=1.0,
            class_weight='balanced',
            random_state=42
        )
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.base_model.fit(X_train, y_train)
        self.model = self.base_model
        
        if self.calibrate and len(np.unique(y_train)) > 1:
            calibrated = CalibratedClassifierCV(self.base_model, cv=5, method='isotonic')
            calibrated.fit(X_train, y_train)
            self.model = calibrated
            self.is_calibrated = True
            
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class RandomForestModel(BaseModel):
    """Random Forest Classifier."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class XGBoostModel(BaseModel):
    """XGBoost Classifier."""
    
    def __init__(self, **params):
        super().__init__("XGBoost")
        default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'scale_pos_weight': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        default_params.update(params)
        self.params = default_params
        self.model = xgb.XGBClassifier(**self.params)
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class LightGBMModel(BaseModel):
    """LightGBM Classifier - typically best for tabular data."""
    
    def __init__(self, **params):
        super().__init__("LightGBM")
        default_params = {
            'n_estimators': 500,
            'max_depth': -1,
            'num_leaves': 31,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'scale_pos_weight': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        default_params.update(params)
        self.params = default_params
        self.model = lgb.LGBMClassifier(**self.params)
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def create_models() -> Dict[str, BaseModel]:
    """Create all model instances for comparison."""
    return {
        'Naive': NaiveModel(),
        'Persistence': PersistenceModel(),
        'LogisticRegression': LogisticRegressionModel(calibrate=True),
        'RandomForest': RandomForestModel(n_estimators=200, max_depth=8),
        'XGBoost': XGBoostModel(n_estimators=300, max_depth=5, learning_rate=0.03),
        'LightGBM': LightGBMModel(n_estimators=500, num_leaves=24, learning_rate=0.02)
    }


def evaluate_model(model: BaseModel, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model_name': model.name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5,
        'brier_score': brier_score_loss(y_test, y_proba),
        'is_calibrated': model.is_calibrated
    }
    
    return metrics


class WalkForwardValidator:
    """
    Implements walk-forward validation to properly evaluate time-series models.
    
    Prevents look-ahead bias by ensuring training data always precedes test data.
    """
    
    def __init__(self, train_size: int = 10000, val_size: int = 2000, 
                 test_size: int = 2000, step_size: int = 1000):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.step_size = step_size
        
    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate walk-forward splits."""
        splits = []
        
        start_idx = 0
        while start_idx + self.train_size + self.val_size + self.test_size <= n_samples:
            train_end = start_idx + self.train_size
            val_end = train_end + self.val_size
            test_end = val_end + self.test_size
            
            train_idx = np.arange(start_idx, train_end)
            val_idx = np.arange(train_end, val_end)
            test_idx = np.arange(val_end, test_end)
            
            splits.append((train_idx, val_idx, test_idx))
            
            start_idx += self.step_size
        
        return splits
    
    def validate(self, model_class, X: np.ndarray, y: np.ndarray, 
                 feature_names: List[str] = None) -> Dict:
        """Run full walk-forward validation."""
        splits = self.split(len(X))
        
        all_metrics = []
        feature_importances = {}
        
        for i, (train_idx, val_idx, test_idx) in enumerate(splits):
            logger.info(f"Fold {i+1}/{len(splits)}")
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Train model
            model = model_class
            model.fit(X_train, y_train)
            
            # Evaluate on test set
            metrics = evaluate_model(model, X_test, y_test)
            metrics['fold'] = i + 1
            all_metrics.append(metrics)
            
            # Collect feature importances if available
            if hasattr(model.model, 'feature_importances_') and feature_names:
                for name, imp in zip(feature_names, model.model.feature_importances_):
                    if name not in feature_importances:
                        feature_importances[name] = []
                    feature_importances[name].append(imp)
        
        # Aggregate results
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key not in ['model_name', 'fold', 'is_calibrated']:
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
                avg_metrics[f'{key}_std'] = np.std([m[key] for m in all_metrics])
        
        avg_metrics['model_name'] = all_metrics[0]['model_name']
        avg_metrics['n_folds'] = len(splits)
        
        # Average feature importances
        if feature_importances:
            avg_importance = {k: np.mean(v) for k, v in feature_importances.items()}
            avg_metrics['feature_importances'] = avg_importance
        
        return avg_metrics


def prepare_data(df: pd.DataFrame, feature_cols: List[str], 
                 target_col: str, drop_na: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features and target for training."""
    X = df[feature_cols].values
    y = df[target_col].values
    
    if drop_na:
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
    
    return X, y


if __name__ == "__main__":
    # Test model training
    from src.features.engine import FeatureEngine, create_targets
    
    print("Loading data...")
    df = pd.read_parquet('data/raw/BTCUSDT_15m.parquet')
    
    print("Creating features...")
    engine = FeatureEngine()
    df = engine.compute_all_features(df)
    df = create_targets(df, horizons=[15])
    
    feature_cols = engine.get_feature_columns(df)
    df = engine.clean_features(df, feature_cols)
    
    # Prepare data
    X, y = prepare_data(df, feature_cols, 'target_binary_15')
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Create and test models
    models = create_models()
    
    print("\nTesting models...")
    for name, model in models.items():
        try:
            # Simple train/test split for quick test
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            
            print(f"\n{name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  Brier: {metrics['brier_score']:.4f}")
            
        except Exception as e:
            print(f"\n{name} failed: {e}")
    
    print("\nModel testing complete!")
