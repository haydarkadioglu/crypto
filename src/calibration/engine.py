"""
Probability Calibration Module

Ensures predicted probabilities match observed frequencies.
Methods: Platt Scaling, Isotonic Regression, Binning-based calibration.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Calibrates model output probabilities to match observed frequencies.
    
    A well-calibrated model satisfies: P(UP | p=0.7) ≈ 0.7
    """
    
    def __init__(self, method: str = 'isotonic'):
        """
        Args:
            method: Calibration method ('platt', 'isotonic', 'binning')
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Fit calibrator on validation data.
        
        Args:
            y_true: Binary true labels (0 or 1)
            y_proba: Uncalibrated probability predictions
        """
        if self.method == 'platt':
            # Platt scaling (logistic regression on probabilities)
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_proba.reshape(-1, 1), y_true)
            
        elif self.method == 'isotonic':
            # Isotonic regression (non-parametric, monotonic)
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_proba, y_true)
            
        elif self.method == 'binning':
            # Quantile binning with empirical probabilities
            self.calibrator = self._fit_binning(y_true, y_proba)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted = True
        logger.info(f"Calibrator fitted using {self.method} method")
        
    def _fit_binning(self, y_true: np.ndarray, y_proba: np.ndarray, 
                     n_bins: int = 10) -> dict:
        """Fit binning-based calibrator."""
        # Create bins
        bin_edges = np.percentile(y_proba, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)  # Remove duplicates
        
        bin_centers = []
        bin_probs = []
        
        for i in range(len(bin_edges) - 1):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_centers.append(np.mean(y_proba[mask]))
                bin_probs.append(y_true[mask].mean())
        
        return {
            'bin_edges': bin_edges,
            'bin_centers': np.array(bin_centers),
            'bin_probs': np.array(bin_probs)
        }
    
    def predict(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probabilities.
        
        Args:
            y_proba: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning uncalibrated probabilities")
            return y_proba
        
        if self.method in ['platt', 'isotonic']:
            return self.calibrator.predict(y_proba.reshape(-1, 1))
        
        elif self.method == 'binning':
            # Map each probability to its bin's empirical probability
            calibrated = np.zeros_like(y_proba)
            edges = self.calibrator['bin_edges']
            centers = self.calibrator['bin_centers']
            probs = self.calibrator['bin_probs']
            
            for i in range(len(y_proba)):
                p = y_proba[i]
                # Find bin
                bin_idx = np.searchsorted(edges[1:], p)
                if bin_idx < len(centers):
                    calibrated[i] = probs[bin_idx]
                else:
                    calibrated[i] = p  # Out of bounds, keep original
            
            return calibrated
        
        return y_proba
    
    def evaluate(self, y_true: np.ndarray, y_proba_uncal: np.ndarray, 
                 y_proba_cal: np.ndarray) -> dict:
        """
        Evaluate calibration quality.
        
        Returns metrics comparing calibrated vs uncalibrated probabilities.
        """
        # Brier score (lower is better)
        brier_uncal = brier_score_loss(y_true, y_proba_uncal)
        brier_cal = brier_score_loss(y_true, y_proba_cal)
        
        # Log loss (lower is better)
        logloss_uncal = log_loss(y_true, y_proba_uncal)
        logloss_cal = log_loss(y_true, y_proba_cal)
        
        # Calibration error (average absolute difference between predicted and actual)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        calibration_errors = []
        
        for i in range(n_bins):
            mask = (y_proba_cal >= bin_edges[i]) & (y_proba_cal < bin_edges[i + 1])
            if mask.sum() > 0:
                avg_pred = y_proba_cal[mask].mean()
                avg_actual = y_true[mask].mean()
                calibration_errors.append(abs(avg_pred - avg_actual))
        
        ece = np.mean(calibration_errors) if calibration_errors else 0.0  # Expected Calibration Error
        
        # Reliability curve data
        reliability_data = self._compute_reliability(y_true, y_proba_cal, n_bins)
        
        return {
            'brier_score_uncalibrated': brier_uncal,
            'brier_score_calibrated': brier_cal,
            'brier_improvement': brier_uncal - brier_cal,
            'log_loss_uncalibrated': logloss_uncal,
            'log_loss_calibrated': logloss_cal,
            'expected_calibration_error': ece,
            'reliability_data': reliability_data
        }
    
    def _compute_reliability(self, y_true: np.ndarray, y_proba: np.ndarray, 
                            n_bins: int = 10) -> dict:
        """Compute data for reliability diagram."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        fraction_positives = []
        counts = []
        
        for i in range(n_bins):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            count = mask.sum()
            if count > 0:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                fraction_positives.append(y_true[mask].mean())
                counts.append(count)
            else:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                fraction_positives.append(np.nan)
                counts.append(0)
        
        return {
            'bin_centers': bin_centers,
            'fraction_positives': fraction_positives,
            'counts': counts
        }
    
    def save(self, filepath: str):
        """Save calibrator to disk."""
        joblib.dump({
            'method': self.method,
            'calibrator': self.calibrator,
            'is_fitted': self.is_fitted
        }, filepath)
        logger.info(f"Calibrator saved to {filepath}")
    
    def load(self, filepath: str):
        """Load calibrator from disk."""
        data = joblib.load(filepath)
        self.method = data['method']
        self.calibrator = data['calibrator']
        self.is_fitted = data['is_fitted']
        logger.info(f"Calibrator loaded from {filepath}")


def cross_validate_calibration(model, X: np.ndarray, y: np.ndarray, 
                               n_folds: int = 5, 
                               method: str = 'isotonic') -> dict:
    """
    Evaluate calibration using cross-validation.
    
    This prevents overfitting by ensuring calibration is evaluated on unseen data.
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_proba_uncal = []
    all_y_proba_cal = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get uncalibrated probabilities
        y_proba_uncal = model.predict_proba(X_val)[:, 1]
        
        # Fit calibrator on training predictions
        y_proba_train = model.predict_proba(X_train)[:, 1]
        calibrator = ProbabilityCalibrator(method=method)
        calibrator.fit(y_train, y_proba_train)
        
        # Calibrate validation probabilities
        y_proba_cal = calibrator.predict(y_proba_uncal)
        
        all_y_true.extend(y_val)
        all_y_proba_uncal.extend(y_proba_uncal)
        all_y_proba_cal.extend(y_proba_cal)
    
    # Evaluate overall calibration
    y_true = np.array(all_y_true)
    y_proba_uncal = np.array(all_y_proba_uncal)
    y_proba_cal = np.array(all_y_proba_cal)
    
    evaluation = calibrator.evaluate(y_true, y_proba_uncal, y_proba_cal)
    
    return {
        'y_true': y_true,
        'y_proba_uncalibrated': y_proba_uncal,
        'y_proba_calibrated': y_proba_cal,
        **evaluation
    }


if __name__ == "__main__":
    # Test calibration
    from src.features.engine import FeatureEngine, create_targets
    from src.models.trainer import LightGBMModel, prepare_data
    from sklearn.model_selection import train_test_split
    
    print("Loading data...")
    df = pd.read_parquet('data/raw/BTCUSDT_15m.parquet').head(10000)
    
    print("Creating features...")
    engine = FeatureEngine()
    df = engine.compute_all_features(df)
    df = create_targets(df, horizons=[15])
    
    feature_cols = engine.get_feature_columns(df)
    df = engine.clean_features(df, feature_cols)
    
    X, y = prepare_data(df, feature_cols, 'target_binary_15')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    # Train model
    print("Training model...")
    model = LightGBMModel()
    model.fit(X_train, y_train)
    
    # Get predictions
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Fit calibrator
    print("\nFitting calibrator...")
    y_proba_train = model.predict_proba(X_train)[:, 1]
    
    calibrator = ProbabilityCalibrator(method='isotonic')
    calibrator.fit(y_train, y_proba_train)
    
    # Calibrate test predictions
    y_proba_calibrated = calibrator.predict(y_proba_test)
    
    # Evaluate
    print("\n=== CALIBRATION RESULTS ===")
    eval_results = calibrator.evaluate(y_test, y_proba_test, y_proba_calibrated)
    
    print(f"Brier Score (uncalibrated): {eval_results['brier_score_uncalibrated']:.4f}")
    print(f"Brier Score (calibrated):   {eval_results['brier_score_calibrated']:.4f}")
    print(f"Improvement: {eval_results['brier_improvement']:.4f}")
    print(f"Log Loss (uncalibrated): {eval_results['log_loss_uncalibrated']:.4f}")
    print(f"Log Loss (calibrated):   {eval_results['log_loss_calibrated']:.4f}")
    print(f"Expected Calibration Error: {eval_results['expected_calibration_error']:.4f}")
    
    # Show reliability data
    rel = eval_results['reliability_data']
    print("\nReliability Data:")
    print("Bin Center | Actual Frequency | Count")
    for i, (center, freq, count) in enumerate(zip(rel['bin_centers'], rel['fraction_positives'], rel['counts'])):
        if not np.isnan(freq):
            print(f"{center:.2f}       | {freq:.3f}            | {count}")
