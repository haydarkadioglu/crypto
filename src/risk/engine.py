"""
Risk Assessment Module

Estimates trading risk based on multiple factors:
- Market volatility
- Prediction confidence
- Historical model performance
- Market regime
- Liquidity conditions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Result of risk assessment."""
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME
    risk_score: float  # 0.0 to 1.0
    volatility_risk: float
    confidence_risk: float
    model_risk: float
    regime_risk: float
    factors: Dict[str, float]


class RiskEngine:
    """
    Comprehensive risk assessment engine.
    
    Evaluates multiple dimensions of risk to produce an overall risk score.
    """
    
    def __init__(self, lookback_period: int = 100):
        """
        Args:
            lookback_period: Number of historical periods for calculations
        """
        self.lookback_period = lookback_period
        self.model_history = []  # Track model predictions vs actuals
        
    def assess(self, df: pd.DataFrame, 
               current_idx: int,
               predicted_proba: float,
               horizon: int = 15,
               model_predictions: Optional[pd.DataFrame] = None) -> RiskAssessment:
        """
        Assess risk for a given prediction point.
        
        Args:
            df: DataFrame with OHLCV and features
            current_idx: Current row index
            predicted_proba: Model's predicted probability of UP
            horizon: Prediction horizon in bars
            model_predictions: Historical model predictions for performance tracking
            
        Returns:
            RiskAssessment with detailed risk breakdown
        """
        # Get current market state
        current_row = df.iloc[current_idx]
        historical = df.iloc[max(0, current_idx - self.lookback_period):current_idx + 1]
        
        # Calculate component risks
        vol_risk = self._calculate_volatility_risk(historical)
        conf_risk = self._calculate_confidence_risk(predicted_proba)
        model_perf_risk = self._calculate_model_performance_risk(model_predictions) if model_predictions is not None else 0.5
        regime_risk = self._calculate_regime_risk(historical)
        
        # Calculate composite risk score
        # Weighted average of component risks
        weights = {
            'volatility': 0.30,
            'confidence': 0.25,
            'model_performance': 0.25,
            'regime': 0.20
        }
        
        risk_score = (
            weights['volatility'] * vol_risk +
            weights['confidence'] * conf_risk +
            weights['model_performance'] * model_perf_risk +
            weights['regime'] * regime_risk
        )
        
        # Clip to valid range
        risk_score = np.clip(risk_score, 0.0, 1.0)
        
        # Determine risk level
        risk_level = self._score_to_level(risk_score)
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            volatility_risk=vol_risk,
            confidence_risk=conf_risk,
            model_risk=model_perf_risk,
            regime_risk=regime_risk,
            factors={
                'atr_pct': historical['close'].pct_change().std() if len(historical) > 1 else 0,
                'probability_distance_from_05': abs(predicted_proba - 0.5),
                'recent_accuracy': 1 - model_perf_risk if model_predictions is not None else 0.5,
                'vol_regime': self._get_vol_regime(historical)
            }
        )
    
    def _calculate_volatility_risk(self, historical: pd.DataFrame) -> float:
        """
        Calculate risk from market volatility.
        
        Higher volatility = higher risk
        """
        if len(historical) < 10:
            return 0.5  # Default for insufficient data
        
        # Realized volatility (annualized)
        returns = historical['close'].pct_change().dropna()
        realized_vol = returns.std() * np.sqrt(35040)  # 15-min bars per year
        
        # ATR as percentage of price
        high_low = historical['high'] - historical['low']
        atr = high_low.rolling(14).mean().iloc[-1] / historical['close'].iloc[-1] * 100
        
        # Normalize to 0-1 scale (using empirical thresholds)
        # Typical crypto 15min vol: 0.5-2%, ATR: 0.3-1%
        vol_normalized = min(realized_vol / 0.5, 2.0) / 2.0  # Cap at 2x normal
        atr_normalized = min(atr / 1.0, 2.0) / 2.0
        
        return 0.6 * vol_normalized + 0.4 * atr_normalized
    
    def _calculate_confidence_risk(self, predicted_proba: float) -> float:
        """
        Calculate risk from prediction uncertainty.
        
        Probabilities near 0.5 indicate high uncertainty.
        Probabilities near 0 or 1 indicate confidence.
        """
        # Distance from maximum uncertainty (0.5)
        distance = abs(predicted_proba - 0.5)
        
        # Convert to risk (closer to 0.5 = higher risk)
        # At 0.5: risk = 1.0, at 0 or 1: risk = 0.0
        confidence_risk = 1.0 - (distance * 2)
        
        return confidence_risk
    
    def _calculate_model_performance_risk(self, predictions: pd.DataFrame) -> float:
        """
        Calculate risk from recent model performance.
        
        Poor recent performance = higher risk
        """
        if predictions is None or len(predictions) < 20:
            return 0.5  # Default for insufficient data
        
        # Recent accuracy
        if 'prediction' in predictions.columns and 'actual' in predictions.columns:
            recent = predictions.tail(50)
            accuracy = (recent['prediction'] == recent['actual']).mean()
            
            # Convert accuracy to risk (accuracy < 0.5 = high risk)
            # At 0.5 accuracy: risk = 1.0, at 1.0 accuracy: risk = 0.0
            accuracy_risk = 1.0 - ((accuracy - 0.5) * 2)
            return np.clip(accuracy_risk, 0.0, 1.0)
        
        return 0.5
    
    def _calculate_regime_risk(self, historical: pd.DataFrame) -> float:
        """
        Calculate risk from market regime.
        
        Certain regimes are inherently riskier:
        - High volatility regimes
        - Transitioning regimes
        - Low liquidity regimes
        """
        if len(historical) < 50:
            return 0.5
        
        regime = self._get_vol_regime(historical)
        
        # Map regime to risk
        regime_risk_map = {
            'very_low': 0.3,   # Low vol can mean opportunity or stagnation
            'low': 0.4,
            'normal': 0.5,
            'high': 0.7,
            'very_high': 0.9,
            'extreme': 1.0
        }
        
        return regime_risk_map.get(regime, 0.5)
    
    def _get_vol_regime(self, historical: pd.DataFrame) -> str:
        """Determine volatility regime."""
        if len(historical) < 20:
            return 'normal'
        
        # Current volatility
        current_vol = historical['close'].pct_change().rolling(20).std().iloc[-1]
        
        # Historical average
        long_vol = historical['close'].pct_change().rolling(100).std().mean()
        
        if pd.isna(current_vol) or pd.isna(long_vol) or long_vol == 0:
            return 'normal'
        
        ratio = current_vol / long_vol
        
        if ratio < 0.5:
            return 'very_low'
        elif ratio < 0.8:
            return 'low'
        elif ratio < 1.2:
            return 'normal'
        elif ratio < 1.5:
            return 'high'
        elif ratio < 2.0:
            return 'very_high'
        else:
            return 'extreme'
    
    def _score_to_level(self, score: float) -> str:
        """Convert numeric risk score to categorical level."""
        if score < 0.25:
            return 'LOW'
        elif score < 0.50:
            return 'MEDIUM'
        elif score < 0.75:
            return 'HIGH'
        else:
            return 'EXTREME'
    
    def update_history(self, prediction: int, actual: int):
        """Update model performance history."""
        self.model_history.append({
            'prediction': prediction,
            'actual': actual,
            'timestamp': pd.Timestamp.now()
        })
        
        # Keep only recent history
        if len(self.model_history) > 500:
            self.model_history = self.model_history[-500:]
    
    def get_recent_accuracy(self, n: int = 50) -> float:
        """Get recent model accuracy."""
        if len(self.model_history) < n:
            return 0.5  # Default
        
        recent = self.model_history[-n:]
        correct = sum(1 for r in recent if r['prediction'] == r['actual'])
        return correct / len(recent)


def create_risk_summary(df: pd.DataFrame, 
                        predictions: pd.Series,
                        proba_col: str,
                        horizon: int = 15) -> pd.DataFrame:
    """
    Create risk assessment for all predictions in a DataFrame.
    
    Returns DataFrame with risk scores and levels.
    """
    risk_engine = RiskEngine()
    
    risk_scores = []
    risk_levels = []
    
    for idx in range(len(df)):
        if pd.isna(predictions.iloc[idx]):
            risk_scores.append(np.nan)
            risk_levels.append('UNKNOWN')
            continue
        
        proba = predictions.iloc[idx]
        
        try:
            assessment = risk_engine.assess(df, idx, proba, horizon)
            risk_scores.append(assessment.risk_score)
            risk_levels.append(assessment.risk_level)
        except Exception as e:
            risk_scores.append(0.5)
            risk_levels.append('UNKNOWN')
    
    result = df.copy()
    result['risk_score'] = risk_scores
    result['risk_level'] = risk_levels
    
    return result


if __name__ == "__main__":
    # Test risk engine
    from src.features.engine import FeatureEngine, create_targets
    from src.models.trainer import LightGBMModel, prepare_data
    
    print("Loading data...")
    df = pd.read_parquet('data/raw/BTCUSDT_15m.parquet').head(5000)
    
    print("Creating features...")
    engine = FeatureEngine()
    df = engine.compute_all_features(df)
    df = create_targets(df, horizons=[15])
    
    feature_cols = engine.get_feature_columns(df)
    df = engine.clean_features(df, feature_cols)
    
    # Train model and get predictions
    X, y = prepare_data(df, feature_cols, 'target_binary_15')
    split = int(len(X) * 0.8)
    
    model = LightGBMModel()
    model.fit(X[:split], y[:split])
    
    df['proba'] = 0.5
    proba_values = model.predict_proba(X[split:])[:, 1]
    df.iloc[split:, df.columns.get_loc('proba')] = proba_values
    df['prediction'] = (df['proba'] >= 0.5).astype(int)
    df['actual'] = df['target_binary_15']
    
    # Test risk engine
    risk_engine = RiskEngine()
    
    print("\n=== RISK ASSESSMENT SAMPLE ===")
    sample_indices = [split + i * 100 for i in range(5)]
    
    for idx in sample_indices:
        if idx >= len(df):
            break
        
        proba = df['proba'].iloc[idx]
        assessment = risk_engine.assess(df, idx, proba, horizon=15)
        
        print(f"\nIndex {idx}:")
        print(f"  Predicted Probability: {proba:.3f}")
        print(f"  Risk Level: {assessment.risk_level}")
        print(f"  Risk Score: {assessment.risk_score:.3f}")
        print(f"  Volatility Risk: {assessment.volatility_risk:.3f}")
        print(f"  Confidence Risk: {assessment.confidence_risk:.3f}")
        print(f"  Regime: {assessment.factors['vol_regime']}")
    
    # Overall risk distribution
    print("\n=== RISK DISTRIBUTION ===")
    test_df = df[split:].copy()
    test_df['risk_score'] = test_df.apply(
        lambda row: risk_engine.assess(test_df, test_df.index.get_loc(row.name), row['proba'], 15).risk_score 
        if not pd.isna(row['proba']) else np.nan, 
        axis=1
    )
    
    print(test_df['risk_score'].describe())
