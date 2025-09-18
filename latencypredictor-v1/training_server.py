import json
import os
import random
import time
import logging
import threading
import joblib
import uvicorn
import numpy as np
import pandas as pd

from datetime import datetime, timezone
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from fastapi.responses import Response  # Fixed import
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from skl2onnx import update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx, convert_sklearn
from xgboost import XGBRegressor
from skl2onnx.common.shape_calculator import (
    calculate_linear_regressor_output_shapes,
)

import tempfile
import shutil
import os  # Added this import
import onnxmltools


update_registered_converter(
    XGBRegressor,
    "XGBoostXGBRegressor",
    calculate_linear_regressor_output_shapes,
    convert_xgboost,
)


try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Please install with: pip install xgboost")


class ModelType(str, Enum):
    BAYESIAN_RIDGE = "bayesian_ridge"
    XGBOOST = "xgboost"


class RandomDropDeque(deque):
    def __init__(self, maxlen):
        super().__init__()
        self._maxlen = maxlen

    def append(self, item):
        if len(self) >= self._maxlen:
            # pick a random index to evict
            idx = random.randrange(len(self))
            # rotate so that element at idx moves to the left end
            self.rotate(-idx)
            # remove it
            self.popleft()
            # rotate back to original ordering
            self.rotate(idx)
        super().append(item)

    def appendleft(self, item):
        if len(self) >= self._maxlen:
            idx = random.randrange(len(self))
            # rotate so that element at idx moves to the right end
            self.rotate(len(self) - idx - 1)
            self.pop()
            # rotate back
            self.rotate(-(len(self) - idx - 1))
        super().appendleft(item)


# --- Configuration ---
class Settings:
    """
    Configuration class for the latency predictor server.
    Reads settings from environment variables with sensible defaults.
    """
    TTFT_MODEL_PATH: str = os.getenv("LATENCY_TTFT_MODEL_PATH", "/tmp/models/ttft.joblib")
    TPOT_MODEL_PATH: str = os.getenv("LATENCY_TPOT_MODEL_PATH", "/tmp/models/tpot.joblib")
    TTFT_SCALER_PATH: str = os.getenv("LATENCY_TTFT_SCALER_PATH", "/tmp/models/ttft_scaler.joblib")
    TPOT_SCALER_PATH: str = os.getenv("LATENCY_TPOT_SCALER_PATH", "/tmp/models/tpot_scaler.joblib")
    RETRAINING_INTERVAL_SEC: int = int(os.getenv("LATENCY_RETRAINING_INTERVAL_SEC", 1800))
    MIN_SAMPLES_FOR_RETRAIN_FRESH: int = int(os.getenv("LATENCY_MIN_SAMPLES_FOR_RETRAIN_FRESH", 10))
    MIN_SAMPLES_FOR_RETRAIN: int = int(os.getenv("LATENCY_MIN_SAMPLES_FOR_RETRAIN", 1000))
    MAX_TRAINING_DATA_SIZE_PER_BUCKET: int = int(os.getenv("LATENCY_MAX_TRAINING_DATA_SIZE_PER_BUCKET", 10000))
    TEST_TRAIN_RATIO: float = float(os.getenv("LATENCY_TEST_TRAIN_RATIO", "0.1"))  # Default 1:10 (10% test, 90% train)
    MAX_TEST_DATA_SIZE: int = int(os.getenv("LATENCY_MAX_TEST_DATA_SIZE", "1000"))  # Max test samples to keep
    MODEL_TYPE: str = os.getenv("LATENCY_MODEL_TYPE", "xgboost")  # Default to XGBoost
    QUANTILE_ALPHA: float = float(os.getenv("LATENCY_QUANTILE_ALPHA", "0.9"))  # p90 quantile

settings = Settings()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add this to your Pydantic models section
class ModelInfoResponse(BaseModel):
    model_type: str
    xgboost_available: bool
    is_ready: bool
    ttft_training_samples: int = Field(default=0, description="Number of TTFT training samples")
    tpot_training_samples: int = Field(default=0, description="Number of TPOT training samples") 
    ttft_test_samples: int = Field(default=0, description="Number of TTFT test samples")
    tpot_test_samples: int = Field(default=0, description="Number of TPOT test samples")
    last_retrain_time: Optional[datetime] = Field(default=None, description="Last retraining timestamp")
    min_samples_for_retrain: int = Field(default=0, description="Minimum samples required for retraining")
    retraining_interval_sec: int = Field(default=0, description="Retraining interval in seconds")


def quantile_loss(y_true, y_pred, quantile):
    """
    Calculate quantile loss (also known as pinball loss).
    
    For quantile τ (tau), the loss is:
    - (τ - 1) * (y_true - y_pred) if y_true < y_pred (under-prediction)
    - τ * (y_true - y_pred) if y_true >= y_pred (over-prediction)
    
    Args:
        y_true: actual values
        y_pred: predicted quantile values
        quantile: the quantile being predicted (e.g., 0.9 for p90)
    
    Returns:
        Mean quantile loss
    """
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return np.mean(loss)


def quantile_coverage(y_true, y_pred, quantile):
    """
    Calculate quantile coverage - the proportion of actual values that fall below the predicted quantile.
    
    For a well-calibrated p90 model, this should be close to 0.9 (90%).
    
    Args:
        y_true: actual values
        y_pred: predicted quantile values
        quantile: the quantile being predicted (e.g., 0.9 for p90)
    
    Returns:
        Coverage percentage (0-100)
    """
    below_prediction = np.sum(y_true <= y_pred)
    coverage = below_prediction / len(y_true)
    return coverage * 100


def quantile_violation_rate(y_true, y_pred, quantile):
    """
    Calculate quantile violation rate - the proportion of times actual values exceed the predicted quantile.
    
    For a well-calibrated p90 model, this should be close to 0.1 (10%).
    
    Args:
        y_true: actual values
        y_pred: predicted quantile values
        quantile: the quantile being predicted (e.g., 0.9 for p90)
    
    Returns:
        Violation rate percentage (0-100)
    """
    violations = np.sum(y_true > y_pred)
    violation_rate = violations / len(y_true)
    return violation_rate * 100


class LatencyPredictor:
    """
    Manages model training, prediction, and data handling.
    """
    def __init__(self, model_type: str = None):
        # Set model type with validation
        if model_type is None:
            model_type = settings.MODEL_TYPE
        
        if model_type not in [ModelType.BAYESIAN_RIDGE, ModelType.XGBOOST]:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(ModelType)}")
        
        if model_type == ModelType.XGBOOST and not XGBOOST_AVAILABLE:
            logging.warning("XGBoost requested but not available. Falling back to Bayesian Ridge.")
            model_type = ModelType.BAYESIAN_RIDGE
        
        self.model_type = ModelType(model_type)
        self.quantile = settings.QUANTILE_ALPHA
        logging.info(f"Initialized LatencyPredictor with model type: {self.model_type}, quantile: {self.quantile}")

        # Data buckets for sampling
        self.cache_buckets = int(1.0 / 0.05)  # 20 buckets for cache percentage (0-100% in 5% increments)
        self.queue_buckets = 5  # 0, 1-2, 3-5, 6-10, 11+ waiting requests
        self.bucket_size = settings.MAX_TRAINING_DATA_SIZE_PER_BUCKET 

        # Data buckets with tuple keys: (queue_bucket, cache_bucket)
        self.ttft_data_buckets = {
            (q, c): deque(maxlen=self.bucket_size)
            for q in range(self.queue_buckets)
            for c in range(self.cache_buckets)
        }
        self.tpot_data_buckets = {
            (q, c): deque(maxlen=self.bucket_size)
            for q in range(self.queue_buckets)
            for c in range(self.cache_buckets)
        }
    
        
        # Test data storage with configurable max size
        self.ttft_test_data = deque(maxlen=settings.MAX_TEST_DATA_SIZE)
        self.tpot_test_data = deque(maxlen=settings.MAX_TEST_DATA_SIZE)
        
        # Quantile-specific metric tracking (store last 5 scores)
        self.ttft_quantile_loss_scores = deque(maxlen=5)
        self.tpot_quantile_loss_scores = deque(maxlen=5)
        self.ttft_coverage_scores = deque(maxlen=5)
        self.tpot_coverage_scores = deque(maxlen=5)
        self.ttft_violation_rates = deque(maxlen=5)
        self.tpot_violation_rates = deque(maxlen=5)

        self.ttft_model = None
        self.tpot_model = None
        self.ttft_scaler = None
        self.tpot_scaler = None
        
        self.ttft_coefficients = None  # Will store descaled coefficients as dict
        self.tpot_coefficients = None  # Will store descaled coefficients as dict

        self.lock = threading.Lock()
        self.last_retrain_time = None
        self._shutdown_event = threading.Event()
        self._training_thread: threading.Thread = None
        self.guess_type = None
        
    def _get_queue_bucket(self, num_waiting: int) -> int:
        """Map number of waiting requests to queue bucket index."""
        if num_waiting == 0:
            return 0
        elif num_waiting <= 2:
            return 1
        elif num_waiting <= 5:
            return 2
        elif num_waiting <= 10:
            return 3
        else:
            return 4  # 11+ requests

    def _get_cache_bucket(self, cache_percentage: float) -> int:
        """Map cache percentage to cache bucket index."""
        pct = max(0.0, min(1.0, cache_percentage))
        return min(int(pct * self.cache_buckets), self.cache_buckets - 1)

    def _get_bucket_key(self, sample: dict) -> tuple:
        """Get (queue_bucket, cache_bucket) tuple key for a sample."""
        queue_bucket = self._get_queue_bucket(sample['num_request_waiting'])
        cache_bucket = self._get_cache_bucket(sample['kv_cache_percentage'])
        return (queue_bucket, cache_bucket)
        
    def _store_descaled_coefficients(self, model, scaler, feature_names, model_name):
        """
        Store descaled coefficients for Bayesian Ridge models.
        Returns a dict with feature names as keys and coefficients as values.
        """
        if self.model_type != ModelType.BAYESIAN_RIDGE or model is None or scaler is None:
            return None
            
        try:
            # Get scaled coefficients and scaler parameters
            coef_scaled = model.coef_
            scale, mean = scaler.scale_, scaler.mean_
            
            # Descale coefficients: w_original = w_scaled / scale
            w_orig = coef_scaled / scale
            
            # Calculate descaled intercept: b_orig = b_scaled - sum(w_scaled * mean / scale)
            intercept = float(model.intercept_) - float(np.dot(coef_scaled, mean / scale))
            
            # Create coefficient dictionary
            coefficients = {"intercept": intercept}
            for feature, coef in zip(feature_names, w_orig):
                coefficients[feature] = float(coef)
                
            logging.info(f"Stored descaled coefficients for {model_name}: {coefficients}")
            return coefficients
            
        except Exception as e:
            logging.error(f"Error storing descaled coefficients for {model_name}: {e}")
            return None

    def shutdown(self):
        """Signal the training thread to exit and join it."""
        self._shutdown_event.set()
        if self._training_thread is not None:
            self._training_thread.join()

    @property
    def is_ready(self) -> bool:
        """Checks if all models and scalers are loaded/trained."""
        if self.model_type == ModelType.BAYESIAN_RIDGE:
            return all([self.ttft_model, self.tpot_model, self.ttft_scaler, self.tpot_scaler])
        else:  # XGBoost
            return all([self.ttft_model, self.tpot_model])

    @is_ready.setter
    def is_ready(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("is_ready must be a boolean value.")
        self._is_ready_override = value

    def _all_samples(self, buckets: dict) -> list:
        samples = []
        for bucket_deque in buckets.values():
            samples.extend(bucket_deque)
        return samples

    def _train_model_with_scaling(self, features: pd.DataFrame, target: pd.Series) -> Union[Tuple[BayesianRidge, StandardScaler], xgb.XGBRegressor]:
        logging.info("_train_model_with_scaling")
        try:
            if len(features) == 0 or len(target) == 0:
                raise ValueError("Empty training data")
            if features.isnull().any().any() or target.isnull().any():
                raise ValueError("Training data contains NaN values")
            if np.isinf(features.values).any() or np.isinf(target.values).any():
                raise ValueError("Training data contains infinite values")

            if self.model_type == ModelType.BAYESIAN_RIDGE:
                print("bayesian ridge")
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
                    raise ValueError("Scaling produced invalid values")

                # For Bayesian Ridge, we'll approximate quantile regression by training on the mean
                # but adjusting predictions later. This is not ideal but Bayesian Ridge doesn't
                # natively support quantile regression.
                model = BayesianRidge(compute_score=True)
                print("calling fit")
                model.fit(features_scaled, target)
                self.guess_type = [('float_input', FloatTensorType([None, features_scaled.shape[1]]))]
                return model, scaler
            
            else:  # XGBoost with quantile regression
                print("xgbregressor")
                model = xgb.XGBRegressor(
                    n_estimators=200,            # Number of trees to build (moderate value for balanced accuracy and speed)
                    max_depth=6,                 # Depth of trees; 6 is typically a sweet spot balancing bias/variance
                    learning_rate=0.05,          # Smaller learning rate to achieve stable convergence
                    subsample=0.8,               # Use 80% of data per tree (adds regularization & reduces overfitting)
                    colsample_bytree=0.8,        # Use 80% of features per tree (improves generalization)
                    min_child_weight=5,          # Helps control tree splits, reducing overfitting on small datasets
                    gamma=0.1,                   # Adds conservative regularization; prevents overfitting
                    objective="reg:quantileerror",    # quantile regression
                    quantile_alpha=self.quantile,    # Use configured quantile (e.g., 0.9 for p90)
                    tree_method='hist',          # Efficient histogram algorithm; optimal for large datasets
                    n_jobs=-1,                   # Utilize all CPU cores for parallel training
                    random_state=42,             # Ensures reproducible results
                    verbosity=1   
                )
                
                model.fit(features, target)
                self.guess_type = [('float_input', FloatTensorType([None, features.shape[1]]))]
                return model
                
        except Exception as e:
            logging.error(f"Error in _train_model_with_scaling: {e}", exc_info=True)
            raise
        
    def _calculate_quantile_metrics_on_test(self, model, scaler, test_data, feature_cols, target_col):
        """Calculate quantile-specific metrics on test data"""
        try:
            df = pd.DataFrame(test_data).dropna()
            df = df[df[target_col] > 0]
            
            if len(df) < 2:
                return None, None, None
            
            X = df[feature_cols]
            if self.model_type == ModelType.BAYESIAN_RIDGE and scaler is not None:
                X = scaler.transform(X)
            
            y_true = df[target_col].values
            y_pred = model.predict(X)
            
            # For Bayesian Ridge (which doesn't do true quantile regression), 
            # we'll estimate the quantile by adding a factor to the mean prediction
            if self.model_type == ModelType.BAYESIAN_RIDGE:
                # Rough approximation: add some multiple of std to get to desired quantile
                # This is a simplification - in practice you'd want proper quantile regression
                std_factor = 1.28 if self.quantile == 0.9 else (2.0 if self.quantile == 0.95 else 0.674)
                _, y_std = model.predict(X, return_std=True)
                y_pred = y_pred + std_factor * y_std
            
            # Calculate quantile-specific metrics
            ql = quantile_loss(y_true, y_pred, self.quantile)
            coverage = quantile_coverage(y_true, y_pred, self.quantile)
            violation_rate = quantile_violation_rate(y_true, y_pred, self.quantile)
            
            return ql, coverage, violation_rate
            
        except Exception as e:
            logging.error(f"Error calculating quantile metrics: {e}", exc_info=True)
            return None, None, None

    def _create_default_model(self, model_type: str) -> Union[Tuple[BayesianRidge, StandardScaler], xgb.XGBRegressor]:
        """Creates and trains a simple default model with initial priors."""
        try:
            logging.info(f"Creating default '{model_type}' model with priors.")
            if model_type == "ttft":
                features = pd.DataFrame({
                    'kv_cache_percentage': [0.0, ],
                    'input_token_length': [1, ],
                    'num_request_waiting': [0, ],
                    'num_request_running': [0, ],
                    'prefix_cache_score': [0.0, ]  # Added prefix_cache_score
                })
                target = pd.Series([10,])
            else:
                features = pd.DataFrame({
                    'kv_cache_percentage': [0.0],
                    'input_token_length': [1],  # Added input_token_length
                    'num_request_waiting': [0, ],
                    'num_request_running': [0, ],
                    'num_tokens_generated': [1,]
                })
                target = pd.Series([10.0])
            return self._train_model_with_scaling(features, target)
        except Exception as e:
            logging.error(f"Error creating default model for {model_type}: {e}", exc_info=True)
            raise

    def train(self):
        try:
            with self.lock:
                ttft_snap = list(self._all_samples(self.ttft_data_buckets))
                tpot_snap = list(self._all_samples(self.tpot_data_buckets))
                total = len(ttft_snap) + len(tpot_snap)
                if total < settings.MIN_SAMPLES_FOR_RETRAIN:
                    logging.info(f"Skipping training: only {total} samples (< {settings.MIN_SAMPLES_FOR_RETRAIN}).")
                    return
                logging.info(f"Initiating training with {total} samples using {self.model_type} for quantile {self.quantile}.")

            new_ttft_model = new_ttft_scaler = None
            new_tpot_model = new_tpot_scaler = None

            # Train TTFT
            if ttft_snap:
                df_ttft = pd.DataFrame(ttft_snap).dropna()
                df_ttft = df_ttft[df_ttft['actual_ttft_ms'] > 0]
                print(f"TTFT training data size: {len(df_ttft)} with sample data: {df_ttft.columns.tolist()}")
                if len(df_ttft) >= settings.MIN_SAMPLES_FOR_RETRAIN:
                    # Updated TTFT features to include prefix_cache_score
                    X_ttft = df_ttft[['kv_cache_percentage', 'input_token_length', 'num_request_waiting', 'num_request_running', 'prefix_cache_score']]
                    y_ttft = df_ttft['actual_ttft_ms']
                    try:
                        result = self._train_model_with_scaling(X_ttft, y_ttft)
                        if self.model_type == ModelType.BAYESIAN_RIDGE:
                            new_ttft_model, new_ttft_scaler = result
                        else:
                            new_ttft_model = result
                            new_ttft_scaler = None
                        
                        # Calculate quantile metrics on test data
                        ttft_feature_cols = ['kv_cache_percentage', 'input_token_length', 'num_request_waiting', 'num_request_running', 'prefix_cache_score']
                        ql, coverage, violation_rate = self._calculate_quantile_metrics_on_test(
                            new_ttft_model, new_ttft_scaler, 
                            list(self.ttft_test_data), ttft_feature_cols, 'actual_ttft_ms'
                        )
                        
                        if ql is not None:
                            self.ttft_quantile_loss_scores.append(ql)
                            self.ttft_coverage_scores.append(coverage)
                            self.ttft_violation_rates.append(violation_rate)
                            logging.info(f"TTFT model trained on {len(df_ttft)} samples. "
                                       f"Quantile Loss = {ql:.4f}, "
                                       f"Coverage = {coverage:.2f}% (target: {self.quantile*100:.0f}%), "
                                       f"Violation Rate = {violation_rate:.2f}% (target: {(1-self.quantile)*100:.0f}%)")
                        else:
                            logging.info(f"TTFT model trained on {len(df_ttft)} samples. Quantile metrics = N/A (insufficient test data)")

                    except Exception:
                        logging.error("Error training TTFT model", exc_info=True)
                else:
                    logging.warning("Not enough TTFT samples, skipping TTFT training.")

            # Train TPOT
            if tpot_snap:
                df_tpot = pd.DataFrame(tpot_snap).dropna()
                df_tpot = df_tpot[df_tpot['actual_tpot_ms'] > 0]
                if len(df_tpot) >= settings.MIN_SAMPLES_FOR_RETRAIN:
                    # TPOT features remain unchanged
                    X_tpot = df_tpot[['kv_cache_percentage', 'input_token_length', 'num_request_waiting', 'num_request_running', 'num_tokens_generated']]
                    y_tpot = df_tpot['actual_tpot_ms']
                    try:
                        result = self._train_model_with_scaling(X_tpot, y_tpot)
                        if self.model_type == ModelType.BAYESIAN_RIDGE:
                            new_tpot_model, new_tpot_scaler = result
                        else:
                            new_tpot_model = result
                            new_tpot_scaler = None
                        
                        # Calculate quantile metrics on test data
                        tpot_feature_cols = ['kv_cache_percentage', 'input_token_length', 'num_request_waiting', 'num_request_running', 'num_tokens_generated']
                        ql, coverage, violation_rate = self._calculate_quantile_metrics_on_test(
                            new_tpot_model, new_tpot_scaler, 
                            list(self.tpot_test_data), tpot_feature_cols, 'actual_tpot_ms'
                        )
                        
                        if ql is not None:
                            self.tpot_quantile_loss_scores.append(ql)
                            self.tpot_coverage_scores.append(coverage)
                            self.tpot_violation_rates.append(violation_rate)
                            logging.info(f"TPOT model trained on {len(df_tpot)} samples. "
                                       f"Quantile Loss = {ql:.4f}, "
                                       f"Coverage = {coverage:.2f}% (target: {self.quantile*100:.0f}%), "
                                       f"Violation Rate = {violation_rate:.2f}% (target: {(1-self.quantile)*100:.0f}%)")
                        else:
                            logging.info(f"TPOT model trained on {len(df_tpot)} samples. Quantile metrics = N/A (insufficient test data)")
                            
                    except Exception:
                        logging.error("Error training TPOT model", exc_info=True)
                else:
                    logging.warning("Not enough TPOT samples, skipping TPOT training.")

            with self.lock:
                if new_ttft_model:
                    self.ttft_model = new_ttft_model
                    if new_ttft_scaler is not None:
                        self.ttft_scaler = new_ttft_scaler
                    
                    # Store descaled coefficients for Bayesian Ridge
                    if self.model_type == ModelType.BAYESIAN_RIDGE:
                        ttft_features = ['kv_cache_percentage', 'input_token_length', 
                                       'num_request_waiting', 'num_request_running', 'prefix_cache_score']
                        self.ttft_coefficients = self._store_descaled_coefficients(
                            new_ttft_model, new_ttft_scaler, ttft_features, "TTFT"
                        )
                        
                if new_tpot_model:
                    self.tpot_model = new_tpot_model
                    if new_tpot_scaler is not None:
                        self.tpot_scaler = new_tpot_scaler
                    
                    # Store descaled coefficients for Bayesian Ridge
                    if self.model_type == ModelType.BAYESIAN_RIDGE:
                        tpot_features = ['kv_cache_percentage', 'input_token_length', 
                                       'num_request_waiting', 'num_request_running', 'num_tokens_generated']
                        self.tpot_coefficients = self._store_descaled_coefficients(
                            new_tpot_model, new_tpot_scaler, tpot_features, "TPOT"
                        )
                
                if self.is_ready:
                    self.last_retrain_time = datetime.now(timezone.utc)
                    try:
                        self._save_models_unlocked()
                    except Exception:
                        logging.error("Error saving models after training.", exc_info=True)
        except Exception as e:
            logging.error(f"Critical error in train(): {e}", exc_info=True)

    def predict(self, features: dict) -> Tuple[float, float, float, float]:
        try:
            with self.lock:
                if not self.is_ready:
                    raise HTTPException(status_code=503, detail="Models not ready")
                required = ['kv_cache_percentage', 'input_token_length', 'num_request_waiting', 'num_request_running', 'num_tokens_generated', 'prefix_cache_score']
                for f in required:
                    if f not in features:
                        raise ValueError(f"Missing required feature: {f}")
                    if not isinstance(features[f], (int, float)):
                        raise ValueError(f"Invalid type for feature {f}: expected number")

                # Updated TTFT features to include prefix_cache_score
                ttft_cols = ['kv_cache_percentage','input_token_length','num_request_waiting','num_request_running','prefix_cache_score']
                tpot_cols = ['kv_cache_percentage','input_token_length','num_request_waiting','num_request_running','num_tokens_generated']
                
                # Create DataFrames for predictions
                df_ttft = pd.DataFrame([{col: features[col] for col in ttft_cols}])
                df_tpot = pd.DataFrame([{col: features[col] for col in tpot_cols}])

                if self.model_type == ModelType.BAYESIAN_RIDGE:
                    # Use scaling for Bayesian Ridge
                    ttft_scaled = self.ttft_scaler.transform(df_ttft)
                    tpot_scaled = self.tpot_scaler.transform(df_tpot)

                    ttft_pred_mean, ttft_std = self.ttft_model.predict(ttft_scaled, return_std=True)
                    tpot_pred_mean, tpot_std = self.tpot_model.predict(tpot_scaled, return_std=True)
                    
                    # Approximate quantile prediction by adding factor to mean
                    std_factor = 1.28 if self.quantile == 0.9 else (2.0 if self.quantile == 0.95 else 0.674)
                    ttft_pred = ttft_pred_mean[0] + std_factor * ttft_std[0]
                    tpot_pred = tpot_pred_mean[0] + std_factor * tpot_std[0]
                    
                    return ttft_pred, tpot_pred, ttft_std[0], tpot_std[0]
                
                else:  # XGBoost with true quantile regression
                    # XGBoost quantile regression directly predicts the quantile
                    ttft_pred = self.ttft_model.predict(df_ttft)
                    tpot_pred = self.tpot_model.predict(df_tpot)
                    
                    # For XGBoost quantile regression, uncertainty estimation is more complex
                    # We'll use a simple heuristic based on the quantile value
                    ttft_std = ttft_pred[0] * 0.1  # 10% of prediction as uncertainty estimate
                    tpot_std = tpot_pred[0] * 0.1
                    
                    return ttft_pred[0], tpot_pred[0], ttft_std, tpot_std
                    
        except ValueError as ve:
            logging.warning(f"Client error in predict(): {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except HTTPException:
            raise
        except Exception as e:
            logging.error("Error in predict():", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error during prediction")

    def add_training_sample(self, sample: dict):
        try:
            required = ['kv_cache_percentage', 'actual_ttft_ms', 'actual_tpot_ms', 'num_tokens_generated', 'input_token_length', 'num_request_waiting', 'num_request_running', 'prefix_cache_score']
            for field in required:
                if field not in sample or not isinstance(sample[field], (int, float)):
                    logging.warning(f"Invalid sample field: {field}")
                    return
        
            # Use hash-based deterministic split to ensure consistent train/test assignment
            # This ensures the same sample always goes to the same split
            sample_hash = hash(str(sorted(sample.items())))
            is_test = (sample_hash % 100) < (settings.TEST_TRAIN_RATIO * 100)
        
            # Create subsets based on conditions
            ttft_valid = sample['actual_ttft_ms'] > 0
            tpot_valid = sample['actual_tpot_ms'] > 0

            if is_test:
                # Add to test data only if the respective metric is valid
                if ttft_valid:
                    self.ttft_test_data.append(sample.copy())
                if tpot_valid:
                    self.tpot_test_data.append(sample.copy())
            else:
                # Add to training buckets only if the respective metric is valid
                bucket_key = self._get_bucket_key(sample)
            
                if ttft_valid:
                    self.ttft_data_buckets[bucket_key].append(sample)
                if tpot_valid:
                    self.tpot_data_buckets[bucket_key].append(sample)
                
        except Exception as e:
            logging.error(f"Error adding training sample: {e}", exc_info=True)
            
    
    def add_training_samples(self, samples: list):
        """Bulk-add multiple training samples in one go."""
        with self.lock:
            for sample in samples:
                try:
                    # reuse the single-sample logic
                    self.add_training_sample(sample)
                except Exception:
                    # log & continue on individual failures
                    logging.exception("Failed to add one sample in bulk ingestion")


    def _save_models_unlocked(self):
        try:
            if self.ttft_model:
                os.makedirs(os.path.dirname(settings.TTFT_MODEL_PATH), exist_ok=True)
                joblib.dump(self.ttft_model, settings.TTFT_MODEL_PATH)
                logging.info("TTFT model saved.")
            
                # Save XGBoost booster trees as JSON
                if self.model_type == ModelType.XGBOOST:
                    try:
                        booster = self.ttft_model.get_booster()
                        raw_trees = booster.get_dump(dump_format="json")
                        trees = [json.loads(t) for t in raw_trees]
                    
                        # Save to JSON file alongside the model
                        ttft_json_path = settings.TTFT_MODEL_PATH.replace('.joblib', '_trees.json')
                        with open(ttft_json_path, 'w') as f:
                            json.dump(trees, f, indent=2)
                        logging.info(f"TTFT XGBoost trees saved to {ttft_json_path}")
                    except Exception as e:
                        logging.error(f"Error saving TTFT XGBoost trees: {e}", exc_info=True)
            
            if self.ttft_scaler and self.model_type == ModelType.BAYESIAN_RIDGE:
                os.makedirs(os.path.dirname(settings.TTFT_SCALER_PATH), exist_ok=True)
                joblib.dump(self.ttft_scaler, settings.TTFT_SCALER_PATH)
                logging.info("TTFT scaler saved.")
            
            if self.tpot_model:
                os.makedirs(os.path.dirname(settings.TPOT_MODEL_PATH), exist_ok=True)
                joblib.dump(self.tpot_model, settings.TPOT_MODEL_PATH)
                logging.info("TPOT model saved.")
            
                # Save XGBoost booster trees as JSON
                if self.model_type == ModelType.XGBOOST:
                    try:
                        booster = self.tpot_model.get_booster()
                        raw_trees = booster.get_dump(dump_format="json")
                        trees = [json.loads(t) for t in raw_trees]
                    
                        # Save to JSON file alongside the model
                        tpot_json_path = settings.TPOT_MODEL_PATH.replace('.joblib', '_trees.json')
                        with open(tpot_json_path, 'w') as f:
                            json.dump(trees, f, indent=2)
                        logging.info(f"TPOT XGBoost trees saved to {tpot_json_path}")
                    except Exception as e:
                        logging.error(f"Error saving TPOT XGBoost trees: {e}", exc_info=True)
            
            if self.tpot_scaler and self.model_type == ModelType.BAYESIAN_RIDGE:
                os.makedirs(os.path.dirname(settings.TPOT_SCALER_PATH), exist_ok=True)
                joblib.dump(self.tpot_scaler, settings.TPOT_SCALER_PATH)
                logging.info("TPOT scaler saved.")
            
        except Exception as e:
            logging.error(f"Error saving models: {e}", exc_info=True)

    def load_models(self):
        try:
            with self.lock:
                if os.path.exists(settings.TTFT_MODEL_PATH):
                    self.ttft_model = joblib.load(settings.TTFT_MODEL_PATH)
                    if self.model_type == ModelType.BAYESIAN_RIDGE and os.path.exists(settings.TTFT_SCALER_PATH):
                        self.ttft_scaler = joblib.load(settings.TTFT_SCALER_PATH)
                else:
                    result = self._create_default_model("ttft")
                    if self.model_type == ModelType.BAYESIAN_RIDGE:
                        self.ttft_model, self.ttft_scaler = result
                    else:
                        self.ttft_model = result
                    settings.MIN_SAMPLES_FOR_RETRAIN = settings.MIN_SAMPLES_FOR_RETRAIN_FRESH
                    self._save_models_unlocked()

                if os.path.exists(settings.TPOT_MODEL_PATH):
                    self.tpot_model = joblib.load(settings.TPOT_MODEL_PATH)
                    if self.model_type == ModelType.BAYESIAN_RIDGE and os.path.exists(settings.TPOT_SCALER_PATH):
                        self.tpot_scaler = joblib.load(settings.TPOT_SCALER_PATH)
                else:
                    result = self._create_default_model("tpot")
                    if self.model_type == ModelType.BAYESIAN_RIDGE:
                        self.tpot_model, self.tpot_scaler = result
                    else:
                        self.tpot_model = result
                    settings.MIN_SAMPLES_FOR_RETRAIN = settings.MIN_SAMPLES_FOR_RETRAIN_FRESH
                    self._save_models_unlocked()

                if not self.is_ready:
                    raise RuntimeError("Failed to initialize models/scalers")
        except Exception as e:
            logging.error(f"Critical error in load_models: {e}", exc_info=True)
            raise
        
    def get_metrics(self) -> str:
        """Render Prometheus-style metrics: model, coefficients/importances, bucket counts, and quantile-specific scores."""
        try:
            # Snapshot models & scalers
            ttft_model, tpot_model = self.ttft_model, self.tpot_model
            ttft_scaler, tpot_scaler = self.ttft_scaler, self.tpot_scaler

            lines: List[str] = []
            # 1) Model type and quantile info
            lines.append(f'model_type{{type="{self.model_type.value}"}} 1')
            lines.append(f'model_quantile{{}} {self.quantile}')

            # Helper: emit linear‐model coefs or tree importances
            def emit_metrics(model, coefficients, feats, prefix):
                if model is None:
                    # placeholders
                    lines.append(f'{prefix}_intercept{{}} 0.0')
                    kind = "coef" if self.model_type == ModelType.BAYESIAN_RIDGE else "importance"
                    for f in feats:
                        lines.append(f'{prefix}_{kind}{{feature="{f}"}} 0.0')
                    return

                if self.model_type == ModelType.BAYESIAN_RIDGE:
                    # Use stored descaled coefficients
                    if coefficients:
                        lines.append(f'{prefix}_intercept{{}} {coefficients.get("intercept", 0.0):.6f}')
                        for f in feats:
                            coef_value = coefficients.get(f, 0.0)
                            lines.append(f'{prefix}_coef{{feature="{f}"}} {coef_value:.6f}')
                    else:
                        # Fallback to zeros if coefficients not available
                        lines.append(f'{prefix}_intercept{{}} 0.0')
                        for f in feats:
                            lines.append(f'{prefix}_coef{{feature="{f}"}} 0.0')
                else:
                    # XGBoost importances
                    try:
                        imps = model.feature_importances_
                    except Exception:
                        imps = [0.0]*len(feats)
                    lines.append(f'{prefix}_intercept{{}} 0.0')
                    for f, imp in zip(feats, imps):
                        lines.append(f'{prefix}_importance{{feature="{f}"}} {imp:.6f}')

            # Updated TTFT features to include prefix_cache_score
            ttft_feats = ["kv_cache_percentage","input_token_length","num_request_waiting","num_request_running","prefix_cache_score"]
            tpot_feats = ["kv_cache_percentage","input_token_length","num_request_waiting","num_request_running","num_tokens_generated"]
            emit_metrics(ttft_model, self.ttft_coefficients, ttft_feats, "ttft")
            emit_metrics(tpot_model, self.tpot_coefficients, tpot_feats, "tpot")

            # 3) Multi-dimensional bucket counts
            for (queue_bucket, cache_bucket), bucket_deque in self.ttft_data_buckets.items():
                count = len(bucket_deque)
                lines.append(f'training_samples_count{{model="ttft",queue_bucket="{queue_bucket}",cache_bucket="{cache_bucket}"}} {count}')
        
            for (queue_bucket, cache_bucket), bucket_deque in self.tpot_data_buckets.items():
                count = len(bucket_deque)
                lines.append(f'training_samples_count{{model="tpot",queue_bucket="{queue_bucket}",cache_bucket="{cache_bucket}"}} {count}')
        
            # Summary metrics by queue state
            for q in range(self.queue_buckets):
                ttft_total = sum(len(self.ttft_data_buckets[(q, c)]) for c in range(self.cache_buckets))
                tpot_total = sum(len(self.tpot_data_buckets[(q, c)]) for c in range(self.cache_buckets))
                lines.append(f'training_samples_queue_total{{model="ttft",queue_bucket="{q}"}} {ttft_total}')
                lines.append(f'training_samples_queue_total{{model="tpot",queue_bucket="{q}"}} {tpot_total}')
        
            # Summary metrics by cache state  
            for c in range(self.cache_buckets):
                ttft_total = sum(len(self.ttft_data_buckets[(q, c)]) for q in range(self.queue_buckets))
                tpot_total = sum(len(self.tpot_data_buckets[(q, c)]) for q in range(self.queue_buckets))
                lines.append(f'training_samples_cache_total{{model="ttft",cache_bucket="{c}"}} {ttft_total}')
                lines.append(f'training_samples_cache_total{{model="tpot",cache_bucket="{c}"}} {tpot_total}')

            # 4) Quantile Loss scores (last up to 5)
            for idx, score in enumerate(self.ttft_quantile_loss_scores):
                lines.append(f'ttft_quantile_loss{{idx="{idx}"}} {score:.6f}')
            for idx, score in enumerate(self.tpot_quantile_loss_scores):
                lines.append(f'tpot_quantile_loss{{idx="{idx}"}} {score:.6f}')

            # 5) Coverage scores (should be close to quantile * 100)
            for idx, coverage in enumerate(self.ttft_coverage_scores):
                lines.append(f'ttft_coverage_percent{{idx="{idx}"}} {coverage:.6f}')
            for idx, coverage in enumerate(self.tpot_coverage_scores):
                lines.append(f'tpot_coverage_percent{{idx="{idx}"}} {coverage:.6f}')

            # 6) Violation rates (should be close to (1-quantile) * 100)
            for idx, violation_rate in enumerate(self.ttft_violation_rates):
                lines.append(f'ttft_violation_rate_percent{{idx="{idx}"}} {violation_rate:.6f}')
            for idx, violation_rate in enumerate(self.tpot_violation_rates):
                lines.append(f'tpot_violation_rate_percent{{idx="{idx}"}} {violation_rate:.6f}')

            # 7) Target metrics for reference
            target_coverage = self.quantile * 100
            target_violation_rate = (1 - self.quantile) * 100
            lines.append(f'target_coverage_percent{{}} {target_coverage:.1f}')
            lines.append(f'target_violation_rate_percent{{}} {target_violation_rate:.1f}')

            return "\n".join(lines) + "\n"

        except Exception as e:
            logging.error(f"Error generating metrics: {e}", exc_info=True)
            return "# error_generating_metrics 1\n"

                

# --- FastAPI Application ---
app = FastAPI(
    title="Latency Predictor Service",
    description="A service to predict TTFT and TPOT using quantile regression with continuous training and feature scaling.",
)

predictor = LatencyPredictor()

# --- Pydantic Models for API ---
class TrainingEntry(BaseModel):
    kv_cache_percentage: float = Field(..., ge=0.0, le=1.0)
    input_token_length: int = Field(..., ge=0)
    num_request_waiting: int = Field(..., ge=0)
    num_request_running: int = Field(..., ge=0)
    actual_ttft_ms: float = Field(..., ge=0.0)
    actual_tpot_ms: float = Field(..., ge=0.0)
    num_tokens_generated: int = Field(..., ge=0)
    prefix_cache_score: float = Field(..., ge=0.0, le=1.0, description="Prefix cache hit ratio score (0.0 to 1.0)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PredictionRequest(BaseModel):
    kv_cache_percentage: float = Field(..., ge=0.0, le=1.0)
    input_token_length: int = Field(..., ge=0)
    num_request_waiting: int = Field(..., ge=0)
    num_request_running: int = Field(..., ge=0)
    num_tokens_generated: int = Field(..., ge=0)
    prefix_cache_score: float = Field(..., ge=0.0, le=1.0, description="Prefix cache hit ratio score (0.0 to 1.0)")

class PredictionResponse(BaseModel):
    ttft_ms: float = Field(..., description=f"Predicted {settings.QUANTILE_ALPHA:.0%} quantile TTFT in milliseconds")
    tpot_ms: float = Field(..., description=f"Predicted {settings.QUANTILE_ALPHA:.0%} quantile TPOT in milliseconds")
    ttft_uncertainty: float = Field(..., description="Uncertainty estimate for TTFT prediction")
    tpot_uncertainty: float = Field(..., description="Uncertainty estimate for TPOT prediction")
    ttft_prediction_bounds: Tuple[float, float] = Field(..., description="Approximate prediction bounds for TTFT")
    tpot_prediction_bounds: Tuple[float, float] = Field(..., description="Approximate prediction bounds for TPOT")
    predicted_at: datetime
    model_type: ModelType = Field(default=predictor.model_type.value, description="Type of model used for prediction")
    quantile: float = Field(default=settings.QUANTILE_ALPHA, description="Quantile being predicted")
    
class BulkTrainingRequest(BaseModel):
    entries: List[TrainingEntry]

# --- Background Training Loop ---
def continuous_training_loop():
    time.sleep(10)
    while not predictor._shutdown_event.is_set():
        try:
            logging.debug("Checking if training should run...")
            predictor.train()
        except Exception:
            logging.error("Error in periodic retraining", exc_info=True)
        if predictor._shutdown_event.wait(timeout=settings.RETRAINING_INTERVAL_SEC):
            break
    logging.info("Training loop exiting.")

# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    logging.info("Server starting up...")
    predictor.load_models()
    t = threading.Thread(target=continuous_training_loop, daemon=True)
    predictor._training_thread = t
    t.start()
    logging.info("Background training started.")

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Server shutting down...")
    predictor.shutdown()
    

@app.post("/add_training_data_bulk", status_code=status.HTTP_202_ACCEPTED)
async def add_training_data_bulk(batch: BulkTrainingRequest):
     """
     Accepts a JSON body like:
       { "entries": [ { …TrainingEntry… }, { … }, … ] }
     """
     try:
        predictor.add_training_samples([e.dict() for e in batch.entries])
        return {"message": f"Accepted {len(batch.entries)} training samples."}
     except Exception:
         logging.error("Failed to add bulk training data", exc_info=True)
         raise HTTPException(status_code=500, detail="Failed to add training data in bulk")

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    try:
        ttft_pred, tpot_pred, ttft_std, tpot_std = predictor.predict(request.dict())
        ttft_pred = max(0, ttft_pred)
        tpot_pred = max(0, tpot_pred)
        ttft_bounds = (max(0, ttft_pred - 2*ttft_std), ttft_pred + 2*ttft_std)
        tpot_bounds = (max(0, tpot_pred - 2*tpot_std), tpot_pred + 2*tpot_std)
        return PredictionResponse(
            ttft_ms=ttft_pred,
            tpot_ms=tpot_pred,
            ttft_uncertainty=ttft_std,
            tpot_uncertainty=tpot_std,
            ttft_prediction_bounds=ttft_bounds,
            tpot_prediction_bounds=tpot_bounds,
            predicted_at=datetime.now(timezone.utc),
            model_type=predictor.model_type.value,
            quantile=predictor.quantile
        )
    except HTTPException:
        raise
    except Exception:
        logging.error("Prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")



@app.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}

@app.get("/readyz", status_code=status.HTTP_200_OK)
async def readiness_check():
    if not predictor.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models are not ready.")
    return {"status": "ready"}


@app.get("/metrics", status_code=status.HTTP_200_OK)
async def metrics():
    """Prometheus metrics including coefficients/importances, bucket counts, and quantile-specific metrics."""
    try:
        content = predictor.get_metrics()
        return Response(content, media_type="text/plain; version=0.0.4")
    except Exception as e:
        logging.error(f"Error in metrics endpoint: {e}", exc_info=True)
        return Response("# Error generating metrics\n", media_type="text/plain; version=0.0.4")
    
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Latency Predictor is running.",
        "model_type": predictor.model_type.value,
        "quantile": predictor.quantile,
        "description": f"Predicting {predictor.quantile:.0%} quantile for TTFT and TPOT latencies"
    }
 
@app.get("/model/download/info")
async def model_download_info():
    """
    Get information about available model downloads and coefficients.
    """
    info = {
        "model_type": predictor.model_type.value,
        "quantile": predictor.quantile,
        "available_endpoints": {}
    }
    
    if predictor.model_type == ModelType.BAYESIAN_RIDGE:
        info["available_endpoints"]["coefficients"] = "/metrics"
        info["coefficients_info"] = {
            "ttft_coefficients_available": predictor.ttft_coefficients is not None,
            "tpot_coefficients_available": predictor.tpot_coefficients is not None,
            "description": "Descaled coefficients available in Prometheus metrics endpoint"
        }
    else:  # XGBoost
        info["available_endpoints"]["trees"] = {
            "ttft_trees": "/model/ttft/xgb/json",
            "tpot_trees": "/model/tpot/xgb/json"
        }
    
    info["model_status"] = {
        "ttft_model_ready": predictor.ttft_model is not None,
        "tpot_model_ready": predictor.tpot_model is not None,
    }
    
    if predictor.model_type == ModelType.BAYESIAN_RIDGE:
        info["model_status"]["ttft_coefficients_ready"] = predictor.ttft_coefficients is not None
        info["model_status"]["tpot_coefficients_ready"] = predictor.tpot_coefficients is not None
    
    # Add quantile-specific evaluation info
    info["evaluation_info"] = {
        "quantile_loss": "Pinball loss for quantile regression evaluation",
        "coverage_percent": f"Percentage of actual values below predicted {predictor.quantile:.0%} quantile (target: {predictor.quantile*100:.1f}%)",
        "violation_rate_percent": f"Percentage of actual values above predicted {predictor.quantile:.0%} quantile (target: {(1-predictor.quantile)*100:.1f}%)"
    }
    
    return info

@app.get("/model/ttft/xgb/json")
async def ttft_xgb_json():
    """
    Dump the TTFT XGBoost model as JSON trees.
    """
    if predictor.model_type != ModelType.XGBOOST:
        raise HTTPException(status_code=404, detail="TTFT model is not XGBoost")
    
    if not predictor.ttft_model:
        raise HTTPException(status_code=404, detail="TTFT model not available")
        
    try:
        booster = predictor.ttft_model.get_booster()
        # get_dump with dump_format="json" gives one JSON string per tree
        raw_trees = booster.get_dump(dump_format="json")
        # parse each string into a dict so the response is a JSON array of objects
        trees = [json.loads(t) for t in raw_trees]
        return JSONResponse(content=trees)
    except Exception as e:
        logging.error(f"Error dumping TTFT XGBoost trees: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error dumping TTFT XGBoost trees")


@app.get("/model/tpot/xgb/json")
async def tpot_xgb_json():
    """
    Dump the TPOT XGBoost model as JSON trees.
    """
    if predictor.model_type != ModelType.XGBOOST:
        raise HTTPException(status_code=404, detail="TPOT model is not XGBoost")
    
    if not predictor.tpot_model:
        raise HTTPException(status_code=404, detail="TPOT model not available")
        
    try:
        booster = predictor.tpot_model.get_booster()
        raw_trees = booster.get_dump(dump_format="json")
        trees = [json.loads(t) for t in raw_trees]
        return JSONResponse(content=trees)
    except Exception as e:
        logging.error(f"Error dumping TPOT XGBoost trees: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error dumping TPOT XGBoost trees")



@app.get("/model/{model_name}/info")
async def model_info(model_name: str):
    """Get model file information including last modified time."""
    model_paths = {
        "ttft": settings.TTFT_MODEL_PATH,
        "tpot": settings.TPOT_MODEL_PATH,
        "ttft_scaler": settings.TTFT_SCALER_PATH,
        "tpot_scaler": settings.TPOT_SCALER_PATH
    }
    
    if model_name not in model_paths:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
    
    model_path = model_paths[model_name]
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Get file stats
    stat = os.stat(model_path)
    last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    
    return {
        "model_name": model_name,
        "path": model_path,
        "size_bytes": stat.st_size,
        "last_modified": last_modified.isoformat(),
        "exists": True,
        "model_type": predictor.model_type.value,
        "quantile": predictor.quantile if model_name in ["ttft", "tpot"] else None
    }


@app.get("/model/{model_name}/download")
async def download_model(model_name: str):
    """Download a model file."""
    model_paths = {
        "ttft": settings.TTFT_MODEL_PATH,
        "tpot": settings.TPOT_MODEL_PATH,
        "ttft_scaler": settings.TTFT_SCALER_PATH,
        "tpot_scaler": settings.TPOT_SCALER_PATH
    }
    
    if model_name not in model_paths:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
    
    model_path = model_paths[model_name]
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Return the file
    filename = f"{model_name}.joblib"
    return FileResponse(
        model_path,
        media_type='application/octet-stream',
        filename=filename
    )


@app.get("/models/list")
async def list_models():
    """List all available models with their status."""
    models = {}
    model_paths = {
        "ttft": settings.TTFT_MODEL_PATH,
        "tpot": settings.TPOT_MODEL_PATH,
        "ttft_scaler": settings.TTFT_SCALER_PATH,
        "tpot_scaler": settings.TPOT_SCALER_PATH
    }
    
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            stat = os.stat(model_path)
            models[model_name] = {
                "exists": True,
                "size_bytes": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            }
        else:
            models[model_name] = {
                "exists": False,
                "size_bytes": 0,
                "last_modified": None
            }
    
    return {
        "models": models,
        "model_type": predictor.model_type.value,
        "quantile": predictor.quantile,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "evaluation_metrics": {
            "quantile_loss": "Lower is better",
            "coverage_percent": f"Target: {predictor.quantile*100:.1f}%",
            "violation_rate_percent": f"Target: {(1-predictor.quantile)*100:.1f}%"
        }
    }


@app.get("/model/{model_name}/download_onnx")
async def download_model_onnx(model_name: str):
    """Download a model file."""
    if model_name == 'tpot':
        booster = predictor.tpot_model.get_booster()
        original_feature_names = booster.feature_names
        if original_feature_names is not None:
            onnx_converter_conform_feature_names = [f"f{num}" for num in range(len(original_feature_names))]
            booster.feature_names = onnx_converter_conform_feature_names
        print(type(booster))
        print(predictor.guess_type)
        onx = to_onnx(predictor.tpot_model, initial_types=predictor.guess_type, target_opset={'ai.onnx.ml': 3})
        with open("predictor.onnx", "wb") as f:
            f.write(onx.SerializeToString())
    return ""

# @app.get("/models/dump")
# async def dump_models():
#     """Dump all available models with their status."""
#     onnx_model = onnxmltools.convert_xgboost(predictor.tpot_model)
#     onnxmltools.utils.save_model(onnx_model, model_name + ".onnx")
    
#     return ""


if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)