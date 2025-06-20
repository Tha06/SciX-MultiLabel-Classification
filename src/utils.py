"""
Utility Functions Module

This module provides helper functions for:
- Configuration management
- Model persistence
- Data validation
- Logging utilities

Functions:
    load_config: Load YAML configuration file
    save_model: Save model and related artifacts
    setup_logging: Configure logging for the project
"""

from typing import Dict, Any
import yaml
import joblib
import logging
from pathlib import Path
import pandas as pd
import numpy as np

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load model configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Dict containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
        
    Example:
        >>> config = load_config('configs/base_config.yaml')
        >>> print(config['max_features'])
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {e}")

def save_model(model: Any, path: str) -> None:
    """
    Save model and related artifacts.
    
    Args:
        model: Trained model instance
        path: Path to save the model
        
    Raises:
        IOError: If saving fails
        
    Example:
        >>> save_model(trained_model, 'models/baseline/model.pkl')
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
    except Exception as e:
        raise IOError(f"Error saving model: {e}")

def load_model(path: str) -> Any:
    """Load trained model"""
    return joblib.load(path)

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save evaluation metrics"""
    pd.DataFrame.from_dict(metrics, orient='index').to_csv(path)

def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    import random
    import torch
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_class_distribution(y: np.ndarray, class_names: list) -> None:
    """Print class distribution"""
    class_counts = y.sum(axis=0)
    for name, count in zip(class_names, class_counts):
        print(f"{name}: {count} samples")

def analyze_errors(model, X, y_true, class_names, target_class, n_samples=5):
    class_idx = list(class_names).index(target_class)
    y_pred = model.predict(X)
    
    # False positives
    fp_mask = (y_pred[:,class_idx] == 1) & (y_true[:,class_idx] == 0)
    print(f"\nFalse Positives for {target_class}:")
    for text in X[fp_mask][:n_samples]:
        print(f"- {text[:150]}...")
    
    # False negatives
    fn_mask = (y_pred[:,class_idx] == 0) & (y_true[:,class_idx] == 1)
    print(f"\nFalse Negatives for {target_class}:")
    for text in X[fn_mask][:n_samples]:
        print(f"- {text[:150]}...")