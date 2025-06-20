"""
Model Evaluation Module

This module provides comprehensive evaluation utilities for multi-label
classification models, including:
- Standard metrics (F1, Precision, Recall)
- Per-class performance analysis
- Error analysis tools
- Threshold optimization

Functions:
    evaluate_model: Main evaluation function
    print_metrics: Pretty-print evaluation metrics
"""

import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, 
    hamming_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)

def evaluate_model(
    model_path: str, 
    mlb_path: str, 
    test_df_path: str, 
    label_column: str = 'verified_uat_labels'
) -> tuple:
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        model_path: Path to trained model pickle file
        mlb_path: Path to MultiLabelBinarizer pickle file
        test_df_path: Path to test data parquet file
        label_column: Name of column containing ground truth labels
    
    Returns:
        tuple: (metrics_dict, classification_report_dict)
        
    Example:
        >>> metrics, report = evaluate_model('model.pkl', 'mlb.pkl', 'test.parquet')
        >>> print_metrics(metrics)
    """
    # Load model and binarizer
    model = joblib.load(model_path)
    mlb = joblib.load(mlb_path)
    
    # Load test data
    test_df = pd.read_parquet(test_df_path)
    
    # Prepare test data
    X_test = test_df['cleaned_text']
    y_test = mlb.transform(test_df[label_column])  # Use specified label column
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'hamming_loss': hamming_loss(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'micro_f1': f1_score(y_test, y_pred, average='micro'),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'micro_precision': precision_score(y_test, y_pred, average='micro'),
        'macro_precision': precision_score(y_test, y_pred, average='macro'),
        'micro_recall': recall_score(y_test, y_pred, average='micro'),
        'macro_recall': recall_score(y_test, y_pred, average='macro')
    }
    
    # Classification report
    report = classification_report(
        y_test, y_pred, 
        target_names=mlb.classes_,
        output_dict=True
    )
    
    return metrics, report


def print_metrics(metrics):
    print("\nEvaluation Metrics:")
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title()}: {value:.4f}")

if __name__ == "__main__":
    metrics, report = evaluate_model(
        '../models/baseline_model.pkl',
        '../models/label_binarizer.pkl',
        '../data/processed/val_processed.parquet',
        label_column='verified_uat_labels'  # Specify correct column name
    )
    print_metrics(metrics)