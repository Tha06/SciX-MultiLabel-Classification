"""
Model Training Pipeline Module

This module handles the training workflow for multi-label classification models including:
- Data preparation
- Model initialization
- Training loop management
- Validation steps
- Threshold optimization
- Model persistence

Classes:
    ModelTrainer: Orchestrates the training process
"""

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import logging
from pathlib import Path
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import yaml
from sklearn.metrics import precision_recall_curve

class ModelTrainer:
    """
    Handles end-to-end training of multi-label classification models.
    
    Features:
    - Configurable training parameters
    - Dynamic threshold optimization
    - Cross-validation support
    - Progress tracking and early stopping
    - Model checkpointing
    """
    
    def __init__(self, config: dict):
        """
        Initialize trainer with configuration.
        
        Args:
            config (dict): Training configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_paths()
        
    def _setup_paths(self):
        """Set up paths for saving models and logs."""
        Path(self.config['model_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['log_dir']).mkdir(parents=True, exist_ok=True)
        
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train model with validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            trained_model: The best performing model
        """
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.config['max_features'],
                ngram_range=(1, self.config['ngram_range'])
            )),
            ('clf', OneVsRestClassifier(
                LogisticRegression(
                    solver='liblinear',
                    penalty='l2',
                    C=self.config['C'],
                    random_state=self.config['random_state']
                )
            ))
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Validate and find optimal thresholds
        thresholds = self.find_optimal_thresholds(pipeline, X_val, y_val)
        
        # TODO: Implement model saving and logging
        
        return pipeline

    def find_optimal_thresholds(self, model, X_val, y_val):
        """
        Find optimal thresholds for multi-label classification.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            thresholds: List of optimal thresholds for each label
        """
        y_probs = model.predict_proba(X_val)
        thresholds = []
        for i in range(y_probs.shape[1]):
            prec, rec, thresh = precision_recall_curve(y_val[:,i], y_probs[:,i])
            f1 = 2*prec*rec/(prec+rec+1e-8)
            thresholds.append(thresh[np.argmax(f1)])
        return thresholds

def main():
    # Load config
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    train_df = pd.read_parquet('data/processed/train_processed.parquet')
    
    # Prepare labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(train_df['keywords'])
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['cleaned_text'], y, test_size=config['test_size'], random_state=config['random_state']
    )
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train model
    model = trainer.train(X_train, y_train, X_val, y_val)
    
    # Save model and label binarizer
    joblib.dump(model, 'models/baseline_model.pkl')
    joblib.dump(mlb, 'models/label_binarizer.pkl')

if __name__ == "__main__":
    main()
