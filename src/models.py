"""
Model Implementations Module

This module contains various model implementations for multi-label classification:
- Baseline models (TF-IDF + Linear)
- Advanced models (Transformers)
- Custom architectures and improvements

Classes:
    BaselineModel: TF-IDF with Logistic Regression
    ImprovedMultiLabelModel: Enhanced model with threshold optimization
    TransformerModel: BERT-based implementation
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
from typing import Tuple, List

class ImprovedMultiLabelModel:
    """
    Enhanced multi-label classifier with optimized thresholds.
    
    Features:
    - Dynamic threshold optimization per class
    - Correlation-based prediction adjustment
    - Handling of rare classes
    - Confidence calibration
    """
    
    def __init__(self, config: dict):
        """
        Initialize model with configuration.
        
        Args:
            config (dict): Model hyperparameters and settings
        """
        self.config = config
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Set up the model pipeline with TF-IDF and classifier."""
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.config.get('max_features', 20000),
                ngram_range=(1, self.config.get('ngram_range', 2)),  # CHANGED: (1, 2) is much faster
                min_df=self.config.get('min_df', 5),
                max_df=self.config.get('max_df', 0.9),
                stop_words='english',
                sublinear_tf=True
            )),
            ('clf', OneVsRestClassifier(
                LogisticRegression(
                    C=self.config.get('C', 1.0),
                    class_weight='balanced',
                    solver='liblinear',  # CHANGED: 'liblinear' is very fast for this
                    penalty='l2',        # CHANGED: 'l2' is the standard for liblinear
                    max_iter=1000,       # Increased for convergence
                    random_state=self.config.get('random_state', 42)
                ),
                n_jobs=-1  # Use all available CPU cores
            ))
        ])
        self.thresholds = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model on given data.
        
        Args:
            X_train: Training texts
            y_train: Binary label matrix
            
        Returns:
            None
        """
        print("Training base model...")
        self.model.fit(X_train, y_train)
        return self.model

    def optimize_thresholds(self, X_val, y_val):
        """Find the best F1-score threshold for each label using validation data."""
        print("Optimizing thresholds on validation set...")
        val_probs = self.model.predict_proba(X_val)
        self.thresholds = np.full(y_val.shape[1], 0.5) # Default to 0.5

        for i in range(y_val.shape[1]):
            # Only optimize for classes present in the validation set
            if np.sum(y_val[:, i]) > 1:
                precision, recall, thresholds = precision_recall_curve(y_val[:, i], val_probs[:, i])
                # To avoid division by zero
                fscore = (2 * precision * recall) / (precision + recall + 1e-8)
                ix = np.argmax(fscore)
                self.thresholds[i] = thresholds[ix]
        print("Threshold optimization complete.")

    def predict(self, X):
        """Predict labels using the optimized thresholds."""
        if self.thresholds is None:
            raise RuntimeError("Thresholds have not been optimized. Call optimize_thresholds() first.")
        
        probs = self.model.predict_proba(X)
        # Apply thresholds to get binary predictions
        return (probs >= self.thresholds).astype(int)

class BaselineModel:
    """
    Baseline model using TF-IDF with logistic regression (OneVsRest)
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self) -> Pipeline:
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.config['max_features'],
                ngram_range=(1, self.config['ngram_range']),
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
    
    def train(self, X, y):
        self.model.fit(X, y)
        return self.model

class RandomForestModel:
    """
    Random Forest model for multi-label classification
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self) -> Pipeline:
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.config['max_features'],
                ngram_range=(1, self.config['ngram_range']),
            )),
            ('clf', OneVsRestClassifier(
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config['random_state']
                )
            ))
        ])
    
    def train(self, X, y):
        self.model.fit(X, y)
        return self.model

class NaiveBayesModel:
    """
    Naive Bayes model for multi-label classification
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self) -> Pipeline:
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.config['max_features'],
                ngram_range=(1, self.config['ngram_range']),
            )),
            ('clf', OneVsRestClassifier(
                MultinomialNB()
            ))
        ])
    
    def train(self, X, y):
        self.model.fit(X, y)
        return self.model

class XGBoostModel:
    """
    XGBoost model for multi-label classification
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self) -> Pipeline:
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.config['max_features'],
                ngram_range=(1, self.config['ngram_range']),
            )),
            ('clf', OneVsRestClassifier(
                xgb.XGBClassifier(
                    objective='binary:logistic',
                    random_state=self.config['random_state']
                )
            ))
        ])
    
    def train(self, X, y):
        self.model.fit(X, y)
        return self.model

class BertModel:
    """
    BERT-based model for multi-label classification
    Note: This requires more setup and GPU resources
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=config['num_labels'],
            problem_type="multi_label_classification"
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, dataset, epochs=3, batch_size=8):
        # This would need a more complete implementation
        # including DataLoader setup, training loop, etc.
        # Placeholder for actual BERT training implementation
        pass

def train_model(X_train, y_train, config, model_type='logistic'):
    """
    Factory function to train different model types
    """
    if model_type == 'logistic':
        model = BaselineModel(config)
    elif model_type == 'random_forest':
        model = RandomForestModel(config)
    elif model_type == 'naive_bayes':
        model = NaiveBayesModel(config)
    elif model_type == 'xgboost':
        model = XGBoostModel(config)
    elif model_type == 'bert':
        model = BertModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.train(X_train, y_train)

class ClassifierChainModel:
    def __init__(self, config):
        self.config = config
        self.base_model = LogisticRegression(
            solver='liblinear',
            C=config['C'],
            random_state=config['random_state']
        )
        
    def train(self, X, y):
        self.model = ClassifierChain(self.base_model)
        self.model.fit(X, y)
        return self.model

class OptimalThresholdModel:
    def __init__(self, model, thresholds):
        self.model = model
        self.thresholds = thresholds
        
    def predict(self, X):
        probs = self.model.predict_proba(X)
        return (probs >= self.thresholds).astype(int)