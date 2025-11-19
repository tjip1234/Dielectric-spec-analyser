"""
Random Forest model for multi-label functional group classification.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path


class FunctionalGroupClassifier:
    """
    Multi-label Random Forest classifier for predicting functional groups
    from dielectric spectroscopy polynomial features.
    """
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 15, 
                 min_samples_split: int = 5, random_state: int = 42):
        """
        Initialize classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            random_state: Random seed for reproducibility
        """
        base_estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        
        self.model = MultiOutputClassifier(base_estimator)
        self.feature_names = None
        self.functional_groups = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
           feature_names: Optional[List[str]] = None,
           functional_groups: Optional[List[str]] = None):
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label matrix (n_samples, n_functional_groups)
            feature_names: Names of features
            functional_groups: Names of functional groups
        """
        self.model.fit(X, y)
        self.feature_names = feature_names
        self.functional_groups = functional_groups
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict functional groups.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Binary predictions (n_samples, n_functional_groups)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Predict probabilities for each functional group.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            List of probability arrays, one per functional group
        """
        # MultiOutputClassifier returns a list of arrays
        # Each array is (n_samples, 2) for binary classification
        probas = []
        for estimator in self.model.estimators_:
            proba = estimator.predict_proba(X)
            # Take probability of positive class
            probas.append(proba[:, 1])
        
        return probas
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance averaged across all functional group classifiers.
        
        Returns:
            Dictionary with 'importances' array and optionally 'feature_names'
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Average feature importance across all estimators
        importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
        
        result = {'importances': importances}
        if self.feature_names:
            result['feature_names'] = self.feature_names
        
        return result
    
    def get_per_group_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance for each functional group separately.
        
        Returns:
            Dictionary mapping functional group names to importance arrays
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if not self.functional_groups:
            raise ValueError("Functional group names not provided during training")
        
        importance_dict = {}
        for i, group in enumerate(self.functional_groups):
            importance_dict[group] = self.model.estimators_[i].feature_importances_
        
        return importance_dict
    
    def save(self, filepath: Path):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'functional_groups': self.functional_groups,
                'is_fitted': self.is_fitted
            }, f)
    
    @classmethod
    def load(cls, filepath: Path) -> 'FunctionalGroupClassifier':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        classifier = cls()
        classifier.model = data['model']
        classifier.feature_names = data['feature_names']
        classifier.functional_groups = data['functional_groups']
        classifier.is_fitted = data['is_fitted']
        
        return classifier


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                  functional_groups: List[str], verbose: bool = True) -> Dict:
    """
    Evaluate multi-label classification performance.
    
    Args:
        y_true: True labels (n_samples, n_groups)
        y_pred: Predicted labels (n_samples, n_groups)
        functional_groups: Names of functional groups
        verbose: Print detailed report
        
    Returns:
        Dictionary with evaluation metrics
    """
    results = {}
    
    # Overall metrics
    results['hamming_loss'] = hamming_loss(y_true, y_pred)
    results['exact_match_ratio'] = accuracy_score(y_true, y_pred)
    
    # Per-group metrics
    results['per_group'] = {}
    
    if verbose:
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(f"\nOverall Metrics:")
        print(f"  Hamming Loss: {results['hamming_loss']:.4f}")
        print(f"  Exact Match Ratio: {results['exact_match_ratio']:.4f}")
        print(f"  (Exact match = all functional groups correct for a sample)")
        print("\n" + "-"*70)
    
    for i, group in enumerate(functional_groups):
        # Only evaluate groups that appear in the dataset
        if np.sum(y_true[:, i]) == 0:
            continue
        
        if verbose:
            print(f"\nFunctional Group: {group.upper()}")
            print("-" * 70)
            print(classification_report(y_true[:, i], y_pred[:, i], 
                                       target_names=['Absent', 'Present'],
                                       zero_division=0))
        
        # Store metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        
        results['per_group'][group] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    return results


def train_and_evaluate(X: np.ndarray, y: np.ndarray, 
                      feature_names: List[str],
                      functional_groups: List[str],
                      metadata: Optional[List[Dict]] = None,
                      test_size: float = 0.2,
                      random_state: int = 42,
                      split_by_compound: bool = True) -> Tuple[FunctionalGroupClassifier, Dict]:
    """
    Complete training and evaluation pipeline.
    
    Args:
        X: Feature matrix
        y: Label matrix
        feature_names: Feature names
        functional_groups: Functional group names
        metadata: Sample metadata (required if split_by_compound=True)
        test_size: Fraction of data for testing
        random_state: Random seed
        split_by_compound: If True, split by compound to avoid data leakage
        
    Returns:
        Trained classifier and evaluation results
    """
    # Split data - compound-level split to avoid data leakage
    if split_by_compound:
        if metadata is None:
            raise ValueError("metadata required for compound-level splitting")
        
        from .dataset_builder import split_by_compound as split_fn
        X_train, X_test, y_train, y_test = split_fn(
            X, y, metadata, test_size=test_size, random_state=random_state
        )
    else:
        # Standard random split (WARNING: may have data leakage!)
        print("⚠ WARNING: Using random split. This may cause data leakage if the same")
        print("   compound has multiple measurements in both train and test sets.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("\nTraining Random Forest classifier...")
    classifier = FunctionalGroupClassifier(random_state=random_state)
    classifier.fit(X_train, y_train, feature_names, functional_groups)
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = classifier.predict(X_test)
    results = evaluate_model(y_test, y_pred, functional_groups, verbose=True)
    
    return classifier, results
