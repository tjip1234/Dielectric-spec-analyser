"""
Visualization utilities for ML classification results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns


def plot_feature_importance(importance_dict: Dict, 
                           top_n: int = 20,
                           save_path: Optional[Path] = None):
    """
    Plot feature importance bar chart.
    
    Args:
        importance_dict: Dict from get_feature_importance()
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    importances = importance_dict['importances']
    feature_names = importance_dict.get('feature_names', 
                                       [f'Feature {i}' for i in range(len(importances))])
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances (Averaged Across All Functional Groups)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")
    
    plt.show()


def plot_per_group_importance(per_group_importance: Dict,
                             feature_names: List[str],
                             top_n: int = 10,
                             save_path: Optional[Path] = None):
    """
    Plot feature importance for each functional group separately.
    
    Args:
        per_group_importance: Dict from get_per_group_importance()
        feature_names: List of feature names
        top_n: Number of top features to show per group
        save_path: Path to save figure
    """
    n_groups = len(per_group_importance)
    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 4 * n_groups))
    
    if n_groups == 1:
        axes = [axes]
    
    for ax, (group, importances) in zip(axes, per_group_importance.items()):
        indices = np.argsort(importances)[::-1][:top_n]
        
        ax.barh(range(top_n), importances[indices])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'{group.upper()}')
        ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-group importance plot to {save_path}")
    
    plt.show()


def plot_confusion_matrix_per_group(y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   functional_groups: List[str],
                                   save_path: Optional[Path] = None):
    """
    Plot confusion matrix for each functional group.
    
    Args:
        y_true: True labels (n_samples, n_groups)
        y_pred: Predicted labels (n_samples, n_groups)
        functional_groups: Names of functional groups
        save_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    n_groups = y_true.shape[1]
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    
    for i, group in enumerate(functional_groups):
        # Skip groups with no positive samples
        if np.sum(y_true[:, i]) == 0:
            axes[i].text(0.5, 0.5, f'{group}\n(No samples)', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            continue
        
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Absent', 'Present'],
                   yticklabels=['Absent', 'Present'])
        axes[i].set_title(f'{group.upper()}')
        axes[i].set_ylabel('True')
        axes[i].set_xlabel('Predicted')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrices to {save_path}")
    
    plt.show()


def plot_prediction_probabilities(probabilities: List[np.ndarray],
                                 functional_groups: List[str],
                                 true_labels: Optional[np.ndarray] = None,
                                 sample_indices: Optional[List[int]] = None,
                                 save_path: Optional[Path] = None):
    """
    Plot prediction probabilities for selected samples.
    
    Args:
        probabilities: List of probability arrays from predict_proba()
        functional_groups: Names of functional groups
        true_labels: True labels for comparison (optional)
        sample_indices: Which samples to plot (default: first 10)
        save_path: Path to save figure
    """
    if sample_indices is None:
        sample_indices = list(range(min(10, len(probabilities[0]))))
    
    # Stack probabilities into matrix
    prob_matrix = np.column_stack([p[sample_indices] for p in probabilities])
    
    fig, ax = plt.subplots(figsize=(12, len(sample_indices) * 0.5 + 2))
    
    im = ax.imshow(prob_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(functional_groups)))
    ax.set_xticklabels(functional_groups, rotation=45, ha='right')
    ax.set_yticks(range(len(sample_indices)))
    ax.set_yticklabels([f'Sample {i}' for i in sample_indices])
    ax.set_xlabel('Functional Group')
    ax.set_ylabel('Sample')
    ax.set_title('Prediction Probabilities')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability')
    
    # Overlay true labels if provided
    if true_labels is not None:
        for i, sample_idx in enumerate(sample_indices):
            for j, group in enumerate(functional_groups):
                if true_labels[sample_idx, j] == 1:
                    ax.add_patch(plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8,
                                              fill=False, edgecolor='blue', linewidth=2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved probability plot to {save_path}")
    
    plt.show()


def plot_performance_summary(results: Dict, save_path: Optional[Path] = None):
    """
    Plot summary of model performance metrics.
    
    Args:
        results: Results dict from evaluate_model()
        save_path: Path to save figure
    """
    per_group = results['per_group']
    
    groups = list(per_group.keys())
    precision = [per_group[g]['precision'] for g in groups]
    recall = [per_group[g]['recall'] for g in groups]
    f1 = [per_group[g]['f1'] for g in groups]
    
    x = np.arange(len(groups))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Functional Group')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance by Functional Group')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance summary to {save_path}")
    
    plt.show()


def visualize_polynomial_coefficients(X: np.ndarray, 
                                     y: np.ndarray,
                                     feature_names: List[str],
                                     functional_groups: List[str],
                                     poly_degree: int = 6,
                                     save_path: Optional[Path] = None):
    """
    Visualize distribution of polynomial coefficients for different functional groups.
    
    Args:
        X: Feature matrix
        y: Label matrix
        feature_names: Feature names
        functional_groups: Functional group names
        poly_degree: Polynomial degree
        save_path: Path to save figure
    """
    # Extract polynomial coefficient indices
    db_coeff_indices = [feature_names.index(f'db_coeff_{i}') 
                        for i in range(poly_degree, -1, -1)]
    phase_coeff_indices = [feature_names.index(f'phase_coeff_{i}') 
                          for i in range(poly_degree, -1, -1)]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot DB coefficients
    ax = axes[0]
    for i, group in enumerate(functional_groups):
        if np.sum(y[:, i]) == 0:
            continue
        
        mask = y[:, i] == 1
        group_coeffs = X[mask][:, db_coeff_indices]
        mean_coeffs = np.mean(group_coeffs, axis=0)
        
        ax.plot(range(poly_degree + 1), mean_coeffs, marker='o', label=group, alpha=0.7)
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Mean Coefficient Value')
    ax.set_title('S11 Magnitude (dB) Polynomial Coefficients by Functional Group')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xticks(range(poly_degree + 1))
    
    # Plot Phase coefficients
    ax = axes[1]
    for i, group in enumerate(functional_groups):
        if np.sum(y[:, i]) == 0:
            continue
        
        mask = y[:, i] == 1
        group_coeffs = X[mask][:, phase_coeff_indices]
        mean_coeffs = np.mean(group_coeffs, axis=0)
        
        ax.plot(range(poly_degree + 1), mean_coeffs, marker='o', label=group, alpha=0.7)
    
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Mean Coefficient Value')
    ax.set_title('Phase Polynomial Coefficients by Functional Group')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xticks(range(poly_degree + 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved polynomial coefficient visualization to {save_path}")
    
    plt.show()
