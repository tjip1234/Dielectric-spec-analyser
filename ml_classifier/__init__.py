"""
ML Classifier Package for Functional Group Prediction from Dielectric Spectroscopy.

This package provides tools for:
- Loading S1P measurement files and YAML metadata
- Extracting polynomial features from frequency-dependent dielectric behavior
- Training multi-label Random Forest classifiers
- Evaluating and visualizing results
"""

__version__ = '1.0.0'

from .data_loader import (
    parse_s1p_file,
    load_compound_yaml,
    find_all_compounds,
    find_pure_solvent_file,
    extract_concentration_percent,
    normalize_solvent_name
)

from .feature_extractor import (
    filter_frequency_range,
    extract_polynomial_coefficients,
    apply_solvent_correction,
    extract_features_from_s1p,
    create_feature_names
)

from .dataset_builder import (
    encode_functional_groups,
    encode_solvent_onehot,
    build_dataset,
    get_functional_group_names,
    split_by_compound,
    ALL_FUNCTIONAL_GROUPS
)

from .model import (
    FunctionalGroupClassifier,
    evaluate_model,
    train_and_evaluate
)

from .visualize import (
    plot_feature_importance,
    plot_per_group_importance,
    plot_confusion_matrix_per_group,
    plot_prediction_probabilities,
    plot_performance_summary,
    visualize_polynomial_coefficients
)

__all__ = [
    # Data loading
    'parse_s1p_file',
    'load_compound_yaml',
    'find_all_compounds',
    'find_pure_solvent_file',
    'extract_concentration_percent',
    'normalize_solvent_name',
    
    # Feature extraction
    'filter_frequency_range',
    'extract_polynomial_coefficients',
    'apply_solvent_correction',
    'extract_features_from_s1p',
    'create_feature_names',
    
    # Dataset building
    'encode_functional_groups',
    'encode_solvent_onehot',
    'build_dataset',
    'get_functional_group_names',
    'split_by_compound',
    'ALL_FUNCTIONAL_GROUPS',
    
    # Model
    'FunctionalGroupClassifier',
    'evaluate_model',
    'train_and_evaluate',
    
    # Visualization
    'plot_feature_importance',
    'plot_per_group_importance',
    'plot_confusion_matrix_per_group',
    'plot_prediction_probabilities',
    'plot_performance_summary',
    'visualize_polynomial_coefficients',
]
