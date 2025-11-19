"""
Dataset construction for ML classification.
Builds feature matrix X and label matrix y from database.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from .data_loader import (
    find_all_compounds, 
    find_pure_solvent_file,
    extract_concentration_percent,
    normalize_solvent_name
)
from .feature_extractor import extract_features_from_s1p, create_feature_names


# All functional groups found in the database
ALL_FUNCTIONAL_GROUPS = [
    'alcohol', 'hydroxyl', 'ether', 'ketone', 'carbonyl',
    'carboxylate', 'amino', 'sodium_salt', 'ionic', 
    'diol', 'halide', 'alkali_metal', 'nitrate'
]


def encode_functional_groups(groups: List[str]) -> np.ndarray:
    """
    Encode functional groups as a multi-hot binary vector.
    
    Args:
        groups: List of functional group names
        
    Returns:
        Binary array of length len(ALL_FUNCTIONAL_GROUPS)
    """
    encoding = np.zeros(len(ALL_FUNCTIONAL_GROUPS), dtype=int)
    
    for group in groups:
        if group in ALL_FUNCTIONAL_GROUPS:
            idx = ALL_FUNCTIONAL_GROUPS.index(group)
            encoding[idx] = 1
    
    return encoding


def encode_solvent_onehot(solvent_name: str) -> Dict[str, int]:
    """
    One-hot encode solvent type.
    
    Args:
        solvent_name: Normalized solvent name ('water', 'ethanol', 'isopropanol', 'other', 'none')
        
    Returns:
        Dictionary with binary flags for each solvent type
    """
    solvents = ['water', 'ethanol', 'isopropanol', 'other', 'none']
    encoding = {f'solvent_{s}': 0 for s in solvents}
    
    if solvent_name in solvents:
        encoding[f'solvent_{solvent_name}'] = 1
    
    return encoding


def build_dataset(database_dir: Path, poly_degree: int = 6, 
                 min_freq_hz: float = 100e6, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
    """
    Build complete dataset from database directory.
    
    Args:
        database_dir: Root database directory
        poly_degree: Polynomial degree for feature extraction
        min_freq_hz: Minimum frequency threshold
        verbose: Print progress messages
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Label matrix (n_samples, n_functional_groups)
        feature_names: List of feature names
        metadata: List of metadata dicts for each sample
    """
    compounds = find_all_compounds(database_dir)
    
    if verbose:
        print(f"Found {len(compounds)} compounds in database")
    
    X_list = []
    y_list = []
    metadata_list = []
    
    # Build feature names
    poly_feature_names = create_feature_names(poly_degree)
    context_features = ['is_pure',
                       'solvent_water', 'solvent_ethanol', 'solvent_isopropanol', 
                       'solvent_other', 'solvent_none']
    feature_names = poly_feature_names + context_features
    
    for compound in compounds:
        compound_info = compound['compound']
        compound_dir = compound['_dir']
        
        # Skip if no functional groups defined
        if 'functional_groups' not in compound_info or not compound_info['functional_groups']:
            if verbose:
                print(f"Skipping {compound_info.get('common_name', 'unknown')}: no functional groups")
            continue
        
        functional_groups = compound_info['functional_groups']
        molecular_weight = compound_info.get('molecular_weight', 0.0)
        
        # Process each measurement
        for measurement in compound.get('measurements', []):
            data_file = measurement.get('data_file')
            if not data_file:
                continue
            
            s1p_path = compound_dir / data_file
            if not s1p_path.exists():
                if verbose:
                    print(f"Warning: S1P file not found: {s1p_path}")
                continue
            
            # Determine if solvent correction is needed
            solvent = measurement.get('solvent')
            is_pure = measurement.get('is_pure', True)
            solvent_s1p_path = None
            
            if not is_pure and solvent:
                solvent_s1p_path = find_pure_solvent_file(compound_dir, solvent, database_dir)
                if solvent_s1p_path and verbose:
                    print(f"  Using solvent correction: {solvent_s1p_path.name}")
            
            # Extract polynomial features
            poly_features = extract_features_from_s1p(s1p_path, solvent_s1p_path, 
                                                     min_freq_hz, poly_degree)
            
            if poly_features is None:
                continue
            
            # Flatten polynomial coefficients
            poly_feature_vector = np.concatenate([
                poly_features['db_coeffs'],
                poly_features['phase_coeffs']
            ])
            
            # Extract context features
            normalized_solvent = normalize_solvent_name(solvent)
            solvent_encoding = encode_solvent_onehot(normalized_solvent)
            
            context_vector = np.array([
                1 if is_pure else 0,
                solvent_encoding['solvent_water'],
                solvent_encoding['solvent_ethanol'],
                solvent_encoding['solvent_isopropanol'],
                solvent_encoding['solvent_other'],
                solvent_encoding['solvent_none']
            ])
            
            # Combine features
            feature_vector = np.concatenate([poly_feature_vector, context_vector])
            
            # Encode labels
            label_vector = encode_functional_groups(functional_groups)
            
            X_list.append(feature_vector)
            y_list.append(label_vector)
            metadata_list.append({
                'compound_name': compound_info.get('common_name', 'unknown'),
                'compound_id': len([c for c in compounds if c == compound]),  # Unique compound index
                'chemical_name': compound_info.get('chemical_name', ''),
                'data_file': data_file,
                'is_pure': is_pure,
                'solvent': solvent,
                'functional_groups': functional_groups
            })
            
            if verbose:
                print(f"✓ {compound_info['common_name']}: {data_file} -> {len(functional_groups)} groups")
    
    if len(X_list) == 0:
        raise ValueError("No valid samples found in database")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    if verbose:
        print(f"\nDataset built: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Label distribution:")
        for i, group in enumerate(ALL_FUNCTIONAL_GROUPS):
            count = np.sum(y[:, i])
            if count > 0:
                print(f"  {group}: {count} samples")
    
    return X, y, feature_names, metadata_list


def get_functional_group_names() -> List[str]:
    """Return the list of all functional group names (for label interpretation)."""
    return ALL_FUNCTIONAL_GROUPS.copy()


def split_by_compound(X: np.ndarray, y: np.ndarray, metadata: List[Dict],
                     test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset by compound to avoid data leakage.
    
    All measurements from the same compound stay together in either train or test set.
    This prevents the model from learning compound-specific characteristics and being
    tested on other measurements of the same compound.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Label matrix (n_samples, n_labels)
        metadata: List of metadata dicts (must include 'compound_name')
        test_size: Fraction of compounds (not samples) for test set
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Group samples by compound
    compound_to_indices = {}
    for i, meta in enumerate(metadata):
        compound_name = meta['compound_name']
        if compound_name not in compound_to_indices:
            compound_to_indices[compound_name] = []
        compound_to_indices[compound_name].append(i)
    
    # Get list of unique compounds
    unique_compounds = list(compound_to_indices.keys())
    n_compounds = len(unique_compounds)
    n_test_compounds = max(1, int(n_compounds * test_size))
    
    # Randomly select test compounds
    rng = np.random.RandomState(random_state)
    test_compounds = rng.choice(unique_compounds, size=n_test_compounds, replace=False)
    train_compounds = [c for c in unique_compounds if c not in test_compounds]
    
    # Get indices for train and test
    train_indices = []
    test_indices = []
    
    for compound in train_compounds:
        train_indices.extend(compound_to_indices[compound])
    
    for compound in test_compounds:
        test_indices.extend(compound_to_indices[compound])
    
    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    print(f"\nCompound-level split:")
    print(f"  Train: {len(train_compounds)} compounds, {len(train_indices)} samples")
    print(f"  Test: {len(test_compounds)} compounds, {len(test_indices)} samples")
    print(f"  Train compounds: {', '.join(sorted(train_compounds)[:5])}{'...' if len(train_compounds) > 5 else ''}")
    print(f"  Test compounds: {', '.join(sorted(test_compounds)[:5])}{'...' if len(test_compounds) > 5 else ''}")
    
    return X_train, X_test, y_train, y_test
