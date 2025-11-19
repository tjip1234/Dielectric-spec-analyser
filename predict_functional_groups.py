"""
Prediction script for classifying unknown compounds.

Load a trained model and predict functional groups for new S1P measurements.
"""

import argparse
from pathlib import Path
import numpy as np
from ml_classifier import (
    FunctionalGroupClassifier,
    extract_features_from_s1p,
    get_functional_group_names,
    parse_s1p_file,
    filter_frequency_range
)


def predict_single_file(model_path: Path, s1p_path: Path, 
                       solvent_path: Path = None,
                       threshold: float = 0.5,
                       min_freq_hz: float = 100e6):
    """
    Predict functional groups for a single S1P file.
    
    Args:
        model_path: Path to trained model
        s1p_path: Path to S1P file to classify
        solvent_path: Path to pure solvent S1P (for correction)
        threshold: Probability threshold for positive prediction
        min_freq_hz: Minimum frequency
    """
    # Load model
    print(f"Loading model from {model_path}...")
    classifier = FunctionalGroupClassifier.load(model_path)
    
    # Extract features
    print(f"Extracting features from {s1p_path.name}...")
    poly_features = extract_features_from_s1p(s1p_path, solvent_path, min_freq_hz)
    
    if poly_features is None:
        print("Error: Failed to extract features")
        return
    
    # Add context features (using defaults since we don't have metadata)
    poly_feature_vector = np.concatenate([
        poly_features['db_coeffs'],
        poly_features['phase_coeffs']
    ])
    
    # Context: assume pure compound, no solvent
    context_vector = np.array([
        1,      # is_pure
        0, 0, 0, 0, 1  # solvent_none
    ])
    
    feature_vector = np.concatenate([poly_feature_vector, context_vector])
    X = feature_vector.reshape(1, -1)
    
    # Predict
    print("\nPredicting functional groups...")
    predictions = classifier.predict(X)[0]
    probabilities = [p[0] for p in classifier.predict_proba(X)]
    
    functional_groups = get_functional_group_names()
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"\nFile: {s1p_path.name}")
    if solvent_path:
        print(f"Solvent correction: {solvent_path.name}")
    
    print(f"\nPredicted Functional Groups (threshold = {threshold}):")
    print("-" * 70)
    
    predicted_groups = []
    for i, (group, pred, prob) in enumerate(zip(functional_groups, predictions, probabilities)):
        if pred == 1 or prob >= threshold:
            predicted_groups.append(group)
            print(f"  ✓ {group.upper():30s}  (probability: {prob:.3f})")
    
    if not predicted_groups:
        print("  (No functional groups predicted)")
    
    print("\nAll probabilities:")
    print("-" * 70)
    for group, prob in sorted(zip(functional_groups, probabilities), 
                             key=lambda x: x[1], reverse=True):
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {group:20s} {bar} {prob:.3f}")
    
    print("="*70 + "\n")


def predict_batch(model_path: Path, s1p_files: list, 
                 output_csv: Path = None,
                 threshold: float = 0.5):
    """
    Predict functional groups for multiple S1P files.
    
    Args:
        model_path: Path to trained model
        s1p_files: List of S1P file paths
        output_csv: Path to save results as CSV
        threshold: Probability threshold
    """
    classifier = FunctionalGroupClassifier.load(model_path)
    functional_groups = get_functional_group_names()
    
    results = []
    
    for s1p_path in s1p_files:
        print(f"Processing {s1p_path.name}...")
        
        poly_features = extract_features_from_s1p(s1p_path)
        if poly_features is None:
            print(f"  Skipping (failed to extract features)")
            continue
        
        poly_feature_vector = np.concatenate([
            poly_features['db_coeffs'],
            poly_features['phase_coeffs']
        ])
        
        context_vector = np.array([1, 0, 0, 0, 0, 1])  # is_pure, solvent_none
        feature_vector = np.concatenate([poly_feature_vector, context_vector])
        X = feature_vector.reshape(1, -1)
        
        probabilities = [p[0] for p in classifier.predict_proba(X)]
        
        result = {
            'filename': s1p_path.name,
            'predicted_groups': [functional_groups[i] for i, p in enumerate(probabilities) if p >= threshold],
            'probabilities': {group: prob for group, prob in zip(functional_groups, probabilities)}
        }
        results.append(result)
        
        print(f"  Predicted: {', '.join(result['predicted_groups']) if result['predicted_groups'] else 'None'}")
    
    # Save to CSV if requested
    if output_csv:
        import csv
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Predicted_Groups'] + functional_groups)
            
            for result in results:
                row = [
                    result['filename'],
                    ', '.join(result['predicted_groups'])
                ] + [f"{result['probabilities'][g]:.3f}" for g in functional_groups]
                writer.writerow(row)
        
        print(f"\nResults saved to {output_csv}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Predict functional groups from S1P files using trained model'
    )
    parser.add_argument(
        'model',
        type=str,
        help='Path to trained model (.pkl file)'
    )
    parser.add_argument(
        's1p_files',
        nargs='+',
        type=str,
        help='S1P file(s) to classify'
    )
    parser.add_argument(
        '--solvent',
        type=str,
        default=None,
        help='Path to pure solvent S1P file (for solvent correction)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probability threshold for positive prediction (default: 0.5)'
    )
    parser.add_argument(
        '--min-freq',
        type=float,
        default=100e6,
        help='Minimum frequency in Hz (default: 100e6 = 100 MHz)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: process multiple files and save CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Output CSV file for batch mode (default: predictions.csv)'
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    s1p_paths = [Path(f) for f in args.s1p_files]
    for p in s1p_paths:
        if not p.exists():
            print(f"Error: S1P file not found: {p}")
            return
    
    solvent_path = Path(args.solvent) if args.solvent else None
    if solvent_path and not solvent_path.exists():
        print(f"Error: Solvent file not found: {solvent_path}")
        return
    
    if args.batch:
        predict_batch(
            model_path,
            s1p_paths,
            output_csv=Path(args.output),
            threshold=args.threshold
        )
    else:
        if len(s1p_paths) > 1:
            print("Warning: Multiple files provided but not in batch mode. Only processing first file.")
        predict_single_file(
            model_path,
            s1p_paths[0],
            solvent_path=solvent_path,
            threshold=args.threshold,
            min_freq_hz=args.min_freq
        )


if __name__ == '__main__':
    main()
