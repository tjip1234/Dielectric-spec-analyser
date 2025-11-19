"""
Main training pipeline for functional group classification.

This script orchestrates the complete ML workflow:
1. Load database
2. Extract features
3. Build dataset
4. Train model
5. Evaluate and visualize results
"""

import argparse
from pathlib import Path
import numpy as np
from ml_classifier import (
    build_dataset,
    get_functional_group_names,
    train_and_evaluate,
    plot_feature_importance,
    plot_per_group_importance,
    plot_confusion_matrix_per_group,
    plot_performance_summary,
    visualize_polynomial_coefficients
)


def main():
    parser = argparse.ArgumentParser(
        description='Train ML classifier for functional group prediction from dielectric spectroscopy'
    )
    parser.add_argument(
        '--database',
        type=str,
        default='Database',
        help='Path to database directory (default: Database)'
    )
    parser.add_argument(
        '--poly-degree',
        type=int,
        default=6,
        help='Polynomial degree for feature extraction (default: 6)'
    )
    parser.add_argument(
        '--min-freq',
        type=float,
        default=100e6,
        help='Minimum frequency in Hz (default: 100e6 = 100 MHz)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--save-model',
        type=str,
        default='trained_model.pkl',
        help='Path to save trained model (default: trained_model.pkl)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save plots and results (default: results)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    database_dir = Path(args.database)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("FUNCTIONAL GROUP CLASSIFICATION FROM DIELECTRIC SPECTROSCOPY")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Database: {database_dir}")
    print(f"  Polynomial degree: {args.poly_degree}")
    print(f"  Minimum frequency: {args.min_freq/1e6:.0f} MHz")
    print(f"  Test size: {args.test_size:.1%}")
    print(f"  Random seed: {args.random_seed}")
    print(f"  Output directory: {output_dir}")
    
    # Step 1: Build dataset
    print("\n" + "="*70)
    print("STEP 1: BUILDING DATASET")
    print("="*70)
    
    X, y, feature_names, metadata = build_dataset(
        database_dir,
        poly_degree=args.poly_degree,
        min_freq_hz=args.min_freq,
        verbose=True
    )
    
    functional_groups = get_functional_group_names()
    
    # Save dataset info
    dataset_info_path = output_dir / 'dataset_info.txt'
    with open(dataset_info_path, 'w') as f:
        f.write(f"Dataset Information\n")
        f.write(f"==================\n\n")
        f.write(f"Total samples: {X.shape[0]}\n")
        f.write(f"Total features: {X.shape[1]}\n")
        f.write(f"Polynomial degree: {args.poly_degree}\n")
        f.write(f"Minimum frequency: {args.min_freq/1e6:.0f} MHz\n\n")
        f.write(f"Functional Groups:\n")
        for i, group in enumerate(functional_groups):
            count = np.sum(y[:, i])
            if count > 0:
                f.write(f"  {group}: {count} samples ({count/len(y)*100:.1f}%)\n")
        f.write(f"\nFeatures:\n")
        for name in feature_names:
            f.write(f"  - {name}\n")
    
    print(f"\nDataset info saved to {dataset_info_path}")
    
    # Step 2: Train and evaluate
    print("\n" + "="*70)
    print("STEP 2: TRAINING AND EVALUATION")
    print("="*70)
    print("\n⚠ Using COMPOUND-LEVEL SPLIT to prevent data leakage")
    print("   (All measurements from same compound stay in train OR test, not both)\n")
    
    classifier, results = train_and_evaluate(
        X, y,
        feature_names,
        functional_groups,
        metadata=metadata,
        test_size=args.test_size,
        random_state=args.random_seed,
        split_by_compound=True
    )
    
    # Save model
    model_path = Path(args.save_model)
    classifier.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save evaluation results
    results_path = output_dir / 'evaluation_results.txt'
    with open(results_path, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"==================\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Hamming Loss: {results['hamming_loss']:.4f}\n")
        f.write(f"  Exact Match Ratio: {results['exact_match_ratio']:.4f}\n\n")
        f.write(f"Per-Group Performance:\n")
        f.write(f"-" * 50 + "\n")
        for group, metrics in results['per_group'].items():
            f.write(f"\n{group.upper()}:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1']:.4f}\n")
            f.write(f"  Support: {metrics['support']}\n")
    
    print(f"Evaluation results saved to {results_path}")
    
    # Step 3: Generate visualizations
    if not args.no_plots:
        print("\n" + "="*70)
        print("STEP 3: GENERATING VISUALIZATIONS")
        print("="*70)
        
        from ml_classifier import split_by_compound
        
        # Re-split for visualization (use same random seed and compound-level split)
        X_train, X_test, y_train, y_test = split_by_compound(
            X, y, metadata, test_size=args.test_size, random_state=args.random_seed
        )
        y_pred = classifier.predict(X_test)
        
        # Feature importance
        print("\nPlotting feature importance...")
        importance = classifier.get_feature_importance()
        plot_feature_importance(
            importance,
            top_n=20,
            save_path=output_dir / 'feature_importance.png'
        )
        
        # Per-group importance
        print("Plotting per-group feature importance...")
        per_group_importance = classifier.get_per_group_importance()
        plot_per_group_importance(
            per_group_importance,
            feature_names,
            top_n=10,
            save_path=output_dir / 'per_group_importance.png'
        )
        
        # Confusion matrices
        print("Plotting confusion matrices...")
        plot_confusion_matrix_per_group(
            y_test,
            y_pred,
            functional_groups,
            save_path=output_dir / 'confusion_matrices.png'
        )
        
        # Performance summary
        print("Plotting performance summary...")
        plot_performance_summary(
            results,
            save_path=output_dir / 'performance_summary.png'
        )
        
        # Polynomial coefficient visualization
        print("Plotting polynomial coefficient distributions...")
        visualize_polynomial_coefficients(
            X, y,
            feature_names,
            functional_groups,
            poly_degree=args.poly_degree,
            save_path=output_dir / 'polynomial_coefficients.png'
        )
        
        print(f"\nAll plots saved to {output_dir}/")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  Model: {model_path}")
    print(f"  Plots: {output_dir}/")
    print(f"  Dataset info: {dataset_info_path}")
    print(f"  Evaluation: {results_path}")
    print("\n")


if __name__ == '__main__':
    main()
