"""
Quick demo of the ML classification pipeline.

This script demonstrates the complete workflow with minimal configuration.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_classifier import (
    build_dataset,
    get_functional_group_names,
    train_and_evaluate,
    plot_feature_importance,
    plot_performance_summary,
    visualize_polynomial_coefficients,
    FunctionalGroupClassifier
)


def main():
    print("="*70)
    print("ML FUNCTIONAL GROUP CLASSIFICATION - QUICK DEMO")
    print("="*70)
    
    # Configuration
    database_dir = Path('Database')
    output_dir = Path('demo_results')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nStep 1: Loading database from {database_dir}...")
    print("-" * 70)
    
    try:
        # Build dataset
        X, y, feature_names, metadata = build_dataset(
            database_dir,
            poly_degree=6,
            min_freq_hz=100e6,
            verbose=True
        )
        
        functional_groups = get_functional_group_names()
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  - Samples: {X.shape[0]}")
        print(f"  - Features: {X.shape[1]}")
        print(f"  - Functional groups: {len(functional_groups)}")
        
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        print("\nMake sure:")
        print("  1. The 'Database' directory exists")
        print("  2. It contains compound.yaml files with functional_groups defined")
        print("  3. S1P files are present and readable")
        return
    
    print(f"\nStep 2: Training Random Forest classifier...")
    print("-" * 70)
    print("Using compound-level split to avoid data leakage...\n")
    
    try:
        # Train and evaluate with compound-level split
        classifier, results = train_and_evaluate(
            X, y,
            feature_names,
            functional_groups,
            metadata=metadata,
            test_size=0.2,
            random_state=42,
            split_by_compound=True
        )
        
        print(f"\n✓ Model trained successfully!")
        
        # Save model
        model_path = output_dir / 'demo_model.pkl'
        classifier.save(model_path)
        print(f"\n✓ Model saved to {model_path}")
        
    except Exception as e:
        print(f"\n✗ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nStep 3: Generating visualizations...")
    print("-" * 70)
    
    try:
        # Feature importance
        importance = classifier.get_feature_importance()
        plot_feature_importance(
            importance,
            top_n=15,
            save_path=output_dir / 'feature_importance.png'
        )
        print(f"✓ Feature importance plot saved")
        
        # Performance summary
        plot_performance_summary(
            results,
            save_path=output_dir / 'performance_summary.png'
        )
        print(f"✓ Performance summary plot saved")
        
        # Polynomial coefficients
        visualize_polynomial_coefficients(
            X, y,
            feature_names,
            functional_groups,
            poly_degree=6,
            save_path=output_dir / 'polynomial_coefficients.png'
        )
        print(f"✓ Polynomial coefficient plot saved")
        
    except Exception as e:
        print(f"\n⚠ Warning: Could not generate all plots: {e}")
        print("  (This is often due to matplotlib backend issues)")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nModel performance:")
    print(f"  - Hamming Loss: {results['hamming_loss']:.4f}")
    print(f"  - Exact Match Ratio: {results['exact_match_ratio']:.4f}")
    
    print(f"\nTop 5 most important features:")
    importances = importance['importances']
    feature_names_list = importance['feature_names']
    top_indices = importances.argsort()[::-1][:5]
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. {feature_names_list[idx]}: {importances[idx]:.4f}")
    
    print(f"\nPer-group F1 scores:")
    for group, metrics in results['per_group'].items():
        print(f"  - {group:20s}: {metrics['f1']:.3f}")
    
    print(f"\nTo use the trained model:")
    print(f"  python predict_functional_groups.py {model_path} <your_file.s1p>")
    print()


if __name__ == '__main__':
    main()
