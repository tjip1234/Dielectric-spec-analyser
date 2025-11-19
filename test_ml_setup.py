"""
Test script to verify ML classifier installation and basic functionality.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import yaml
        print("  ✓ pyyaml")
    except ImportError as e:
        print(f"  ✗ pyyaml: {e}")
        return False
    
    try:
        import sklearn
        print("  ✓ scikit-learn")
    except ImportError as e:
        print(f"  ✗ scikit-learn: {e}")
        return False
    
    try:
        import matplotlib
        print("  ✓ matplotlib")
    except ImportError as e:
        print(f"  ✗ matplotlib: {e}")
        return False
    
    try:
        import seaborn
        print("  ✓ seaborn")
    except ImportError as e:
        print(f"  ✗ seaborn: {e}")
        return False
    
    return True


def test_ml_classifier_imports():
    """Test that ml_classifier package can be imported."""
    print("\nTesting ml_classifier package...")
    
    try:
        from ml_classifier import (
            parse_s1p_file,
            extract_features_from_s1p,
            build_dataset,
            FunctionalGroupClassifier,
            plot_feature_importance
        )
        print("  ✓ All ml_classifier modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import ml_classifier: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_access():
    """Test that database directory exists and has data."""
    print("\nTesting database access...")
    
    database_dir = Path('Database')
    
    if not database_dir.exists():
        print(f"  ✗ Database directory not found: {database_dir}")
        return False
    
    print(f"  ✓ Database directory exists")
    
    # Count compound directories
    yaml_files = list(database_dir.rglob('compound.yaml'))
    print(f"  ✓ Found {len(yaml_files)} compound.yaml files")
    
    if len(yaml_files) == 0:
        print("  ⚠ Warning: No compound.yaml files found")
        return False
    
    # Check for S1P files
    s1p_files = list(database_dir.rglob('*.s1p'))
    print(f"  ✓ Found {len(s1p_files)} S1P files")
    
    return True


def test_feature_extraction():
    """Test feature extraction on a sample file."""
    print("\nTesting feature extraction...")
    
    database_dir = Path('Database')
    s1p_files = list(database_dir.rglob('*.s1p'))
    
    if not s1p_files:
        print("  ⚠ Skipping: No S1P files found")
        return True
    
    test_file = s1p_files[0]
    print(f"  Testing with: {test_file.name}")
    
    try:
        from ml_classifier import parse_s1p_file, extract_features_from_s1p
        
        # Test parsing
        freq, s11_db, phase = parse_s1p_file(test_file)
        print(f"  ✓ Parsed S1P file: {len(freq)} frequency points")
        
        # Test feature extraction
        features = extract_features_from_s1p(test_file, min_freq_hz=100e6, poly_degree=6)
        
        if features is not None:
            print(f"  ✓ Extracted polynomial features")
            print(f"    - DB coefficients: {len(features['db_coeffs'])}")
            print(f"    - Phase coefficients: {len(features['phase_coeffs'])}")
            return True
        else:
            print(f"  ✗ Feature extraction returned None")
            return False
            
    except Exception as e:
        print(f"  ✗ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("ML CLASSIFIER INSTALLATION TEST")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Import test", test_imports()))
    results.append(("ML classifier import test", test_ml_classifier_imports()))
    results.append(("Database access test", test_database_access()))
    results.append(("Feature extraction test", test_feature_extraction()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n✓ All tests passed! The ML classifier is ready to use.")
        print("\nNext steps:")
        print("  1. Run the demo: python demo_ml_classifier.py")
        print("  2. Train a model: python train_classifier.py")
        print("  3. Make predictions: python predict_functional_groups.py <model> <s1p_file>")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        if failed == 1 and results[0][1]:  # Only import failed
            print("\nTo install missing packages, run:")
            print("  pip install -r requirements_ml.txt")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
