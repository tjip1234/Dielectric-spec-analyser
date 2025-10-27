"""
Test script to verify S1P GUI modules are working
"""

import sys
import numpy as np
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from s1p_gui import constants, formulas, data_loader, gui_main
        print("  ✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_formulas():
    """Test formula calculations"""
    print("\nTesting formulas...")
    from s1p_gui.formulas import calculate_dielectric_properties, calculate_slope, calculate_integral
    
    # Create mock S11 data
    freq = np.linspace(500e6, 3e9, 100)
    s11 = 0.5 * np.exp(1j * np.linspace(0, np.pi, 100))
    
    # Calculate properties
    data = calculate_dielectric_properties(s11, freq)
    
    assert 'epsilon_prime' in data
    assert 'epsilon_double_prime' in data
    assert len(data['frequency']) == 100
    
    # Test slope calculation
    slope, intercept = calculate_slope(freq, data['epsilon_prime'])
    assert not np.isnan(slope)
    
    # Test integral
    integral = calculate_integral(freq, data['epsilon_prime'])
    assert not np.isnan(integral)
    
    print("  ✓ Formula calculations working")
    return True

def test_data_structures():
    """Test data loader structures"""
    print("\nTesting data structures...")
    from s1p_gui.data_loader import DataManager, S1PDataFile
    
    # Test DataManager
    manager = DataManager()
    assert len(manager) == 0
    
    manager.files.append(None)  # Mock file
    assert len(manager) == 1
    
    manager.clear_all()
    assert len(manager) == 0
    
    print("  ✓ Data structures working")
    return True

def test_constants():
    """Test constants module"""
    print("\nTesting constants...")
    from s1p_gui.constants import EPSILON_0, FREQ_RANGES, AVAILABLE_METRICS
    
    assert EPSILON_0 > 0
    assert 'low' in FREQ_RANGES
    assert 'mid' in FREQ_RANGES
    assert 'full' in FREQ_RANGES
    assert len(AVAILABLE_METRICS) > 0
    
    print("  ✓ Constants loaded")
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    print("\nChecking dependencies...")
    
    packages = {
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'PyQt5': 'PyQt5',
        'skrf': 'scikit-rf'
    }
    
    all_installed = True
    
    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
            print(f"  ✓ {package_name} installed")
        except ImportError:
            print(f"  ✗ {package_name} NOT installed - run: pip install {package_name}")
            all_installed = False
    
    return all_installed

def main():
    """Run all tests"""
    print("="*60)
    print("S1P GUI - Module Tests")
    print("="*60)
    
    all_passed = True
    
    # Check dependencies first
    if not check_dependencies():
        print("\n" + "="*60)
        print("⚠ Install missing dependencies first:")
        print("  pip install -r requirements_gui.txt")
        print("="*60)
        return
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_constants()
    all_passed &= test_formulas()
    all_passed &= test_data_structures()
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nYou can now run the GUI:")
        print("  python run_s1p_gui.py")
    else:
        print("✗ Some tests failed")
        print("\nCheck the errors above and fix issues")
    print("="*60)

if __name__ == '__main__':
    main()
