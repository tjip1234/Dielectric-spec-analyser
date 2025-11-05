#!/usr/bin/env python3
"""
Example: How to use the probe calibration system

This script shows how to:
1. Create a calibration from your reference liquid S1P files
2. Use it to convert unknown samples
3. Save/load calibrations
"""

import numpy as np
from pathlib import Path
from s1p_gui.calibration import ProbeCalibration, ReferenceLibrary
from s1p_gui.data_loader import S1PDataFile
import skrf as rf


def load_s1p_as_dict(filepath):
    """Load S1P file and return as dict with 's11_complex' and 'frequency'"""
    network = rf.Network(str(filepath))
    return {
        'frequency': network.f,
        's11_complex': network.s[:, 0, 0]
    }


def example_basic_calibration():
    """Basic example: Create calibration from three reference liquids"""
    print("=" * 60)
    print("EXAMPLE 1: Basic 3-Liquid Calibration")
    print("=" * 60)

    # Step 1: Create calibration object
    calibration = ProbeCalibration()

    # Step 2: Load reference permittivity data
    print("\n1. Loading reference data for 25°C...")
    reference_data = calibration.load_reference_data(temperature=25.0)
    print("   Loaded reference data for water, ethanol, isopropanol")

    # Step 3: Load your S1P measurements
    # NOTE: You need to have these files!
    print("\n2. Loading S11 measurements for reference liquids...")
    try:
        s11_measurements = {
            'water': load_s1p_as_dict('references/DI-water.s1p'),
            'ethanol': load_s1p_as_dict('references/ethanol.s1p'),
            'isopropanol': load_s1p_as_dict('references/propan-2-ol.s1p'),
        }
        print("   Successfully loaded all reference measurements")
    except FileNotFoundError:
        print("   ERROR: Could not find S1P files. Update the paths!")
        print("   Expected files: water.s1p, ethanol.s1p, isopropanol.s1p")
        return None

    # Step 4: Perform calibration
    print("\n3. Performing calibration...")
    if calibration.calibrate(s11_measurements, reference_data):
        print("   ✓ Calibration successful!")
        return calibration
    else:
        print("   ✗ Calibration failed!")
        return None


def example_convert_unknown(calibration):
    """Example: Use calibration to convert unknown sample"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Convert Unknown Sample")
    print("=" * 60)

    if not calibration:
        print("No calibration available!")
        return

    print("\n1. Loading unknown sample...")
    try:
        unknown_data = load_s1p_as_dict('path/to/unknown_sample.s1p')
    except FileNotFoundError:
        print("   ERROR: Could not find unknown_sample.s1p")
        return

    print("   Loaded unknown sample")

    print("\n2. Converting S11 to permittivity...")
    epsilon_real, epsilon_imag = calibration.convert_s11_to_permittivity(
        unknown_data['s11_complex'],
        unknown_data['frequency']
    )

    print("   ✓ Conversion complete!")
    print(f"\n   Frequency range: {unknown_data['frequency'][0]/1e9:.2f} - {unknown_data['frequency'][-1]/1e9:.2f} GHz")
    print(f"   ε' range: {np.nanmin(epsilon_real):.2f} - {np.nanmax(epsilon_real):.2f}")
    print(f"   ε'' range: {np.nanmin(epsilon_imag):.2f} - {np.nanmax(epsilon_imag):.2f}")

    return epsilon_real, epsilon_imag


def example_save_load_calibration(calibration):
    """Example: Save and load calibration"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Save and Load Calibration")
    print("=" * 60)

    if not calibration:
        print("No calibration to save!")
        return

    filepath = 'my_probe_calibration.pkl'

    print(f"\n1. Saving calibration to {filepath}...")
    if calibration.save(filepath):
        print("   ✓ Save successful!")

        print(f"\n2. Loading calibration from {filepath}...")
        loaded_cal = ProbeCalibration.load(filepath)
        if loaded_cal and loaded_cal.is_calibrated():
            print("   ✓ Load successful! Calibration is ready to use")
        else:
            print("   ✗ Load failed!")
    else:
        print("   ✗ Save failed!")


def example_with_data_manager():
    """Example: Using calibration with DataManager"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Integration with DataManager")
    print("=" * 60)

    from s1p_gui.data_loader import DataManager

    # Create data manager
    print("\n1. Creating data manager...")
    data_manager = DataManager()

    # Load calibration (from previous save, or new)
    print("2. Loading calibration...")
    calibration = ProbeCalibration()
    calibration.load_reference_data()

    # Create dummy measurements for demo
    s11_measurements = {
        'water': {
            'frequency': np.linspace(0.5e9, 3e9, 16),
            's11_complex': 0.5 * np.ones(16)
        },
        'ethanol': {
            'frequency': np.linspace(0.5e9, 3e9, 16),
            's11_complex': 0.4 * np.ones(16)
        },
        'isopropanol': {
            'frequency': np.linspace(0.5e9, 3e9, 16),
            's11_complex': 0.45 * np.ones(16)
        }
    }

    if calibration.calibrate(s11_measurements):
        print("   ✓ Calibration complete")

        # Set calibration in data manager
        print("\n3. Setting calibration in data manager...")
        data_manager.set_calibration(calibration)
        print("   ✓ All future files will use this calibration")

        # When you add files, they automatically use the calibration
        print("\n4. Adding files (they use calibration automatically)...")
        try:
            data_file = data_manager.add_file(Path('path/to/sample.s1p'))
            if data_file:
                print("   ✓ File loaded with automatic calibration applied")
        except:
            print("   (File not found, but calibration is set up)")


def example_reference_data_structure():
    """Example: Understanding the reference data structure"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Reference Data Structure")
    print("=" * 60)

    refs = ReferenceLibrary.get_water_25c()

    print("\nWater reference data at 25°C:")
    print(f"  Frequencies: {len(refs['frequency'])} points")
    print(f"  Range: {refs['frequency'][0]/1e9:.2f} - {refs['frequency'][-1]/1e9:.2f} GHz")
    print(f"\n  ε' (real permittivity):")
    print(f"    Min: {np.min(refs['epsilon_real']):.2f}")
    print(f"    Max: {np.max(refs['epsilon_real']):.2f}")
    print(f"\n  ε'' (imaginary permittivity):")
    print(f"    Min: {np.min(refs['epsilon_imag']):.2f}")
    print(f"    Max: {np.max(refs['epsilon_imag']):.2f}")


if __name__ == '__main__':
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Probe Calibration Examples".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    # Run examples
    example_reference_data_structure()

    cal = example_basic_calibration()

    if cal:
        example_convert_unknown(cal)
        example_save_load_calibration(cal)
        example_with_data_manager()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60 + "\n")
