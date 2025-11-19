"""
Analyze the database to understand dataset composition and quality.

This script provides insights into:
- Number of compounds and measurements
- Functional group distribution
- Frequency range coverage
- Data quality issues
"""

import argparse
from pathlib import Path
from collections import Counter
import numpy as np

from ml_classifier import (
    find_all_compounds,
    parse_s1p_file,
    filter_frequency_range
)


def analyze_database(database_dir: Path, min_freq_hz: float = 100e6):
    """
    Comprehensive analysis of the database.
    
    Args:
        database_dir: Path to database directory
        min_freq_hz: Minimum frequency threshold
    """
    print("="*70)
    print("DATABASE ANALYSIS")
    print("="*70)
    
    compounds = find_all_compounds(database_dir)
    
    print(f"\n📁 Database: {database_dir}")
    print(f"📊 Total compounds: {len(compounds)}")
    
    # Analyze compounds
    total_measurements = 0
    functional_group_counts = Counter()
    measurement_types = {'pure': 0, 'solution': 0}
    solvent_types = Counter()
    issues = []
    
    print("\n" + "-"*70)
    print("COMPOUNDS")
    print("-"*70)
    
    for i, compound in enumerate(compounds, 1):
        comp_info = compound['compound']
        comp_dir = compound['_dir']
        measurements = compound.get('measurements', [])
        
        name = comp_info.get('common_name', 'Unknown')
        chem_name = comp_info.get('chemical_name', '')
        groups = comp_info.get('functional_groups', [])
        
        print(f"\n{i}. {name}")
        if chem_name and chem_name != name:
            print(f"   Chemical name: {chem_name}")
        print(f"   Functional groups: {', '.join(groups) if groups else 'None defined'}")
        print(f"   Measurements: {len(measurements)}")
        
        if not groups:
            issues.append(f"{name}: No functional groups defined")
        
        # Count measurements
        total_measurements += len(measurements)
        for group in groups:
            functional_group_counts[group] += len(measurements)
        
        # Analyze measurements
        for j, meas in enumerate(measurements, 1):
            is_pure = meas.get('is_pure', True)
            solvent = meas.get('solvent')
            data_file = meas.get('data_file', '')
            
            if is_pure:
                measurement_types['pure'] += 1
            else:
                measurement_types['solution'] += 1
                if solvent:
                    solvent_types[solvent] += 1
            
            # Check if file exists
            s1p_path = comp_dir / data_file
            if not s1p_path.exists():
                issues.append(f"{name}/{data_file}: File not found")
                print(f"     {j}. {data_file} - ✗ FILE NOT FOUND")
            else:
                print(f"     {j}. {data_file} - ✓")
    
    # Functional group distribution
    print("\n" + "-"*70)
    print("FUNCTIONAL GROUP DISTRIBUTION")
    print("-"*70)
    
    if functional_group_counts:
        max_count = max(functional_group_counts.values())
        for group, count in functional_group_counts.most_common():
            bar_length = int((count / max_count) * 40)
            bar = "█" * bar_length
            print(f"{group:20s} {bar} {count:3d} samples")
    else:
        print("No functional groups found in database")
    
    # Measurement statistics
    print("\n" + "-"*70)
    print("MEASUREMENT STATISTICS")
    print("-"*70)
    print(f"Total measurements: {total_measurements}")
    print(f"  Pure compounds: {measurement_types['pure']}")
    print(f"  Solutions: {measurement_types['solution']}")
    
    if solvent_types:
        print(f"\nSolvent distribution:")
        for solvent, count in solvent_types.most_common():
            print(f"  {solvent}: {count}")
    
    # Analyze S1P files
    print("\n" + "-"*70)
    print("S1P FILE ANALYSIS")
    print("-"*70)
    
    freq_ranges = []
    point_counts = []
    usable_measurements = 0
    
    for compound in compounds:
        comp_dir = compound['_dir']
        for meas in compound.get('measurements', []):
            data_file = meas.get('data_file', '')
            s1p_path = comp_dir / data_file
            
            if s1p_path.exists():
                try:
                    freq, s11_db, phase = parse_s1p_file(s1p_path)
                    freq_filt, _, _ = filter_frequency_range(freq, s11_db, phase, min_freq_hz)
                    
                    if len(freq_filt) >= 7:  # Enough for 6th degree polynomial
                        usable_measurements += 1
                        freq_ranges.append((freq.min(), freq.max()))
                        point_counts.append(len(freq))
                    else:
                        issues.append(f"{s1p_path.name}: Only {len(freq_filt)} points above {min_freq_hz/1e6:.0f} MHz")
                
                except Exception as e:
                    issues.append(f"{s1p_path.name}: Parse error - {e}")
    
    if freq_ranges:
        min_freqs, max_freqs = zip(*freq_ranges)
        print(f"Usable measurements (≥7 points above {min_freq_hz/1e6:.0f} MHz): {usable_measurements}/{total_measurements}")
        print(f"\nFrequency range:")
        print(f"  Minimum start: {min(min_freqs)/1e6:.2f} MHz")
        print(f"  Maximum end: {max(max_freqs)/1e9:.3f} GHz")
        print(f"\nPoints per file:")
        print(f"  Mean: {np.mean(point_counts):.0f}")
        print(f"  Min: {min(point_counts)}")
        print(f"  Max: {max(point_counts)}")
    
    # Data quality issues
    if issues:
        print("\n" + "-"*70)
        print("⚠ DATA QUALITY ISSUES")
        print("-"*70)
        for issue in issues:
            print(f"  • {issue}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ {len(compounds)} compounds")
    print(f"✓ {total_measurements} total measurements")
    print(f"✓ {usable_measurements} usable for ML ({usable_measurements/total_measurements*100:.1f}%)")
    print(f"✓ {len(functional_group_counts)} unique functional groups")
    
    if issues:
        print(f"⚠ {len(issues)} data quality issues found")
    else:
        print(f"✓ No data quality issues detected")
    
    # Recommendations
    print("\n" + "-"*70)
    print("RECOMMENDATIONS")
    print("-"*70)
    
    min_samples_per_group = 5
    rare_groups = [g for g, c in functional_group_counts.items() if c < min_samples_per_group]
    
    if rare_groups:
        print(f"⚠ Low sample count for functional groups (< {min_samples_per_group} samples):")
        for group in rare_groups:
            print(f"   • {group}: {functional_group_counts[group]} samples")
        print(f"   Consider adding more samples or merging similar groups")
    
    if usable_measurements < 50:
        print(f"⚠ Dataset is small ({usable_measurements} samples)")
        print(f"   Consider:")
        print(f"   • Adding more compound measurements")
        print(f"   • Using cross-validation instead of train/test split")
        print(f"   • Reducing model complexity (lower poly degree, simpler model)")
    
    if measurement_types['solution'] > 0 and measurement_types['pure'] == 0:
        print(f"⚠ Only solution measurements found")
        print(f"   Solvent correction requires pure solvent measurements")
    
    if not issues and usable_measurements >= 50 and not rare_groups:
        print("✓ Dataset looks good! Ready for model training.")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze database composition and quality'
    )
    parser.add_argument(
        '--database',
        type=str,
        default='Database',
        help='Path to database directory (default: Database)'
    )
    parser.add_argument(
        '--min-freq',
        type=float,
        default=100e6,
        help='Minimum frequency in Hz (default: 100e6 = 100 MHz)'
    )
    
    args = parser.parse_args()
    
    database_dir = Path(args.database)
    
    if not database_dir.exists():
        print(f"Error: Database directory not found: {database_dir}")
        return 1
    
    analyze_database(database_dir, args.min_freq)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
