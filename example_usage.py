"""
Quick example demonstrating the S1P GUI modules programmatically
(without launching the GUI)
"""

from pathlib import Path
import numpy as np
from s1p_gui.data_loader import DataManager
from s1p_gui.formulas import calculate_statistics, calculate_slope, calculate_integral

# Example: Load and analyze S1P files programmatically
def analyze_s1p_files(file_paths):
    """
    Example function showing how to use the modules programmatically
    
    Args:
        file_paths: List of paths to S1P files
    """
    # Create data manager
    manager = DataManager()
    
    # Load files
    print("Loading files...")
    for fp in file_paths:
        data_file = manager.add_file(Path(fp))
        if data_file:
            print(f"  ✓ Loaded: {data_file.name}")
        else:
            print(f"  ✗ Failed: {fp}")
    
    if len(manager) == 0:
        print("No files loaded!")
        return
    
    # Apply frequency filter (e.g., 500 MHz - 3 GHz)
    print("\nApplying frequency filter: 500 MHz - 3 GHz")
    manager.apply_frequency_filter_all(500e6, 3e9)
    
    # Analyze each file
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    for data_file in manager.files:
        data = data_file.get_data(use_filtered=True)
        
        if data is None:
            continue
        
        print(f"\n{data_file.name}")
        print("-" * 60)
        
        # Get frequency range
        freq_min, freq_max = data_file.get_frequency_range()
        print(f"Frequency range: {freq_min/1e6:.2f} - {freq_max/1e6:.2f} MHz")
        print(f"Number of points: {len(data['frequency'])}")
        
        # Calculate statistics for different metrics
        for metric in ['epsilon_prime', 'epsilon_double_prime', 'epsilon_magnitude']:
            print(f"\n{metric.replace('_', ' ').title()}:")
            
            stats = calculate_statistics(data, metric)
            
            if stats:
                print(f"  Mean:     {stats.get('mean', np.nan):.4e}")
                print(f"  Std Dev:  {stats.get('std', np.nan):.4e}")
                print(f"  Min/Max:  {stats.get('min', np.nan):.4e} / {stats.get('max', np.nan):.4e}")
                print(f"  Slope:    {stats.get('slope', np.nan):.4e} per GHz")
                print(f"  Integral: {stats.get('integral', np.nan):.4e}")
                
                # Sub-range statistics (if available)
                if 'mean_low_0.5_1GHz' in stats:
                    print(f"\n  Sub-range means:")
                    print(f"    0.5-1 GHz: {stats.get('mean_low_0.5_1GHz', np.nan):.4e}")
                    print(f"    1-2 GHz:   {stats.get('mean_mid_1_2GHz', np.nan):.4e}")
                    print(f"    2-3 GHz:   {stats.get('mean_high_2_3GHz', np.nan):.4e}")
                    print(f"    Ratio L/H: {stats.get('ratio_low_high', np.nan):.4f}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


def compare_two_files(file1, file2, metric='epsilon_double_prime'):
    """
    Example: Compare two S1P files
    
    Args:
        file1: Path to first file
        file2: Path to second file
        metric: Metric to compare (default: epsilon_double_prime)
    """
    from s1p_gui.data_loader import S1PDataFile
    
    print(f"\nComparing {Path(file1).name} vs {Path(file2).name}")
    print(f"Metric: {metric}")
    print("="*60)
    
    # Load files
    df1 = S1PDataFile(Path(file1))
    df2 = S1PDataFile(Path(file2))
    
    if not df1.load() or not df2.load():
        print("Error loading files!")
        return
    
    # Apply same frequency range to both
    df1.apply_frequency_filter(500e6, 3e9)
    df2.apply_frequency_filter(500e6, 3e9)
    
    # Get statistics
    stats1 = calculate_statistics(df1.get_data(), metric)
    stats2 = calculate_statistics(df2.get_data(), metric)
    
    # Compare
    print(f"\n{'Parameter':<25} {df1.name:<20} {df2.name:<20} Difference")
    print("-" * 90)
    
    params = ['mean', 'std', 'slope', 'integral']
    for param in params:
        val1 = stats1.get(param, np.nan)
        val2 = stats2.get(param, np.nan)
        diff = val2 - val1 if not np.isnan(val1) and not np.isnan(val2) else np.nan
        print(f"{param.capitalize():<25} {val1:<20.4e} {val2:<20.4e} {diff:<.4e}")
    
    print("="*90)


if __name__ == '__main__':
    # Example usage
    print("="*60)
    print("S1P GUI - Programmatic Example")
    print("="*60)
    print("\nThis demonstrates using the modules without launching the GUI")
    print("\nTo use this example:")
    print("  1. Replace the file paths below with your actual S1P files")
    print("  2. Run this script: python example_usage.py")
    print("\n" + "="*60)
    
    # Example file paths (replace with your actual files)
    example_files = [
        # "path/to/your/file1.s1p",
        # "path/to/your/file2.s1p",
    ]
    
    if example_files and all(Path(f).exists() for f in example_files):
        analyze_s1p_files(example_files)
        
        if len(example_files) >= 2:
            print("\n\nCOMPARISON ANALYSIS")
            compare_two_files(example_files[0], example_files[1])
    else:
        print("\nNo files to analyze. Please edit this script and add file paths.")
        print("\nTo run the GUI instead, use:")
        print("  python run_s1p_gui.py")
