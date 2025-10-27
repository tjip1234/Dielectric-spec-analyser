#!/usr/bin/env python3
"""
Interactive Dielectric Analysis Script for S1P Files
Allows user to select a file and generates plots for different frequency ranges
Also supports S2P files.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import skrf as rf
import glob
from matplotlib.ticker import ScalarFormatter, FuncFormatter

# Physical constants
EPSILON_0 = 8.854e-12  # Permittivity of free space (F/m)

# Frequency ranges for analysis (Hz)
FREQ_RANGES = {
    'low': (10e6, 150e6, '10 MHz - 500 MHz'),
    'mid': (500e6, 3e9, '1 GHz - 3 GHz'),
    'full': (0, np.inf, 'Full Sweep')
}

def find_result_folders():
    """Find all folders that start with 'result' in the current directory"""
    current_dir = Path('.')
    result_folders = [d for d in current_dir.iterdir() 
                     if d.is_dir() and d.name.lower().startswith('result')]
    return sorted(result_folders, key=lambda x: x.name)

def select_folder():
    """Let user select a folder to analyze"""
    folders = find_result_folders()
    
    if not folders:
        print("\nNo folders starting with 'result' found in current directory!")
        print("Looking in current directory instead...")
        return Path('.')
    
    print("\n" + "="*80)
    print("Available Result Folders:")
    print("="*80)
    for idx, folder in enumerate(folders, 1):
        print(f"  {idx}. {folder.name}")
    print(f"  {len(folders)+1}. Current directory (no subfolder)")
    print("="*80)
    
    while True:
        try:
            choice = input(f"\nSelect folder to analyze [1-{len(folders)+1}, default=1]: ").strip()
            if choice == '':
                return folders[0]
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(folders):
                selected = folders[choice_num - 1]
                print(f"  ✓ Selected folder: {selected.name}")
                return selected
            elif choice_num == len(folders) + 1:
                print(f"  ✓ Using current directory")
                return Path('.')
            else:
                print(f"Please enter a number between 1 and {len(folders)+1}")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            return None

def find_s1p_files(folder=None):
    """Find all .s1p files in the specified folder (or current directory)"""
    if folder is None:
        folder = Path('.')
    
    search_path = folder / "*.s1p"
    s1p_files = glob.glob(str(search_path))
    
    if not s1p_files:
        print(f"No .s1p files found in {folder}!")
        return []
    
    # Return just the filenames relative to the folder
    return sorted([Path(f) for f in s1p_files])

def find_s2p_files(folder=None):
    """Find all .s2p files in the specified folder (or current directory)"""
    if folder is None:
        folder = Path('.')
    
    search_path = folder / "*.s2p"
    s2p_files = glob.glob(str(search_path))
    
    # Return just the filenames relative to the folder
    return sorted([Path(f) for f in s2p_files])

def select_files(files, file_type="S1P"):
    """Interactive file selection - allows selecting 1-3 files (first is optional baseline/solvent)"""
    print("\n" + "="*80)
    print(f"Available {file_type} Files:")
    print("="*80)
    for idx, filename in enumerate(files, 1):
        print(f"  {idx}. {filename}")
    print("="*80)
    
    selected = []
    baseline_mode = False
    
    # Ask if user wants to use a baseline/solvent
    while True:
        try:
            choice = input("\nUse a SOLVENT/BASELINE file for comparison? (y/n, default=n): ").strip().lower()
            if choice == '' or choice == 'n':
                baseline_mode = False
                print("  ✓ No baseline - analyzing pure samples only")
                break
            elif choice == 'y':
                baseline_mode = True
                break
            else:
                print("Please enter 'y' or 'n'")
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            return None
    
    # If baseline mode, select baseline file first
    if baseline_mode:
        while True:
            try:
                choice = input("\nSelect SOLVENT/BASELINE file (will be shown as dotted line) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return None
                choice_num = int(choice)
                if 1 <= choice_num <= len(files):
                    selected.append(files[choice_num - 1])
                    print(f"  ✓ Selected baseline: {files[choice_num - 1]}")
                    break
                else:
                    print(f"Please enter a number between 1 and {len(files)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n\nCancelled by user.")
                return None
    
    # Select sample files
    sample_num = 1
    max_samples = 2 if baseline_mode else 3
    
    while sample_num <= max_samples:
        try:
            if sample_num == 1:
                prompt = f"\nSelect SAMPLE {sample_num} file (or press Enter to finish): "
            else:
                prompt = f"\nSelect SAMPLE {sample_num} file (or press Enter to skip): "
            
            choice = input(prompt).strip()
            if choice == '':
                if sample_num == 1 and len(selected) == 0:
                    print("You must select at least one file!")
                    continue
                break
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(files):
                if files[choice_num - 1] in selected:
                    print("This file is already selected. Please select a different file.")
                else:
                    selected.append(files[choice_num - 1])
                    print(f"  ✓ Selected sample {sample_num}: {files[choice_num - 1]}")
                    sample_num += 1
            else:
                print(f"Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("Invalid input. Please enter a number or press Enter to finish.")
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            return None
    
    if len(selected) == 0:
        return None
    
    # Return tuple: (selected_files, baseline_mode)
    return (selected, baseline_mode)

def load_s1p_file(filename):
    """Load S1P file using scikit-rf"""
    # Handle both string and Path objects
    network = rf.Network(str(filename))
    return network

def load_s2p_file(filename):
    """Load S2P file using scikit-rf"""
    # Handle both string and Path objects
    network = rf.Network(str(filename))
    return network

def calculate_dielectric_properties(network, freq_min, freq_max):
    """
    Calculate dielectric properties from S11 measurements using open coaxial probe
    """
    freq = network.f  # Frequency in Hz
    s11 = network.s[:, 0, 0]  # S11 parameters (complex)
    
    # Filter frequency range
    freq_mask = (freq >= freq_min) & (freq <= freq_max)
    freq = freq[freq_mask]
    s11 = s11[freq_mask]
    
    if len(freq) == 0:
        return None
    
    # Calculate reflection coefficient
    gamma = s11
    
    # For open coaxial probe measurement:
    # Simplified model: epsilon_r = ((1+gamma)/(1-gamma))^2
    epsilon_complex = ((1 + gamma) / (1 - gamma)) ** 2
    
    epsilon_prime = np.real(epsilon_complex)
    epsilon_double_prime = np.imag(epsilon_complex)
    
    # Complex permittivity magnitude
    epsilon_magnitude = np.abs(epsilon_complex)
    
    # S11 properties
    s11_mag = np.abs(s11)
    s11_phase = np.angle(s11, deg=True)
    s11_db = 20 * np.log10(np.abs(s11))
    
    return {
        'frequency': freq,
        's11': s11,
        's11_mag': s11_mag,
        's11_phase': s11_phase,
        's11_db': s11_db,
        'epsilon_prime': epsilon_prime,
        'epsilon_double_prime': epsilon_double_prime,
        'epsilon_magnitude': epsilon_magnitude,
    }

def calculate_s21_properties(network, freq_min, freq_max):
    """
    Calculate S21 transmission properties from S2P measurements
    Uses simplified NRW method for dielectric extraction
    """
    CELL_LENGTH = 2e-3  # 2mm transmission cell
    
    freq = network.f  # Frequency in Hz
    s21 = network.s[:, 1, 0]  # S21 parameters (complex transmission)
    
    # Filter frequency range
    freq_mask = (freq >= freq_min) & (freq <= freq_max)
    freq = freq[freq_mask]
    s21 = s21[freq_mask]
    
    if len(freq) == 0:
        return None
    
    # Calculate transmission properties
    s21_mag = np.abs(s21)
    s21_phase = np.angle(s21, deg=True)
    s21_db = 20 * np.log10(np.abs(s21))
    
    # Simplified Nicholson-Ross-Weir (NRW) method approximation
    c = 299792458  # Speed of light (m/s)
    
    # Attenuation (Np/m)
    alpha = -np.log(s21_mag) / CELL_LENGTH if CELL_LENGTH > 0 else np.zeros_like(s21_mag)
    
    # Phase constant (rad/m)
    beta = -np.angle(s21) / CELL_LENGTH if CELL_LENGTH > 0 else np.zeros_like(s21_mag)
    
    # Complex propagation constant
    gamma = alpha + 1j * beta
    
    # Simplified permittivity estimation: epsilon_r = (c * gamma / (2*pi*f*j))^2
    omega = 2 * np.pi * freq
    epsilon_complex = (c * gamma / (1j * omega)) ** 2
    
    # Handle numerical issues
    epsilon_complex = np.where(np.isfinite(epsilon_complex), epsilon_complex, 0+0j)
    
    epsilon_prime = np.real(epsilon_complex)
    epsilon_double_prime = np.abs(np.imag(epsilon_complex))
    
    # Filter outliers using percentile-based method
    def remove_outliers(data, percentile=90):
        threshold = np.percentile(data[data > 0], percentile) if np.any(data > 0) else np.inf
        return np.where(data > threshold, np.nan, data)
    
    epsilon_prime = remove_outliers(epsilon_prime, percentile=90)
    epsilon_double_prime = remove_outliers(epsilon_double_prime, percentile=90)
    
    # Apply physical upper limits
    MAX_EPSILON_PRIME = 1000
    MAX_EPSILON_LOSS = 10000
    
    epsilon_prime = np.where(epsilon_prime > MAX_EPSILON_PRIME, np.nan, epsilon_prime)
    epsilon_double_prime = np.where(epsilon_double_prime > MAX_EPSILON_LOSS, np.nan, epsilon_double_prime)
    
    # Complex permittivity magnitude
    epsilon_magnitude = np.sqrt(np.nan_to_num(epsilon_prime)**2 + np.nan_to_num(epsilon_double_prime)**2)
    epsilon_magnitude = np.where(np.isfinite(epsilon_magnitude) & (epsilon_magnitude > 0), 
                                 epsilon_magnitude, np.nan)
    
    return {
        'frequency': freq,
        's21': s21,
        's21_mag': s21_mag,
        's21_phase': s21_phase,
        's21_db': s21_db,
        'epsilon_prime': epsilon_prime,
        'epsilon_double_prime': epsilon_double_prime,
        'epsilon_magnitude': epsilon_magnitude,
    }

def calculate_mid_range_features(data, metric='epsilon_double_prime'):
    """
    Calculate 6 features for mid-range frequency analysis (0.5-3 GHz)
    
    Args:
        data: Data dictionary containing 'frequency' and dielectric properties
        metric: Which property to analyze ('epsilon_prime', 'epsilon_double_prime', or 'epsilon_magnitude')
    
    Returns:
        Dictionary with 6 features
    """
    if data is None:
        return None
    
    freq = data['frequency']
    values = data[metric]
    
    # Filter valid (non-NaN) data
    valid_mask = np.isfinite(values)
    freq_valid = freq[valid_mask]
    values_valid = values[valid_mask]
    
    if len(freq_valid) == 0:
        return None
    
    # Define sub-ranges (Hz)
    low_range = (0.5e9, 1.0e9)   # 0.5-1 GHz
    mid_range = (1.0e9, 2.0e9)   # 1-2 GHz
    high_range = (2.0e9, 3.0e9)  # 2-3 GHz
    full_range = (0.5e9, 3.0e9)  # 0.5-3 GHz
    
    def get_range_data(freq_arr, val_arr, f_min, f_max):
        """Extract data within frequency range"""
        mask = (freq_arr >= f_min) & (freq_arr <= f_max)
        return freq_arr[mask], val_arr[mask]
    
    # Extract sub-range data
    freq_low, val_low = get_range_data(freq_valid, values_valid, *low_range)
    freq_mid, val_mid = get_range_data(freq_valid, values_valid, *mid_range)
    freq_high, val_high = get_range_data(freq_valid, values_valid, *high_range)
    freq_full, val_full = get_range_data(freq_valid, values_valid, *full_range)
    
    # Calculate features
    features = {}
    
    # 1. Mean low frequency (0.5-1 GHz)
    features['mean_low_freq'] = np.mean(val_low) if len(val_low) > 0 else np.nan
    
    # 2. Mean mid frequency (1-2 GHz)
    features['mean_mid_freq'] = np.mean(val_mid) if len(val_mid) > 0 else np.nan
    
    # 3. Mean high frequency (2-3 GHz)
    features['mean_high_freq'] = np.mean(val_high) if len(val_high) > 0 else np.nan
    
    # 4. Slope (linear fit over full range 0.5-3 GHz)
    if len(freq_full) > 1:
        # Use polyfit for linear regression, convert frequency to GHz for better units
        freq_ghz = freq_full / 1e9
        coeffs = np.polyfit(freq_ghz, val_full, 1)
        features['slope'] = coeffs[0]  # Slope in units per GHz
        features['slope_per_hz'] = coeffs[0] / 1e9  # Also keep per Hz for backwards compatibility
    else:
        features['slope'] = np.nan
        features['slope_per_hz'] = np.nan
    
    # 5. Total area (integrate 0.5-3 GHz using trapezoidal rule)
    if len(freq_full) > 1:
        features['total_area'] = np.trapz(val_full, freq_full)
    else:
        features['total_area'] = np.nan
    
    # 6. Ratio low/high
    if features['mean_high_freq'] != 0 and not np.isnan(features['mean_high_freq']):
        features['ratio_low_high'] = features['mean_low_freq'] / features['mean_high_freq']
    else:
        features['ratio_low_high'] = np.nan
    
    return features

def get_slope_label(data, metric='epsilon_double_prime'):
    """
    Get slope value formatted as a string for plot labels
    
    Args:
        data: Data dictionary
        metric: Which property to analyze
    
    Returns:
        Formatted string like "slope: 1.23e-3 /GHz" or empty string if not mid-range
    """
    if data is None:
        return ""
    
    freq = data['frequency']
    freq_min_ghz = freq[0] / 1e9
    freq_max_ghz = freq[-1] / 1e9
    
    # Only calculate for mid-range data (0.5-3 GHz)
    if not (0.4 <= freq_min_ghz <= 0.6 and 2.5 <= freq_max_ghz <= 3.5):
        return ""
    
    values = data[metric]
    valid_mask = np.isfinite(values)
    freq_valid = freq[valid_mask]
    values_valid = values[valid_mask]
    
    if len(freq_valid) < 2:
        return ""
    
    # Get data in 0.5-3 GHz range and fit
    mask = (freq_valid >= 0.5e9) & (freq_valid <= 3.0e9)
    freq_range = freq_valid[mask]
    val_range = values_valid[mask]
    
    if len(freq_range) < 2:
        return ""
    
    # Linear fit with frequency in GHz
    freq_ghz = freq_range / 1e9
    coeffs = np.polyfit(freq_ghz, val_range, 1)
    slope = coeffs[0]
    
    return f"slope: {slope:.2e}/GHz"

def save_mid_range_features(data_list, sample_names, output_folder, file_type='s1p'):
    """
    Save mid-range features to a CSV file
    
    Args:
        data_list: List of data dictionaries
        sample_names: List of sample names
        output_folder: Path to output folder
        file_type: 's1p' or 's2p'
    """
    import csv
    
    features_list = []
    
    for data, name in zip(data_list, sample_names):
        if data is None:
            continue
        
        # Check if this is mid-range data (0.5-3 GHz approximately)
        freq_min_ghz = data['frequency'][0] / 1e9
        freq_max_ghz = data['frequency'][-1] / 1e9
        
        if 0.4 <= freq_min_ghz <= 0.6 and 2.5 <= freq_max_ghz <= 3.5:
            features = calculate_mid_range_features(data, 'epsilon_double_prime')
            if features:
                features_dict = {
                    'sample_name': name,
                    'mean_low_freq_0.5_1GHz': features['mean_low_freq'],
                    'mean_mid_freq_1_2GHz': features['mean_mid_freq'],
                    'mean_high_freq_2_3GHz': features['mean_high_freq'],
                    'slope_per_Hz': features['slope'],
                    'total_area_integral': features['total_area'],
                    'ratio_low_high': features['ratio_low_high']
                }
                features_list.append(features_dict)
    
    if features_list:
        csv_filename = output_folder / f'{file_type}_mid_range_features.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['sample_name', 'mean_low_freq_0.5_1GHz', 'mean_mid_freq_1_2GHz', 
                         'mean_high_freq_2_3GHz', 'slope_per_Hz', 'total_area_integral', 
                         'ratio_low_high']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for features_dict in features_list:
                writer.writerow(features_dict)
        
        print(f"  ✓ Saved mid-range features to {csv_filename}")
        return csv_filename
    
    return None

def setup_scientific_plot_style():
    """Set up professional scientific plot style"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.linewidth': 1.2,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
    })

def format_freq_axis(ax, freq):
    """Format frequency axis with appropriate units"""
    max_freq = np.max(freq)
    if max_freq < 1e6:
        # Use kHz
        ax.set_xlabel('Frequency (kHz)', fontweight='bold')
        return freq / 1e3
    elif max_freq < 1e9:
        # Use MHz
        ax.set_xlabel('Frequency (MHz)', fontweight='bold')
        return freq / 1e6
    else:
        # Use GHz
        ax.set_xlabel('Frequency (GHz)', fontweight='bold')
        return freq / 1e9


def compute_auto_regions(freq, metric, n_regions=2, window_frac=0.12, min_points=20):
    """Automatically find n_regions stable fingerprint regions in the data.

    Args:
        freq: frequency array (Hz)
        metric: array to evaluate stability (e.g., s11_db or s21_db)
        n_regions: number of regions to return
        window_frac: fraction of total points used as window length (approx)
        min_points: minimum number of points per window

    Returns: list of (fmin, fmax, label, key)
    """
    freq = np.asarray(freq)
    metric = np.asarray(metric)
    N = len(freq)
    if N < min_points * 2:
        return []

    # Replace NaNs in metric with median to avoid spurious large stds
    if np.any(~np.isfinite(metric)):
        med = np.nanmedian(metric)
        metric = np.where(np.isfinite(metric), metric, med)

    window_pts = max(min_points, int(np.round(N * window_frac)))
    if window_pts >= N:
        window_pts = max(min_points, N // 4)

    # rolling std over windows
    stds = np.full(N, np.nan)
    for i in range(0, N - window_pts + 1):
        w = metric[i:i + window_pts]
        stds[i + window_pts // 2] = np.nanstd(w)

    # Ignore edges (require center away from edges)
    edge_margin = int(window_pts // 2 + 1)
    valid_idx = np.arange(edge_margin, N - edge_margin)
    # sort candidate centers by increasing std (more stable first)
    candidates = [idx for idx in valid_idx if np.isfinite(stds[idx])]
    if not candidates:
        return []
    candidates_sorted = sorted(candidates, key=lambda i: stds[i])

    selected = []
    min_separation = max(int(window_pts * 1.5), int(0.05 * N))
    for c in candidates_sorted:
        # ensure separation from previously selected centers
        if any(abs(c - s) < min_separation for s in selected):
            continue
        selected.append(c)
        if len(selected) >= n_regions:
            break

    regions = []
    for idx, center in enumerate(selected):
        start = max(0, center - window_pts // 2)
        end = min(N, start + window_pts)
        fmin = float(freq[start])
        fmax = float(freq[end - 1])
        label = f"Auto Region {idx+1} ({fmin:.3e} - {fmax:.3e} Hz)"
        key = f"auto{idx+1}"
        regions.append((fmin, fmax, label, key))

    return regions

def create_plots_for_range(data_list, range_name, sample_names, baseline_mode=True):
    """Create all plots for a specific frequency range
    
    Args:
        data_list: List of data dictionaries (1 or 2 samples)
        range_name: Name of frequency range
        sample_names: List of sample names
        baseline_mode: Whether first sample is a baseline (enables normalized plots)
    """
    
    # Filter out None data
    valid_data = [(data, name, idx) for idx, (data, name) in enumerate(zip(data_list, sample_names)) if data is not None]
    
    if not valid_data:
        print(f"  No data in {range_name} range, skipping...")
        return []
    
    # Scientific color palette (professional and distinguishable)
    colors = ['#666666', '#d62728', '#2ca02c', '#ff7f0e']  # Gray (baseline), Red, Green, Orange
    comparison_mode = len(valid_data) > 1
    
    figures = []
    output_prefix = '_'.join(sample_names) if comparison_mode else sample_names[0]
    title_suffix = f"{' + '.join(sample_names)}" if comparison_mode else sample_names[0]
    
    # 0. S11 Overview (Magnitude and Phase) - Log Frequency Scale
    fig0, (ax0_mag, ax0_phase) = plt.subplots(2, 1, figsize=(12, 10))
    
    for data, name, orig_idx in valid_data:
        # Use Hz for log scale
        freq_hz = data['frequency']
        linestyle = '--' if orig_idx == 0 else '-'
        linewidth = 2.5 if orig_idx == 0 else 2.0
        label = f"{name} (baseline)" if orig_idx == 0 else name
        
        # Magnitude plot (top)
        ax0_mag.semilogx(freq_hz, data['s11_db'], color=colors[orig_idx], 
                        linewidth=linewidth, linestyle=linestyle, 
                        label=label if comparison_mode else None)
        
        # Phase plot (bottom)
        ax0_phase.semilogx(freq_hz, data['s11_phase'], color=colors[orig_idx], 
                          linewidth=linewidth, linestyle=linestyle,
                          label=label if comparison_mode else None)
    
    # Magnitude subplot
    ax0_mag.set_ylabel('|S₁₁| (dB)', fontweight='bold', fontsize=12)
    ax0_mag.set_title(f"S11 Overview - {title_suffix} - {range_name}", fontweight='bold', fontsize=13)
    if comparison_mode:
        ax0_mag.legend(loc='best')
    ax0_mag.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Phase subplot
    ax0_phase.set_xlabel('Frequency (Hz)', fontweight='bold', fontsize=12)
    ax0_phase.set_ylabel('∠S₁₁ (degrees)', fontweight='bold', fontsize=12)
    if comparison_mode:
        ax0_phase.legend(loc='best')
    ax0_phase.grid(True, alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    figures.append((f's1p_{output_prefix}_s11_overview_{range_name}.png', fig0))
    
    # 1. Cole-Cole Plot
    fig1 = plt.figure(figsize=(8, 7))
    ax1 = fig1.add_subplot(111)
    
    for data, name, orig_idx in valid_data:
        valid_mask = np.isfinite(data['epsilon_prime']) & np.isfinite(data['epsilon_double_prime'])
        if np.any(valid_mask):
            # First file (baseline) gets dotted line
            linestyle = '--' if orig_idx == 0 else '-'
            linewidth = 2.0 if orig_idx == 0 else 1.5
            marker = 'o' if orig_idx == 0 else 'o'
            markersize = 3 if orig_idx == 0 else 4
            label = f"{name} (baseline)" if orig_idx == 0 else name
            
            ax1.plot(data['epsilon_prime'][valid_mask], 
                    data['epsilon_double_prime'][valid_mask],
                    marker=marker, linestyle=linestyle, color=colors[orig_idx], 
                    markersize=markersize, alpha=0.7, linewidth=linewidth, 
                    label=label if comparison_mode else None)
    
    ax1.set_xlabel("Real Permittivity (ε')", fontweight='bold')
    ax1.set_ylabel("Imaginary Permittivity (ε'')", fontweight='bold')
    ax1.set_title(f"Cole-Cole Plot\n{title_suffix} - {range_name}", fontweight='bold')
    if comparison_mode:
        ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.ticklabel_format(style='scientific', axis='both', scilimits=(-2,2))
    plt.tight_layout()
    figures.append((f's1p_{output_prefix}_cole_cole_{range_name}.png', fig1))
    
    # 2. Real Permittivity vs Frequency
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    
    use_log_scale = (range_name == 'full')
    
    for data, name, orig_idx in valid_data:
        if use_log_scale:
            freq_to_plot = data['frequency']
        else:
            freq_to_plot = format_freq_axis(ax2, data['frequency'])
        
        valid_mask = np.isfinite(data['epsilon_prime'])
        if np.any(valid_mask):
            linestyle = '-'  # Always solid
            linewidth = 2.0
            
            # Add slope to label if this is mid-range data
            slope_label = get_slope_label(data, 'epsilon_prime')
            if baseline_mode and orig_idx == 0:
                label = f"{name} (baseline)"
            elif slope_label:
                label = f"{name} ({slope_label})"
            else:
                label = name
            
            if use_log_scale:
                ax2.semilogx(freq_to_plot[valid_mask], data['epsilon_prime'][valid_mask], 
                            color=colors[orig_idx], linewidth=linewidth, linestyle=linestyle,
                            label=label if comparison_mode else None)
            else:
                ax2.plot(freq_to_plot[valid_mask], data['epsilon_prime'][valid_mask], 
                        color=colors[orig_idx], linewidth=linewidth, linestyle=linestyle,
                        label=label if comparison_mode else None)
    
    if use_log_scale:
        ax2.set_xlabel('Frequency (Hz)', fontweight='bold')
    
    ax2.set_ylabel("Real Permittivity (ε')", fontweight='bold')
    ax2.set_title(f"Real Part of Permittivity\n{title_suffix} - {range_name}", fontweight='bold')
    if comparison_mode:
        ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--', which='both' if use_log_scale else 'major')
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(-2,2))
    plt.tight_layout()
    figures.append((f's1p_{output_prefix}_epsilon_real_{range_name}.png', fig2))
    
    # 3. Imaginary Permittivity vs Frequency
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    
    for data, name, orig_idx in valid_data:
        if use_log_scale:
            freq_to_plot = data['frequency']
        else:
            freq_to_plot = format_freq_axis(ax3, data['frequency'])
        
        valid_mask = np.isfinite(data['epsilon_double_prime'])
        if np.any(valid_mask):
            linestyle = '-'  # Always solid
            linewidth = 2.0
            
            # Add slope to label if this is mid-range data
            slope_label = get_slope_label(data, 'epsilon_double_prime')
            if baseline_mode and orig_idx == 0:
                label = f"{name} (baseline)"
            elif slope_label:
                label = f"{name} ({slope_label})"
            else:
                label = name
            
            if use_log_scale:
                ax3.semilogx(freq_to_plot[valid_mask], data['epsilon_double_prime'][valid_mask], 
                            color=colors[orig_idx], linewidth=linewidth, linestyle=linestyle,
                            label=label if comparison_mode else None)
            else:
                ax3.plot(freq_to_plot[valid_mask], data['epsilon_double_prime'][valid_mask], 
                        color=colors[orig_idx], linewidth=linewidth, linestyle=linestyle,
                        label=label if comparison_mode else None)
    
    if use_log_scale:
        ax3.set_xlabel('Frequency (Hz)', fontweight='bold')
    
    ax3.set_ylabel("Imaginary Permittivity (ε'')", fontweight='bold')
    ax3.set_title(f"Imaginary Part of Permittivity\n{title_suffix} - {range_name}", fontweight='bold')
    if comparison_mode:
        ax3.legend()
    ax3.grid(True, alpha=0.3, linestyle='--', which='both' if use_log_scale else 'major')
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(-2,2))
    plt.tight_layout()
    figures.append((f's1p_{output_prefix}_epsilon_imag_{range_name}.png', fig3))
    
    # 4. Complex Permittivity Magnitude vs Frequency
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = fig4.add_subplot(111)
    
    for data, name, orig_idx in valid_data:
        if use_log_scale:
            freq_to_plot = data['frequency']
        else:
            freq_to_plot = format_freq_axis(ax4, data['frequency'])
        
        valid_mask = np.isfinite(data['epsilon_magnitude']) & (data['epsilon_magnitude'] > 0)
        if np.any(valid_mask):
            linestyle = '-'  # Always solid
            linewidth = 2.0
            
            # Add slope to label if this is mid-range data
            slope_label = get_slope_label(data, 'epsilon_magnitude')
            if baseline_mode and orig_idx == 0:
                label = f"{name} (baseline)"
            elif slope_label:
                label = f"{name} ({slope_label})"
            else:
                label = name
            
            if use_log_scale:
                ax4.loglog(freq_to_plot[valid_mask], data['epsilon_magnitude'][valid_mask], 
                          color=colors[orig_idx], linewidth=linewidth, linestyle=linestyle,
                          label=label if comparison_mode else None)
            else:
                ax4.semilogy(freq_to_plot[valid_mask], data['epsilon_magnitude'][valid_mask], 
                            color=colors[orig_idx], linewidth=linewidth, linestyle=linestyle,
                            label=label if comparison_mode else None)
    
    if use_log_scale:
        ax4.set_xlabel('Frequency (Hz)', fontweight='bold')
    
    ax4.set_ylabel('|ε*|', fontweight='bold')
    ax4.set_title(f"Complex Permittivity Magnitude\n{title_suffix} - {range_name}", fontweight='bold')
    if comparison_mode:
        ax4.legend()
    ax4.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.tight_layout()
    figures.append((f's1p_{output_prefix}_epsilon_magnitude_{range_name}.png', fig4))
    
    # 5. Normalized Real Permittivity vs Frequency (Baseline-subtracted)
    if baseline_mode and comparison_mode and len(valid_data) > 1:
        fig5 = plt.figure(figsize=(10, 6))
        ax5 = fig5.add_subplot(111)
        
        # Get baseline data (first file)
        baseline_data = valid_data[0][0]
        baseline_freq = baseline_data['frequency']
        baseline_eps_prime = baseline_data['epsilon_prime']
        
        # Plot zero line for baseline
        if use_log_scale:
            freq_for_baseline = baseline_freq
        else:
            freq_for_baseline = format_freq_axis(ax5, baseline_freq)
        ax5.axhline(y=0, color=colors[0], linestyle='--', linewidth=2.5, 
                   label=f"{valid_data[0][1]} (baseline)", alpha=0.7)
        
        # Plot normalized samples (subtract baseline)
        for data, name, orig_idx in valid_data[1:]:
            if use_log_scale:
                freq_to_plot = data['frequency']
            else:
                freq_to_plot = format_freq_axis(ax5, data['frequency'])
            
            # Interpolate baseline to match sample frequencies
            baseline_interp = np.interp(data['frequency'], baseline_freq, baseline_eps_prime)
            normalized_eps_prime = data['epsilon_prime'] - baseline_interp
            
            valid_mask = np.isfinite(normalized_eps_prime)
            if np.any(valid_mask):
                if use_log_scale:
                    ax5.semilogx(freq_to_plot[valid_mask], normalized_eps_prime[valid_mask], 
                                color=colors[orig_idx], linewidth=2.0, linestyle='-',
                                label=f"{name} (Δε')")
                else:
                    ax5.plot(freq_to_plot[valid_mask], normalized_eps_prime[valid_mask], 
                            color=colors[orig_idx], linewidth=2.0, linestyle='-',
                            label=f"{name} (Δε')")
        
        if use_log_scale:
            ax5.set_xlabel('Frequency (Hz)', fontweight='bold')
        
        ax5.set_ylabel("Normalized Real Permittivity (Δε')", fontweight='bold')
        ax5.set_title(f"Baseline-Subtracted Real Permittivity\n{title_suffix} - {range_name}", 
                     fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, linestyle='--', which='both' if use_log_scale else 'major')
        ax5.ticklabel_format(style='scientific', axis='y', scilimits=(-2,2))
        plt.tight_layout()
        figures.append((f's1p_{output_prefix}_epsilon_real_normalized_{range_name}.png', fig5))
    
    # 6. Normalized Imaginary Permittivity vs Frequency (Baseline-subtracted)
    if baseline_mode and comparison_mode and len(valid_data) > 1:
        fig6 = plt.figure(figsize=(10, 6))
        ax6 = fig6.add_subplot(111)
        
        # Get baseline data (first file)
        baseline_data = valid_data[0][0]
        baseline_freq = baseline_data['frequency']
        baseline_eps_imag = baseline_data['epsilon_double_prime']
        
        # Plot zero line for baseline
        ax6.axhline(y=0, color=colors[0], linestyle='--', linewidth=2.5, 
                   label=f"{valid_data[0][1]} (baseline)", alpha=0.7)
        
        # Plot normalized samples (subtract baseline)
        for data, name, orig_idx in valid_data[1:]:
            if use_log_scale:
                freq_to_plot = data['frequency']
            else:
                freq_to_plot = format_freq_axis(ax6, data['frequency'])
            
            # Interpolate baseline to match sample frequencies
            baseline_interp = np.interp(data['frequency'], baseline_freq, baseline_eps_imag)
            normalized_eps_imag = data['epsilon_double_prime'] - baseline_interp
            
            valid_mask = np.isfinite(normalized_eps_imag)
            if np.any(valid_mask):
                if use_log_scale:
                    ax6.semilogx(freq_to_plot[valid_mask], normalized_eps_imag[valid_mask], 
                                color=colors[orig_idx], linewidth=2.0, linestyle='-',
                                label=f"{name} (Δε'')")
                else:
                    ax6.plot(freq_to_plot[valid_mask], normalized_eps_imag[valid_mask], 
                            color=colors[orig_idx], linewidth=2.0, linestyle='-',
                            label=f"{name} (Δε'')")
        
        if use_log_scale:
            ax6.set_xlabel('Frequency (Hz)', fontweight='bold')
        
        ax6.set_ylabel("Normalized Imaginary Permittivity (Δε'')", fontweight='bold')
        ax6.set_title(f"Baseline-Subtracted Imaginary Permittivity\n{title_suffix} - {range_name}", 
                     fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, linestyle='--', which='both' if use_log_scale else 'major')
        ax6.ticklabel_format(style='scientific', axis='y', scilimits=(-2,2))
        plt.tight_layout()
        figures.append((f's1p_{output_prefix}_epsilon_imag_normalized_{range_name}.png', fig6))
    
    # 7. Normalized Complex Permittivity Magnitude vs Frequency (Baseline-subtracted)
    if baseline_mode and comparison_mode and len(valid_data) > 1:
        fig7 = plt.figure(figsize=(10, 6))
        ax7 = fig7.add_subplot(111)
        
        # Get baseline data (first file)
        baseline_data = valid_data[0][0]
        baseline_freq = baseline_data['frequency']
        baseline_eps_mag = baseline_data['epsilon_magnitude']
        
        # Plot zero line for baseline
        ax7.axhline(y=0, color=colors[0], linestyle='--', linewidth=2.5, 
                   label=f"{valid_data[0][1]} (baseline)", alpha=0.7)
        
        # Plot normalized samples (subtract baseline)
        for data, name, orig_idx in valid_data[1:]:
            if use_log_scale:
                freq_to_plot = data['frequency']
            else:
                freq_to_plot = format_freq_axis(ax7, data['frequency'])
            
            # Interpolate baseline to match sample frequencies
            baseline_interp = np.interp(data['frequency'], baseline_freq, baseline_eps_mag)
            normalized_eps_mag = data['epsilon_magnitude'] - baseline_interp
            
            valid_mask = np.isfinite(normalized_eps_mag)
            if np.any(valid_mask):
                if use_log_scale:
                    ax7.semilogx(freq_to_plot[valid_mask], normalized_eps_mag[valid_mask], 
                                color=colors[orig_idx], linewidth=2.0, linestyle='-',
                                label=f"{name} (Δ|ε*|)")
                else:
                    ax7.plot(freq_to_plot[valid_mask], normalized_eps_mag[valid_mask], 
                            color=colors[orig_idx], linewidth=2.0, linestyle='-',
                            label=f"{name} (Δ|ε*|)")
        
        if use_log_scale:
            ax7.set_xlabel('Frequency (Hz)', fontweight='bold')
        
        ax7.set_ylabel("Normalized Complex Permittivity Magnitude (Δ|ε*|)", fontweight='bold')
        ax7.set_title(f"Baseline-Subtracted Complex Permittivity Magnitude\n{title_suffix} - {range_name}", 
                     fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3, linestyle='--', which='both' if use_log_scale else 'major')
        ax7.ticklabel_format(style='scientific', axis='y', scilimits=(-2,2))
        plt.tight_layout()
        figures.append((f's1p_{output_prefix}_epsilon_magnitude_normalized_{range_name}.png', fig7))
        ax7.grid(True, alpha=0.3, linestyle='--')
        ax7.ticklabel_format(style='scientific', axis='y', scilimits=(-2,2))
        plt.tight_layout()
        figures.append((f's1p_{output_prefix}_epsilon_magnitude_normalized_{range_name}.png', fig7))
    
    return figures

def print_statistics(data, range_name):
    """Print statistics for the data"""
    if data is None:
        return
    
    print(f"\n  {range_name}:")
    print(f"    Frequency range: {data['frequency'][0]/1e6:.2f} - {data['frequency'][-1]/1e6:.2f} MHz")
    print(f"    Number of points: {len(data['frequency'])}")
    
    valid_prime = data['epsilon_prime'][np.isfinite(data['epsilon_prime'])]
    valid_imag = data['epsilon_double_prime'][np.isfinite(data['epsilon_double_prime'])]
    
    if len(valid_prime) > 0:
        print(f"    ε' range: {np.min(valid_prime):.3e} to {np.max(valid_prime):.3e}")
    if len(valid_imag) > 0:
        print(f"    ε'' range: {np.min(valid_imag):.3e} to {np.max(valid_imag):.3e}")
    print(f"    S11 (dB) range: {np.min(data['s11_db']):.2f} to {np.max(data['s11_db']):.2f} dB")
    
    # Calculate and print mid-range features if this is the mid range (0.5-3 GHz)
    freq_min_ghz = data['frequency'][0] / 1e9
    freq_max_ghz = data['frequency'][-1] / 1e9
    if 0.4 <= freq_min_ghz <= 0.6 and 2.5 <= freq_max_ghz <= 3.5:
        print(f"\n    MID-RANGE FEATURES (0.5-3 GHz):")
        features = calculate_mid_range_features(data, 'epsilon_double_prime')
        if features:
            print(f"      Mean ε'' (0.5-1 GHz):   {features['mean_low_freq']:.3e}")
            print(f"      Mean ε'' (1-2 GHz):     {features['mean_mid_freq']:.3e}")
            print(f"      Mean ε'' (2-3 GHz):     {features['mean_high_freq']:.3e}")
            print(f"      Slope (ε'' vs freq):    {features['slope']:.3e} /GHz")
            print(f"      Total area (integral):  {features['total_area']:.3e}")
            print(f"      Ratio (low/high):       {features['ratio_low_high']:.3f}")

def print_statistics_s21(data, range_name):
    """Print statistics for S21 data"""
    if data is None:
        return
    
    print(f"\n  {range_name}:")
    print(f"    Frequency range: {data['frequency'][0]/1e6:.2f} - {data['frequency'][-1]/1e6:.2f} MHz")
    print(f"    Number of points: {len(data['frequency'])}")
    
    valid_prime = data['epsilon_prime'][np.isfinite(data['epsilon_prime'])]
    valid_imag = data['epsilon_double_prime'][np.isfinite(data['epsilon_double_prime'])]
    
    if len(valid_prime) > 0:
        print(f"    ε' range: {np.min(valid_prime):.3e} to {np.max(valid_prime):.3e}")
    if len(valid_imag) > 0:
        print(f"    ε'' range: {np.min(valid_imag):.3e} to {np.max(valid_imag):.3e}")
    print(f"    S21 (dB) range: {np.min(data['s21_db']):.2f} to {np.max(data['s21_db']):.2f} dB")
    
    # Calculate and print mid-range features if this is the mid range (0.5-3 GHz)
    freq_min_ghz = data['frequency'][0] / 1e9
    freq_max_ghz = data['frequency'][-1] / 1e9
    if 0.4 <= freq_min_ghz <= 0.6 and 2.5 <= freq_max_ghz <= 3.5:
        print(f"\n    MID-RANGE FEATURES (0.5-3 GHz):")
        features = calculate_mid_range_features(data, 'epsilon_double_prime')
        if features:
            print(f"      Mean ε'' (0.5-1 GHz):   {features['mean_low_freq']:.3e}")
            print(f"      Mean ε'' (1-2 GHz):     {features['mean_mid_freq']:.3e}")
            print(f"      Mean ε'' (2-3 GHz):     {features['mean_high_freq']:.3e}")
            print(f"      Slope (ε'' vs freq):    {features['slope']:.3e} /GHz")
            print(f"      Total area (integral):  {features['total_area']:.3e}")
            print(f"      Ratio (low/high):       {features['ratio_low_high']:.3f}")

def create_s21_plots_for_range(data_list, range_name, sample_names, baseline_mode=True):
    """Create S21 plots for a specific frequency range
    
    Args:
        data_list: List of S21 data dictionaries
        range_name: Name of frequency range
        sample_names: List of sample names
        baseline_mode: Whether first sample is a baseline (enables normalized plots)
    """
    
    # Filter out None data
    valid_data = [(data, name, idx) for idx, (data, name) in enumerate(zip(data_list, sample_names)) if data is not None]
    
    if not valid_data:
        print(f"  No data in {range_name} range, skipping...")
        return []
    
    # Scientific color palette
    colors = ['#666666', '#d62728', '#2ca02c', '#ff7f0e']
    comparison_mode = len(valid_data) > 1
    
    figures = []
    output_prefix = '_'.join(sample_names) if comparison_mode else sample_names[0]
    title_suffix = f"{' + '.join(sample_names)}" if comparison_mode else sample_names[0]
    
    use_log_scale = (range_name == 'full')
    
    # 1. S21 Overview (Magnitude and Phase) - Log Frequency Scale
    fig1, (ax1_mag, ax1_phase) = plt.subplots(2, 1, figsize=(12, 10))
    
    for data, name, orig_idx in valid_data:
        freq_hz = data['frequency']
        linestyle = '--' if orig_idx == 0 else '-'
        linewidth = 2.5 if orig_idx == 0 else 2.0
        label = f"{name} (baseline)" if orig_idx == 0 else name
        
        # Magnitude plot (top)
        ax1_mag.semilogx(freq_hz, data['s21_db'], color=colors[orig_idx], 
                        linewidth=linewidth, linestyle=linestyle, 
                        label=label if comparison_mode else None)
        
        # Phase plot (bottom)
        ax1_phase.semilogx(freq_hz, data['s21_phase'], color=colors[orig_idx], 
                          linewidth=linewidth, linestyle=linestyle,
                          label=label if comparison_mode else None)
    
    # Magnitude subplot
    ax1_mag.set_ylabel('|S₂₁| (dB)', fontweight='bold', fontsize=12)
    ax1_mag.set_title(f"S21 Overview - {title_suffix} - {range_name}", fontweight='bold', fontsize=13)
    if comparison_mode:
        ax1_mag.legend(loc='best')
    ax1_mag.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Phase subplot
    ax1_phase.set_xlabel('Frequency (Hz)', fontweight='bold', fontsize=12)
    ax1_phase.set_ylabel('∠S₂₁ (degrees)', fontweight='bold', fontsize=12)
    if comparison_mode:
        ax1_phase.legend(loc='best')
    ax1_phase.grid(True, alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    figures.append((f's2p_{output_prefix}_s21_overview_{range_name}.png', fig1))
    
    # 2. Cole-Cole Plot
    fig2 = plt.figure(figsize=(8, 7))
    ax2 = fig2.add_subplot(111)
    
    for data, name, orig_idx in valid_data:
        valid_mask = np.isfinite(data['epsilon_prime']) & np.isfinite(data['epsilon_double_prime'])
        if np.any(valid_mask):
            linestyle = '--' if orig_idx == 0 else '-'
            linewidth = 2.0 if orig_idx == 0 else 1.5
            marker = 'o'
            markersize = 3 if orig_idx == 0 else 4
            label = f"{name} (baseline)" if orig_idx == 0 else name
            
            ax2.plot(data['epsilon_prime'][valid_mask], 
                    data['epsilon_double_prime'][valid_mask],
                    marker=marker, linestyle=linestyle, color=colors[orig_idx], 
                    markersize=markersize, alpha=0.7, linewidth=linewidth, 
                    label=label if comparison_mode else None)
    
    ax2.set_xlabel("Real Permittivity (ε')", fontweight='bold')
    ax2.set_ylabel("Imaginary Permittivity (ε'')", fontweight='bold')
    ax2.set_title(f"Cole-Cole Plot\n{title_suffix} - {range_name}", fontweight='bold')
    if comparison_mode:
        ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.ticklabel_format(style='scientific', axis='both', scilimits=(-2,2))
    plt.tight_layout()
    figures.append((f's2p_{output_prefix}_cole_cole_{range_name}.png', fig2))
    
    # 3. Real Permittivity vs Frequency
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    
    for data, name, orig_idx in valid_data:
        if use_log_scale:
            freq_to_plot = data['frequency']
        else:
            freq_to_plot = format_freq_axis(ax3, data['frequency'])
        
        valid_mask = np.isfinite(data['epsilon_prime'])
        if np.any(valid_mask):
            linestyle = '-'  # Always solid
            linewidth = 2.0
            
            # Add slope to label if this is mid-range data
            slope_label = get_slope_label(data, 'epsilon_prime')
            if baseline_mode and orig_idx == 0:
                label = f"{name} (baseline)"
            elif slope_label:
                label = f"{name} ({slope_label})"
            else:
                label = name
            
            if use_log_scale:
                ax3.semilogx(freq_to_plot[valid_mask], data['epsilon_prime'][valid_mask], 
                            color=colors[orig_idx], linewidth=linewidth, linestyle=linestyle,
                            label=label if comparison_mode else None)
            else:
                ax3.plot(freq_to_plot[valid_mask], data['epsilon_prime'][valid_mask], 
                        color=colors[orig_idx], linewidth=linewidth, linestyle=linestyle,
                        label=label if comparison_mode else None)
    
    if use_log_scale:
        ax3.set_xlabel('Frequency (Hz)', fontweight='bold')
    
    ax3.set_ylabel("Real Permittivity (ε')", fontweight='bold')
    ax3.set_title(f"Real Part of Permittivity\n{title_suffix} - {range_name}", fontweight='bold')
    if comparison_mode:
        ax3.legend()
    ax3.grid(True, alpha=0.3, linestyle='--', which='both' if use_log_scale else 'major')
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(-2,2))
    plt.tight_layout()
    figures.append((f's2p_{output_prefix}_epsilon_real_{range_name}.png', fig3))
    
    # 4. Imaginary Permittivity vs Frequency
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = fig4.add_subplot(111)
    
    for data, name, orig_idx in valid_data:
        if use_log_scale:
            freq_to_plot = data['frequency']
        else:
            freq_to_plot = format_freq_axis(ax4, data['frequency'])
        
        valid_mask = np.isfinite(data['epsilon_double_prime'])
        if np.any(valid_mask):
            linestyle = '-'  # Always solid
            linewidth = 2.0
            
            # Add slope to label if this is mid-range data
            slope_label = get_slope_label(data, 'epsilon_double_prime')
            if baseline_mode and orig_idx == 0:
                label = f"{name} (baseline)"
            elif slope_label:
                label = f"{name} ({slope_label})"
            else:
                label = name
            
            if use_log_scale:
                ax4.semilogx(freq_to_plot[valid_mask], data['epsilon_double_prime'][valid_mask], 
                            color=colors[orig_idx], linewidth=linewidth, linestyle=linestyle,
                            label=label if comparison_mode else None)
            else:
                ax4.plot(freq_to_plot[valid_mask], data['epsilon_double_prime'][valid_mask], 
                        color=colors[orig_idx], linewidth=linewidth, linestyle=linestyle,
                        label=label if comparison_mode else None)
    
    if use_log_scale:
        ax4.set_xlabel('Frequency (Hz)', fontweight='bold')
    
    ax4.set_ylabel("Imaginary Permittivity (ε'')", fontweight='bold')
    ax4.set_title(f"Imaginary Part of Permittivity\n{title_suffix} - {range_name}", fontweight='bold')
    if comparison_mode:
        ax4.legend()
    ax4.grid(True, alpha=0.3, linestyle='--', which='both' if use_log_scale else 'major')
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(-2,2))
    plt.tight_layout()
    figures.append((f's2p_{output_prefix}_epsilon_imag_{range_name}.png', fig4))
    
    # 5. Complex Permittivity Magnitude vs Frequency
    fig5 = plt.figure(figsize=(10, 6))
    ax5 = fig5.add_subplot(111)
    
    for data, name, orig_idx in valid_data:
        if use_log_scale:
            freq_to_plot = data['frequency']
        else:
            freq_to_plot = format_freq_axis(ax5, data['frequency'])
        
        valid_mask = np.isfinite(data['epsilon_magnitude']) & (data['epsilon_magnitude'] > 0)
        if np.any(valid_mask):
            linestyle = '-'  # Always solid
            linewidth = 2.0
            
            # Add slope to label if this is mid-range data
            slope_label = get_slope_label(data, 'epsilon_magnitude')
            if baseline_mode and orig_idx == 0:
                label = f"{name} (baseline)"
            elif slope_label:
                label = f"{name} ({slope_label})"
            else:
                label = name
            
            if use_log_scale:
                ax5.loglog(freq_to_plot[valid_mask], data['epsilon_magnitude'][valid_mask], 
                          color=colors[orig_idx], linewidth=linewidth, linestyle=linestyle,
                          label=label if comparison_mode else None)
            else:
                ax5.semilogy(freq_to_plot[valid_mask], data['epsilon_magnitude'][valid_mask], 
                            color=colors[orig_idx], linewidth=linewidth, linestyle=linestyle,
                            label=label if comparison_mode else None)
    
    if use_log_scale:
        ax5.set_xlabel('Frequency (Hz)', fontweight='bold')
    
    ax5.set_ylabel('|ε*|', fontweight='bold')
    ax5.set_title(f"Complex Permittivity Magnitude\n{title_suffix} - {range_name}", fontweight='bold')
    if comparison_mode:
        ax5.legend()
    ax5.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.tight_layout()
    figures.append((f's2p_{output_prefix}_epsilon_magnitude_{range_name}.png', fig5))
    
    return figures

def run_s1p_workflow():
    """Main S1P analysis function"""
    print("\n" + "="*80)
    print("Interactive S1P Dielectric Analysis Tool")
    print("="*80)
    
    # Set up scientific plot style
    setup_scientific_plot_style()
    
    # First, let user select a folder
    selected_folder = select_folder()
    if selected_folder is None:
        print("\nNo folder selected. Exiting.")
        return
    
    # Find all S1P files in the selected folder
    files = find_s1p_files(selected_folder)
    if not files:
        print(f"No .s1p files found in {selected_folder}!")
        return
    
    # Let user select file(s)
    result = select_files(files, "S1P")
    if result is None:
        print("\nNo file selected. Exiting.")
        return
    
    selected_files, baseline_mode = result
    if len(selected_files) == 0:
        print("\nNo file selected. Exiting.")
        return
    
    sample_names = [Path(f).stem for f in selected_files]
    comparison_mode = len(selected_files) > 1
    
    if baseline_mode and comparison_mode:
        files_str = ' + '.join([f"'{f}'" for f in selected_files])
        names_str = ' + '.join(sample_names)
        print(f"\nAnalyzing: {files_str}")
        print(f"  Baseline (dotted): {sample_names[0]}")
        if len(sample_names) > 1:
            print(f"  Sample 1: {sample_names[1]}")
        if len(sample_names) > 2:
            print(f"  Sample 2: {sample_names[2]}")
        output_folder = Path(f"s1p_analysis_{'_'.join(sample_names)}")
    elif comparison_mode:
        files_str = ' + '.join([f"'{f}'" for f in selected_files])
        print(f"\nAnalyzing multiple samples: {files_str}")
        for idx, name in enumerate(sample_names, 1):
            print(f"  Sample {idx}: {name}")
        output_folder = Path(f"s1p_analysis_{'_'.join(sample_names)}")
    else:
        print(f"\nAnalyzing: {selected_files[0]}")
        print(f"Sample name: {sample_names[0]}")
        output_folder = Path(f"s1p_analysis_{sample_names[0]}")
    
    # Create output folder
    output_folder.mkdir(exist_ok=True)
    print(f"Output folder: {output_folder}")
    
    # Load the network(s)
    networks = []
    for selected_file in selected_files:
        try:
            network = load_s1p_file(selected_file)
            networks.append(network)
            print(f"Loaded {selected_file}: {len(network.f)} frequency points")
        except Exception as e:
            print(f"Error loading file {selected_file}: {e}")
            return
    
    # Process each frequency range
    all_figures = []
    print("\nProcessing frequency ranges...")
    
    # Ask user whether to use predefined ranges or automatic region detection
    mode_choice = ''
    while mode_choice not in ('m', 'a'):
        mode_choice = input("Choose ranges: (m)anual predefined or (a)utomatic detection? [m/a, default=a]: ").strip().lower()
        if mode_choice == '':
            mode_choice = 'a'

    if mode_choice == 'm':
        for range_key, (freq_min, freq_max, range_label) in FREQ_RANGES.items():
            print(f"\n  Processing {range_label}...")
            # Calculate properties for all networks
            data_list = []
            for network in networks:
                data = calculate_dielectric_properties(network, freq_min, freq_max)
                data_list.append(data)
            # Print statistics for each
            for i, (data, name) in enumerate(zip(data_list, sample_names)):
                if data is not None:
                    print(f"\n  {name}:")
                    print_statistics(data, range_label)
            # Create plots
            figures = create_plots_for_range(data_list, range_key, sample_names, baseline_mode)
            all_figures.extend(figures)
            
            # Save mid-range features if this is the 'mid' range
            if range_key == 'mid':
                save_mid_range_features(data_list, sample_names, output_folder, 's1p')
    else:
        # Automatic detection: use S11 dB from first (baseline) if available otherwise aggregate
        ref_network = networks[0]
        full_data = calculate_dielectric_properties(ref_network, 0, np.inf)
        if full_data is None:
            print("Could not compute full sweep for automatic detection. Falling back to manual ranges.")
            for range_key, (freq_min, freq_max, range_label) in FREQ_RANGES.items():
                data_list = [calculate_dielectric_properties(network, freq_min, freq_max) for network in networks]
                figures = create_plots_for_range(data_list, range_key, sample_names, baseline_mode)
                all_figures.extend(figures)
        else:
            freq = full_data['frequency']
            metric = full_data['s11_db']
            regions = compute_auto_regions(freq, metric, n_regions=2, window_frac=0.12, min_points=30)
            # Always include full sweep as well
            regions.append((freq[0], freq[-1], 'Full Sweep', 'full'))
            for (fmin, fmax, label, key) in regions:
                print(f"\n  Processing {label}...")
                data_list = [calculate_dielectric_properties(network, fmin, fmax) for network in networks]
                for i, (data, name) in enumerate(zip(data_list, sample_names)):
                    if data is not None:
                        print(f"\n  {name}:")
                        print_statistics(data, label)
                figures = create_plots_for_range(data_list, key, sample_names, baseline_mode)
                all_figures.extend(figures)
    
    # Save all figures
    print("\n\nSaving figures...")
    for filename, fig in all_figures:
        output_path = output_folder / filename
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved {output_path}")
        plt.close(fig)
    
    print(f"\n✓ Analysis complete! Generated {len(all_figures)} plots in '{output_folder}' folder.")
    print("="*80)

def run_s2p_workflow():
    """Main S2P analysis function"""
    print("\n" + "="*80)
    print("Interactive S2P Transmission Analysis Tool")
    print("="*80)
    
    # Set up scientific plot style
    setup_scientific_plot_style()
    
    # First, let user select a folder
    selected_folder = select_folder()
    if selected_folder is None:
        print("\nNo folder selected. Exiting.")
        return
    
    # Find all S2P files in the selected folder
    files = find_s2p_files(selected_folder)
    if not files:
        print(f"No .s2p files found in {selected_folder}!")
        return
    
    # Let user select file(s)
    result = select_files(files, "S2P")
    if result is None:
        print("\nNo file selected. Exiting.")
        return
    
    selected_files, baseline_mode = result
    if len(selected_files) == 0:
        print("\nNo file selected. Exiting.")
        return
    
    sample_names = [Path(f).stem for f in selected_files]
    comparison_mode = len(selected_files) > 1
    
    if baseline_mode and comparison_mode:
        files_str = ' + '.join([f"'{f}'" for f in selected_files])
        print(f"\nAnalyzing: {files_str}")
        print(f"  Baseline (dotted): {sample_names[0]}")
        if len(sample_names) > 1:
            print(f"  Sample 1: {sample_names[1]}")
        if len(sample_names) > 2:
            print(f"  Sample 2: {sample_names[2]}")
        output_folder = Path(f"s2p_analysis_{'_'.join(sample_names)}")
    elif comparison_mode:
        files_str = ' + '.join([f"'{f}'" for f in selected_files])
        print(f"\nAnalyzing multiple samples: {files_str}")
        for idx, name in enumerate(sample_names, 1):
            print(f"  Sample {idx}: {name}")
        output_folder = Path(f"s2p_analysis_{'_'.join(sample_names)}")
    else:
        print(f"\nAnalyzing: {selected_files[0]}")
        print(f"Sample name: {sample_names[0]}")
        output_folder = Path(f"s2p_analysis_{sample_names[0]}")
    
    # Create output folder
    output_folder.mkdir(exist_ok=True)
    print(f"Output folder: {output_folder}")
    
    # Load the network(s)
    networks = []
    for selected_file in selected_files:
        try:
            network = load_s2p_file(selected_file)
            networks.append(network)
            print(f"Loaded {selected_file}: {len(network.f)} frequency points")
        except Exception as e:
            print(f"Error loading file {selected_file}: {e}")
            return
    
    # Process each frequency range
    all_figures = []
    print("\nProcessing frequency ranges...")
    
    # Ask user whether to use predefined ranges or automatic region detection
    mode_choice = ''
    while mode_choice not in ('m', 'a'):
        mode_choice = input("Choose ranges: (m)anual predefined or (a)utomatic detection? [m/a, default=a]: ").strip().lower()
        if mode_choice == '':
            mode_choice = 'a'

    if mode_choice == 'm':
        for range_key, (freq_min, freq_max, range_label) in FREQ_RANGES.items():
            print(f"\n  Processing {range_label}...")
            # Calculate properties for all networks
            data_list = []
            for network in networks:
                data = calculate_s21_properties(network, freq_min, freq_max)
                data_list.append(data)
            # Print statistics for each
            for i, (data, name) in enumerate(zip(data_list, sample_names)):
                if data is not None:
                    print(f"\n  {name}:")
                    print_statistics_s21(data, range_label)
            # Create plots
            figures = create_s21_plots_for_range(data_list, range_key, sample_names, baseline_mode)
            all_figures.extend(figures)
            
            # Save mid-range features if this is the 'mid' range
            if range_key == 'mid':
                save_mid_range_features(data_list, sample_names, output_folder, 's2p')
    else:
        # Automatic detection using first network's full sweep s21_db
        ref_network = networks[0]
        full_data = calculate_s21_properties(ref_network, 0, np.inf)
        if full_data is None:
            print("Could not compute full sweep for automatic detection. Falling back to manual ranges.")
            for range_key, (freq_min, freq_max, range_label) in FREQ_RANGES.items():
                data_list = [calculate_s21_properties(network, freq_min, freq_max) for network in networks]
                figures = create_s21_plots_for_range(data_list, range_key, sample_names, baseline_mode)
                all_figures.extend(figures)
        else:
            freq = full_data['frequency']
            metric = full_data['s21_db']
            regions = compute_auto_regions(freq, metric, n_regions=2, window_frac=0.12, min_points=30)
            regions.append((freq[0], freq[-1], 'Full Sweep', 'full'))
            for (fmin, fmax, label, key) in regions:
                print(f"\n  Processing {label}...")
                data_list = [calculate_s21_properties(network, fmin, fmax) for network in networks]
                for i, (data, name) in enumerate(zip(data_list, sample_names)):
                    if data is not None:
                        print(f"\n  {name}:")
                        print_statistics_s21(data, label)
                figures = create_s21_plots_for_range(data_list, key, sample_names, baseline_mode)
                all_figures.extend(figures)
    
    # Save all figures
    print("\n\nSaving figures...")
    for filename, fig in all_figures:
        output_path = output_folder / filename
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved {output_path}")
        plt.close(fig)
    
    print(f"\n✓ Analysis complete! Generated {len(all_figures)} plots in '{output_folder}' folder.")
    print("="*80)

def main():
    print("="*80)
    print("Dielectric/Network Analysis Tool")
    print("="*80)
    print("Select measurement type:")
    print("  1. S1P (Reflection, S11)")
    print("  2. S2P (Transmission, S21)")
    
    while True:
        choice = input("\nAnalyze S1P or S2P files? [1/2, default=1]: ").strip()
        if choice == '' or choice == '1':
            mode = 's1p'
            break
        elif choice == '2':
            mode = 's2p'
            break
        else:
            print("Please enter 1 for S1P or 2 for S2P.")
    
    if mode == 's1p':
        run_s1p_workflow()
    else:
        run_s2p_workflow()

if __name__ == "__main__":
    main()
