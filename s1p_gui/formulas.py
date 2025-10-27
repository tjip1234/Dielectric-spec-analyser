"""
Dielectric property calculations and analysis formulas
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .constants import EPSILON_0


def calculate_dielectric_properties(s11_complex: np.ndarray, 
                                    frequencies: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate dielectric properties from S11 measurements using open coaxial probe
    
    Args:
        s11_complex: Complex S11 parameters
        frequencies: Frequency array in Hz
        
    Returns:
        Dictionary containing all calculated properties
    """
    # Calculate reflection coefficient
    gamma = s11_complex
    
    # For open coaxial probe measurement:
    # Simplified model: epsilon_r = ((1+gamma)/(1-gamma))^2
    epsilon_complex = ((1 + gamma) / (1 - gamma)) ** 2
    
    epsilon_prime = np.real(epsilon_complex)
    epsilon_double_prime = np.imag(epsilon_complex)
    
    # Complex permittivity magnitude
    epsilon_magnitude = np.abs(epsilon_complex)
    
    # S11 properties
    s11_mag = np.abs(s11_complex)
    s11_phase = np.angle(s11_complex, deg=True)
    # Avoid log10(0) by adding small epsilon
    s11_db = 20 * np.log10(np.maximum(s11_mag, 1e-12))
    
    return {
        'frequency': frequencies,
        's11': s11_complex,
        's11_mag': s11_mag,
        's11_phase': s11_phase,
        's11_db': s11_db,
        'epsilon_prime': epsilon_prime,
        'epsilon_double_prime': epsilon_double_prime,
        'epsilon_magnitude': epsilon_magnitude,
    }


def calculate_slope(frequencies: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    """
    Calculate linear slope using least squares fitting
    
    Args:
        frequencies: Frequency array in Hz
        values: Metric values to fit
        
    Returns:
        (slope, intercept) in units per Hz
    """
    # Filter out NaN values
    valid_mask = np.isfinite(values) & np.isfinite(frequencies)
    if np.sum(valid_mask) < 2:
        return np.nan, np.nan
    
    freq_valid = frequencies[valid_mask]
    val_valid = values[valid_mask]
    
    # Convert to GHz for better numerical stability
    freq_ghz = freq_valid / 1e9
    coeffs = np.polyfit(freq_ghz, val_valid, 1)
    
    return coeffs[0], coeffs[1]  # slope per GHz, intercept

def calculate_log_slope(frequencies: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    """
    Calculate linear slope using least squares fitting
    
    Args:
        frequencies: Frequency array in Hz
        values: Metric values to fit

    Returns:
        (slope, intercept) in units per Hz
    """
    # Filter out NaN values
    valid_mask = np.isfinite(values) & np.isfinite(frequencies)
    if np.sum(valid_mask) < 2:
        return np.nan, np.nan

    freq_valid = frequencies[valid_mask]
    val_valid = values[valid_mask]

    # Convert to GHz for better numerical stability
    freq_ghz = freq_valid / 1e9
    val_log = np.log10(np.maximum(val_valid, 1e-12))  # Avoid log10(0)

    coeffs = np.polyfit(freq_ghz, val_log, 1)

    return coeffs[0], coeffs[1]  # slope per GHz, intercept


def calculate_integral(frequencies: np.ndarray, values: np.ndarray) -> float:
    """
    Calculate area under curve using trapezoidal integration
    
    Args:
        frequencies: Frequency array in Hz
        values: Metric values to integrate
        
    Returns:
        Integrated area
    """
    # Filter out NaN values
    valid_mask = np.isfinite(values) & np.isfinite(frequencies)
    if np.sum(valid_mask) < 2:
        return np.nan
    
    freq_valid = frequencies[valid_mask]
    val_valid = values[valid_mask]
    
    return np.trapz(val_valid, freq_valid)


def calculate_mean_in_range(frequencies: np.ndarray, 
                            values: np.ndarray,
                            freq_min: float,
                            freq_max: float) -> float:
    """
    Calculate mean value in a specific frequency range
    
    Args:
        frequencies: Frequency array in Hz
        values: Metric values
        freq_min: Minimum frequency in Hz
        freq_max: Maximum frequency in Hz
        
    Returns:
        Mean value in range
    """
    mask = (frequencies >= freq_min) & (frequencies <= freq_max)
    mask &= np.isfinite(values)
    
    if np.sum(mask) == 0:
        return np.nan
    
    return np.mean(values[mask])


def calculate_statistics(data: Dict[str, np.ndarray], 
                         metric: str = 'epsilon_double_prime') -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a given metric
    
    Args:
        data: Data dictionary containing frequencies and metrics
        metric: Which metric to analyze
        
    Returns:
        Dictionary with statistical measures
    """
    if data is None or metric not in data:
        return {}
    
    freq = data['frequency']
    values = data[metric]
    
    # Filter valid data
    valid_mask = np.isfinite(values)
    if np.sum(valid_mask) == 0:
        return {}
    
    freq_valid = freq[valid_mask]
    val_valid = values[valid_mask]
    
    # Calculate statistics
    stats = {
        'mean': np.mean(val_valid),
        'median': np.median(val_valid),
        'std': np.std(val_valid),
        'min': np.min(val_valid),
        'max': np.max(val_valid),
        'range': np.max(val_valid) - np.min(val_valid),
    }
    
    # Calculate slope
    slope, intercept = calculate_slope(freq_valid, val_valid)
    stats['slope'] = slope
    stats['slope_per_ghz'] = slope  # Already in per GHz
    
    # Calculate integral
    stats['integral'] = calculate_integral(freq_valid, val_valid)
    
    # Calculate means in sub-ranges (if in 0.5-3 GHz range)
    freq_min_ghz = freq_valid[0] / 1e9
    freq_max_ghz = freq_valid[-1] / 1e9
    
    if 0.4 <= freq_min_ghz <= 0.6 and 2.5 <= freq_max_ghz <= 3.5:
        stats['mean_low_0.5_1GHz'] = calculate_mean_in_range(freq_valid, val_valid, 0.5e9, 1.0e9)
        stats['mean_mid_1_2GHz'] = calculate_mean_in_range(freq_valid, val_valid, 1.0e9, 2.0e9)
        stats['mean_high_2_3GHz'] = calculate_mean_in_range(freq_valid, val_valid, 2.0e9, 3.0e9)
        
        # Calculate ratio
        if stats['mean_high_2_3GHz'] != 0 and not np.isnan(stats['mean_high_2_3GHz']):
            stats['ratio_low_high'] = stats['mean_low_0.5_1GHz'] / stats['mean_high_2_3GHz']
        else:
            stats['ratio_low_high'] = np.nan
    
    return stats


def filter_frequency_range(data: Dict[str, np.ndarray],
                           freq_min: float,
                           freq_max: float) -> Optional[Dict[str, np.ndarray]]:
    """
    Filter data to a specific frequency range
    
    Args:
        data: Data dictionary containing frequencies and metrics
        freq_min: Minimum frequency in Hz
        freq_max: Maximum frequency in Hz
        
    Returns:
        Filtered data dictionary or None if no data in range
    """
    if data is None:
        return None
    
    freq = data['frequency']
    freq_mask = (freq >= freq_min) & (freq <= freq_max)
    
    if not np.any(freq_mask):
        return None
    
    filtered_data = {}
    for key, values in data.items():
        if isinstance(values, np.ndarray):
            filtered_data[key] = values[freq_mask]
        else:
            filtered_data[key] = values
    
    return filtered_data


def auto_detect_stable_regions(frequencies: np.ndarray,
                               metric: np.ndarray,
                               n_regions: int = 2,
                               window_frac: float = 0.12,
                               min_points: int = 20) -> list:
    """
    Automatically detect stable regions in the frequency spectrum
    
    Args:
        frequencies: Frequency array in Hz
        metric: Metric array to analyze for stability
        n_regions: Number of regions to detect
        window_frac: Fraction of total points for window size
        min_points: Minimum points per window
        
    Returns:
        List of (freq_min, freq_max, label) tuples
    """
    freq = np.asarray(frequencies)
    met = np.asarray(metric)
    N = len(freq)
    
    if N < min_points * 2:
        return []
    
    # Replace NaNs with median
    if np.any(~np.isfinite(met)):
        med = np.nanmedian(met)
        met = np.where(np.isfinite(met), met, med)
    
    window_pts = max(min_points, int(np.round(N * window_frac)))
    if window_pts >= N:
        window_pts = max(min_points, N // 4)
    
    # Calculate rolling standard deviation
    stds = np.full(N, np.nan)
    for i in range(0, N - window_pts + 1):
        w = met[i:i + window_pts]
        stds[i + window_pts // 2] = np.nanstd(w)
    
    # Find stable regions (low std)
    edge_margin = int(window_pts // 2 + 1)
    valid_idx = np.arange(edge_margin, N - edge_margin)
    candidates = [idx for idx in valid_idx if np.isfinite(stds[idx])]
    
    if not candidates:
        return []
    
    candidates_sorted = sorted(candidates, key=lambda i: stds[i])
    
    selected = []
    min_separation = max(int(window_pts * 1.5), int(0.05 * N))
    
    for c in candidates_sorted:
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
        label = f"Auto Region {idx+1}"
        regions.append((fmin, fmax, label))
    
    return regions
