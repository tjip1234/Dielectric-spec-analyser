"""
Feature extraction from S1P data using polynomial fitting.
Extracts frequency-dependent behavior (slopes/curvature) rather than absolute values.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from .data_loader import parse_s1p_file


def filter_frequency_range(freq_hz: np.ndarray, s11_db: np.ndarray, 
                          phase_deg: np.ndarray, min_freq_hz: float = 100e6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter data to only include frequencies >= min_freq_hz.
    
    Args:
        freq_hz: Frequency array in Hz
        s11_db: S11 magnitude in dB
        phase_deg: S11 phase in degrees
        min_freq_hz: Minimum frequency threshold (default 100 MHz)
        
    Returns:
        Filtered freq_hz, s11_db, phase_deg
    """
    mask = freq_hz >= min_freq_hz
    return freq_hz[mask], s11_db[mask], phase_deg[mask]


def extract_polynomial_coefficients(freq_hz: np.ndarray, s11_db: np.ndarray, 
                                   phase_deg: np.ndarray, degree: int = 6) -> Dict[str, np.ndarray]:
    """
    Fit polynomial to S11 magnitude and phase, return coefficients.
    
    Frequency is normalized to GHz for numerical stability.
    Returns coefficients from highest to lowest degree: [a_n, a_{n-1}, ..., a_1, a_0]
    
    Args:
        freq_hz: Frequency array in Hz
        s11_db: S11 magnitude in dB
        phase_deg: S11 phase in degrees
        degree: Polynomial degree (default 6)
        
    Returns:
        Dictionary with 'db_coeffs' and 'phase_coeffs' arrays
    """
    if len(freq_hz) < degree + 1:
        raise ValueError(f"Not enough data points ({len(freq_hz)}) for degree {degree} polynomial")
    
    # Normalize frequency to GHz for numerical stability
    freq_ghz = freq_hz / 1e9
    
    # Fit polynomials
    db_coeffs = np.polyfit(freq_ghz, s11_db, degree)
    phase_coeffs = np.polyfit(freq_ghz, phase_deg, degree)
    
    return {
        'db_coeffs': db_coeffs,
        'phase_coeffs': phase_coeffs
    }


def apply_solvent_correction(compound_coeffs: Dict[str, np.ndarray], 
                            solvent_coeffs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Subtract solvent polynomial coefficients from compound coefficients.
    
    This isolates the compound's intrinsic dielectric behavior.
    
    Args:
        compound_coeffs: Polynomial coefficients for compound in solution
        solvent_coeffs: Polynomial coefficients for pure solvent
        
    Returns:
        Corrected coefficients
    """
    return {
        'db_coeffs': compound_coeffs['db_coeffs'] - solvent_coeffs['db_coeffs'],
        'phase_coeffs': compound_coeffs['phase_coeffs'] - solvent_coeffs['phase_coeffs']
    }


def extract_features_from_s1p(s1p_path: Path, solvent_s1p_path: Optional[Path] = None,
                              min_freq_hz: float = 100e6, poly_degree: int = 6) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract polynomial features from an S1P file.
    
    Args:
        s1p_path: Path to S1P file
        solvent_s1p_path: Path to pure solvent S1P file (for solvent correction)
        min_freq_hz: Minimum frequency threshold (default 100 MHz)
        poly_degree: Polynomial degree (default 6)
        
    Returns:
        Dictionary with 'db_coeffs' and 'phase_coeffs', or None if extraction fails
    """
    try:
        # Load and filter S1P data
        freq_hz, s11_db, phase_deg = parse_s1p_file(s1p_path)
        freq_hz, s11_db, phase_deg = filter_frequency_range(freq_hz, s11_db, phase_deg, min_freq_hz)
        
        if len(freq_hz) < poly_degree + 1:
            print(f"Warning: Not enough data points in {s1p_path.name} after filtering")
            return None
        
        # Extract polynomial coefficients
        coeffs = extract_polynomial_coefficients(freq_hz, s11_db, phase_deg, poly_degree)
        
        # Apply solvent correction if provided
        if solvent_s1p_path is not None:
            try:
                solv_freq, solv_db, solv_phase = parse_s1p_file(solvent_s1p_path)
                solv_freq, solv_db, solv_phase = filter_frequency_range(solv_freq, solv_db, solv_phase, min_freq_hz)
                
                if len(solv_freq) >= poly_degree + 1:
                    solvent_coeffs = extract_polynomial_coefficients(solv_freq, solv_db, solv_phase, poly_degree)
                    coeffs = apply_solvent_correction(coeffs, solvent_coeffs)
            except Exception as e:
                print(f"Warning: Failed to apply solvent correction: {e}")
        
        return coeffs
    
    except Exception as e:
        print(f"Error extracting features from {s1p_path}: {e}")
        return None


def create_feature_names(poly_degree: int = 6) -> list:
    """
    Create human-readable feature names for polynomial coefficients.
    
    Args:
        poly_degree: Polynomial degree
        
    Returns:
        List of feature names
    """
    feature_names = []
    
    # DB coefficients (highest to lowest degree)
    for i in range(poly_degree, -1, -1):
        feature_names.append(f'db_coeff_{i}')
    
    # Phase coefficients (highest to lowest degree)
    for i in range(poly_degree, -1, -1):
        feature_names.append(f'phase_coeff_{i}')
    
    return feature_names
