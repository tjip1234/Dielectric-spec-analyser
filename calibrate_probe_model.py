"""
Probe Calibration using Aperture Admittance Model (C0, Cf)
Comparison of Dual-Reference vs Single-Reference Calibration.

Model:
Y_meas = j*omega*Cf + j*omega*C0 * epsilon*

Modes:
1. Dual-Reference (Water + Iso): Solves for both C0 and Cf.
2. Single-Reference (Water): Assumes Cf=0, solves for C0.
3. Single-Reference (Iso): Assumes Cf=0, solves for C0.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.optimize import minimize

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from ml_classifier.data_loader import parse_s1p_file

# =============================================================================
# DATA LOADING
# =============================================================================

def load_npl_isopropanol():
    """Load NPL reference data for isopropanol at 15C"""
    csv_path = Path("isopropanol_permittivity_15C_NPL.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")
        
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    return {
        'freq_hz': data[:, 0] * 1e9,
        'eps_complex': data[:, 1] - 1j * data[:, 3]
    }

def load_literature_water():
    """Load literature water permittivity data"""
    csv_path = Path("water_permittivity_literature.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")
        
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    return {
        'freq_hz': data[:, 0] * 1e9,
        'eps_complex': data[:, 3] - 1j * data[:, 4]
    }

def load_s1p_file(filepath):
    """Load S1P file and return frequency, magnitude (linear), phase (rad)"""
    freq_hz, s11_db, phase_deg = parse_s1p_file(filepath)
    return {
        'freq_hz': freq_hz,
        's11_mag': 10 ** (s11_db / 20),
        'phase_rad': np.deg2rad(phase_deg)
    }

def s11_to_admittance(s11_mag, phase_rad, y0=1/50.0):
    """Convert S11 to Admittance Y = (1-Gamma)/(1+Gamma) * Y0"""
    gamma = s11_mag * np.exp(1j * phase_rad)
    # Avoid division by zero
    gamma = np.where(gamma == -1, -0.999999, gamma)
    y = (1 - gamma) / (1 + gamma) * y0
    return y

def interpolate_to_freqs(target_freqs, source_data):
    """Interpolate source data (freq, complex) to target frequencies"""
    # Handle both dictionary with 'eps_complex' and raw S1P dicts
    if 'eps_complex' in source_data:
        val_real = np.interp(target_freqs, source_data['freq_hz'], np.real(source_data['eps_complex']))
        val_imag = np.interp(target_freqs, source_data['freq_hz'], np.imag(source_data['eps_complex']))
        return val_real + 1j * val_imag
    elif 's11_mag' in source_data:
        mag = np.interp(target_freqs, source_data['freq_hz'], source_data['s11_mag'])
        phase = np.interp(target_freqs, source_data['freq_hz'], source_data['phase_rad'])
        return s11_to_admittance(mag, phase)
    return None

# =============================================================================
# CALIBRATION ROUTINES
# =============================================================================

def calibrate_dual(npl_iso, lit_water, s1p_iso, s1p_water, freq_range=(0.1e9, 3e9)):
    """
    Solve for C0 and Cf using Water and Isopropanol standards.
    """
    # Use NPL frequencies as base grid
    mask = (npl_iso['freq_hz'] >= freq_range[0]) & (npl_iso['freq_hz'] <= freq_range[1])
    freqs = npl_iso['freq_hz'][mask]
    omega = 2 * np.pi * freqs
    
    # Interpolate everything to these frequencies
    eps_iso = interpolate_to_freqs(freqs, npl_iso)
    eps_water = interpolate_to_freqs(freqs, lit_water)
    y_iso = interpolate_to_freqs(freqs, s1p_iso)
    y_water = interpolate_to_freqs(freqs, s1p_water)
    
    # Solve system:
    # Y_w = jwCf + jwC0*eps_w
    # Y_i = jwCf + jwC0*eps_i
    # => Y_w - Y_i = jwC0(eps_w - eps_i)
    
    delta_y = y_water - y_iso
    delta_eps = eps_water - eps_iso
    
    c0 = delta_y / (1j * omega * delta_eps)
    cf = (y_water / (1j * omega)) - c0 * eps_water
    
    return {'type': 'Dual (Water+Iso)', 'freq_hz': freqs, 'c0': c0, 'cf': cf}

def calibrate_single(ref_std, s1p_std, name="Single", freq_range=(0.1e9, 3e9)):
    """
    Solve for C0 assuming Cf=0.
    Y = jwC0*eps => C0 = Y / (jw*eps)
    """
    mask = (ref_std['freq_hz'] >= freq_range[0]) & (ref_std['freq_hz'] <= freq_range[1])
    freqs = ref_std['freq_hz'][mask]
    omega = 2 * np.pi * freqs
    
    eps_ref = interpolate_to_freqs(freqs, ref_std)
    y_ref = interpolate_to_freqs(freqs, s1p_std)
    
    c0 = y_ref / (1j * omega * eps_ref)
    cf = np.zeros_like(c0)
    
    return {'type': f'Single ({name})', 'freq_hz': freqs, 'c0': c0, 'cf': cf}

# =============================================================================
# ANALYSIS & PLOTTING
# =============================================================================

def plot_c_comparison(cals, output_path):
    """Plot C0 and Cf for multiple calibrations"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    for cal in cals:
        freq_ghz = cal['freq_hz'] / 1e9
        lbl = cal['type']
        
        # Plot Real(C0) - The main capacitance
        ax1.plot(freq_ghz, np.real(cal['c0'])*1e15, label=f"{lbl} (Real)", linewidth=2)
        # Optional: Plot Imag(C0) to show loss/error
        # ax1.plot(freq_ghz, np.imag(cal['c0'])*1e15, '--', label=f"{lbl} (Imag)", alpha=0.5)

        # Plot Real(Cf)
        ax2.plot(freq_ghz, np.real(cal['cf'])*1e15, label=f"{lbl} (Real)", linewidth=2)
    
    ax1.set_title("Aperture Capacitance C0")
    ax1.set_ylabel("Capacitance (fF)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Fringing Capacitance Cf")
    ax2.set_ylabel("Capacitance (fF)")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved C-value comparison to {output_path}")

def print_c_stats(cal):
    """Print mean values of C0 and Cf"""
    c0_mean = np.mean(cal['c0'])
    cf_mean = np.mean(cal['cf'])
    print(f"[{cal['type']}]")
    print(f"  Mean C0: {np.real(c0_mean)*1e15:.2f} + j{np.imag(c0_mean)*1e15:.2f} fF")
    print(f"  Mean Cf: {np.real(cf_mean)*1e15:.2f} + j{np.imag(cf_mean)*1e15:.2f} fF")
    print("-" * 40)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("PROBE PARAMETER EXTRACTION (C0, Cf)")
    print("="*80)
    
    # 1. Load Data
    try:
        npl_iso = load_npl_isopropanol()
        lit_water = load_literature_water()
        
        s1p_iso = load_s1p_file("Database/Isopropanol/propan-2-ol.s1p")
        s1p_water = load_s1p_file("Database/Water/DI-water.s1p")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. Run Calibrations
    print("\nCalculating Probe Parameters...")
    
    # Dual Reference
    cal_dual = calibrate_dual(npl_iso, lit_water, s1p_iso, s1p_water)
    
    # Single Reference (Water) - Assumes Cf=0
    cal_water = calibrate_single(lit_water, s1p_water, name="Water-Only")
    
    # Single Reference (Iso) - Assumes Cf=0
    cal_iso = calibrate_single(npl_iso, s1p_iso, name="Iso-Only")
    
    # 3. Output Stats
    print("\n--- Extracted Capacitance Values (Mean 0.1-3 GHz) ---")
    print_c_stats(cal_dual)
    print_c_stats(cal_water)
    print_c_stats(cal_iso)
    
    # 4. Plot Comparison
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    plot_c_comparison([cal_dual, cal_water, cal_iso], output_dir / "probe_C_values_comparison.png")

if __name__ == "__main__":
    main()
