"""
Calibrate S11->Permittivity using NPL Isopropanol data, then apply to Diethyl Ether.

Method:
1. Load NPL isopropanol reference (epsilon', epsilon'' at each frequency)
2. Load your isopropanol S1P measurement
3. Create calibration mapping: epsilon = Y_measured * CalibrationFactor
   where Y_measured = (1-S11)/(1+S11) (Admittance)
4. Load your Diethyl Ether S1P measurement
5. Apply the same mapping to get Et2O's epsilon', epsilon''
6. Fit Debye model to Et2O's extracted permittivity
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d
from scipy.stats import linregress

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from ml_classifier.data_loader import parse_s1p_file

# =============================================================================
# MODELS
# =============================================================================

def debye_model(f, eps_s, eps_inf, tau):
    """Single Debye relaxation - returns complex permittivity"""
    omega = 2 * np.pi * f
    denom = 1 + (omega * tau)**2
    eps_prime = eps_inf + (eps_s - eps_inf) / denom
    eps_double_prime = (eps_s - eps_inf) * omega * tau / denom
    return eps_prime, eps_double_prime

def s11_to_admittance(s11_mag, phase_rad):
    """Convert S11 (polar) to Admittance Y = (1-Gamma)/(1+Gamma)"""
    gamma = s11_mag * np.exp(1j * phase_rad)
    # Avoid division by zero if gamma is exactly -1
    gamma = np.where(gamma == -1, -0.999999, gamma)
    y = (1 - gamma) / (1 + gamma)
    return y

# =============================================================================
# DATA LOADING
# =============================================================================

def load_npl_isopropanol():
    """Load NPL reference data for isopropanol at 15C"""
    csv_path = Path("isopropanol_permittivity_15C_NPL.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")
        
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    
    freq_ghz = data[:, 0]
    eps_prime = data[:, 1]
    eps_double_prime = data[:, 3]
    
    return {
        'freq_hz': freq_ghz * 1e9,
        'freq_ghz': freq_ghz,
        'eps_prime': eps_prime,
        'eps_double_prime': eps_double_prime
    }

def load_s1p_file(filepath):
    """Load S1P file and return frequency, magnitude (linear), phase (rad)"""
    freq_hz, s11_db, phase_deg = parse_s1p_file(filepath)
    
    # Convert to linear magnitude and radians
    s11_mag = 10 ** (s11_db / 20)
    phase_rad = np.deg2rad(phase_deg)
    
    return {
        'freq_hz': freq_hz,
        'freq_ghz': freq_hz / 1e9,
        's11_mag': s11_mag,
        'phase_rad': phase_rad,
        's11_db': s11_db,
        'phase_deg': phase_deg
    }

# =============================================================================
# CALIBRATION & FITTING
# =============================================================================

def create_calibration_from_isopropanol(npl_data, s1p_iso, freq_range=(0.1e9, 3e9)):
    """
    Create a calibration factor using Isopropanol as a standard.
    Assumption: epsilon_complex = Y_measured * CalibrationFactor
    CalibrationFactor = epsilon_ref / Y_measured_iso
    """
    # Filter NPL data to range
    mask_npl = (npl_data['freq_hz'] >= freq_range[0]) & (npl_data['freq_hz'] <= freq_range[1])
    freqs = npl_data['freq_hz'][mask_npl]
    eps_ref_complex = npl_data['eps_prime'][mask_npl] - 1j * npl_data['eps_double_prime'][mask_npl]
    
    # Interpolate S1P data to NPL frequencies
    s11_mag_interp = np.interp(freqs, s1p_iso['freq_hz'], s1p_iso['s11_mag'])
    phase_rad_interp = np.interp(freqs, s1p_iso['freq_hz'], s1p_iso['phase_rad'])
    
    # Calculate Admittance Y for Isopropanol
    y_iso = s11_to_admittance(s11_mag_interp, phase_rad_interp)
    
    # Calculate Calibration Factor (Complex)
    # C_factor = epsilon / Y
    cal_factor = eps_ref_complex / y_iso
    
    return {
        'freq_hz': freqs,
        'cal_factor': cal_factor,
        'y_iso': y_iso,
        'eps_ref': eps_ref_complex
    }

def apply_calibration(calibration, s1p_target):
    """Apply calibration to target S1P data"""
    cal_freqs = calibration['freq_hz']
    cal_factor = calibration['cal_factor']
    
    # Interpolate target S1P to calibration frequencies
    s11_mag_interp = np.interp(cal_freqs, s1p_target['freq_hz'], s1p_target['s11_mag'])
    phase_rad_interp = np.interp(cal_freqs, s1p_target['freq_hz'], s1p_target['phase_rad'])
    
    # Calculate Admittance Y for Target
    y_target = s11_to_admittance(s11_mag_interp, phase_rad_interp)
    
    # Apply Calibration: epsilon = Y * C_factor
    eps_complex = y_target * cal_factor
    
    return {
        'freq_hz': cal_freqs,
        'freq_ghz': cal_freqs / 1e9,
        'eps_prime': np.real(eps_complex),
        'eps_double_prime': -np.imag(eps_complex), # Convention: eps'' is positive loss
        'y_target': y_target
    }

def fit_debye_parameters(freq_hz, eps_prime, eps_double_prime):
    """Fit Debye model to extracted permittivity"""
    
    def residual(params):
        eps_s, eps_inf, tau = params
        ep, edp = debye_model(freq_hz, eps_s, eps_inf, tau)
        
        # Weighted residual (normalize by magnitude)
        res_p = np.sum(((eps_prime - ep) / np.mean(eps_prime))**2)
        res_dp = np.sum(((eps_double_prime - edp) / (np.mean(eps_double_prime)+0.1))**2)
        return res_p + res_dp

    # Initial guesses for Diethyl Ether (Low permittivity)
    # Expected: eps_s ~ 4.3, eps_inf ~ 1.8
    x0 = [4.3, 1.8, 1e-12] # eps_s, eps_inf, tau
    bounds = [(2, 10), (1, 5), (1e-14, 1e-10)]
    
    result = minimize(residual, x0, bounds=bounds, method='L-BFGS-B')
    
    eps_s, eps_inf, tau = result.x
    
    # Calculate R2
    ep_fit, edp_fit = debye_model(freq_hz, eps_s, eps_inf, tau)
    ss_res = np.sum((eps_prime - ep_fit)**2) + np.sum((eps_double_prime - edp_fit)**2)
    ss_tot = np.sum((eps_prime - np.mean(eps_prime))**2) + np.sum((eps_double_prime - np.mean(eps_double_prime))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'eps_s': eps_s,
        'eps_inf': eps_inf,
        'tau': tau,
        'f_relax': 1/(2*np.pi*tau),
        'r2': r2
    }

# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(npl_iso, calibration, et2o_result, et2o_fit, output_path):
    """Plot the results"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    # 1. Reference Data (Isopropanol)
    ax = axes[0, 0]
    ax.plot(npl_iso['freq_ghz'], npl_iso['eps_prime'], 'b-', label="Iso ε' (Ref)")
    ax.plot(npl_iso['freq_ghz'], npl_iso['eps_double_prime'], 'r-', label="Iso ε'' (Ref)")
    ax.set_title("Reference: Isopropanol (NPL)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Permittivity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Calibration Factor (Mapping)
    ax = axes[0, 1]
    cal_mag = np.abs(calibration['cal_factor'])
    cal_phase = np.angle(calibration['cal_factor'], deg=True)
    
    ax2 = ax.twinx()
    l1 = ax.plot(calibration['freq_hz']/1e9, cal_mag, 'g-', label="|Cal Factor|")
    l2 = ax2.plot(calibration['freq_hz']/1e9, cal_phase, 'm--', label="∠Cal Factor")
    
    ax.set_title("Calibration Mapping (Factor = ε_ref / Y_iso)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude")
    ax2.set_ylabel("Phase (deg)")
    
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. Extracted Et2O Permittivity
    ax = axes[1, 0]
    ax.plot(et2o_result['freq_ghz'], et2o_result['eps_prime'], 'b-', label="Et2O ε' (Extracted)")
    ax.plot(et2o_result['freq_ghz'], et2o_result['eps_double_prime'], 'r-', label="Et2O ε'' (Extracted)")
    
    ax.set_title("Extracted: Diethyl Ether Permittivity")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Permittivity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Admittance Comparison
    ax = axes[1, 1]
    ax.plot(calibration['freq_hz']/1e9, np.abs(calibration['y_iso']), 'k-', label="|Y_iso|")
    ax.plot(et2o_result['freq_ghz'], np.abs(et2o_result['y_target']), 'b--', label="|Y_Et2O|")
    ax.set_title("Admittance Comparison (|Y|)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Admittance Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Debye Fit (Real Part)
    ax = axes[2, 0]
    freq_smooth = np.linspace(et2o_result['freq_hz'].min(), et2o_result['freq_hz'].max(), 100)
    ep_fit, edp_fit = debye_model(freq_smooth, et2o_fit['eps_s'], et2o_fit['eps_inf'], et2o_fit['tau'])
    
    ax.plot(et2o_result['freq_ghz'], et2o_result['eps_prime'], 'bo', alpha=0.5, label="Data")
    ax.plot(freq_smooth/1e9, ep_fit, 'k--', linewidth=2, label="Debye Fit")
    
    ax.set_title(f"Et2O ε' Fit (εs={et2o_fit['eps_s']:.1f})")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Real Permittivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Debye Fit (Imaginary Part)
    ax = axes[2, 1]
    ax.plot(et2o_result['freq_ghz'], et2o_result['eps_double_prime'], 'ro', alpha=0.5, label="Data")
    ax.plot(freq_smooth/1e9, edp_fit, 'k--', linewidth=2, label="Debye Fit")
    
    ax.set_title(f"Et2O ε'' Fit (τ={et2o_fit['tau']*1e12:.1f}ps)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Imaginary Permittivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Diethyl Ether Debye Analysis via Isopropanol Calibration\nR²={et2o_fit['r2']:.4f}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("DIETHYL ETHER DEBYE ANALYSIS (Isopropanol Calibration)")
    print("="*80)
    
    # 1. Load Reference
    print("Loading NPL Isopropanol Reference...")
    try:
        npl_iso = load_npl_isopropanol()
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Load Isopropanol S1P
    print("Loading Isopropanol S1P...")
    iso_path = Path("Database/Isopropanol/propan-2-ol.s1p")
    if not iso_path.exists():
        iso_path = Path("compounds/propan-2-ol.s1p")
    
    if not iso_path.exists():
        print("Error: Could not find isopropanol S1P file")
        return
    s1p_iso = load_s1p_file(iso_path)
    
    # 3. Create Calibration
    print("Creating Calibration Mapping...")
    cal = create_calibration_from_isopropanol(npl_iso, s1p_iso)
    
    # 4. Load Diethyl Ether S1P
    print("Loading Diethyl Ether S1P...")
    et2o_path = Path("Database/Diethylether/diethyl-ether.s1p")
    if not et2o_path.exists():
        et2o_path = Path("compounds/diethyl-ether.s1p")
        
    if not et2o_path.exists():
        print("Error: Could not find Diethyl Ether S1P file")
        return
    s1p_et2o = load_s1p_file(et2o_path)
    
    # 5. Apply Calibration
    print("Applying Calibration to Diethyl Ether...")
    et2o_result = apply_calibration(cal, s1p_et2o)
    
    # 6. Fit Debye
    print("Fitting Debye Model...")
    fit = fit_debye_parameters(et2o_result['freq_hz'], 
                               et2o_result['eps_prime'], 
                               et2o_result['eps_double_prime'])
    
    print("\nRESULTS:")
    print(f"Static Permittivity (εs): {fit['eps_s']:.2f} (Expected ~4.3)")
    print(f"High-Freq Permittivity (ε∞): {fit['eps_inf']:.2f} (Expected ~1.8)")
    print(f"Relaxation Time (τ): {fit['tau']*1e12:.2f} ps")
    print(f"Relaxation Freq: {fit['f_relax']/1e9:.2f} GHz")
    print(f"Fit Quality (R²): {fit['r2']:.4f}")
    
    # 7. Plot
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    plot_results(npl_iso, cal, et2o_result, fit, output_dir / "et2o_debye_calibrated.png")

if __name__ == "__main__":
    main()
