"""
Dual-Reference Calibration (Water + Isopropanol) for Acetone (Propan-2-one) Analysis

Method:
1. Load Reference Data:
   - Isopropanol (NPL 15°C)
   - Water (Literature 20°C)
2. Load S1P Measurements:
   - Isopropanol
   - Water
   - Acetone (Propan-2-one)
3. Create Dual Calibration Mapping:
   Model: ε = α * Y + β
   Where Y = (1-S11)/(1+S11) (Admittance)
   We solve for complex coefficients α and β at each frequency using the two standards.
4. Apply mapping to Acetone S1P to extract permittivity.
5. Fit Debye model to the result.
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
    return {
        'freq_hz': data[:, 0] * 1e9,
        'freq_ghz': data[:, 0],
        'eps_prime': data[:, 1],
        'eps_double_prime': data[:, 3]
    }

def load_literature_water():
    """Load literature water permittivity data"""
    csv_path = Path("water_permittivity_literature.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")
        
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    return {
        'freq_ghz': data[:, 0],
        'eps_prime_20C': data[:, 3],
        'eps_double_prime_20C': data[:, 4]
    }

def load_s1p_file(filepath):
    """Load S1P file and return frequency, magnitude (linear), phase (rad)"""
    freq_hz, s11_db, phase_deg = parse_s1p_file(filepath)
    return {
        'freq_hz': freq_hz,
        'freq_ghz': freq_hz / 1e9,
        's11_mag': 10 ** (s11_db / 20),
        'phase_rad': np.deg2rad(phase_deg),
        's11_db': s11_db,
        'phase_deg': phase_deg
    }

# =============================================================================
# DUAL CALIBRATION
# =============================================================================

def create_dual_calibration(npl_iso, lit_water, s1p_iso, s1p_water, freq_range=(0.1e9, 3e9)):
    """
    Create calibration using TWO standards: Isopropanol and Water.
    Model: epsilon = alpha * Y + beta
    """
    # Common frequencies (use NPL iso frequencies as base)
    mask = (npl_iso['freq_hz'] >= freq_range[0]) & (npl_iso['freq_hz'] <= freq_range[1])
    freqs = npl_iso['freq_hz'][mask]
    
    # 1. Reference Permittivities at these frequencies
    # Isopropanol
    eps_iso = npl_iso['eps_prime'][mask] - 1j * npl_iso['eps_double_prime'][mask]
    
    # Water (Interpolate literature data)
    # Use 20C data as it's closest to 18C measurements
    eps_water_prime = np.interp(freqs/1e9, lit_water['freq_ghz'], lit_water['eps_prime_20C'])
    eps_water_dprime = np.interp(freqs/1e9, lit_water['freq_ghz'], lit_water['eps_double_prime_20C'])
    eps_water = eps_water_prime - 1j * eps_water_dprime
    
    # 2. Measured Admittances at these frequencies
    # Isopropanol
    s11_iso_mag = np.interp(freqs, s1p_iso['freq_hz'], s1p_iso['s11_mag'])
    phase_iso = np.interp(freqs, s1p_iso['freq_hz'], s1p_iso['phase_rad'])
    y_iso = s11_to_admittance(s11_iso_mag, phase_iso)
    
    # Water
    s11_water_mag = np.interp(freqs, s1p_water['freq_hz'], s1p_water['s11_mag'])
    phase_water = np.interp(freqs, s1p_water['freq_hz'], s1p_water['phase_rad'])
    y_water = s11_to_admittance(s11_water_mag, phase_water)
    
    # 3. Solve for alpha and beta
    # epsilon = alpha * Y + beta
    # eps_iso = alpha * y_iso + beta
    # eps_water = alpha * y_water + beta
    
    # Subtracting: eps_water - eps_iso = alpha * (y_water - y_iso)
    # alpha = (eps_water - eps_iso) / (y_water - y_iso)
    
    denom = y_water - y_iso
    # Avoid division by zero
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    
    alpha = (eps_water - eps_iso) / denom
    beta = eps_water - alpha * y_water
    
    return {
        'freq_hz': freqs,
        'alpha': alpha,
        'beta': beta,
        'y_iso': y_iso,
        'y_water': y_water,
        'eps_iso': eps_iso,
        'eps_water': eps_water
    }

def apply_dual_calibration(calibration, s1p_target):
    """Apply dual calibration to target S1P data"""
    cal_freqs = calibration['freq_hz']
    alpha = calibration['alpha']
    beta = calibration['beta']
    
    # Interpolate target S1P to calibration frequencies
    s11_mag_interp = np.interp(cal_freqs, s1p_target['freq_hz'], s1p_target['s11_mag'])
    phase_rad_interp = np.interp(cal_freqs, s1p_target['freq_hz'], s1p_target['phase_rad'])
    
    # Calculate Admittance Y for Target
    y_target = s11_to_admittance(s11_mag_interp, phase_rad_interp)
    
    # Apply Calibration: epsilon = alpha * Y + beta
    eps_complex = alpha * y_target + beta
    
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
        
        # Weighted residual
        res_p = np.sum(((eps_prime - ep) / np.mean(eps_prime))**2)
        res_dp = np.sum(((eps_double_prime - edp) / (np.mean(eps_double_prime)+0.1))**2)
        return res_p + res_dp

    # Initial guesses for Acetone
    # Expected: eps_s ~ 20-21, eps_inf ~ 1.8-2.0, tau ~ 3-4 ps
    x0 = [20.0, 1.9, 3e-12] # eps_s, eps_inf, tau
    bounds = [(5, 40), (1, 10), (1e-13, 1e-10)]
    
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

def plot_results(calibration, target_result, target_fit, output_path):
    """Plot the results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Calibration Standards (Admittance Plane)
    ax = axes[0, 0]
    ax.plot(np.real(calibration['y_iso']), np.imag(calibration['y_iso']), 'b-', label='Iso (Ref)')
    ax.plot(np.real(calibration['y_water']), np.imag(calibration['y_water']), 'c-', label='Water (Ref)')
    ax.plot(np.real(target_result['y_target']), np.imag(target_result['y_target']), 'r-', label='Acetone (Target)')
    ax.set_title("Admittance Plane (Y)")
    ax.set_xlabel("Real(Y)")
    ax.set_ylabel("Imag(Y)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Calibration Coefficients
    ax = axes[0, 1]
    freq_ghz = calibration['freq_hz'] / 1e9
    ax.plot(freq_ghz, np.abs(calibration['alpha']), 'g-', label='|α|')
    ax.plot(freq_ghz, np.abs(calibration['beta']), 'm-', label='|β|')
    ax.set_title("Calibration Coeffs (ε = αY + β)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Extracted Permittivity & Fit
    ax = axes[1, 0]
    freq_smooth = np.linspace(target_result['freq_hz'].min(), target_result['freq_hz'].max(), 100)
    ep_fit, edp_fit = debye_model(freq_smooth, target_fit['eps_s'], target_fit['eps_inf'], target_fit['tau'])
    
    ax.plot(target_result['freq_ghz'], target_result['eps_prime'], 'bo', alpha=0.5, label="Data")
    ax.plot(freq_smooth/1e9, ep_fit, 'k--', linewidth=2, label="Debye Fit")
    
    ax.set_title(f"Acetone ε' Fit (εs={target_fit['eps_s']:.2f})")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Real Permittivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Imaginary Part
    ax = axes[1, 1]
    ax.plot(target_result['freq_ghz'], target_result['eps_double_prime'], 'ro', alpha=0.5, label="Data")
    ax.plot(freq_smooth/1e9, edp_fit, 'k--', linewidth=2, label="Debye Fit")
    
    ax.set_title(f"Acetone ε'' Fit (τ={target_fit['tau']*1e12:.1f}ps)")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Imaginary Permittivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Acetone Dual-Cal Analysis (Water+Iso)\nR²={target_fit['r2']:.4f}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("ACETONE (PROPAN-2-ONE) DUAL-CALIBRATION ANALYSIS")
    print("Using both Water and Isopropanol as references")
    print("="*80)
    
    # 1. Load References
    print("Loading Reference Data...")
    try:
        npl_iso = load_npl_isopropanol()
        lit_water = load_literature_water()
    except Exception as e:
        print(f"Error loading references: {e}")
        return

    # 2. Load S1P Files
    print("Loading S1P Measurements...")
    try:
        # Isopropanol
        iso_path = Path("Database/Isopropanol/propan-2-ol.s1p")
        if not iso_path.exists(): iso_path = Path("compounds/propan-2-ol.s1p")
        s1p_iso = load_s1p_file(iso_path)
        
        # Water
        water_path = Path("Database/Water/DI-water.s1p")
        if not water_path.exists(): water_path = Path("compounds/DI-water.s1p")
        s1p_water = load_s1p_file(water_path)
        
        # Acetone
        acetone_path = Path("Database/Propan-2-one/propan-2-one.s1p")
        if not acetone_path.exists(): acetone_path = Path("compounds/propan-2-one.s1p")
        s1p_acetone = load_s1p_file(acetone_path)
        
    except Exception as e:
        print(f"Error loading S1P files: {e}")
        return
    
    # 3. Create Dual Calibration
    print("Creating Dual Calibration (Water + Iso)...")
    cal = create_dual_calibration(npl_iso, lit_water, s1p_iso, s1p_water)
    
    # 4. Apply to Acetone
    print("Applying Calibration to Acetone...")
    acetone_result = apply_dual_calibration(cal, s1p_acetone)
    
    # 5. Fit Debye
    print("Fitting Debye Model...")
    fit = fit_debye_parameters(acetone_result['freq_hz'], 
                               acetone_result['eps_prime'], 
                               acetone_result['eps_double_prime'])
    
    print("\nRESULTS:")
    print(f"Static Permittivity (εs): {fit['eps_s']:.2f} (Expected ~20-21)")
    print(f"High-Freq Permittivity (ε∞): {fit['eps_inf']:.2f} (Expected ~1.8-2.0)")
    print(f"Relaxation Time (τ): {fit['tau']*1e12:.2f} ps (Expected ~3-4 ps)")
    print(f"Relaxation Freq: {fit['f_relax']/1e9:.2f} GHz")
    print(f"Fit Quality (R²): {fit['r2']:.4f}")
    
    # 6. Plot
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    plot_results(cal, acetone_result, fit, output_dir / "acetone_dual_cal.png")

if __name__ == "__main__":
    main()
