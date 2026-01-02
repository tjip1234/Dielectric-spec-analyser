import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from ml_classifier.data_loader import parse_s1p_file

def load_reference_data(csv_path):
    """Load reference permittivity data using numpy"""
    # Check if file exists
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        return None, None, None

    try:
        # Try reading with header
        data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
        
        # Check shape to determine format
        if data.shape[1] >= 4: # Assuming NPL format or similar
            # NPL: Freq_GHz, eps_prime, eps_double_prime (cols 0, 1, 3 usually)
            # But let's look at the file content structure from previous context
            # isopropanol_permittivity_15C_NPL.csv: Freq_GHz, eps_prime_bestfit, ..., eps_double_prime_bestfit
            freq = data[:, 0] * 1e9
            ep = data[:, 1]
            # Check if col 3 is double prime (standard in this workspace)
            edp = data[:, 3] 
            return freq, ep, edp
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None, None, None

def calculate_admittance(s1p_path, freq_target):
    """Load S1P and convert to Admittance Y at target frequencies"""
    if not Path(s1p_path).exists():
        print(f"Error: {s1p_path} not found")
        return None

    freq_meas, s11_db, phase_deg = parse_s1p_file(s1p_path)
    
    # Interpolate S11 to target frequencies
    s11_mag = 10**(s11_db/20)
    s11_complex = s11_mag * np.exp(1j * np.deg2rad(phase_deg))
    
    # Interpolate real and imag parts separately
    s11_real_interp = np.interp(freq_target, freq_meas, np.real(s11_complex))
    s11_imag_interp = np.interp(freq_target, freq_meas, np.imag(s11_complex))
    gamma = s11_real_interp + 1j * s11_imag_interp
    
    # Convert Gamma to Admittance (normalized to Y0)
    # Y = (1 - Gamma) / (1 + Gamma) * Y0
    # We'll work with normalized Y first, Y0 scales C0/Cf later
    # Y_norm = (1 - gamma) / (1 + gamma)
    
    # Handle singularity at gamma=-1
    denom = 1 + gamma
    denom[np.abs(denom) < 1e-10] = 1e-10
    
    Y_norm = (1 - gamma) / denom
    
    return Y_norm

def solve_c_single_std(freq, Y_meas, eps_ref):
    """
    Solve C0, Cf using a single standard with loss.
    Y = j*w*Cf + j*w*C0*(e' - j*e'')
    Y = w*C0*e'' + j*w*(Cf + C0*e')
    
    Real(Y) = w*C0*e''  -> C0 = Real(Y) / (w*e'')
    Imag(Y) = w*Cf + w*C0*e' -> Cf = Imag(Y)/w - C0*e'
    """
    omega = 2 * np.pi * freq
    Y0 = 1/50.0 # 50 ohm system
    Y_abs = Y_meas * Y0
    
    G = np.real(Y_abs)
    B = np.imag(Y_abs)
    
    eps_p = np.real(eps_ref)
    eps_dp = -np.imag(eps_ref) # eps* = e' - j e''
    
    # Avoid division by zero
    mask = eps_dp > 0.1
    
    C0 = np.zeros_like(freq)
    Cf = np.zeros_like(freq)
    
    # Only calculate where we have enough loss to distinguish C0
    C0[mask] = G[mask] / (omega[mask] * eps_dp[mask])
    Cf[mask] = (B[mask] / omega[mask]) - (C0[mask] * eps_p[mask])
    
    return C0, Cf

def main():
    print("="*80)
    print("PROBE PARAMETER EXTRACTION (C0, Cf)")
    print("="*80)
    
    # 1. Load Reference Data (Interpolate to common grid)
    # Use range 500 MHz to 3 GHz where probe is usually well behaved
    freq_grid = np.arange(0.5e9, 3.01e9, 0.05e9) 
    
    # Water Reference
    f_w, ep_w, edp_w = load_reference_data("water_permittivity_literature.csv")
    if f_w is None: return
    eps_water_ref = np.interp(freq_grid, f_w, ep_w) - 1j * np.interp(freq_grid, f_w, edp_w)

    # Isopropanol Reference
    f_i, ep_i, edp_i = load_reference_data("isopropanol_permittivity_15C_NPL.csv")
    if f_i is None: return
    eps_iso_ref = np.interp(freq_grid, f_i, ep_i) - 1j * np.interp(freq_grid, f_i, edp_i)

    # 2. Load Measurements
    Y_water = calculate_admittance("Database/Water/DI-water.s1p", freq_grid)
    Y_iso = calculate_admittance("Database/Isopropanol/propan-2-ol.s1p", freq_grid)
    
    if Y_water is None or Y_iso is None: return

    # 3. Calculate C0, Cf
    
    # Method A: Water Only
    C0_w, Cf_w = solve_c_single_std(freq_grid, Y_water, eps_water_ref)
    
    # Method B: Isopropanol Only
    C0_i, Cf_i = solve_c_single_std(freq_grid, Y_iso, eps_iso_ref)
    
    # Method C: Dual Reference (Water + Iso)
    # Y_w - Y_i = j*w*C0 * (eps_w - eps_i)
    # C0 = (Y_w - Y_i) / (j*w * (eps_w - eps_i))
    omega = 2 * np.pi * freq_grid
    Y0 = 1/50.0
    
    delta_Y = (Y_water - Y_iso) * Y0
    delta_eps = eps_water_ref - eps_iso_ref
    
    C0_dual = delta_Y / (1j * omega * delta_eps)
    # Take real part (physically C0 is real)
    C0_dual_real = np.real(C0_dual)
    
    # Calculate Cf using Water and the derived C0
    # Y_w = j*w*Cf + j*w*C0*eps_w
    # Cf = Y_w/(j*w) - C0*eps_w
    Cf_dual = (Y_water * Y0) / (1j * omega) - C0_dual_real * eps_water_ref
    Cf_dual_real = np.real(Cf_dual)

    # 4. Print Results at key frequencies
    print(f"\n{'Freq (GHz)':<10} | {'C0 (fF) [Water]':<15} | {'C0 (fF) [Iso]':<15} | {'C0 (fF) [Dual]':<15}")
    print("-" * 65)
    
    # Select a few points to print
    indices = np.linspace(0, len(freq_grid)-1, 10, dtype=int)
    
    for idx in indices:
        f = freq_grid[idx]/1e9
        print(f"{f:<10.2f} | {C0_w[idx]*1e15:<15.2f} | {C0_i[idx]*1e15:<15.2f} | {C0_dual_real[idx]*1e15:<15.2f}")

    print(f"\n{'Freq (GHz)':<10} | {'Cf (fF) [Water]':<15} | {'Cf (fF) [Iso]':<15} | {'Cf (fF) [Dual]':<15}")
    print("-" * 65)
    for idx in indices:
        f = freq_grid[idx]/1e9
        print(f"{f:<10.2f} | {Cf_w[idx]*1e15:<15.2f} | {Cf_i[idx]*1e15:<15.2f} | {Cf_dual_real[idx]*1e15:<15.2f}")

    # 5. Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(freq_grid/1e9, C0_w*1e15, label='Water Only')
    plt.plot(freq_grid/1e9, C0_i*1e15, label='Iso Only')
    plt.plot(freq_grid/1e9, C0_dual_real*1e15, 'k--', linewidth=2, label='Dual Ref')
    plt.title('Probe Aperture Capacitance C0')
    plt.ylabel('Capacitance (fF)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(freq_grid/1e9, Cf_w*1e15, label='Water Only')
    plt.plot(freq_grid/1e9, Cf_i*1e15, label='Iso Only')
    plt.plot(freq_grid/1e9, Cf_dual_real*1e15, 'k--', linewidth=2, label='Dual Ref')
    plt.title('Probe Fringing Capacitance Cf')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Capacitance (fF)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = Path('demo_results/probe_calibration_C_values.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")
    # plt.show() # Commented out for non-interactive environments

if __name__ == "__main__":
    main()
