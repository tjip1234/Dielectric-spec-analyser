"""
Debye Relaxation Fitting for Dielectric Materials
Fits complex permittivity data to single and double Debye models

Debye Model:
ε*(ω) = ε∞ + (εs - ε∞) / (1 + jωτ)

Where:
  ε∞  = high-frequency permittivity (optical)
  εs  = static (low-frequency) permittivity  
  τ   = relaxation time (seconds)
  ω   = angular frequency = 2πf

Real part:      ε' = ε∞ + (εs - ε∞) / (1 + (ωτ)²)
Imaginary part: ε" = (εs - ε∞) * ωτ / (1 + (ωτ)²)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ml_classifier.data_loader import parse_s1p_file


# =============================================================================
# DEBYE RELAXATION MODELS
# =============================================================================

def debye_single(freq_hz, eps_s, eps_inf, tau):
    """
    Single Debye relaxation model
    
    Parameters:
        freq_hz: frequency in Hz
        eps_s: static permittivity
        eps_inf: high-frequency permittivity
        tau: relaxation time in seconds
    
    Returns:
        eps_prime, eps_double_prime (real and imaginary parts)
    """
    omega = 2 * np.pi * freq_hz
    omega_tau = omega * tau
    
    eps_prime = eps_inf + (eps_s - eps_inf) / (1 + omega_tau**2)
    eps_double_prime = (eps_s - eps_inf) * omega_tau / (1 + omega_tau**2)
    
    return eps_prime, eps_double_prime


def debye_double(freq_hz, eps_s, eps_2, eps_inf, tau1, tau2):
    """
    Double Debye relaxation model (two relaxation processes)
    
    ε* = ε∞ + (εs - ε2)/(1 + jωτ1) + (ε2 - ε∞)/(1 + jωτ2)
    """
    omega = 2 * np.pi * freq_hz
    
    # First relaxation
    omega_tau1 = omega * tau1
    delta_eps1 = eps_s - eps_2
    eps_prime1 = delta_eps1 / (1 + omega_tau1**2)
    eps_double_prime1 = delta_eps1 * omega_tau1 / (1 + omega_tau1**2)
    
    # Second relaxation
    omega_tau2 = omega * tau2
    delta_eps2 = eps_2 - eps_inf
    eps_prime2 = delta_eps2 / (1 + omega_tau2**2)
    eps_double_prime2 = delta_eps2 * omega_tau2 / (1 + omega_tau2**2)
    
    eps_prime = eps_inf + eps_prime1 + eps_prime2
    eps_double_prime = eps_double_prime1 + eps_double_prime2
    
    return eps_prime, eps_double_prime


def cole_cole(freq_hz, eps_s, eps_inf, tau, alpha):
    """
    Cole-Cole model (broadened relaxation)
    
    ε* = ε∞ + (εs - ε∞) / (1 + (jωτ)^(1-α))
    
    α = 0 gives standard Debye
    α > 0 gives broadened relaxation peak
    """
    omega = 2 * np.pi * freq_hz
    omega_tau = omega * tau
    
    # Cole-Cole with distribution parameter
    phi = (1 - alpha) * np.pi / 2
    
    denom = 1 + 2 * omega_tau**(1-alpha) * np.cos(phi) + omega_tau**(2*(1-alpha))
    
    eps_prime = eps_inf + (eps_s - eps_inf) * (1 + omega_tau**(1-alpha) * np.cos(phi)) / denom
    eps_double_prime = (eps_s - eps_inf) * omega_tau**(1-alpha) * np.sin(phi) / denom
    
    return eps_prime, eps_double_prime


def havriliak_negami(freq_hz, eps_s, eps_inf, tau, alpha, beta):
    """
    Havriliak-Negami model (asymmetric broadening)
    
    ε* = ε∞ + (εs - ε∞) / (1 + (jωτ)^α)^β
    
    α controls symmetric broadening (0 < α ≤ 1)
    β controls asymmetric broadening (0 < β ≤ 1)
    α=1, β=1 gives Debye; α<1, β=1 gives Cole-Cole
    """
    omega = 2 * np.pi * freq_hz
    omega_tau = omega * tau
    
    # Calculate using polar form
    phi = alpha * np.pi / 2
    r = (1 + 2 * omega_tau**alpha * np.cos(phi) + omega_tau**(2*alpha))**(beta/2)
    theta = beta * np.arctan2(omega_tau**alpha * np.sin(phi), 
                               1 + omega_tau**alpha * np.cos(phi))
    
    eps_prime = eps_inf + (eps_s - eps_inf) * np.cos(theta) / r
    eps_double_prime = (eps_s - eps_inf) * np.sin(theta) / r
    
    return eps_prime, eps_double_prime


# =============================================================================
# FITTING FUNCTIONS
# =============================================================================

def fit_single_debye(freq_hz, eps_prime, eps_double_prime, 
                     initial_guess=None, bounds=None):
    """Fit single Debye model to permittivity data"""
    
    if initial_guess is None:
        # Estimate initial parameters
        eps_s_guess = eps_prime[0]  # Low frequency
        eps_inf_guess = eps_prime[-1]  # High frequency
        # Estimate tau from peak in eps''
        peak_idx = np.argmax(eps_double_prime)
        tau_guess = 1 / (2 * np.pi * freq_hz[peak_idx])
        initial_guess = [eps_s_guess, eps_inf_guess, tau_guess]
    
    if bounds is None:
        bounds = ([1, 1, 1e-15], [100, 50, 1e-6])
    
    def objective(params):
        eps_s, eps_inf, tau = params
        eps_p_model, eps_dp_model = debye_single(freq_hz, eps_s, eps_inf, tau)
        
        # Weighted residuals (normalize by magnitude)
        resid_prime = (eps_prime - eps_p_model) / np.max(eps_prime)
        resid_double = (eps_double_prime - eps_dp_model) / np.max(eps_double_prime)
        
        return np.sum(resid_prime**2 + resid_double**2)
    
    result = minimize(objective, initial_guess, method='L-BFGS-B',
                     bounds=list(zip(bounds[0], bounds[1])))
    
    eps_s, eps_inf, tau = result.x
    
    # Calculate R² for both components
    eps_p_fit, eps_dp_fit = debye_single(freq_hz, eps_s, eps_inf, tau)
    
    ss_res_p = np.sum((eps_prime - eps_p_fit)**2)
    ss_tot_p = np.sum((eps_prime - np.mean(eps_prime))**2)
    r2_prime = 1 - ss_res_p / ss_tot_p
    
    ss_res_dp = np.sum((eps_double_prime - eps_dp_fit)**2)
    ss_tot_dp = np.sum((eps_double_prime - np.mean(eps_double_prime))**2)
    r2_double = 1 - ss_res_dp / ss_tot_dp
    
    return {
        'model': 'Single Debye',
        'eps_s': eps_s,
        'eps_inf': eps_inf,
        'tau': tau,
        'tau_ps': tau * 1e12,  # Convert to picoseconds
        'f_relax_ghz': 1 / (2 * np.pi * tau) / 1e9,  # Relaxation frequency
        'r2_prime': r2_prime,
        'r2_double': r2_double,
        'r2_avg': (r2_prime + r2_double) / 2
    }


def fit_double_debye(freq_hz, eps_prime, eps_double_prime,
                     initial_guess=None, bounds=None):
    """Fit double Debye model to permittivity data"""
    
    if initial_guess is None:
        eps_s_guess = eps_prime[0]
        eps_inf_guess = eps_prime[-1]
        eps_2_guess = (eps_s_guess + eps_inf_guess) / 2
        tau1_guess = 1e-10  # 100 ps
        tau2_guess = 1e-11  # 10 ps
        initial_guess = [eps_s_guess, eps_2_guess, eps_inf_guess, tau1_guess, tau2_guess]
    
    if bounds is None:
        bounds = ([1, 1, 1, 1e-13, 1e-14], [100, 50, 20, 1e-8, 1e-9])
    
    def objective(params):
        eps_s, eps_2, eps_inf, tau1, tau2 = params
        if tau1 <= tau2:  # Ensure tau1 > tau2
            return 1e10
        eps_p_model, eps_dp_model = debye_double(freq_hz, eps_s, eps_2, eps_inf, tau1, tau2)
        
        resid_prime = (eps_prime - eps_p_model) / np.max(eps_prime)
        resid_double = (eps_double_prime - eps_dp_model) / np.max(eps_double_prime)
        
        return np.sum(resid_prime**2 + resid_double**2)
    
    result = minimize(objective, initial_guess, method='L-BFGS-B',
                     bounds=list(zip(bounds[0], bounds[1])))
    
    eps_s, eps_2, eps_inf, tau1, tau2 = result.x
    
    eps_p_fit, eps_dp_fit = debye_double(freq_hz, eps_s, eps_2, eps_inf, tau1, tau2)
    
    ss_res_p = np.sum((eps_prime - eps_p_fit)**2)
    ss_tot_p = np.sum((eps_prime - np.mean(eps_prime))**2)
    r2_prime = 1 - ss_res_p / ss_tot_p
    
    ss_res_dp = np.sum((eps_double_prime - eps_dp_fit)**2)
    ss_tot_dp = np.sum((eps_double_prime - np.mean(eps_double_prime))**2)
    r2_double = 1 - ss_res_dp / ss_tot_dp
    
    return {
        'model': 'Double Debye',
        'eps_s': eps_s,
        'eps_2': eps_2,
        'eps_inf': eps_inf,
        'tau1': tau1,
        'tau2': tau2,
        'tau1_ps': tau1 * 1e12,
        'tau2_ps': tau2 * 1e12,
        'f_relax1_ghz': 1 / (2 * np.pi * tau1) / 1e9,
        'f_relax2_ghz': 1 / (2 * np.pi * tau2) / 1e9,
        'r2_prime': r2_prime,
        'r2_double': r2_double,
        'r2_avg': (r2_prime + r2_double) / 2
    }


def fit_cole_cole(freq_hz, eps_prime, eps_double_prime,
                  initial_guess=None, bounds=None):
    """Fit Cole-Cole model to permittivity data"""
    
    if initial_guess is None:
        eps_s_guess = eps_prime[0]
        eps_inf_guess = eps_prime[-1]
        peak_idx = np.argmax(eps_double_prime)
        tau_guess = 1 / (2 * np.pi * freq_hz[peak_idx])
        alpha_guess = 0.1
        initial_guess = [eps_s_guess, eps_inf_guess, tau_guess, alpha_guess]
    
    if bounds is None:
        bounds = ([1, 1, 1e-15, 0], [100, 50, 1e-6, 0.5])
    
    def objective(params):
        eps_s, eps_inf, tau, alpha = params
        eps_p_model, eps_dp_model = cole_cole(freq_hz, eps_s, eps_inf, tau, alpha)
        
        resid_prime = (eps_prime - eps_p_model) / np.max(eps_prime)
        resid_double = (eps_double_prime - eps_dp_model) / np.max(eps_double_prime)
        
        return np.sum(resid_prime**2 + resid_double**2)
    
    result = minimize(objective, initial_guess, method='L-BFGS-B',
                     bounds=list(zip(bounds[0], bounds[1])))
    
    eps_s, eps_inf, tau, alpha = result.x
    
    eps_p_fit, eps_dp_fit = cole_cole(freq_hz, eps_s, eps_inf, tau, alpha)
    
    ss_res_p = np.sum((eps_prime - eps_p_fit)**2)
    ss_tot_p = np.sum((eps_prime - np.mean(eps_prime))**2)
    r2_prime = 1 - ss_res_p / ss_tot_p
    
    ss_res_dp = np.sum((eps_double_prime - eps_dp_fit)**2)
    ss_tot_dp = np.sum((eps_double_prime - np.mean(eps_double_prime))**2)
    r2_double = 1 - ss_res_dp / ss_tot_dp
    
    return {
        'model': 'Cole-Cole',
        'eps_s': eps_s,
        'eps_inf': eps_inf,
        'tau': tau,
        'tau_ps': tau * 1e12,
        'alpha': alpha,
        'f_relax_ghz': 1 / (2 * np.pi * tau) / 1e9,
        'r2_prime': r2_prime,
        'r2_double': r2_double,
        'r2_avg': (r2_prime + r2_double) / 2
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_npl_reference(csv_path):
    """Load NPL reference permittivity data from CSV"""
    import csv
    
    freq_ghz = []
    eps_prime = []
    eps_double_prime = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            freq_ghz.append(float(row['Freq_GHz']))
            eps_prime.append(float(row['eps_prime_bestfit']))
            eps_double_prime.append(float(row['eps_double_prime_bestfit']))
    
    return (np.array(freq_ghz) * 1e9,  # Convert to Hz
            np.array(eps_prime), 
            np.array(eps_double_prime))


def s11_to_permittivity_approx(s11_db, phase_deg, freq_hz, 
                                z0=50, substrate_eps=1):
    """
    Approximate conversion from S11 to permittivity
    This is a simplified model - actual conversion depends on probe geometry
    
    For open-ended coaxial probe:
    Γ = (Z_sample - Z0) / (Z_sample + Z0)
    Z_sample ∝ 1/√ε*
    """
    # Convert S11 to complex reflection coefficient
    s11_mag = 10**(s11_db / 20)
    s11_phase_rad = np.deg2rad(phase_deg)
    gamma = s11_mag * np.exp(1j * s11_phase_rad)
    
    # Simplified inversion (assumes ideal probe)
    # This is approximate - real conversion needs probe calibration
    z_ratio = (1 + gamma) / (1 - gamma)
    
    # Approximate permittivity (this is very rough!)
    eps_complex = (z0 / (z_ratio * z0))**2
    
    eps_prime = np.real(eps_complex)
    eps_double_prime = -np.imag(eps_complex)
    
    return eps_prime, eps_double_prime


def load_s1p_as_permittivity(s1p_path, use_approx=True):
    """
    Load S1P file and optionally convert to permittivity
    
    Note: Without proper calibration, this is only approximate!
    For accurate results, use calibrated permittivity data.
    """
    freq_hz, s11_db, phase_deg = parse_s1p_file(s1p_path)
    
    if use_approx:
        eps_prime, eps_double_prime = s11_to_permittivity_approx(
            s11_db, phase_deg, freq_hz)
        return freq_hz, eps_prime, eps_double_prime
    else:
        # Return raw S11 data
        return freq_hz, s11_db, phase_deg


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_fit_comparison(freq_hz, eps_prime, eps_double_prime, fits, title="Debye Fit Comparison"):
    """Plot data with multiple model fits"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    freq_ghz = freq_hz / 1e9
    freq_plot = np.linspace(freq_hz.min(), freq_hz.max(), 500)
    freq_plot_ghz = freq_plot / 1e9
    
    colors = {'Single Debye': '#E74C3C', 'Double Debye': '#3498DB', 
              'Cole-Cole': '#2ECC71', 'Havriliak-Negami': '#9B59B6'}
    
    # ε' vs frequency
    ax1 = axes[0, 0]
    ax1.scatter(freq_ghz, eps_prime, s=80, c='black', marker='o', 
               label='Data', zorder=3, alpha=0.7)
    
    for fit in fits:
        model = fit['model']
        color = colors.get(model, 'gray')
        
        if model == 'Single Debye':
            eps_p, eps_dp = debye_single(freq_plot, fit['eps_s'], fit['eps_inf'], fit['tau'])
        elif model == 'Double Debye':
            eps_p, eps_dp = debye_double(freq_plot, fit['eps_s'], fit['eps_2'], 
                                         fit['eps_inf'], fit['tau1'], fit['tau2'])
        elif model == 'Cole-Cole':
            eps_p, eps_dp = cole_cole(freq_plot, fit['eps_s'], fit['eps_inf'], 
                                      fit['tau'], fit['alpha'])
        
        ax1.plot(freq_plot_ghz, eps_p, '-', color=color, linewidth=2.5,
                label=f"{model} (R²={fit['r2_prime']:.4f})")
    
    ax1.set_xlabel('Frequency (GHz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel("ε' (Real Permittivity)", fontsize=12, fontweight='bold')
    ax1.set_title("ε' vs Frequency", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ε'' vs frequency
    ax2 = axes[0, 1]
    ax2.scatter(freq_ghz, eps_double_prime, s=80, c='black', marker='o',
               label='Data', zorder=3, alpha=0.7)
    
    for fit in fits:
        model = fit['model']
        color = colors.get(model, 'gray')
        
        if model == 'Single Debye':
            eps_p, eps_dp = debye_single(freq_plot, fit['eps_s'], fit['eps_inf'], fit['tau'])
        elif model == 'Double Debye':
            eps_p, eps_dp = debye_double(freq_plot, fit['eps_s'], fit['eps_2'],
                                         fit['eps_inf'], fit['tau1'], fit['tau2'])
        elif model == 'Cole-Cole':
            eps_p, eps_dp = cole_cole(freq_plot, fit['eps_s'], fit['eps_inf'],
                                      fit['tau'], fit['alpha'])
        
        ax2.plot(freq_plot_ghz, eps_dp, '-', color=color, linewidth=2.5,
                label=f"{model} (R²={fit['r2_double']:.4f})")
    
    ax2.set_xlabel('Frequency (GHz)', fontsize=12, fontweight='bold')
    ax2.set_ylabel("ε'' (Imaginary Permittivity)", fontsize=12, fontweight='bold')
    ax2.set_title("ε'' vs Frequency", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Cole-Cole plot (ε'' vs ε')
    ax3 = axes[1, 0]
    ax3.scatter(eps_prime, eps_double_prime, s=80, c='black', marker='o',
               label='Data', zorder=3, alpha=0.7)
    
    for fit in fits:
        model = fit['model']
        color = colors.get(model, 'gray')
        
        if model == 'Single Debye':
            eps_p, eps_dp = debye_single(freq_plot, fit['eps_s'], fit['eps_inf'], fit['tau'])
        elif model == 'Double Debye':
            eps_p, eps_dp = debye_double(freq_plot, fit['eps_s'], fit['eps_2'],
                                         fit['eps_inf'], fit['tau1'], fit['tau2'])
        elif model == 'Cole-Cole':
            eps_p, eps_dp = cole_cole(freq_plot, fit['eps_s'], fit['eps_inf'],
                                      fit['tau'], fit['alpha'])
        
        ax3.plot(eps_p, eps_dp, '-', color=color, linewidth=2.5, label=model)
    
    ax3.set_xlabel("ε' (Real)", fontsize=12, fontweight='bold')
    ax3.set_ylabel("ε'' (Imaginary)", fontsize=12, fontweight='bold')
    ax3.set_title("Cole-Cole Plot", fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # Model comparison bar chart
    ax4 = axes[1, 1]
    models = [f['model'] for f in fits]
    r2_avgs = [f['r2_avg'] for f in fits]
    bar_colors = [colors.get(m, 'gray') for m in models]
    
    bars = ax4.bar(models, r2_avgs, color=bar_colors, edgecolor='black', linewidth=2)
    ax4.axhline(y=0.99, color='green', linestyle='--', linewidth=2, label='R²=0.99')
    ax4.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='R²=0.95')
    
    ax4.set_ylabel('Average R²', fontsize=12, fontweight='bold')
    ax4.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylim(0.8, 1.02)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add R² values on bars
    for bar, r2 in zip(bars, r2_avgs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{r2:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def print_fit_results(fits):
    """Print detailed fit results"""
    
    print("\n" + "="*100)
    print("DEBYE RELAXATION FIT RESULTS")
    print("="*100)
    
    for fit in fits:
        print(f"\n{'-'*50}")
        print(f"Model: {fit['model']}")
        print(f"{'-'*50}")
        
        print(f"  εs (static permittivity):     {fit['eps_s']:.4f}")
        if 'eps_2' in fit:
            print(f"  ε2 (intermediate):            {fit['eps_2']:.4f}")
        print(f"  ε∞ (high-freq permittivity):  {fit['eps_inf']:.4f}")
        
        if 'tau' in fit:
            print(f"  τ (relaxation time):          {fit['tau']:.4e} s ({fit['tau_ps']:.2f} ps)")
            print(f"  f_relax (relaxation freq):    {fit['f_relax_ghz']:.4f} GHz")
        if 'tau1' in fit:
            print(f"  τ1 (slow relaxation):         {fit['tau1']:.4e} s ({fit['tau1_ps']:.2f} ps)")
            print(f"  τ2 (fast relaxation):         {fit['tau2']:.4e} s ({fit['tau2_ps']:.2f} ps)")
            print(f"  f_relax1:                     {fit['f_relax1_ghz']:.4f} GHz")
            print(f"  f_relax2:                     {fit['f_relax2_ghz']:.4f} GHz")
        if 'alpha' in fit:
            print(f"  α (broadening parameter):     {fit['alpha']:.4f}")
        
        print(f"\n  R² (ε'):   {fit['r2_prime']:.6f}")
        print(f"  R² (ε''):  {fit['r2_double']:.6f}")
        print(f"  R² (avg):  {fit['r2_avg']:.6f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function - fit NPL isopropanol data"""
    
    print("="*100)
    print("DEBYE RELAXATION FITTING")
    print("Isopropanol @ 15°C (NPL Reference Data)")
    print("="*100)
    
    # Load NPL reference data
    npl_csv = Path("isopropanol_permittivity_15C_NPL.csv")
    
    if not npl_csv.exists():
        print(f"Error: {npl_csv} not found!")
        return
    
    freq_hz, eps_prime, eps_double_prime = load_npl_reference(npl_csv)
    
    print(f"\nLoaded {len(freq_hz)} data points")
    print(f"Frequency range: {freq_hz.min()/1e9:.2f} - {freq_hz.max()/1e9:.2f} GHz")
    print(f"ε' range: {eps_prime.min():.2f} - {eps_prime.max():.2f}")
    print(f"ε'' range: {eps_double_prime.min():.2f} - {eps_double_prime.max():.2f}")
    
    # Fit models
    print("\n[1/3] Fitting Single Debye model...")
    fit_single = fit_single_debye(freq_hz, eps_prime, eps_double_prime)
    
    print("[2/3] Fitting Double Debye model...")
    fit_double = fit_double_debye(freq_hz, eps_prime, eps_double_prime)
    
    print("[3/3] Fitting Cole-Cole model...")
    fit_cc = fit_cole_cole(freq_hz, eps_prime, eps_double_prime)
    
    fits = [fit_single, fit_double, fit_cc]
    
    # Print results
    print_fit_results(fits)
    
    # Find best model
    best_fit = max(fits, key=lambda x: x['r2_avg'])
    print(f"\n{'='*100}")
    print(f"BEST MODEL: {best_fit['model']} (R² = {best_fit['r2_avg']:.6f})")
    print(f"{'='*100}")
    
    # Plot
    fig = plot_fit_comparison(freq_hz, eps_prime, eps_double_prime, fits,
                             title="Isopropanol @ 15°C - Debye Relaxation Fits (NPL Data)")
    
    output_path = Path("demo_results/debye_fit_isopropanol.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.show()
    
    # Literature comparison
    print("\n" + "="*100)
    print("LITERATURE COMPARISON (Isopropanol @ ~20°C)")
    print("="*100)
    print("Expected values from literature:")
    print("  εs ≈ 18-20 (static permittivity)")
    print("  ε∞ ≈ 2-4 (high-frequency permittivity)")
    print("  τ ≈ 300-600 ps (relaxation time)")
    print("  f_relax ≈ 0.3-0.5 GHz")
    print("\nYour fit:")
    print(f"  εs = {best_fit['eps_s']:.2f}")
    print(f"  ε∞ = {best_fit['eps_inf']:.2f}")
    if 'tau_ps' in best_fit:
        print(f"  τ = {best_fit['tau_ps']:.1f} ps")
        print(f"  f_relax = {best_fit['f_relax_ghz']:.3f} GHz")


def fit_custom_s1p(s1p_path):
    """Fit Debye model to a custom S1P file"""
    
    print(f"\n{'='*100}")
    print(f"FITTING: {s1p_path}")
    print(f"{'='*100}")
    
    # Load S1P file
    freq_hz, s11_db, phase_deg = parse_s1p_file(s1p_path)
    
    # Filter to 100 MHz - 3 GHz
    mask = (freq_hz >= 100e6) & (freq_hz <= 3e9)
    freq_hz = freq_hz[mask]
    s11_db = s11_db[mask]
    phase_deg = phase_deg[mask]
    
    print(f"Loaded {len(freq_hz)} points ({freq_hz.min()/1e9:.2f} - {freq_hz.max()/1e9:.2f} GHz)")
    
    # Approximate permittivity conversion
    eps_prime, eps_double_prime = s11_to_permittivity_approx(s11_db, phase_deg, freq_hz)
    
    print("\nNote: Using approximate S11 -> permittivity conversion")
    print("      Results may not be accurate without proper probe calibration!")
    
    # Fit models
    fits = []
    
    try:
        fit_single = fit_single_debye(freq_hz, eps_prime, eps_double_prime)
        fits.append(fit_single)
    except Exception as e:
        print(f"Single Debye fit failed: {e}")
    
    try:
        fit_double = fit_double_debye(freq_hz, eps_prime, eps_double_prime)
        fits.append(fit_double)
    except Exception as e:
        print(f"Double Debye fit failed: {e}")
    
    try:
        fit_cc = fit_cole_cole(freq_hz, eps_prime, eps_double_prime)
        fits.append(fit_cc)
    except Exception as e:
        print(f"Cole-Cole fit failed: {e}")
    
    if fits:
        print_fit_results(fits)
        
        fig = plot_fit_comparison(freq_hz, eps_prime, eps_double_prime, fits,
                                 title=f"Debye Fit: {Path(s1p_path).stem}")
        plt.show()
    
    return fits


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Fit custom S1P file
        fit_custom_s1p(sys.argv[1])
    else:
        # Fit NPL reference data
        main()
