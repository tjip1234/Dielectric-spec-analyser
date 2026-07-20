"""
Temperature-dependent dielectric analysis for Isopropanol
For each S1P file: fit polynomial to magnitude and phase vs frequency
Then analyze how those polynomial coefficients change with temperature
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from scipy import stats
from scipy.optimize import curve_fit
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ml_classifier.data_loader import parse_s1p_file

# Target frequencies for table output
TARGET_FREQS_HZ = [100e6, 500e6, 1e9, 2e9, 3e9]
TARGET_FREQS_LABELS = ['100 MHz', '0.5 GHz', '1 GHz', '2 GHz', '3 GHz']

# Polynomial degrees to test
POLY_DEGREES = [1, 2, 3]


def load_temperature_series(compound_dir):
    """Load all temperature measurements from the compound directory"""
    yaml_path = compound_dir / 'compound.yaml'
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    measurements = []
    for m in data.get('measurements', []):
        temp = m.get('temperature_c')
        data_file = compound_dir / m['data_file']
        
        if data_file.exists() and temp is not None:
            try:
                freq_hz, s11_db, phase_deg = parse_s1p_file(data_file)
                measurements.append({
                    'temperature': temp,
                    'freq_hz': freq_hz,
                    's11_db': s11_db,
                    'phase_deg': phase_deg,
                    'file': data_file.name
                })
            except Exception as e:
                print(f"Error loading {data_file}: {e}")
    
    # Sort by temperature
    measurements.sort(key=lambda x: x['temperature'])
    return measurements, data.get('compound', {})


def fit_frequency_polynomial(freq_hz, values, degree, freq_range=(100e6, 3e9)):
    """Fit a polynomial to values vs frequency within a range (100 MHz to 3 GHz)"""
    # Filter to frequency range
    mask = (freq_hz >= freq_range[0]) & (freq_hz <= freq_range[1])
    freq_ghz = freq_hz[mask] / 1e9  # Convert to GHz for numerical stability
    values_filt = values[mask]
    
    # Fit polynomial
    coeffs = np.polyfit(freq_ghz, values_filt, degree)
    poly = np.poly1d(coeffs)
    
    # Calculate R²
    y_pred = poly(freq_ghz)
    ss_res = np.sum((values_filt - y_pred) ** 2)
    ss_tot = np.sum((values_filt - np.mean(values_filt)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'coeffs': coeffs,
        'poly': poly,
        'r_squared': r_squared,
        'degree': degree
    }


def analyze_polynomial_fits(measurements, poly_degrees=POLY_DEGREES):
    """For each measurement, fit polynomials and collect coefficients"""
    
    results = []
    
    for m in measurements:
        temp = m['temperature']
        freq_hz = m['freq_hz']
        
        temp_result = {'temperature': temp, 'mag': {}, 'phase': {}}
        
        for degree in poly_degrees:
            # Fit magnitude
            mag_fit = fit_frequency_polynomial(freq_hz, m['s11_db'], degree)
            temp_result['mag'][degree] = mag_fit
            
            # Fit phase
            phase_fit = fit_frequency_polynomial(freq_hz, m['phase_deg'], degree)
            temp_result['phase'][degree] = phase_fit
        
        results.append(temp_result)
    
    return results


def print_polynomial_coefficients_table(poly_results, param='mag', degree=2):
    """Print table showing polynomial coefficients at each temperature"""
    
    param_name = "MAGNITUDE" if param == 'mag' else "PHASE"
    
    print(f"\n{'='*120}")
    print(f"{param_name} - Polynomial Degree {degree} Coefficients vs Temperature")
    print(f"Fit: y = a₀·f^{degree} + a₁·f^{degree-1} + ... + a_{degree}  (f in GHz)")
    print(f"{'='*120}")
    
    # Header
    header = f"{'Temp (°C)':>10} |"
    for i in range(degree + 1):
        header += f" {'a' + str(i):>14} |"
    header += f" {'R²':>10}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for r in poly_results:
        fit = r[param][degree]
        coeffs = fit['coeffs']
        row = f"{r['temperature']:>10.1f} |"
        for c in coeffs:
            row += f" {c:>14.6f} |"
        row += f" {fit['r_squared']:>10.6f}"
        print(row)


def analyze_coefficient_trends(poly_results, param='mag', degree=2):
    """Analyze how each polynomial coefficient changes with temperature"""
    
    temps = np.array([r['temperature'] for r in poly_results])
    coeffs_by_idx = {}
    
    # Extract coefficients
    n_coeffs = degree + 1
    for i in range(n_coeffs):
        coeffs_by_idx[i] = np.array([r[param][degree]['coeffs'][i] for r in poly_results])
    
    # Analyze trend for each coefficient
    trends = {}
    for i in range(n_coeffs):
        slope, intercept, r_value, p_value, std_err = stats.linregress(temps, coeffs_by_idx[i])
        trends[i] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'values': coeffs_by_idx[i]
        }
    
    return temps, trends


def print_coefficient_trend_analysis(temps, trends, param='mag', degree=2):
    """Print analysis of how coefficients change with temperature"""
    
    param_name = "MAGNITUDE" if param == 'mag' else "PHASE"
    
    print(f"\n{'='*100}")
    print(f"{param_name} - How Poly{degree} Coefficients Change with Temperature")
    print(f"{'='*100}")
    print(f"{'Coeff':>10} | {'Slope (per °C)':>18} | {'R²':>10} | {'p-value':>12} | {'Significant?':>12}")
    print("-" * 80)
    
    for i, trend in trends.items():
        sig = "YES" if trend['p_value'] < 0.05 else "no"
        if trend['r_squared'] > 0.8:
            sig += " (strong)"
        elif trend['r_squared'] > 0.5:
            sig += " (moderate)"
        
        print(f"{'a' + str(i):>10} | {trend['slope']:>18.8f} | {trend['r_squared']:>10.4f} | {trend['p_value']:>12.6f} | {sig:>12}")


def plot_coefficient_analysis(poly_results, compound_info, output_dir):
    """Create visualization of polynomial coefficient changes with temperature"""
    
    output_dir = Path(output_dir)
    
    # Best degree for each parameter (degree 2 is usually good balance)
    degree = 2
    
    temps_mag, trends_mag = analyze_coefficient_trends(poly_results, 'mag', degree)
    temps_phase, trends_phase = analyze_coefficient_trends(poly_results, 'phase', degree)
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    
    temps_fit = np.linspace(min(temps_mag) - 2, max(temps_mag) + 2, 100)
    colors = plt.cm.Set1(np.linspace(0, 1, degree + 1))
    
    # Row 1: Magnitude coefficients
    for i, (coeff_idx, trend) in enumerate(trends_mag.items()):
        ax = axes[0, i]
        ax.scatter(temps_mag, trend['values'], s=100, c=colors[i], 
                  edgecolors='black', linewidth=2, alpha=0.8, zorder=3)
        
        # Linear fit line
        y_fit = trend['slope'] * temps_fit + trend['intercept']
        ax.plot(temps_fit, y_fit, '--', color='red', linewidth=2, alpha=0.8, zorder=2,
               label=f'slope={trend["slope"]:.6f}/°C\nR²={trend["r_squared"]:.4f}')
        
        ax.set_xlabel('Temperature (°C)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'a{coeff_idx}', fontsize=12, fontweight='bold')
        ax.set_title(f'Magnitude: a{coeff_idx} vs Temperature', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    # Summary plot for magnitude (R² of each coefficient trend)
    ax_sum_mag = axes[0, 3]
    coeff_names = [f'a{i}' for i in range(degree + 1)]
    r2_values = [trends_mag[i]['r_squared'] for i in range(degree + 1)]
    bars = ax_sum_mag.bar(coeff_names, r2_values, color=colors[:degree+1], 
                         edgecolor='black', linewidth=2, alpha=0.8)
    ax_sum_mag.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='R²=0.5')
    ax_sum_mag.axhline(y=0.8, color='green', linestyle='--', linewidth=2, label='R²=0.8')
    ax_sum_mag.set_xlabel('Coefficient', fontsize=11, fontweight='bold')
    ax_sum_mag.set_ylabel('R² (trend with temp)', fontsize=11, fontweight='bold')
    ax_sum_mag.set_title('Magnitude: Coefficient Trend Strength', fontsize=12, fontweight='bold', pad=10)
    ax_sum_mag.set_ylim(0, 1.05)
    ax_sum_mag.legend(fontsize=9)
    ax_sum_mag.grid(True, alpha=0.3, axis='y')
    
    # Row 2: Phase coefficients
    for i, (coeff_idx, trend) in enumerate(trends_phase.items()):
        ax = axes[1, i]
        ax.scatter(temps_phase, trend['values'], s=100, c=colors[i], 
                  edgecolors='black', linewidth=2, alpha=0.8, zorder=3)
        
        # Linear fit line
        y_fit = trend['slope'] * temps_fit + trend['intercept']
        ax.plot(temps_fit, y_fit, '--', color='red', linewidth=2, alpha=0.8, zorder=2,
               label=f'slope={trend["slope"]:.4f}/°C\nR²={trend["r_squared"]:.4f}')
        
        ax.set_xlabel('Temperature (°C)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'a{coeff_idx}', fontsize=12, fontweight='bold')
        ax.set_title(f'Phase: a{coeff_idx} vs Temperature', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    # Summary plot for phase
    ax_sum_phase = axes[1, 3]
    r2_values_phase = [trends_phase[i]['r_squared'] for i in range(degree + 1)]
    bars = ax_sum_phase.bar(coeff_names, r2_values_phase, color=colors[:degree+1], 
                           edgecolor='black', linewidth=2, alpha=0.8)
    ax_sum_phase.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='R²=0.5')
    ax_sum_phase.axhline(y=0.8, color='green', linestyle='--', linewidth=2, label='R²=0.8')
    ax_sum_phase.set_xlabel('Coefficient', fontsize=11, fontweight='bold')
    ax_sum_phase.set_ylabel('R² (trend with temp)', fontsize=11, fontweight='bold')
    ax_sum_phase.set_title('Phase: Coefficient Trend Strength', fontsize=12, fontweight='bold', pad=10)
    ax_sum_phase.set_ylim(0, 1.05)
    ax_sum_phase.legend(fontsize=9)
    ax_sum_phase.grid(True, alpha=0.3, axis='y')
    
    compound_name = compound_info.get('chemical_name', 'Unknown')
    temps = [r['temperature'] for r in poly_results]
    plt.suptitle(f'Polynomial Coefficient Analysis: {compound_name}\n' +
                f'Poly Degree {degree} | Temperature Range: {min(temps):.0f}°C to {max(temps):.0f}°C',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Legend at bottom
    fig.text(0.5, 0.01, 
            f'Polynomial: y = a₀·f² + a₁·f + a₂  (f in GHz) | '
            f'High R² = coefficient changes predictably with temperature',
            ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / 'temp_coefficient_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: temp_coefficient_analysis.png")
    plt.close()


def plot_fit_quality_comparison(poly_results, compound_info, output_dir):
    """Compare R² for different polynomial degrees"""
    
    output_dir = Path(output_dir)
    
    temps = [r['temperature'] for r in poly_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#FFD93D']
    
    # Magnitude R² comparison
    for degree, color in zip(POLY_DEGREES, colors):
        r2_values = [r['mag'][degree]['r_squared'] for r in poly_results]
        ax1.plot(temps, r2_values, 'o-', color=color, markersize=8, linewidth=2,
                label=f'Poly{degree} (avg R²={np.mean(r2_values):.4f})', alpha=0.8)
    
    ax1.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R²', fontsize=12, fontweight='bold')
    ax1.set_title('Magnitude: Polynomial Fit Quality vs Temperature', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.9, 1.005)
    
    # Phase R² comparison
    for degree, color in zip(POLY_DEGREES, colors):
        r2_values = [r['phase'][degree]['r_squared'] for r in poly_results]
        ax2.plot(temps, r2_values, 'o-', color=color, markersize=8, linewidth=2,
                label=f'Poly{degree} (avg R²={np.mean(r2_values):.4f})', alpha=0.8)
    
    ax2.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('R²', fontsize=12, fontweight='bold')
    ax2.set_title('Phase: Polynomial Fit Quality vs Temperature', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.9, 1.005)
    
    compound_name = compound_info.get('chemical_name', 'Unknown')
    plt.suptitle(f'Polynomial Fit Quality Comparison: {compound_name}',
                fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temp_fit_quality.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: temp_fit_quality.png")
    plt.close()


def plot_spectra_with_fits(measurements, poly_results, compound_info, output_dir):
    """Plot raw spectra with polynomial fits overlaid"""
    
    output_dir = Path(output_dir)
    degree = 2  # Use degree 2 for visualization
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(measurements)))
    freq_fit = np.linspace(0.1, 3, 200)  # GHz (100 MHz to 3 GHz)
    
    for i, (m, r) in enumerate(zip(measurements, poly_results)):
        temp = m['temperature']
        freq_ghz = m['freq_hz'] / 1e9
        mask = (freq_ghz >= 0.1) & (freq_ghz <= 3)  # 100 MHz to 3 GHz
        
        # Magnitude
        ax1.plot(freq_ghz[mask], m['s11_db'][mask], '-', color=colors[i], 
                linewidth=1.5, alpha=0.6)
        # Fit line (dashed)
        y_fit = r['mag'][degree]['poly'](freq_fit)
        ax1.plot(freq_fit, y_fit, '--', color=colors[i], linewidth=1, alpha=0.8)
        
        # Phase
        ax2.plot(freq_ghz[mask], m['phase_deg'][mask], '-', color=colors[i], 
                linewidth=1.5, alpha=0.6, label=f"{temp}°C")
        y_fit = r['phase'][degree]['poly'](freq_fit)
        ax2.plot(freq_fit, y_fit, '--', color=colors[i], linewidth=1, alpha=0.8)
    
    ax1.set_xlabel('Frequency (GHz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('S11 Magnitude (dB)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Magnitude vs Frequency (solid=data, dashed=Poly{degree} fit)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Frequency (GHz)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('S11 Phase (degrees)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Phase vs Frequency (solid=data, dashed=Poly{degree} fit)', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Legend at bottom
    temps = [m['temperature'] for m in measurements]
    fig.legend(loc='lower center', ncol=min(10, len(temps)), fontsize=9, 
              frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
    
    compound_name = compound_info.get('chemical_name', 'Unknown')
    plt.suptitle(f'Spectra with Polynomial Fits: {compound_name}',
                fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_dir / 'temp_spectra_with_fits.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: temp_spectra_with_fits.png")
    plt.close()


def main():
    """Main function"""
    
    print("="*120)
    print("TEMPERATURE-DEPENDENT POLYNOMIAL ANALYSIS")
    print("For each S1P: fit polynomial to freq-dependent data, track how coefficients change with temp")
    print("="*120)
    
    compound_dir = Path("Database/temp-isopropanol")
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    
    print("\n[1/6] Loading temperature series...")
    measurements, compound_info = load_temperature_series(compound_dir)
    print(f"      ✓ Loaded {len(measurements)} measurements")
    print(f"      Temperature range: {measurements[0]['temperature']}°C to {measurements[-1]['temperature']}°C")
    
    print("\n[2/6] Fitting polynomials to each S1P file...")
    poly_results = analyze_polynomial_fits(measurements, POLY_DEGREES)
    print(f"      ✓ Fitted polynomials of degree {POLY_DEGREES}")
    
    print("\n[3/6] Polynomial coefficient tables...")
    for degree in POLY_DEGREES:
        print_polynomial_coefficients_table(poly_results, 'mag', degree)
        print_polynomial_coefficients_table(poly_results, 'phase', degree)
    
    print("\n[4/6] Analyzing coefficient trends with temperature...")
    for degree in [2]:  # Focus on degree 2
        _, trends_mag = analyze_coefficient_trends(poly_results, 'mag', degree)
        _, trends_phase = analyze_coefficient_trends(poly_results, 'phase', degree)
        print_coefficient_trend_analysis(_, trends_mag, 'mag', degree)
        print_coefficient_trend_analysis(_, trends_phase, 'phase', degree)
    
    print("\n[5/6] Creating visualizations...")
    plot_spectra_with_fits(measurements, poly_results, compound_info, output_dir)
    plot_fit_quality_comparison(poly_results, compound_info, output_dir)
    plot_coefficient_analysis(poly_results, compound_info, output_dir)
    
    # Summary
    print("\n[6/6] Summary")
    print("="*120)
    
    # Check which coefficients show significant temperature dependence
    degree = 2
    _, trends_mag = analyze_coefficient_trends(poly_results, 'mag', degree)
    _, trends_phase = analyze_coefficient_trends(poly_results, 'phase', degree)
    
    print("\nKEY FINDINGS (Poly2 coefficients):")
    print("-" * 80)
    
    print("\nMAGNITUDE coefficients that change with temperature:")
    for i, trend in trends_mag.items():
        if trend['r_squared'] > 0.5:
            strength = "STRONG" if trend['r_squared'] > 0.8 else "moderate"
            print(f"  a{i}: slope = {trend['slope']:.6f}/°C, R² = {trend['r_squared']:.4f} ({strength})")
        else:
            print(f"  a{i}: NO significant trend (R² = {trend['r_squared']:.4f})")
    
    print("\nPHASE coefficients that change with temperature:")
    for i, trend in trends_phase.items():
        if trend['r_squared'] > 0.5:
            strength = "STRONG" if trend['r_squared'] > 0.8 else "moderate"
            print(f"  a{i}: slope = {trend['slope']:.6f}/°C, R² = {trend['r_squared']:.4f} ({strength})")
        else:
            print(f"  a{i}: NO significant trend (R² = {trend['r_squared']:.4f})")
    
    print("\n" + "="*120)
    print(f"Results saved to: {output_dir}/")
    print("="*120 + "\n")


if __name__ == "__main__":
    main()