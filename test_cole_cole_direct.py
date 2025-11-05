#!/usr/bin/env python3
"""Direct test of Cole-Cole function"""

import numpy as np
import warnings
from s1p_gui.cole_cole import calculate_cole_cole_parameters

# Set up warnings to show
warnings.simplefilter("always")

# Need at least 10 points for the function to not return NaN
frequencies = np.linspace(0.5e9, 3.0e9, 16)

# Water-like Cole-Cole model
omega = 2 * np.pi * frequencies
eps_s = 80.0
eps_inf = 4.0
tau = 1.7e-11
alpha = 0.02

term = 1 + (1j * omega * tau) ** (1 - alpha)
eps_complex = eps_inf + (eps_s - eps_inf) / term
eps_real = np.real(eps_complex) + np.random.normal(0, 0.2, len(frequencies))
eps_imag = np.abs(np.imag(eps_complex)) + np.random.normal(0, 0.2, len(frequencies))

print(f"Input frequencies: {frequencies}")
print(f"Input ε' range: {eps_real.min():.2f} - {eps_real.max():.2f}")
print(f"Input ε'' range: {eps_imag.min():.2f} - {eps_imag.max():.2f}")
print(f"Number of points: {len(frequencies)}")
print(f"Valid data mask would filter to: {np.sum(np.isfinite(eps_real) & np.isfinite(eps_imag) & np.isfinite(frequencies))} points")

# Call directly
print("\nCalling calculate_cole_cole_parameters...")
result = calculate_cole_cole_parameters(frequencies, eps_real, eps_imag)

print(f"\nResult:")
print(f"  epsilon_s: {result['epsilon_s']}")
print(f"  epsilon_inf: {result['epsilon_inf']}")
print(f"  fit_success: {result['fit_success']}")
print(f"  tau: {result['tau']}")
print(f"  alpha: {result['alpha']}")
