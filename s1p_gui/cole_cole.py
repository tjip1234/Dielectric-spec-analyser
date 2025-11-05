"""
Cole-Cole model fitting for complex permittivity analysis

Provides functions to fit Cole-Cole relaxation models to dielectric data
and extract relaxation parameters.
"""

import numpy as np
from typing import Dict
import warnings
from scipy.optimize import least_squares


def calculate_cole_cole_parameters(frequencies: np.ndarray,
                                   epsilon_prime: np.ndarray,
                                   epsilon_double_prime: np.ndarray) -> Dict[str, float]:
    """
    Fit Cole-Cole parameters using nonlinear least squares

    Cole-Cole model: ε*(ω) = ε_∞ + (ε_s - ε_∞) / (1 + (iωτ)^(1-α))

    Args:
        frequencies: Frequency array in Hz
        epsilon_prime: Real part of permittivity (ε')
        epsilon_double_prime: Imaginary part of permittivity (ε'')

    Returns:
        Dictionary with fitted Cole-Cole parameters
    """
    # Filter valid data
    valid_mask = (np.isfinite(epsilon_prime) &
                  np.isfinite(epsilon_double_prime) &
                  np.isfinite(frequencies))

    if np.sum(valid_mask) < 10:
        return {
            'epsilon_s': np.nan,
            'epsilon_inf': np.nan,
            'relaxation_strength': np.nan,
            'max_loss': np.nan,
            'loss_tangent': np.nan,
            'tau': np.nan,
            'alpha': np.nan,
            'fit_success': False
        }

    freq = frequencies[valid_mask]
    eps_real = epsilon_prime[valid_mask]
    eps_imag = epsilon_double_prime[valid_mask]

    omega = 2 * np.pi * freq

    # Cole-Cole model
    def cole_cole_complex(omega, eps_s, eps_inf, tau, alpha):
        """Returns (real, imag) parts of complex permittivity"""
        term = 1 + (1j * omega * tau) ** (1 - alpha)
        eps_complex = eps_inf + (eps_s - eps_inf) / term
        return np.real(eps_complex), np.imag(eps_complex)

    # Residual function for fitting
    def residuals(params):
        eps_s, eps_inf, tau, alpha = params
        model_real, model_imag = cole_cole_complex(omega, eps_s, eps_inf, tau, alpha)

        # Weight both real and imaginary parts equally
        res_real = (model_real - eps_real) / np.std(eps_real)
        res_imag = (model_imag - eps_imag) / np.std(eps_imag)

        return np.concatenate([res_real, res_imag])

    # Initial guesses
    eps_s_guess = np.percentile(eps_real, 95)  # Near maximum
    eps_inf_guess = np.percentile(eps_real, 5)   # Near minimum but not outlier
    eps_inf_guess = max(1.0, min(eps_inf_guess, eps_s_guess - 1))  # Ensure sensible

    # Estimate relaxation time from peak loss
    peak_idx = np.argmax(eps_imag)
    tau_guess = 1 / omega[peak_idx] if peak_idx < len(omega) else 1e-10

    alpha_guess = 0.0  # Start with Debye model

    # Bounds: FIXED to prevent eps_s and eps_inf swapping
    # CRITICAL: Must guarantee eps_s > eps_inf at all times!
    # Strategy: Keep eps_s relatively high, eps_inf relatively low
    bounds_lower = [
        max(1.5, eps_s_guess * 0.7),  # eps_s lower bound (allow some flexibility)
        1.0,                           # eps_inf lower (absolute minimum)
        1e-12,                         # tau lower
        0.0                            # alpha lower
    ]
    bounds_upper = [
        eps_s_guess * 1.8,             # eps_s upper (generous upper limit)
        min(eps_s_guess * 0.3, eps_s_guess - 2.0),  # eps_inf upper MUST be well below eps_s
        1e-6,                          # tau upper
        0.5                            # alpha upper
    ]

    # Safety check: Ensure bounds are physically valid
    # eps_inf_upper must be strictly less than eps_s_lower
    if bounds_upper[1] >= bounds_lower[0]:
        # Force separation: eps_inf stays below half of eps_s_lower
        bounds_upper[1] = bounds_lower[0] * 0.4
        if bounds_upper[1] < 1.0:
            bounds_upper[1] = 1.0
            bounds_lower[0] = 2.5

    # Ensure initial guess is within bounds
    eps_s_guess_clipped = np.clip(eps_s_guess, bounds_lower[0], bounds_upper[0])
    eps_inf_guess_clipped = np.clip(eps_inf_guess, bounds_lower[1], bounds_upper[1])
    tau_guess = np.clip(tau_guess, bounds_lower[2], bounds_upper[2])
    alpha_guess = np.clip(alpha_guess, bounds_lower[3], bounds_upper[3])

    try:
        # Use robust fitting with soft_l1 loss to handle outliers
        result = least_squares(
            residuals,
            x0=[eps_s_guess_clipped, eps_inf_guess_clipped, tau_guess, alpha_guess],
            bounds=(bounds_lower, bounds_upper),
            loss='soft_l1',  # Robust to outliers
            max_nfev=1000
        )

        eps_s_fit, eps_inf_fit, tau_fit, alpha_fit = result.x

        # SAFETY CHECK: Detect and fix parameter swapping
        # If optimizer somehow violated the constraint, swap them back
        if eps_s_fit <= eps_inf_fit:
            warnings.warn(
                f"Cole-Cole fit produced invalid order: eps_s={eps_s_fit:.2f} <= eps_inf={eps_inf_fit:.2f}. "
                f"Swapping values to maintain physical correctness.",
                UserWarning
            )
            eps_s_fit, eps_inf_fit = eps_inf_fit, eps_s_fit

        # Calculate derived parameters
        relaxation_strength = eps_s_fit - eps_inf_fit

        # Recalculate model to get max loss
        _, model_imag = cole_cole_complex(omega, eps_s_fit, eps_inf_fit, tau_fit, alpha_fit)
        max_loss = np.max(model_imag)

        # Average loss tangent
        with np.errstate(divide='ignore', invalid='ignore'):
            loss_tangent = np.nanmean(eps_imag / eps_real)

        return {
            'epsilon_s': float(eps_s_fit),
            'epsilon_inf': float(eps_inf_fit),
            'relaxation_strength': float(relaxation_strength),
            'max_loss': float(max_loss),
            'loss_tangent': float(loss_tangent),
            'tau': float(tau_fit),
            'alpha': float(alpha_fit),
            'fit_success': result.success and (eps_s_fit > eps_inf_fit)  # Add order check to success
        }

    except Exception as e:
        error_msg = f"Cole-Cole fitting failed: {e}"
        warnings.warn(error_msg)
        # Also print to help debugging
        import sys
        print(f"DEBUG: {error_msg}", file=sys.stderr)
        return {
            'epsilon_s': np.nan,
            'epsilon_inf': np.nan,
            'relaxation_strength': np.nan,
            'max_loss': np.nan,
            'loss_tangent': np.nan,
            'tau': np.nan,
            'alpha': np.nan,
            'fit_success': False
        }
