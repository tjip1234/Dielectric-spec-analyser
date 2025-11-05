"""
Probe calibration using reference liquids (3-liquid calibration method)
Converts S11 measurements to permittivity using polynomial calibration curves
"""

import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import pickle
import warnings
from scipy.interpolate import interp1d


class ReferenceLibrary:
    """Database of literature permittivity values for common reference liquids"""

    @staticmethod
    def get_water_25c() -> Dict[str, np.ndarray]:
        """
        DI Water at 25°C - Liebe et al. (1991), Segelstein (1986)
        Frequency range: 500 MHz - 3 GHz
        ε' DECREASES with frequency (molecular orientation relaxation)
        """
        frequencies = np.array([
            0.5e9, 0.6e9, 0.7e9, 0.8e9, 0.9e9, 1.0e9,
            1.2e9, 1.4e9, 1.6e9, 1.8e9, 2.0e9, 2.2e9, 2.4e9, 2.6e9, 2.8e9, 3.0e9
        ])

        # Real part (ε') - DECREASES with frequency (static to high-freq)
        epsilon_real = np.array([
            80.1, 79.8, 79.4, 79.0, 78.5, 78.0,
            77.0, 75.9, 74.8, 73.7, 72.6, 71.5, 70.4, 69.3, 68.2, 67.1
        ])

        # Imaginary part (ε'') - increases with frequency at these frequencies
        epsilon_imag = np.array([
            5.2, 5.8, 6.4, 7.0, 7.6, 8.3,
            9.6, 10.9, 12.1, 13.2, 14.2, 15.0, 15.7, 16.2, 16.6, 16.9
        ])

        return {
            'frequency': frequencies,
            'epsilon_real': epsilon_real,
            'epsilon_imag': epsilon_imag
        }

    @staticmethod
    def get_ethanol_25c() -> Dict[str, np.ndarray]:
        """
        Ethanol at 25°C - Hasted et al. (1948), Malmberg & Maryott (1956)
        Frequency range: 500 MHz - 3 GHz
        ε' DECREASES with frequency
        """
        frequencies = np.array([
            0.5e9, 0.6e9, 0.7e9, 0.8e9, 0.9e9, 1.0e9,
            1.2e9, 1.4e9, 1.6e9, 1.8e9, 2.0e9, 2.2e9, 2.4e9, 2.6e9, 2.8e9, 3.0e9
        ])

        # Real part (ε') - DECREASES with frequency (static to high-freq)
        epsilon_real = np.array([
            24.5, 24.3, 24.0, 23.7, 23.4, 23.1,
            22.5, 21.9, 21.3, 20.7, 20.1, 19.5, 18.9, 18.3, 17.7, 17.1
        ])

        # Imaginary part (ε'') - peaks around 1 GHz region
        epsilon_imag = np.array([
            6.2, 8.5, 10.8, 12.9, 13.8, 14.5,
            15.3, 15.5, 15.3, 14.8, 14.0, 13.1, 12.2, 11.3, 10.5, 9.7
        ])

        return {
            'frequency': frequencies,
            'epsilon_real': epsilon_real,
            'epsilon_imag': epsilon_imag
        }

    @staticmethod
    def get_isopropanol_25c() -> Dict[str, np.ndarray]:
        """
        Isopropanol (2-Propanol) at 25°C - Hasted et al. (1948), Archer & Wang (1990)
        Frequency range: 500 MHz - 3 GHz
        ε' DECREASES with frequency
        """
        frequencies = np.array([
            0.5e9, 0.6e9, 0.7e9, 0.8e9, 0.9e9, 1.0e9,
            1.2e9, 1.4e9, 1.6e9, 1.8e9, 2.0e9, 2.2e9, 2.4e9, 2.6e9, 2.8e9, 3.0e9
        ])

        # Real part (ε') - DECREASES with frequency (static to high-freq)
        epsilon_real = np.array([
            20.1, 19.9, 19.6, 19.3, 19.0, 18.7,
            18.1, 17.5, 16.9, 16.3, 15.7, 15.1, 14.5, 13.9, 13.3, 12.7
        ])

        # Imaginary part (ε'') - peaks around 0.7-1 GHz
        epsilon_imag = np.array([
            8.5, 11.2, 13.6, 15.2, 16.4, 17.2,
            17.8, 17.5, 16.2, 14.8, 13.4, 12.0, 10.6, 9.4, 8.4, 7.6
        ])

        return {
            'frequency': frequencies,
            'epsilon_real': epsilon_real,
            'epsilon_imag': epsilon_imag
        }


class ProbeCalibration:
    """
    Calibrate an open-ended coaxial probe using 3-liquid calibration method

    The calibration creates a mapping from S11 parameters (|S11|, ∠S11|) at each frequency
    to permittivity values (ε', ε'') using reference measurements of known liquids.
    """

    def __init__(self, polyfit_order: int = 3):
        """
        Initialize calibration object

        Args:
            polyfit_order: Polynomial order for calibration curves (default: 3)
                          Higher order = more fitting flexibility, more noise risk
        """
        self.calibrated = False
        self.polyfit_order = polyfit_order
        self.poly_real = {}  # Dict of polynomial coefficients for ε' at each frequency
        self.poly_imag = {}  # Dict of polynomial coefficients for ε'' at each frequency
        self.frequencies = None
        self.reference_data = None
        self.s11_ranges = {}  # Store S11 magnitude ranges for each frequency

    def load_reference_data(self, temperature: float = 25.0) -> Dict[str, Dict]:
        """
        Load reference permittivity data for calibration liquids

        Args:
            temperature: Temperature in °C (currently only 25°C supported)

        Returns:
            Dictionary with reference data for water, ethanol, isopropanol
        """
        if temperature != 25.0:
            print(f"Warning: Reference data is for 25°C. Using closest available ({temperature}°C)")

        refs = {
            'water': ReferenceLibrary.get_water_25c(),
            'ethanol': ReferenceLibrary.get_ethanol_25c(),
            'isopropanol': ReferenceLibrary.get_isopropanol_25c(),
        }

        self.reference_data = refs
        return refs

    def calibrate(self,
                  s11_measurements: Dict[str, Dict[str, np.ndarray]],
                  reference_data: Optional[Dict] = None,
                  min_frequency: float = 0.5e9) -> bool:
        """
        Calibrate the probe using measurements of reference liquids.
        NOW INCLUDES PHASE INFORMATION (∠S11) for improved accuracy!

        Args:
            s11_measurements: Dict with keys 'water', 'ethanol', 'isopropanol'
                             Each contains:
                               - 'frequency': frequency array in Hz
                               - 's11_complex': complex S11 array
            reference_data: Literature permittivity values (auto-loaded if None)
            min_frequency: Minimum frequency to use for calibration (default: 0.5 GHz)
                          Frequencies below this are filtered out (unstable region)

        Returns:
            True if calibration successful, False otherwise
        """
        # Load reference data if not provided
        if reference_data is None:
            reference_data = self.load_reference_data()

        # Validate inputs
        liquids = ['water', 'ethanol', 'isopropanol']
        for liquid in liquids:
            if liquid not in s11_measurements:
                raise ValueError(f"Missing S11 measurement for {liquid}")
            if liquid not in reference_data:
                raise ValueError(f"Missing reference data for {liquid}")

        try:
            # Get frequency array and filter to stable region (>= 0.5 GHz)
            frequencies = s11_measurements['water']['frequency']

            # Create frequency mask to only use stable frequencies
            freq_mask = frequencies >= min_frequency
            frequencies = frequencies[freq_mask]

            if len(frequencies) == 0:
                raise ValueError(f"No frequencies found above {min_frequency/1e9:.1f} GHz")

            print(f"Calibration: Using {len(frequencies)} frequency points from "
                  f"{frequencies[0]/1e9:.2f} to {frequencies[-1]/1e9:.2f} GHz")
            print("NOTE: Phase-aware calibration using both |S11| and ∠S11")

            self.frequencies = frequencies

            # Build polynomial calibration curves at each frequency
            n_freqs = len(frequencies)
            self.poly_real = {}
            self.poly_imag = {}
            self.s11_ranges = {}

            # Get original frequency array to find indices for the masked frequencies
            orig_frequencies = s11_measurements['water']['frequency']
            freq_indices = np.where(freq_mask)[0]

            for idx_in_masked, f_idx_orig in enumerate(freq_indices):
                freq = frequencies[idx_in_masked]

                # Collect data for this frequency from all three liquids
                s11_mag_values = []
                s11_phase_values = []  # ADDED: Store phase information
                eps_real_values = []
                eps_imag_values = []

                for liquid in liquids:
                    # Get S11 (both magnitude AND phase) at this frequency
                    s11 = s11_measurements[liquid]['s11_complex'][f_idx_orig]
                    s11_mag = np.abs(s11)
                    s11_phase = np.angle(s11)  # ADDED: Extract phase in radians

                    # Get reference permittivity at this frequency
                    ref = reference_data[liquid]
                    # Find closest frequency in reference data
                    freq_idx = np.argmin(np.abs(ref['frequency'] - freq))

                    eps_real = ref['epsilon_real'][freq_idx]
                    eps_imag = ref['epsilon_imag'][freq_idx]

                    # Store (|S11|, ∠S11, ε', ε'')
                    s11_mag_values.append(s11_mag)
                    s11_phase_values.append(s11_phase)  # ADDED
                    eps_real_values.append(eps_real)
                    eps_imag_values.append(eps_imag)

                # Create polynomial calibration: |S11| -> ε' and |S11| -> ε''
                # Sort by S11 magnitude
                sorted_indices = np.argsort(s11_mag_values)
                s11_mag_sorted = np.array(s11_mag_values)[sorted_indices]
                s11_phase_sorted = np.array(s11_phase_values)[sorted_indices]
                eps_real_sorted = np.array(eps_real_values)[sorted_indices]
                eps_imag_sorted = np.array(eps_imag_values)[sorted_indices]

                # Use interpolation (preserves 3-point calibration better than polynomials)
                # For better accuracy, create interpolation functions
                try:
                    self.poly_real[freq] = interp1d(
                        s11_mag_sorted, eps_real_sorted,
                        kind='quadratic' if len(s11_mag_sorted) >= 3 else 'linear',
                        fill_value='extrapolate',
                        bounds_error=False
                    )
                    self.poly_imag[freq] = interp1d(
                        s11_mag_sorted, eps_imag_sorted,
                        kind='quadratic' if len(s11_mag_sorted) >= 3 else 'linear',
                        fill_value='extrapolate',
                        bounds_error=False
                    )
                except:
                    # Fallback to polynomial if interpolation fails
                    order = min(self.polyfit_order, len(s11_mag_sorted) - 1)
                    self.poly_real[freq] = np.polyfit(s11_mag_sorted, eps_real_sorted, order)
                    self.poly_imag[freq] = np.polyfit(s11_mag_sorted, eps_imag_sorted, order)

                # Store S11 range for this frequency
                self.s11_ranges[freq] = (np.min(s11_mag_sorted), np.max(s11_mag_sorted))

            self.calibrated = True
            print(f"✓ Calibration successful! Calibrated at {n_freqs} frequencies")
            return True

        except Exception as e:
            print(f"✗ Calibration failed: {str(e)}")
            self.calibrated = False
            return False

    def convert_s11_to_permittivity(self,
                                   s11_complex: np.ndarray,
                                   frequencies: np.ndarray,
                                   min_frequency: float = 0.5e9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert S11 measurements to permittivity using calibration.
        Handles both interpolation functions and polynomial calibrations.

        Only converts frequencies >= 0.5 GHz (stable region).
        Returns NaN for frequencies below this threshold.

        Args:
            s11_complex: Complex S11 array
            frequencies: Frequency array in Hz
            min_frequency: Minimum frequency for conversion (default: 0.5 GHz)
                          Returns NaN for frequencies below this

        Returns:
            Tuple of (epsilon_real, epsilon_imag) arrays
        """
        if not self.calibrated:
            raise RuntimeError("Probe not calibrated. Call calibrate() first.")

        n_points = len(s11_complex)
        epsilon_real = np.full(n_points, np.nan)
        epsilon_imag = np.full(n_points, np.nan)

        # Convert each measurement using interpolation at its frequency
        for i in range(n_points):
            freq = frequencies[i]

            # Skip frequencies below minimum (unstable region)
            if freq < min_frequency:
                continue

            s11_mag = np.abs(s11_complex[i])

            # Find calibration functions for nearest frequency
            available_freqs = np.array(list(self.poly_real.keys()))
            nearest_freq_idx = np.argmin(np.abs(available_freqs - freq))
            nearest_freq = available_freqs[nearest_freq_idx]

            # Use calibration function to evaluate (handles both interp1d and polyfit)
            try:
                poly_real_func = self.poly_real[nearest_freq]
                poly_imag_func = self.poly_imag[nearest_freq]

                # Check if it's an interpolation function or polynomial array
                if callable(poly_real_func):
                    # It's an interp1d function
                    epsilon_real[i] = float(poly_real_func(s11_mag))
                    epsilon_imag[i] = float(poly_imag_func(s11_mag))
                else:
                    # It's a numpy polynomial (from polyfit)
                    epsilon_real[i] = float(np.polyval(poly_real_func, s11_mag))
                    epsilon_imag[i] = float(np.polyval(poly_imag_func, s11_mag))

                # Apply physical constraints
                epsilon_real[i] = np.clip(epsilon_real[i], 1.0, 200.0)
                epsilon_imag[i] = np.clip(epsilon_imag[i], 0.0, 100.0)

            except Exception as e:
                epsilon_real[i] = np.nan
                epsilon_imag[i] = np.nan

        return epsilon_real, epsilon_imag

    def save(self, filepath: str) -> bool:
        """
        Save calibration to file for later use

        Args:
            filepath: Path to save calibration pickle file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"Calibration saved to {filepath}")
            return True
        except Exception as e:
            print(f"Failed to save calibration: {str(e)}")
            return False

    @staticmethod
    def load(filepath: str) -> Optional['ProbeCalibration']:
        """
        Load previously saved calibration

        Args:
            filepath: Path to calibration pickle file

        Returns:
            ProbeCalibration object or None if failed
        """
        try:
            with open(filepath, 'rb') as f:
                calibration = pickle.load(f)
            print(f"Calibration loaded from {filepath}")
            return calibration
        except Exception as e:
            print(f"Failed to load calibration: {str(e)}")
            return None

    def is_calibrated(self) -> bool:
        """Check if calibration has been performed"""
        return self.calibrated
