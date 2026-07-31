"""Fast grain radiative transfer shared by the optimized SED and image models.

The calculations assume an optically thin dust disk.  This module handles the
grain physics only: equilibrium temperatures, absorption, thermal emission,
and scattering.  Spatial integration over an entire disk belongs to
``SED``; line-of-sight integration belongs to ``Image``.

Array-dimension notation used below:

``W``
    Number of requested wavelengths.
``S``
    Number of grain sizes used for numerical integration.
``D``
    Number of stellocentric distances.
``A``
    Number of scattering angles.

Unless stated otherwise, wavelengths are in microns, grain sizes are in
metres, stellocentric distances are in au, and returned flux densities are in
Jy before any disk-density or line-of-sight weighting.
"""

import time

import astropy.constants as cst
import numpy as np
from scipy.interpolate import interp1d

from pyGrater.constants import (
    LIGHT_SPEED_CM_S,
    MAX_SAFE_EXPONENT,
    MICRON_TO_CM,
    PLANCK_C1_CGS,
    PLANCK_C2_CM_K,
)
from pyGrater.temperatures import Temperature


try:
    from numba import njit, prange

    # Thermal emission on a complete (wavelength, distance) grid.  The grain
    # size integral and Planck evaluation are fused to avoid a W x S x D cube.
    @njit(parallel=True, cache=True)
    def _calculate_thermal_emission_by_distance_numba(
            wavelength_cm, five_log_wavelength_cm, planck_to_jy_prefactor,
            wavelength_cm_fifth_power, grain_temperatures,
            grains_survive, absorption_cross_section,
            inverse_observer_distance_squared):
        n_wavelengths = wavelength_cm.shape[0]
        n_grain_sizes, n_distances = grain_temperatures.shape
        dust_thermal_emission = np.empty((n_wavelengths, n_distances))
        for wavelength_index in prange(n_wavelengths):
            current_wavelength_cm = wavelength_cm[wavelength_index]
            current_five_log_wavelength = (
                five_log_wavelength_cm[wavelength_index])
            current_planck_prefactor = (
                planck_to_jy_prefactor[wavelength_index])
            current_wavelength_fifth_power = (
                wavelength_cm_fifth_power[wavelength_index])
            for distance_index in range(n_distances):
                emission_at_distance = 0.0
                for grain_size_index in range(n_grain_sizes):
                    if grains_survive[grain_size_index, distance_index]:
                        exponent = (
                            PLANCK_C2_CM_K
                            / (current_wavelength_cm
                               * grain_temperatures[
                                   grain_size_index, distance_index])
                            + current_five_log_wavelength)
                        if exponent > MAX_SAFE_EXPONENT:
                            continue
                        emission_at_distance += (
                            current_planck_prefactor
                            / (np.exp(exponent)
                               - current_wavelength_fifth_power)
                            * absorption_cross_section[
                                grain_size_index, wavelength_index])
                dust_thermal_emission[wavelength_index, distance_index] = (
                    emission_at_distance
                    * inverse_observer_distance_squared)
        return dust_thermal_emission

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


def _trapezoid_integration_coefficients(x):
    """Return coefficients for composite-trapezoid integration on ``x``.

    For sampled values ``f``, ``np.dot(f, coefficients)`` is numerically
    equivalent to ``np.trapezoid(f, x)`` but avoids recomputing the coefficients
    during every MCMC likelihood evaluation.
    """
    dx = np.diff(x)
    w = np.empty_like(x)
    w[0] = dx[0] / 2.0
    w[-1] = dx[-1] / 2.0
    w[1:-1] = (dx[:-1] + dx[1:]) / 2.0
    return w


def _interp_regular_grid_2d(x_grid, y_grid, values, xq, yq):
    """Bilinearly interpolate ``values`` onto the Cartesian product xq x yq."""
    x_lower_index = np.searchsorted(x_grid, xq) - 1
    y_lower_index = np.searchsorted(y_grid, yq) - 1
    x_lower_index = np.clip(x_lower_index, 0, len(x_grid) - 2)
    y_lower_index = np.clip(y_lower_index, 0, len(y_grid) - 2)

    x_lower = x_grid[x_lower_index]
    x_upper = x_grid[x_lower_index + 1]
    y_lower = y_grid[y_lower_index]
    y_upper = y_grid[y_lower_index + 1]
    x_fraction = ((xq - x_lower) / (x_upper - x_lower))[:, None]
    y_fraction = ((yq - y_lower) / (y_upper - y_lower))[None, :]

    x_index_2d = x_lower_index[:, None]
    y_index_2d = y_lower_index[None, :]
    return (
        (1.0 - x_fraction) * (1.0 - y_fraction)
        * values[x_index_2d, y_index_2d]
        + x_fraction * (1.0 - y_fraction)
        * values[x_index_2d + 1, y_index_2d]
        + (1.0 - x_fraction) * y_fraction
        * values[x_index_2d, y_index_2d + 1]
        + x_fraction * y_fraction
        * values[x_index_2d + 1, y_index_2d + 1])


def _interp_rows_by_x(x_grid, values, xq):
    """Interpolate the first axis of a 2-D array and preserve its columns."""
    lower_index = np.searchsorted(x_grid, xq) - 1
    lower_index = np.clip(lower_index, 0, len(x_grid) - 2)
    lower_x = x_grid[lower_index]
    upper_x = x_grid[lower_index + 1]
    fraction_above = ((xq - lower_x) / (upper_x - lower_x))[:, None]
    return ((1.0 - fraction_above) * values[lower_index]
            + fraction_above * values[lower_index + 1])


def _precompute_interpolation_on_second_axis(y_grid, yq):
    """Precompute grid cells and fractions for repeated interpolation at yq."""
    lower_index = np.searchsorted(y_grid, yq) - 1
    lower_index = np.clip(lower_index, 0, len(y_grid) - 2)
    lower_y = y_grid[lower_index]
    upper_y = y_grid[lower_index + 1]
    fraction_above = (yq - lower_y) / (upper_y - lower_y)
    return lower_index, fraction_above


def _interp_two_regular_grid_2d_with_y(x_grid, values_a, values_b,
                                       xq, y_lower_index, y_fraction_above):
    """Bilinearly interpolate two arrays that share a precomputed second axis."""
    x_lower_index = np.searchsorted(x_grid, xq) - 1
    x_lower_index = np.clip(x_lower_index, 0, len(x_grid) - 2)
    x_lower = x_grid[x_lower_index]
    x_upper = x_grid[x_lower_index + 1]
    x_fraction_above = ((xq - x_lower) / (x_upper - x_lower))[:, None]
    y_fraction_above = y_fraction_above[None, :]
    x_index_2d = x_lower_index[:, None]
    y_index_2d = y_lower_index[None, :]

    lower_x_lower_y = (1.0 - x_fraction_above) * (1.0 - y_fraction_above)
    upper_x_lower_y = x_fraction_above * (1.0 - y_fraction_above)
    lower_x_upper_y = (1.0 - x_fraction_above) * y_fraction_above
    upper_x_upper_y = x_fraction_above * y_fraction_above

    interpolated_a = (
        lower_x_lower_y * values_a[x_index_2d, y_index_2d]
        + upper_x_lower_y * values_a[x_index_2d + 1, y_index_2d]
        + lower_x_upper_y * values_a[x_index_2d, y_index_2d + 1]
        + upper_x_upper_y * values_a[x_index_2d + 1, y_index_2d + 1])
    interpolated_b = (
        lower_x_lower_y * values_b[x_index_2d, y_index_2d]
        + upper_x_lower_y * values_b[x_index_2d + 1, y_index_2d]
        + lower_x_upper_y * values_b[x_index_2d, y_index_2d + 1]
        + upper_x_upper_y * values_b[x_index_2d + 1, y_index_2d + 1])
    return interpolated_a, interpolated_b


class Fluxes:
    """Calculate grain emission before integrating over the disk geometry.

    Initialization loads the temperature table and precomputes every quantity
    that depends only on the star, material, wavelength grid, or distance grid.
    A model evaluation therefore only updates grain-size-dependent quantities.
    """

    def __init__(self, grain, star, wavelengths_for_calc,
                 size_distribution_function, scattering_phase_function,
                 N_temp=600, N_distances=400, dist_max_input=1000,
                 N_scattering_angles=500):
        self.absorption_efficiency_grid = grain.Qabs
        self.scattering_efficiency_grid = grain.Qsca
        self.optical_property_grain_sizes_micron = grain.Qabs_sizes
        self.optical_property_wavelengths_micron = grain.Qabs_waves

        self.temperature_model = Temperature(grain, star, N_temp=N_temp)
        self.stellar_spectrum_wavelengths_micron = star.waves
        self.stellar_spectrum_flux_jy = star.flux
        self.sublimation_temperature_k = grain.Tsub
        self.observer_distance_m = star.distance * cst.pc.value

        dist_max = np.min([
            dist_max_input, np.max(self.temperature_model.therm_dist)])
        self.stellocentric_distances_au = np.geomspace(
            np.min(self.temperature_model.therm_dist), dist_max, N_distances)
        self.grain_temperatures_by_size_and_distance = (
            self.temperature_model.get_temperature(
                self.stellocentric_distances_au))

        self.stellar_spectrum_interpolator = interp1d(
            self.stellar_spectrum_wavelengths_micron,
            self.stellar_spectrum_flux_jy,
            kind='linear', bounds_error=False, fill_value=0)
        self.model_wavelengths_micron = np.asarray(
            wavelengths_for_calc, dtype=np.float64)
        self.size_distribution_function = size_distribution_function
        self.scattering_phase_function = scattering_phase_function
        self.n_scattering_angles = N_scattering_angles
        self.scattering_angles_radian = np.linspace(
            0, np.pi, N_scattering_angles)
        self.sizes_for_integral = None
        self.timings = {}
        self._wavelength_cm = np.ascontiguousarray(
            self.model_wavelengths_micron * MICRON_TO_CM)
        self._five_log_wavelength_cm = np.ascontiguousarray(
            5.0 * np.log(self._wavelength_cm))
        self._wavelength_cm_fifth_power = np.ascontiguousarray(
            self._wavelength_cm**5)
        self._planck_to_jy_prefactor = np.ascontiguousarray(
            PLANCK_C1_CGS * self.model_wavelengths_micron**2
            / LIGHT_SPEED_CM_S * 1e15)
        self._stellar_flux_at_model_wavelengths = np.asarray(
            self.stellar_spectrum_interpolator(
                self.model_wavelengths_micron))
        self._zero_scattering_angle_radian = np.array([0.0])
        self._stellocentric_distance_squared_m2 = (
            self.stellocentric_distances_au * cst.au.value)**2
        self._inverse_observer_distance_squared_m2 = (
            1.0 / self.observer_distance_m**2)
        (self._optical_wavelength_grid_index,
         self._optical_wavelength_interpolation_fraction) = (
            _precompute_interpolation_on_second_axis(
                self.optical_property_wavelengths_micron,
                self.model_wavelengths_micron))

    def interpolate_grain_temperatures(
            self, grain_sizes_m, stellocentric_distances_au):
        """Return grain temperatures with shape ``(S, D)`` in kelvin."""
        return _interp_regular_grid_2d(
            self.optical_property_grain_sizes_micron,
            self.stellocentric_distances_au,
            self.grain_temperatures_by_size_and_distance,
            grain_sizes_m / 1e-6,
            stellocentric_distances_au)

    def interpolate_absorption_efficiency(self, grain_sizes_m):
        """Return Q_abs with shape ``(S, W)``."""
        return _interp_regular_grid_2d(
            self.optical_property_grain_sizes_micron,
            self.optical_property_wavelengths_micron,
            self.absorption_efficiency_grid,
            grain_sizes_m / 1e-6,
            self.model_wavelengths_micron)

    def interpolate_scattering_efficiency(self, grain_sizes_m):
        """Return Q_sca with shape ``(S, W)``."""
        return _interp_regular_grid_2d(
            self.optical_property_grain_sizes_micron,
            self.optical_property_wavelengths_micron,
            self.scattering_efficiency_grid,
            grain_sizes_m / 1e-6,
            self.model_wavelengths_micron)

    def planck_function_jy(self, wavelengths_micron, temperatures_k):
        """Evaluate the frequency-form Planck function and return Jy."""
        wavelength_cm = wavelengths_micron * MICRON_TO_CM
        exponent = (PLANCK_C2_CM_K / wavelength_cm / temperatures_k
                    + 5.0 * np.log(wavelength_cm))
        safe_exponent = np.minimum(exponent, MAX_SAFE_EXPONENT)
        planck_cgs = PLANCK_C1_CGS / (
            np.exp(safe_exponent) - wavelength_cm**5)
        conversion_to_jy = (
            wavelengths_micron**2 / LIGHT_SPEED_CM_S * 1e15)
        return planck_cgs * conversion_to_jy

    def _prepare_grain_size_integration(self, model_parameters):
        """Build the grain-size grid and reusable geometric cross section."""
        grain_sizes_m = np.geomspace(
            model_parameters['a_min'],
            model_parameters['a_max'],
            model_parameters['N_sizes_integral'])
        self.sizes_for_integral = grain_sizes_m
        size_distribution = self.size_distribution_function(
            grain_sizes_m, model_parameters)
        geometric_cross_section_distribution = (
            np.pi * grain_sizes_m**2 * size_distribution)
        grain_size_integration_coefficients = (
            _trapezoid_integration_coefficients(grain_sizes_m))
        return (grain_sizes_m, geometric_cross_section_distribution,
                grain_size_integration_coefficients)

    def calculate_dust_thermal_emission(self, model_parameters):
        """Return thermal emission with shape ``(W, D)``.

        The returned array is the size-integrated grain emission evaluated at
        each stellocentric distance.  Grains hotter than the material's
        sublimation temperature contribute zero emission.
        """
        t_start = time.perf_counter()
        (grain_sizes_m, geometric_cross_section_distribution,
         grain_size_integration_coefficients) = (
            self._prepare_grain_size_integration(model_parameters))

        grain_temperatures = self.interpolate_grain_temperatures(
            grain_sizes_m, self.stellocentric_distances_au)
        t_temps = time.perf_counter()
        absorption_efficiency = self.interpolate_absorption_efficiency(
            grain_sizes_m)
        t_qinterp = time.perf_counter()

        grains_survive = (
            grain_temperatures <= self.sublimation_temperature_k)
        absorption_cross_section = absorption_efficiency * (
            geometric_cross_section_distribution
            * grain_size_integration_coefficients)[:, None]

        if _NUMBA_AVAILABLE:
            dust_thermal_emission = (
                _calculate_thermal_emission_by_distance_numba(
                    self._wavelength_cm,
                    self._five_log_wavelength_cm,
                    self._planck_to_jy_prefactor,
                    self._wavelength_cm_fifth_power,
                    np.ascontiguousarray(grain_temperatures),
                    np.ascontiguousarray(grains_survive),
                    np.ascontiguousarray(absorption_cross_section),
                    self._inverse_observer_distance_squared_m2))
        else:
            n_wav = len(self.model_wavelengths_micron)
            n_dist = len(self.stellocentric_distances_au)
            dust_thermal_emission = np.zeros((n_wav, n_dist))
            chunk = 50
            for i0 in range(0, n_wav, chunk):
                i1 = min(i0 + chunk, n_wav)
                wavelength_chunk = self.model_wavelengths_micron[i0:i1]
                planck_emission = self.planck_function_jy(
                    wavelength_chunk[:, None, None],
                    grain_temperatures[None, :, :])
                planck_emission *= grains_survive[None, :, :]
                dust_thermal_emission[i0:i1] = np.einsum(
                    'wsd,sw->wd', planck_emission,
                    absorption_cross_section[:, i0:i1])
                dust_thermal_emission[i0:i1] *= (
                    self._inverse_observer_distance_squared_m2)

        t_end = time.perf_counter()
        self.timings['dust_temperature_interpolation'] = t_temps - t_start
        self.timings['absorption_efficiency_interpolation'] = (
            t_qinterp - t_temps)
        self.timings['planck_emission_and_grain_size_integration'] = (
            t_end - t_qinterp)
        self.timings['dust_thermal_emission_total'] = t_end - t_start
        return dust_thermal_emission

    def thermal_flux(self, params):
        """Return thermal flux with shape ``(wavelength, distance)``.

        This is a public diagnostic convenience method. ``SED`` and
        ``Image`` do not call it; they use specialized combined calculation
        paths to avoid unnecessary arrays and repeated work.
        """
        return self.calculate_dust_thermal_emission(params)

    def calculate_scattered_starlight_for_all_angles(
            self, model_parameters, scattering_parameters=None):
        """Return the two factors needed for scattered-light images.

        Returns
        -------
        scattered_starlight_before_angular_dependence : ndarray, shape (W, D)
            Scattered stellar light including stellar illumination, Q_sca,
            grain-size integration, and sublimation, but not the phase function.
        scattering_phase_function_by_angle : ndarray, shape (A,)
            Phase-function value at each tabulated scattering angle.

        Multiplying these arrays as ``first[:, :, None] * second[None, None, :]``
        gives scattered light as a function of wavelength, distance, and angle
        without storing that much larger cube during normal image calculations.
        """
        if scattering_parameters is None:
            scattering_parameters = model_parameters
        t_start = time.perf_counter()
        (grain_sizes_m, geometric_cross_section_distribution,
         grain_size_integration_coefficients) = (
            self._prepare_grain_size_integration(model_parameters))
        grain_temperatures = self.interpolate_grain_temperatures(
            grain_sizes_m, self.stellocentric_distances_au)
        grains_survive = (
            grain_temperatures <= self.sublimation_temperature_k)
        scattering_efficiency = self.interpolate_scattering_efficiency(
            grain_sizes_m)
        t_q = time.perf_counter()

        surviving_geometric_cross_section = (
            geometric_cross_section_distribution[:, None] * grains_survive)
        scattered_starlight_before_angular_dependence = (
            (scattering_efficiency
             * grain_size_integration_coefficients[:, None]).T
            @ surviving_geometric_cross_section)
        scattered_starlight_before_angular_dependence *= (
            self._stellar_flux_at_model_wavelengths[:, None]
            / self._stellocentric_distance_squared_m2[None, :])
        scattering_phase_function_by_angle = self.scattering_phase_function(
            self.scattering_angles_radian, **scattering_parameters)
        t_end = time.perf_counter()
        self.timings['scattering_efficiency_interpolation'] = (
            t_q - t_start)
        self.timings['scattered_starlight_grain_size_integration'] = (
            t_end - t_q)
        self.timings['scattered_starlight_total'] = t_end - t_start
        return (scattered_starlight_before_angular_dependence,
                scattering_phase_function_by_angle)

    def scattered_flux(self, params):
        """Return scattered flux with shape ``(wavelength, distance, angle)``.

        This diagnostic method materializes the complete three-dimensional
        array. ``SED`` and ``Image`` intentionally do not call it;
        their optimized paths keep distance and angular factors separate.
        """
        (distance_dependent_scattered_starlight,
         scattering_phase_function_by_angle) = (
            self.calculate_scattered_starlight_for_all_angles(params, params))
        return (
            distance_dependent_scattered_starlight[:, :, None]
            * scattering_phase_function_by_angle[None, None, :])

    def calculate_emission_for_image(self, model_parameters):
        """Return thermal and scattered-light terms required by ``Image``."""
        dust_thermal_emission_by_distance = (
            self.calculate_dust_thermal_emission(model_parameters))
        (scattered_starlight_before_angular_dependence,
         scattering_phase_function_by_angle) = (
            self.calculate_scattered_starlight_for_all_angles(
                model_parameters, model_parameters)
        )
        return (dust_thermal_emission_by_distance,
                scattered_starlight_before_angular_dependence,
                scattering_phase_function_by_angle)

    def calculate_emission_by_stellocentric_distance(
            self, model_parameters):
        """Return ``(thermal, scattered)`` arrays, each with shape ``(W, D)``.

        Scattered starlight is evaluated at zero scattering angle, matching
        the isotropic convention used by the disk-integrated SED.
        """
        (dust_thermal_emission_by_distance,
         scattered_starlight_before_angular_dependence,
         scattering_phase_function_by_angle) = (
            self.calculate_emission_for_image(model_parameters))
        scattered_starlight_by_distance = (
            scattered_starlight_before_angular_dependence
            * scattering_phase_function_by_angle[0])
        return (dust_thermal_emission_by_distance,
                scattered_starlight_by_distance)
