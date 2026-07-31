"""Fast disk-integrated SEDs for optically thin circumstellar dust.

``SED`` combines two independent parts of the model:

1. The disk density model is integrated over cylindrical radius and height.
   This produces the amount of density-volume represented at each true
   stellocentric distance.
2. Grain radiative transfer supplies thermal dust emission and scattered
   starlight at those distances.  The two parts are contracted to produce the
   unresolved spectral energy distribution.

The direct calculation avoids constructing wavelength-by-distance emission
tables and is the default for fitting.  Explicit distance tables remain
available for diagnostics and for checking consistency with ``Image``.

Conventions
-----------
Wavelengths are in microns, grain sizes in metres, spatial distances in au,
flux densities in Jy, and returned dust masses in Earth masses.  The model is
optically thin and the disk-integrated SED uses isotropic scattering.
"""
import logging

import time

import astropy.constants as cst
import numpy as np

from pyGrater.constants import (

    DEFAULT_DENSITY_CUTOFF,
    LARGE_WAVELENGTH_GRID_THRESHOLD,
    MAX_SAFE_EXPONENT,
    MINIMUM_SCALED_HEIGHT,
    N_VERTICAL_GRID_POINTS,
    PLANCK_C2_CM_K,
)
from pyGrater.fluxes import (
    Fluxes,
    _NUMBA_AVAILABLE,
    _interp_rows_by_x,
    _interp_two_regular_grid_2d_with_y,
    _trapezoid_integration_coefficients,
)
from pyGrater.phase_functions import isotropic
from pyGrater.utils import (
    calculate_normalization_density_jacobian_sublimation_fast,
)



logger = logging.getLogger(__name__)
if _NUMBA_AVAILABLE:
    from numba import njit, prange

    @njit(cache=True)
    def _deposit_disk_volume_from_distances(
            flattened_stellocentric_distance_au, distance_grid_au,
            vertical_density_volume, radial_integration_coefficients,
            n_scaled_heights):
        """Find bins and deposit density-volume in one compiled pass."""
        output = np.zeros(distance_grid_au.size)
        minimum_distance = distance_grid_au[0]
        maximum_distance = distance_grid_au[-1]
        for radius_index in range(radial_integration_coefficients.size):
            radial_coefficient = radial_integration_coefficients[radius_index]
            row_start = radius_index * n_scaled_heights
            for height_index in range(n_scaled_heights):
                flat_index = row_start + height_index
                distance = flattened_stellocentric_distance_au[flat_index]
                if minimum_distance <= distance <= maximum_distance:
                    lower_bin = (
                        np.searchsorted(distance_grid_au, distance) - 1)
                    if lower_bin < 0:
                        lower_bin = 0
                    elif lower_bin >= distance_grid_au.size - 1:
                        lower_bin = distance_grid_au.size - 2
                    fraction_above = (
                        (distance - distance_grid_au[lower_bin])
                        / (distance_grid_au[lower_bin + 1]
                           - distance_grid_au[lower_bin]))
                    contribution = (
                        vertical_density_volume[flat_index]
                        * radial_coefficient)
                    output[lower_bin] += (
                        contribution * (1.0 - fraction_above))
                    output[lower_bin + 1] += (
                        contribution * fraction_above)
        return output

    # Direct SED path for small and medium W: integrate Planck emission over
    # distance and grain size without constructing emission-by-distance arrays.
    @njit(parallel=True, cache=True)
    def _integrated_dust_thermal_emission(
            wavelength_cm, five_log_wavelength_cm,
            wavelength_cm_fifth_power, planck_to_jy_prefactor,
            grain_temperatures, grains_survive, absorption_cross_section,
            disk_density_volume_by_distance,
            inverse_observer_distance_squared):
        n_wavelengths = wavelength_cm.shape[0]
        n_grain_sizes, n_distances = grain_temperatures.shape
        dust_thermal_sed = np.empty(n_wavelengths)
        for wavelength_index in prange(n_wavelengths):
            current_wavelength_cm = wavelength_cm[wavelength_index]
            current_five_log_wavelength = (
                five_log_wavelength_cm[wavelength_index])
            current_wavelength_fifth_power = (
                wavelength_cm_fifth_power[wavelength_index])
            current_planck_prefactor = (
                planck_to_jy_prefactor[wavelength_index])
            emission_at_wavelength = 0.0
            for grain_size_index in range(n_grain_sizes):
                current_absorption_cross_section = absorption_cross_section[
                    grain_size_index, wavelength_index]
                emission_for_grain_size = 0.0
                for distance_index in range(n_distances):
                    if grains_survive[grain_size_index, distance_index]:
                        exponent = (
                            PLANCK_C2_CM_K
                            / (current_wavelength_cm
                               * grain_temperatures[
                                   grain_size_index, distance_index])
                            + current_five_log_wavelength)
                        if exponent <= MAX_SAFE_EXPONENT:
                            emission_for_grain_size += (
                                current_planck_prefactor
                                / (np.exp(exponent)
                                   - current_wavelength_fifth_power)
                                * disk_density_volume_by_distance[
                                    distance_index])
                emission_at_wavelength += (
                    current_absorption_cross_section
                    * emission_for_grain_size)
            dust_thermal_sed[wavelength_index] = (
                emission_at_wavelength * inverse_observer_distance_squared)
        return dust_thermal_sed

    # Direct SED path for large W: expose wavelength x grain-size jobs to
    # Numba, then finish the grain-size integral with NumPy.
    @njit(parallel=True, cache=True)
    def _planck_emission_integrated_over_disk(
            wavelength_cm, five_log_wavelength_cm,
            wavelength_cm_fifth_power, planck_to_jy_prefactor,
            grain_temperatures, grains_survive,
            disk_density_volume_by_distance):
        n_wavelengths = wavelength_cm.shape[0]
        n_grain_sizes, n_distances = grain_temperatures.shape
        emission_by_wavelength_and_size = np.empty(
            (n_wavelengths, n_grain_sizes))
        for job_index in prange(n_wavelengths * n_grain_sizes):
            wavelength_index = job_index // n_grain_sizes
            grain_size_index = job_index - wavelength_index * n_grain_sizes
            current_wavelength_cm = wavelength_cm[wavelength_index]
            current_five_log_wavelength = (
                five_log_wavelength_cm[wavelength_index])
            current_wavelength_fifth_power = (
                wavelength_cm_fifth_power[wavelength_index])
            current_planck_prefactor = (
                planck_to_jy_prefactor[wavelength_index])
            emission_for_grain_size = 0.0
            for distance_index in range(n_distances):
                if grains_survive[grain_size_index, distance_index]:
                    exponent = (
                        PLANCK_C2_CM_K
                        / (current_wavelength_cm
                           * grain_temperatures[
                               grain_size_index, distance_index])
                        + current_five_log_wavelength)
                    if exponent <= MAX_SAFE_EXPONENT:
                        emission_for_grain_size += (
                            current_planck_prefactor
                            / (np.exp(exponent)
                               - current_wavelength_fifth_power)
                            * disk_density_volume_by_distance[distance_index])
            emission_by_wavelength_and_size[
                wavelength_index, grain_size_index] = emission_for_grain_size
        return emission_by_wavelength_and_size


class SED:
    """Calculate thermal and scattered-light disk SEDs quickly."""

    def __init__(self, grain, star, density_function,
                 size_distribution_function, wavelengths_for_calc,
                 N_distances=800):
        self.grain = grain
        self.star = star
        self.density_function = density_function
        self.size_distribution_function = size_distribution_function
        self.scattering_phase_function = isotropic
        self.model_wavelengths_micron = np.asarray(
            wavelengths_for_calc, dtype=np.float64)
        self.radiative_transfer = Fluxes(
            grain, star, self.model_wavelengths_micron,
            size_distribution_function, isotropic,
            N_distances=N_distances)
        self.absorption_efficiency_grid = (
            self.radiative_transfer.absorption_efficiency_grid)
        self.scattering_efficiency_grid = (
            self.radiative_transfer.scattering_efficiency_grid)
        self.optical_property_grain_sizes_micron = (
            self.radiative_transfer.optical_property_grain_sizes_micron)
        self.temperature_model = self.radiative_transfer.temperature_model
        self.sublimation_temperature_k = (
            self.radiative_transfer.sublimation_temperature_k)
        self.stellocentric_distances_au = (
            self.radiative_transfer.stellocentric_distances_au)
        self.grain_temperatures_by_size_and_distance = (
            self.radiative_transfer.grain_temperatures_by_size_and_distance)
        self.sizes_for_integral = None
        self.radiative_transfer_timings = (
            self.radiative_transfer.timings)

        half_vertical_grid = (N_VERTICAL_GRID_POINTS - 1) // 2
        positive_scaled_heights = np.geomspace(
            MINIMUM_SCALED_HEIGHT, 1.0, half_vertical_grid)
        self.scaled_vertical_coordinate = np.concatenate(
            (-positive_scaled_heights[::-1], [0.0],
             positive_scaled_heights))
        self.vertical_integration_coefficients = (
            _trapezoid_integration_coefficients(
                self.scaled_vertical_coordinate))

        self.timings = {}

    def _build_vertical_grid(self, cylindrical_radius_au):
        """Return height, maximum height, and true distance on the disk grid."""
        maximum_height_au = self.vertical_grid_scale_au * np.sqrt(
            1 + cylindrical_radius_au**2 / self.maximum_disk_radius_au**2)
        height_above_midplane_au = (
            maximum_height_au[:, None]
            * self.scaled_vertical_coordinate[None, :])
        stellocentric_distance_au = np.sqrt(
            cylindrical_radius_au[:, None]**2
            + height_above_midplane_au**2)
        return (height_above_midplane_au, maximum_height_au,
                stellocentric_distance_au)

    def _build_spatial_integration_grid(self, model_parameters):
        """Construct the cylindrical disk grid used for density integration."""
        reference_radius_au = model_parameters['r0']
        inner_slope = model_parameters['alphain']
        outer_slope = model_parameters['alphaout']
        flaring_exponent = model_parameters['beta']
        vertical_exponent = model_parameters['gamma']
        scale_height_at_reference_radius_au = model_parameters['h0']
        
        
        density_cutoff = model_parameters.get(
            'p_cutoff', DEFAULT_DENSITY_CUTOFF)
        maximum_disk_radius_au = model_parameters.get(
            'rmax', reference_radius_au * density_cutoff ** (1.0 / outer_slope))
        
        self.maximum_disk_radius_au = maximum_disk_radius_au
        cylindrical_radius_au = np.geomspace(
            self.stellocentric_distances_au.min(), maximum_disk_radius_au,
            len(self.stellocentric_distances_au))

        effective_inner_slope = inner_slope + flaring_exponent
        effective_outer_slope = outer_slope + flaring_exponent
        peak_radius_au = (
            -effective_inner_slope / effective_outer_slope
        ) ** (1.0 / (2 * effective_inner_slope
                     - 2 * effective_outer_slope)) * reference_radius_au
        height_at_peak_radius_au = (
            scale_height_at_reference_radius_au
            * (peak_radius_au / reference_radius_au)**flaring_exponent
            * np.log(1.0 / density_cutoff)**(1.0 / vertical_exponent))
        self.vertical_grid_scale_au = height_at_peak_radius_au / np.sqrt(
            peak_radius_au**2 / maximum_disk_radius_au**2 + 1)

        (height_above_midplane_au, maximum_height_au,
         stellocentric_distance_au) = self._build_vertical_grid(
            cylindrical_radius_au)
        return (cylindrical_radius_au, height_above_midplane_au,
                maximum_height_au, stellocentric_distance_au)

    def get_total_mass(self, **model_parameters):
        """Return the dust mass in Earth masses for the supplied normalization.

        If ``M_tot`` is supplied, it is already the requested mass and is
        returned directly.  Otherwise the method integrates the density and
        grain-size distributions corresponding to ``A_norm`` while excluding
        grains above their sublimation temperature.
        """
        if 'M_tot' in model_parameters:
            return float(model_parameters['M_tot'])
        normalization_constant = model_parameters.get('A_norm', None)
        if normalization_constant is None:
            raise ValueError('get_total_mass requires either A_norm or M_tot.')

        grain_sizes_m = np.geomspace(
            model_parameters['a_min'], model_parameters['a_max'],
            model_parameters['N_sizes_integral'])
        self.sizes_for_integral = grain_sizes_m
        grain_bulk_density_kg_m3 = (
            self.grain.grain_properties['Density'] * 1000)
        (cylindrical_radius_au, height_above_midplane_au,
         maximum_height_au, stellocentric_distance_au) = (
            self._build_spatial_integration_grid(model_parameters))
        if self.maximum_disk_radius_au <= self.stellocentric_distances_au[0]:
            # The modeled disk ends before this composition has any tabulated
            # surviving grains. Integrating a descending radial grid would
            # create a small, unphysical negative mass.
            return 0.0

        cylindrical_radius_column_au = cylindrical_radius_au[:, None]
        relative_number_density = self.density_function(
            cylindrical_radius_column_au, 0., height_above_midplane_au,
            model_parameters)
        density_times_cylindrical_volume = (
            relative_number_density
            * 2.0 * np.pi * cylindrical_radius_column_au
            * maximum_height_au[:, None])
        vertical_density_volume = (
            density_times_cylindrical_volume
            * self.vertical_integration_coefficients[None, :]).ravel()
        radial_integration_coefficients = (
            _trapezoid_integration_coefficients(cylindrical_radius_au))
        disk_density_volume_by_distance = (
            self._disk_density_volume_by_stellocentric_distance(
                cylindrical_radius_au, stellocentric_distance_au,
                vertical_density_volume,
                radial_integration_coefficients))

        grain_temperatures = _interp_rows_by_x(
            self.optical_property_grain_sizes_micron,
            self.grain_temperatures_by_size_and_distance,
            grain_sizes_m / 1e-6)
        grains_survive = (
            grain_temperatures <= self.sublimation_temperature_k)
        size_distribution = self.size_distribution_function(
            grain_sizes_m, model_parameters)
        grain_mass_distribution = (
            (4.0 * np.pi / 3.0) * grain_bulk_density_kg_m3 * grain_sizes_m**3
            * size_distribution
            * _trapezoid_integration_coefficients(grain_sizes_m))
        surviving_disk_volume_by_size = (
            grains_survive @ disk_density_volume_by_distance)
        total_mass = normalization_constant * np.dot(
            grain_mass_distribution, surviving_disk_volume_by_size)
        return total_mass / cst.M_earth.value

    def _disk_density_volume_by_stellocentric_distance(
            self, cylindrical_radius_au, stellocentric_distance_au,
            vertical_density_volume,
            radial_integration_coefficients):
        """Deposit every cylindrical grid cell into true-distance bins.

        A cell generally lies between two tabulated stellocentric distances.
        Its density-volume contribution is therefore divided linearly between
        those adjacent bins.  The result has shape ``(D,)``.
        """
        flattened_stellocentric_distance_au = (
            stellocentric_distance_au.ravel())
        distance_grid_au = self.stellocentric_distances_au
        if _NUMBA_AVAILABLE:
            return _deposit_disk_volume_from_distances(
                np.ascontiguousarray(flattened_stellocentric_distance_au),
                np.ascontiguousarray(distance_grid_au),
                np.ascontiguousarray(vertical_density_volume),
                np.ascontiguousarray(radial_integration_coefficients),
                len(self.scaled_vertical_coordinate))

        lower_distance_bin = np.searchsorted(
            distance_grid_au, flattened_stellocentric_distance_au) - 1
        lower_distance_bin = np.clip(
            lower_distance_bin, 0, len(distance_grid_au) - 2)
        inside_distance_grid = (
            (flattened_stellocentric_distance_au >= distance_grid_au[0])
            & (flattened_stellocentric_distance_au <= distance_grid_au[-1]))
        fraction_above = (
            (flattened_stellocentric_distance_au
             - distance_grid_au[lower_distance_bin])
            / (distance_grid_au[lower_distance_bin + 1]
               - distance_grid_au[lower_distance_bin]))
        fraction_below = 1.0 - fraction_above
        density_volume_at_grid_points = (
            vertical_density_volume
            * np.repeat(radial_integration_coefficients,
                        len(self.scaled_vertical_coordinate)))
        disk_density_volume_by_distance = np.bincount(
            lower_distance_bin[inside_distance_grid],
            weights=(density_volume_at_grid_points[inside_distance_grid]
                     * fraction_below[inside_distance_grid]),
            minlength=len(distance_grid_au))
        disk_density_volume_by_distance += np.bincount(
            lower_distance_bin[inside_distance_grid] + 1,
            weights=(density_volume_at_grid_points[inside_distance_grid]
                     * fraction_above[inside_distance_grid]),
            minlength=len(distance_grid_au))
        return disk_density_volume_by_distance

    def calculate_disk_integrated_sed(
            self, model_parameters, disk_density_volume_by_distance,
            keep_separate_fluxes):
        """Calculate the SED without constructing emission-by-distance arrays.

        This is the optimized fitting path.  Thermal emission is contracted
        directly over grain size and stellocentric distance.  Scattered light
        is contracted through the total illuminated grain cross section.
        """
        t0 = time.perf_counter()
        radiative_transfer = self.radiative_transfer
        grain_sizes_m = np.geomspace(
            model_parameters['a_min'], model_parameters['a_max'],
            model_parameters['N_sizes_integral'])
        self.sizes_for_integral = grain_sizes_m
        radiative_transfer.sizes_for_integral = grain_sizes_m

        size_distribution = self.size_distribution_function(
            grain_sizes_m, model_parameters)
        grain_cross_section_distribution = (
            np.pi * grain_sizes_m**2 * size_distribution)
        grain_size_integration_coefficients = (
            _trapezoid_integration_coefficients(grain_sizes_m))
        grain_temperatures = _interp_rows_by_x(
            self.optical_property_grain_sizes_micron,
            self.grain_temperatures_by_size_and_distance,
            grain_sizes_m / 1e-6)
        grains_below_sublimation_temperature = (
            grain_temperatures <= self.sublimation_temperature_k)
        t_temperatures = time.perf_counter()

        absorption_efficiency, scattering_efficiency = (
            _interp_two_regular_grid_2d_with_y(
                self.optical_property_grain_sizes_micron,
                self.absorption_efficiency_grid,
                self.scattering_efficiency_grid,
                grain_sizes_m / 1e-6,
                radiative_transfer._optical_wavelength_grid_index,
                radiative_transfer
                ._optical_wavelength_interpolation_fraction))
        t_optical_efficiencies = time.perf_counter()

        absorption_cross_section_by_size_and_wavelength = (
            absorption_efficiency
            * (grain_cross_section_distribution
               * grain_size_integration_coefficients)[:, None])
        disk_density_volume_by_distance = np.ascontiguousarray(
            disk_density_volume_by_distance)

        if (_NUMBA_AVAILABLE
                and len(self.model_wavelengths_micron)
                >= LARGE_WAVELENGTH_GRID_THRESHOLD):
            planck_emission_integrated_over_disk = (
                _planck_emission_integrated_over_disk(
                    radiative_transfer._wavelength_cm,
                    radiative_transfer._five_log_wavelength_cm,
                    radiative_transfer._wavelength_cm_fifth_power,
                    radiative_transfer._planck_to_jy_prefactor,
                    np.ascontiguousarray(grain_temperatures),
                    np.ascontiguousarray(
                        grains_below_sublimation_temperature),
                    disk_density_volume_by_distance))
            dust_thermal_sed = np.sum(
                planck_emission_integrated_over_disk
                * absorption_cross_section_by_size_and_wavelength.T,
                axis=1)
            dust_thermal_sed *= (
                radiative_transfer._inverse_observer_distance_squared_m2)
        elif _NUMBA_AVAILABLE:
            dust_thermal_sed = _integrated_dust_thermal_emission(
                radiative_transfer._wavelength_cm,
                radiative_transfer._five_log_wavelength_cm,
                radiative_transfer._wavelength_cm_fifth_power,
                radiative_transfer._planck_to_jy_prefactor,
                np.ascontiguousarray(grain_temperatures),
                np.ascontiguousarray(grains_below_sublimation_temperature),
                np.ascontiguousarray(
                    absorption_cross_section_by_size_and_wavelength),
                disk_density_volume_by_distance,
                radiative_transfer._inverse_observer_distance_squared_m2)
        else:
            wavelength_cm = self.model_wavelengths_micron * 1e-4
            planck_emission = (
                radiative_transfer._planck_to_jy_prefactor[:, None, None]
                / (
                np.exp(np.minimum(
                    1.43983 / (wavelength_cm[:, None, None]
                               * grain_temperatures[None, :, :])
                    + 5.0 * np.log(wavelength_cm)[:, None, None], 709.0))
                - wavelength_cm[:, None, None]**5))
            planck_emission *= (
                grains_below_sublimation_temperature[None, :, :])
            dust_thermal_sed = np.einsum(
                'wsd,sw,d->w', planck_emission,
                absorption_cross_section_by_size_and_wavelength,
                disk_density_volume_by_distance)
            dust_thermal_sed *= (
                radiative_transfer._inverse_observer_distance_squared_m2)

        t_dust_thermal_emission = time.perf_counter()
        stellar_illumination_by_distance = (
            disk_density_volume_by_distance
            / radiative_transfer._stellocentric_distance_squared_m2)
        scattering_cross_section_by_size = (
            grain_cross_section_distribution
            * grain_size_integration_coefficients
            * (grains_below_sublimation_temperature
               @ stellar_illumination_by_distance))
        scattered_starlight_sed = (
            scattering_efficiency.T @ scattering_cross_section_by_size)
        scattered_starlight_sed *= (
            radiative_transfer._stellar_flux_at_model_wavelengths)
        scattered_starlight_sed *= self.scattering_phase_function(
            radiative_transfer._zero_scattering_angle_radian,
            **model_parameters)[0]
        t_end = time.perf_counter()

        self.radiative_transfer_timings['dust_temperature_interpolation'] = (
            t_temperatures - t0)
        self.radiative_transfer_timings['optical_efficiency_interpolation'] = (
            t_optical_efficiencies - t_temperatures)
        self.radiative_transfer_timings[
            'planck_emission_and_grain_size_integration'] = (
            t_dust_thermal_emission - t_optical_efficiencies)
        self.radiative_transfer_timings['dust_thermal_emission_total'] = (
            t_dust_thermal_emission - t0)
        self.radiative_transfer_timings[
            'scattered_starlight_grain_size_integration'] = (
            t_end - t_dust_thermal_emission)
        self.radiative_transfer_timings['scattered_starlight_total'] = (
            t_end - t_dust_thermal_emission)

        if keep_separate_fluxes:
            return dust_thermal_sed, scattered_starlight_sed
        dust_thermal_sed += scattered_starlight_sed
        return dust_thermal_sed

    def calculate_emission_by_stellocentric_distance(self, model_parameters):
        """Return diagnostic ``(thermal, scattered)`` arrays of shape ``(W,D)``."""
        result = (self.radiative_transfer
                  .calculate_emission_by_stellocentric_distance(
                      model_parameters))
        self.sizes_for_integral = self.radiative_transfer.sizes_for_integral
        return result

    def get_SED(self, keep_separate_fluxes=False,
                verbose_timing=False, return_emission_by_distance=False,
                **model_parameters):
        """Calculate the fastest disk-integrated SED.

        Parameters
        ----------
        keep_separate_fluxes : bool
            If true, return ``(dust_thermal_sed, scattered_starlight_sed)``.
            Otherwise return their sum.
        verbose_timing : bool
            Print the measured runtime of each major calculation stage.
        return_emission_by_distance : bool
            Also return the unnormalized thermal and scattered-light arrays as
            functions of wavelength and stellocentric distance.
        **model_parameters
            Disk-density and grain-size-distribution parameters, including
            either ``A_norm`` or ``M_tot``.

        Returns
        -------
        With ``return_emission_by_distance=False``, return the normalized SED.
        With it enabled, return
        ``(sed_result, dust_thermal_emission_by_distance,
        scattered_starlight_by_distance)``.  The distance-resolved arrays have
        shape ``(W, D)`` and do not include ``A_norm`` or mass normalization.
        """
        timings = {}
        total_start = time.perf_counter()
        (cylindrical_radius_au, height_above_midplane_au,
         maximum_height_au, stellocentric_distance_au) = (
            self._build_spatial_integration_grid(model_parameters))
        grid_ready = time.perf_counter()
        timings['grid_setup'] = grid_ready - total_start

        density_start = time.perf_counter()
        cylindrical_radius_column_au = cylindrical_radius_au[:, None]
        relative_number_density = self.density_function(
            cylindrical_radius_column_au, 0., height_above_midplane_au,
            model_parameters)
        density_times_cylindrical_volume = (
            relative_number_density
            * 2.0 * np.pi * cylindrical_radius_column_au
            * maximum_height_au[:, None])
        del relative_number_density
        density_ready = time.perf_counter()
        timings['density'] = density_ready - density_start

        spatial_integration_start = time.perf_counter()
        vertical_density_volume = (
            density_times_cylindrical_volume
            * self.vertical_integration_coefficients[None, :]).ravel()
        radial_integration_coefficients = (
            _trapezoid_integration_coefficients(cylindrical_radius_au))
        disk_density_volume_by_distance = (
            self._disk_density_volume_by_stellocentric_distance(
                cylindrical_radius_au, stellocentric_distance_au,
                vertical_density_volume,
                radial_integration_coefficients))
        del (density_times_cylindrical_volume, stellocentric_distance_au,
             vertical_density_volume,
             radial_integration_coefficients)
        spatial_integration_ready = time.perf_counter()
        timings['disk_density_volume_by_distance'] = (
            spatial_integration_ready - spatial_integration_start)

        radiative_transfer_start = time.perf_counter()
        unnormalized_sed = self.calculate_disk_integrated_sed(
            model_parameters, disk_density_volume_by_distance,
            keep_separate_fluxes)
        emission_by_distance = None
        if return_emission_by_distance:
            emission_by_distance = (
                self.calculate_emission_by_stellocentric_distance(
                    model_parameters))
        radiative_transfer_ready = time.perf_counter()
        timings['radiative_transfer'] = (
            radiative_transfer_ready - radiative_transfer_start)

        normalization_start = time.perf_counter()
        if 'M_tot' in model_parameters:
            grain_bulk_density_kg_m3 = (
                self.grain.grain_properties['Density'] * 1000)
            total_mass_kg = model_parameters['M_tot'] * cst.M_earth.value
            density_normalization = (
                calculate_normalization_density_jacobian_sublimation_fast(
                self.temperature_model,
                total_mass_kg, self.sizes_for_integral,
                cylindrical_radius_au, height_above_midplane_au,
                maximum_height_au,
                self.scaled_vertical_coordinate,
                grain_bulk_density_kg_m3, self.density_function,
                model_parameters, self.size_distribution_function,
                model_parameters))
        else:
            density_normalization = model_parameters['A_norm']
        self.density_normalization = density_normalization
        del height_above_midplane_au, maximum_height_au
        normalization_ready = time.perf_counter()
        timings['normalisation'] = normalization_ready - normalization_start

        timings['total'] = normalization_ready - total_start
        timings.update(self.radiative_transfer_timings)
        self.timings = timings

        if verbose_timing:
            logger.info('\n=== SED timing breakdown ===')
            for k, v in timings.items():
                pct = v / timings['total'] * 100
                logger.info(f'  {k:30s}: {v:8.3f} s  ({pct:5.1f}%)')
            logger.info('============================\n')

        if keep_separate_fluxes:
            (unnormalized_dust_thermal_sed,
             unnormalized_scattered_starlight_sed) = unnormalized_sed
            result = (
                unnormalized_dust_thermal_sed * density_normalization,
                unnormalized_scattered_starlight_sed * density_normalization)
            self.dust_thermal_sed, self.scattered_starlight_sed = result
        else:
            result = unnormalized_sed * density_normalization
            self.total_sed = result

        if return_emission_by_distance:
            (dust_thermal_emission_by_distance,
             scattered_starlight_by_distance) = emission_by_distance
            self.dust_thermal_emission_by_distance = (
                dust_thermal_emission_by_distance)
            self.scattered_starlight_by_distance = (
                scattered_starlight_by_distance)
            return (result, dust_thermal_emission_by_distance,
                    scattered_starlight_by_distance)
        return result
