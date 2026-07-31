"""Field-of-view transmitted SEDs for inclined optically thin disks.

The ordinary unresolved SED integrates every azimuth with unit transmission.
This module replaces that analytic azimuthal factor with the mean sky-plane
transmission seen by each instrument.  Different wavelengths can therefore
use different apertures or interferometric fiber transmission patterns.

Sky convention: North is positive upward, East is positive leftward, and
``PA`` is measured east of North.  The disk is axisymmetric, matching the
assumption made by ``SED`` when it evaluates density at one azimuth.
"""
import logging

import time

import astropy.constants as cst
import numpy as np

from pyGrater.SED import SED
from pyGrater.fluxes import _trapezoid_integration_coefficients
from pyGrater.utils import (

    calculate_normalization_density_jacobian_sublimation_fast,
)



logger = logging.getLogger(__name__)
try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _gaussian_azimuth_integral(
            cylindrical_radius_au, height_au, inclination_radian,
            fwhm_au, cutoff_radius_au, n_azimuth):
        """Integrate circular Gaussian transmission over disk azimuth."""
        n_radius, n_height = height_au.shape
        result = np.empty((n_radius, n_height))
        cosine_inclination = np.cos(inclination_radian)
        sine_inclination = np.sin(inclination_radian)
        gaussian_coefficient = 4.0 * np.log(2.0) / fwhm_au**2
        azimuth_step = 2.0 * np.pi / n_azimuth
        for radius_index in prange(n_radius):
            radius = cylindrical_radius_au[radius_index]
            for height_index in range(n_height):
                height = height_au[radius_index, height_index]
                transmission_sum = 0.0
                for azimuth_index in range(n_azimuth):
                    azimuth = azimuth_index * azimuth_step
                    projected_minor_axis = (
                        radius * np.cos(azimuth) * cosine_inclination
                        - height * sine_inclination)
                    projected_major_axis = radius * np.sin(azimuth)
                    sky_radius_squared = (
                        projected_minor_axis**2 + projected_major_axis**2)
                    if (cutoff_radius_au <= 0.0
                            or sky_radius_squared <= cutoff_radius_au**2):
                        transmission_sum += np.exp(
                            -gaussian_coefficient * sky_radius_squared)
                result[radius_index, height_index] = (
                    transmission_sum * azimuth_step)
        return result

    @njit(parallel=True, cache=True)
    def _gaussian_density_volume_by_distance(
            cylindrical_radius_au, height_au, density_without_azimuth,
            radial_integration_coefficients, distance_bin_index,
            fraction_above, inside_distance_grid, inclination_radian,
            fwhm_au_by_instrument, cutoff_au_by_instrument,
            full_transmission_by_instrument, cosine_azimuth,
            sine_azimuth, n_distance_bins):
        """Fuse FOV integration and deposition for all Gaussian instruments."""
        n_instruments = len(fwhm_au_by_instrument)
        n_radius, n_height = height_au.shape
        rows = np.zeros((n_instruments, n_radius, n_distance_bins))
        cosine_inclination = np.cos(inclination_radian)
        sine_inclination = np.sin(inclination_radian)
        n_azimuth = len(cosine_azimuth)
        azimuth_step = 2.0 * np.pi / n_azimuth
        gaussian_coefficient = np.empty(n_instruments)
        for instrument_index in range(n_instruments):
            gaussian_coefficient[instrument_index] = (
                4.0 * np.log(2.0)
                / fwhm_au_by_instrument[instrument_index]**2)
        for radius_index in prange(n_radius):
            radius = cylindrical_radius_au[radius_index]
            radial_coefficient = radial_integration_coefficients[radius_index]
            for height_index in range(n_height):
                if not inside_distance_grid[radius_index, height_index]:
                    continue
                base_density_volume = (
                    density_without_azimuth[radius_index, height_index]
                    * radial_coefficient)
                lower_bin = distance_bin_index[radius_index, height_index]
                upper_fraction = fraction_above[radius_index, height_index]
                lower_fraction = 1.0 - upper_fraction
                height = height_au[radius_index, height_index]
                transmission_sum = np.zeros(n_instruments)
                for azimuth_index in range(n_azimuth):
                    projected_minor_axis = (
                        radius * cosine_azimuth[azimuth_index]
                        * cosine_inclination - height * sine_inclination)
                    projected_major_axis = (
                        radius * sine_azimuth[azimuth_index])
                    sky_radius_squared = (
                        projected_minor_axis**2 + projected_major_axis**2)
                    for instrument_index in range(n_instruments):
                        cutoff_au = cutoff_au_by_instrument[instrument_index]
                        if (not full_transmission_by_instrument[instrument_index]
                                and (cutoff_au <= 0.0
                                     or sky_radius_squared <= cutoff_au**2)):
                            transmission_sum[instrument_index] += np.exp(
                                -gaussian_coefficient[instrument_index]
                                * sky_radius_squared)
                for instrument_index in range(n_instruments):
                    azimuth_integral = (
                        2.0 * np.pi
                        if full_transmission_by_instrument[instrument_index]
                        else transmission_sum[instrument_index] * azimuth_step)
                    contribution = base_density_volume * azimuth_integral
                    rows[instrument_index, radius_index, lower_bin] += (
                        contribution * lower_fraction)
                    rows[instrument_index, radius_index, lower_bin + 1] += (
                        contribution * upper_fraction)
        output = np.empty((n_instruments, n_distance_bins))
        for instrument_index in prange(n_instruments):
            for distance_index in range(n_distance_bins):
                total = 0.0
                for radius_index in range(n_radius):
                    total += rows[
                        instrument_index, radius_index, distance_index]
                output[instrument_index, distance_index] = total
        return output

    @njit(parallel=True, cache=True)
    def _tabulated_density_volume_by_distance(
            cylindrical_radius_au, height_au, density_without_azimuth,
            radial_integration_coefficients, distance_bin_index,
            fraction_above, inside_distance_grid, inclination_radian,
            position_angle_radian, distance_pc, east_grid_arcsec,
            north_grid_arcsec, transmission_grid, cosine_azimuth,
            sine_azimuth, n_distance_bins):
        """Fuse tabulated sky transmission and distance-bin deposition."""
        n_radius, n_height = height_au.shape
        rows = np.zeros((n_radius, n_distance_bins))
        cosine_inclination = np.cos(inclination_radian)
        sine_inclination = np.sin(inclination_radian)
        cosine_pa = np.cos(position_angle_radian)
        sine_pa = np.sin(position_angle_radian)
        east_min = east_grid_arcsec[0]
        north_min = north_grid_arcsec[0]
        east_step = east_grid_arcsec[1] - east_grid_arcsec[0]
        north_step = north_grid_arcsec[1] - north_grid_arcsec[0]
        n_east = len(east_grid_arcsec)
        n_north = len(north_grid_arcsec)
        azimuth_step = 2.0 * np.pi / len(cosine_azimuth)
        for radius_index in prange(n_radius):
            radius = cylindrical_radius_au[radius_index]
            radial_coefficient = radial_integration_coefficients[radius_index]
            for height_index in range(n_height):
                if not inside_distance_grid[radius_index, height_index]:
                    continue
                height = height_au[radius_index, height_index]
                transmission_sum = 0.0
                for azimuth_index in range(len(cosine_azimuth)):
                    minor_west_au = (
                        radius * cosine_azimuth[azimuth_index]
                        * cosine_inclination - height * sine_inclination)
                    major_north_au = radius * sine_azimuth[azimuth_index]
                    west_au = (cosine_pa * minor_west_au
                               - sine_pa * major_north_au)
                    north_au = (sine_pa * minor_west_au
                                + cosine_pa * major_north_au)
                    east_arcsec = -west_au / distance_pc
                    north_arcsec = north_au / distance_pc
                    east_position = (east_arcsec - east_min) / east_step
                    north_position = (north_arcsec - north_min) / north_step
                    east_index = int(np.floor(east_position))
                    north_index = int(np.floor(north_position))
                    if (east_index < 0 or east_index >= n_east - 1
                            or north_index < 0 or north_index >= n_north - 1):
                        continue
                    east_fraction = east_position - east_index
                    north_fraction = north_position - north_index
                    transmission_sum += (
                        transmission_grid[north_index, east_index]
                        * (1.0 - north_fraction) * (1.0 - east_fraction)
                        + transmission_grid[north_index + 1, east_index]
                        * north_fraction * (1.0 - east_fraction)
                        + transmission_grid[north_index, east_index + 1]
                        * (1.0 - north_fraction) * east_fraction
                        + transmission_grid[north_index + 1, east_index + 1]
                        * north_fraction * east_fraction)
                contribution = (
                    density_without_azimuth[radius_index, height_index]
                    * radial_coefficient * transmission_sum * azimuth_step)
                lower_bin = distance_bin_index[radius_index, height_index]
                upper_fraction = fraction_above[radius_index, height_index]
                rows[radius_index, lower_bin] += (
                    contribution * (1.0 - upper_fraction))
                rows[radius_index, lower_bin + 1] += (
                    contribution * upper_fraction)
        output = np.empty(n_distance_bins)
        for distance_index in prange(n_distance_bins):
            total = 0.0
            for radius_index in range(n_radius):
                total += rows[radius_index, distance_index]
            output[distance_index] = total
        return output

    _NUMBA_FOV_AVAILABLE = True
except ImportError:
    _NUMBA_FOV_AVAILABLE = False


class GaussianFieldOfView:
    """Circular Gaussian transmission defined by angular FWHM and cutoff.

    Parameters are in arcseconds.  Transmission equals 0.5 at a radius of
    ``fwhm_arcsec / 2``.  Outside ``cutoff_radius_arcsec`` it is exactly zero;
    use ``None`` for no cutoff.
    """

    def __init__(self, fwhm_arcsec, cutoff_radius_arcsec=None):
        self.fwhm_arcsec = float(fwhm_arcsec)
        self.cutoff_radius_arcsec = (
            None if cutoff_radius_arcsec is None
            else float(cutoff_radius_arcsec))
        if self.fwhm_arcsec <= 0:
            raise ValueError('Gaussian FWHM must be positive.')
        if (self.cutoff_radius_arcsec is not None
                and self.cutoff_radius_arcsec <= 0):
            raise ValueError('Gaussian cutoff radius must be positive.')

    def azimuth_integral(
            self, cylindrical_radius_au, height_au, inclination_radian,
            position_angle_radian, distance_pc, n_azimuth):
        """Return integral of transmission over azimuth for every disk cell."""
        del position_angle_radian  # Circular transmission is PA-invariant.
        fwhm_au = self.fwhm_arcsec * distance_pc
        cutoff_au = (0.0 if self.cutoff_radius_arcsec is None else
                     self.cutoff_radius_arcsec * distance_pc)
        if _NUMBA_FOV_AVAILABLE:
            return _gaussian_azimuth_integral(
                np.ascontiguousarray(cylindrical_radius_au),
                np.ascontiguousarray(height_au), inclination_radian,
                fwhm_au, cutoff_au, n_azimuth)

        azimuth = np.arange(n_azimuth) * (2.0 * np.pi / n_azimuth)
        result = np.zeros_like(height_au)
        for angle in azimuth:
            projected_minor = (
                cylindrical_radius_au[:, None] * np.cos(angle)
                * np.cos(inclination_radian)
                - height_au * np.sin(inclination_radian))
            projected_major = (
                cylindrical_radius_au[:, None] * np.sin(angle))
            radius_squared = projected_minor**2 + projected_major**2
            transmission = np.exp(
                -4.0 * np.log(2.0) * radius_squared / fwhm_au**2)
            if cutoff_au > 0:
                transmission[radius_squared > cutoff_au**2] = 0.0
            result += transmission
        return result * (2.0 * np.pi / n_azimuth)

    def sky_transmission(self, east_arcsec, north_arcsec):
        """Evaluate the Gaussian transmission directly in the sky plane."""
        sky_radius_squared_arcsec2 = (
            np.asarray(east_arcsec, dtype=np.float64)**2
            + np.asarray(north_arcsec, dtype=np.float64)**2)
        transmission = np.exp(
            -4.0 * np.log(2.0) * sky_radius_squared_arcsec2
            / self.fwhm_arcsec**2)
        if self.cutoff_radius_arcsec is not None:
            transmission = np.where(
                sky_radius_squared_arcsec2
                <= self.cutoff_radius_arcsec**2,
                transmission, 0.0)
        return transmission


class CallableFieldOfView:
    """A general sky transmission function.

    ``transmission_function(east_arcsec, north_arcsec)`` must accept NumPy
    arrays and return values between zero and one with the same shape.
    """

    def __init__(self, transmission_function):
        if not callable(transmission_function):
            raise TypeError('transmission_function must be callable.')
        self.transmission_function = transmission_function

    def azimuth_integral(
            self, cylindrical_radius_au, height_au, inclination_radian,
            position_angle_radian, distance_pc, n_azimuth):
        cosine_inclination = np.cos(inclination_radian)
        sine_inclination = np.sin(inclination_radian)
        cosine_pa = np.cos(position_angle_radian)
        sine_pa = np.sin(position_angle_radian)
        azimuth_step = 2.0 * np.pi / n_azimuth
        transmission_integral = np.zeros_like(height_au)
        # Batching removes Python-call overhead for complex patterns while
        # limiting temporary coordinate arrays to a few tens of megabytes.
        azimuth_batch_size = min(8, n_azimuth)
        radius = cylindrical_radius_au[None, :, None]
        height = height_au[None, :, :]
        for batch_start in range(0, n_azimuth, azimuth_batch_size):
            batch_stop = min(batch_start + azimuth_batch_size, n_azimuth)
            azimuth = (
                np.arange(batch_start, batch_stop)[:, None, None]
                * azimuth_step)
            # Coordinates before the PA rotation.  ``minor`` is positive
            # toward West in the Image internal convention.
            projected_minor_west_au = (
                radius * np.cos(azimuth) * cosine_inclination
                - height * sine_inclination)
            projected_major_north_au = (
                radius * np.sin(azimuth))
            west_au = (cosine_pa * projected_minor_west_au
                       - sine_pa * projected_major_north_au)
            north_au = (sine_pa * projected_minor_west_au
                        + cosine_pa * projected_major_north_au)
            east_arcsec = -west_au / distance_pc
            north_arcsec = north_au / distance_pc
            transmission = np.asarray(
                self.transmission_function(east_arcsec, north_arcsec),
                dtype=np.float64)
            expected_shape = (
                batch_stop - batch_start, *height_au.shape)
            if transmission.shape != expected_shape:
                transmission = np.broadcast_to(
                    transmission, expected_shape)
            if np.any(~np.isfinite(transmission)):
                raise ValueError('FOV transmission returned non-finite values.')
            transmission_integral += np.sum(
                np.clip(transmission, 0.0, 1.0), axis=0)
        return transmission_integral * azimuth_step

    def sky_transmission(self, east_arcsec, north_arcsec):
        """Evaluate and validate the user-supplied sky transmission."""
        transmission = np.asarray(
            self.transmission_function(east_arcsec, north_arcsec),
            dtype=np.float64)
        expected_shape = np.broadcast_shapes(
            np.shape(east_arcsec), np.shape(north_arcsec))
        if transmission.shape != expected_shape:
            transmission = np.broadcast_to(transmission, expected_shape)
        if np.any(~np.isfinite(transmission)):
            raise ValueError('FOV transmission returned non-finite values.')
        return np.clip(transmission, 0.0, 1.0)


class TabulatedFieldOfView:
    """A fitting-grade 2D transmission map on a regular sky grid.

    ``transmission`` has shape ``(north, east)``.  Both coordinate arrays are
    in arcseconds, strictly increasing, uniformly spaced, and transmission is
    zero outside the table.
    """

    def __init__(self, east_arcsec, north_arcsec, transmission):
        self.east_arcsec = np.asarray(east_arcsec, dtype=np.float64)
        self.north_arcsec = np.asarray(north_arcsec, dtype=np.float64)
        self.transmission = np.asarray(transmission, dtype=np.float64)
        if self.transmission.shape != (
                len(self.north_arcsec), len(self.east_arcsec)):
            raise ValueError(
                'transmission shape must be (north coordinates, east coordinates).')
        for name, coordinates in (
                ('east_arcsec', self.east_arcsec),
                ('north_arcsec', self.north_arcsec)):
            if len(coordinates) < 2 or np.any(np.diff(coordinates) <= 0):
                raise ValueError(f'{name} must be strictly increasing.')
            if not np.allclose(
                    np.diff(coordinates), np.diff(coordinates)[0],
                    rtol=1e-12, atol=0.0):
                raise ValueError(f'{name} must be uniformly spaced.')
        if np.any(~np.isfinite(self.transmission)):
            raise ValueError('Transmission table contains non-finite values.')
        self.transmission = np.clip(self.transmission, 0.0, 1.0)

    def sky_transmission(self, east_arcsec, north_arcsec):
        """Bilinearly interpolate the map at sky-plane coordinates."""
        east_arcsec, north_arcsec = np.broadcast_arrays(
            np.asarray(east_arcsec, dtype=np.float64),
            np.asarray(north_arcsec, dtype=np.float64))
        east_position = (
            (east_arcsec - self.east_arcsec[0])
            / (self.east_arcsec[1] - self.east_arcsec[0]))
        north_position = (
            (north_arcsec - self.north_arcsec[0])
            / (self.north_arcsec[1] - self.north_arcsec[0]))
        east_index = np.floor(east_position).astype(np.int64)
        north_index = np.floor(north_position).astype(np.int64)
        inside = (
            (east_index >= 0) & (east_index < len(self.east_arcsec) - 1)
            & (north_index >= 0)
            & (north_index < len(self.north_arcsec) - 1))
        output = np.zeros(east_arcsec.shape, dtype=np.float64)
        if not np.any(inside):
            return output
        east_lower = east_index[inside]
        north_lower = north_index[inside]
        east_fraction = east_position[inside] - east_lower
        north_fraction = north_position[inside] - north_lower
        output[inside] = (
            self.transmission[north_lower, east_lower]
            * (1.0 - north_fraction) * (1.0 - east_fraction)
            + self.transmission[north_lower + 1, east_lower]
            * north_fraction * (1.0 - east_fraction)
            + self.transmission[north_lower, east_lower + 1]
            * (1.0 - north_fraction) * east_fraction
            + self.transmission[north_lower + 1, east_lower + 1]
            * north_fraction * east_fraction)
        return output

    def azimuth_integral(
            self, cylindrical_radius_au, height_au, inclination_radian,
            position_angle_radian, distance_pc, n_azimuth):
        """Portable fallback used when the fused Numba path is unavailable."""
        return CallableFieldOfView(self.sky_transmission).azimuth_integral(
            cylindrical_radius_au, height_au, inclination_radian,
            position_angle_radian, distance_pc, n_azimuth)


class SEDFOV(SED):
    """SED model with instrument-dependent sky transmission.

    ``instrument_names`` must match the model wavelength array.  Each distinct
    name maps to a ``GaussianFieldOfView``, ``TabulatedFieldOfView``,
    ``CallableFieldOfView``, callable, or ``None`` (full transmission) in
    ``transmission_by_instrument``.  Tabulated maps are the optimized choice
    for complex patterns during fitting.
    """

    def __init__(
            self, grain, star, density_function, size_distribution_function,
            wavelengths_for_calc, instrument_names,
            transmission_by_instrument, N_distances=800, n_azimuth=64,
            spatial_parameter_names=None, spatial_cache=None):
        super().__init__(
            grain, star, density_function, size_distribution_function,
            wavelengths_for_calc, N_distances=N_distances)
        self.instrument_names = np.asarray(instrument_names, dtype=str)
        if self.instrument_names.shape != self.model_wavelengths_micron.shape:
            raise ValueError(
                'instrument_names must have the same shape as wavelengths.')
        self.n_azimuth = int(n_azimuth)
        if self.n_azimuth < 4:
            raise ValueError('n_azimuth must be at least four.')
        azimuth_radian = np.arange(self.n_azimuth) * (
            2.0 * np.pi / self.n_azimuth)
        self._cosine_azimuth = np.cos(azimuth_radian)
        self._sine_azimuth = np.sin(azimuth_radian)
        if spatial_parameter_names is None:
            spatial_parameter_names = (
                'r0', 'h0', 'alphain', 'alphaout', 'beta', 'gamma',
                'itilt', 'PA', 'p_cutoff')
        self.spatial_parameter_names = tuple(spatial_parameter_names)
        # A cache may be shared by compositions belonging to the same ring.
        # It deliberately stores only the current geometry: an MCMC can visit
        # millions of geometries, so retaining historical entries is unsafe.
        self._spatial_cache = (
            {} if spatial_cache is None else spatial_cache)

        self.transmission_by_instrument = {}
        for instrument in np.unique(self.instrument_names):
            if instrument not in transmission_by_instrument:
                raise ValueError(f'Missing FOV transmission for {instrument}.')
            transmission = transmission_by_instrument[instrument]
            if callable(transmission) and not hasattr(
                    transmission, 'azimuth_integral'):
                transmission = CallableFieldOfView(transmission)
            if (transmission is not None
                    and not hasattr(transmission, 'azimuth_integral')):
                raise TypeError(
                    f'Invalid transmission for instrument {instrument}.')
            self.transmission_by_instrument[instrument] = transmission

    def _spatial_cache_key(self, model_parameters):
        """Return values that can change density or projected transmission."""
        return tuple(
            (name, model_parameters.get(name, None))
            for name in self.spatial_parameter_names)

    def _density_volume_by_instrument(
            self, cylindrical_radius_au, height_au, maximum_height_au,
            stellocentric_distance_au, model_parameters):
        relative_density = self.density_function(
            cylindrical_radius_au[:, None], 0.0, height_au,
            model_parameters)
        density_without_azimuth = (
            relative_density * cylindrical_radius_au[:, None]
            * maximum_height_au[:, None]
            * self.vertical_integration_coefficients[None, :])
        radial_coefficients = _trapezoid_integration_coefficients(
            cylindrical_radius_au)
        inclination_radian = np.radians(model_parameters.get('itilt', 0.0))
        position_angle_radian = np.radians(model_parameters.get('PA', 0.0))

        density_volume_by_instrument = {}
        simple_instruments = [
            instrument for instrument, transmission
            in self.transmission_by_instrument.items()
            if transmission is None
            or isinstance(transmission, GaussianFieldOfView)]
        if simple_instruments and _NUMBA_FOV_AVAILABLE:
            distance_grid_au = self.stellocentric_distances_au
            lower_distance_bin = np.searchsorted(
                distance_grid_au, stellocentric_distance_au) - 1
            lower_distance_bin = np.clip(
                lower_distance_bin, 0, len(distance_grid_au) - 2)
            inside_distance_grid = (
                (stellocentric_distance_au >= distance_grid_au[0])
                & (stellocentric_distance_au <= distance_grid_au[-1]))
            fraction_above = (
                (stellocentric_distance_au
                 - distance_grid_au[lower_distance_bin])
                / (distance_grid_au[lower_distance_bin + 1]
                   - distance_grid_au[lower_distance_bin]))
            fwhm_au = []
            cutoff_au = []
            full_transmission = []
            for instrument in simple_instruments:
                transmission = self.transmission_by_instrument[instrument]
                full_transmission.append(transmission is None)
                fwhm_au.append(
                    1.0 if transmission is None else
                    transmission.fwhm_arcsec * self.star.distance)
                cutoff_au.append(
                    0.0 if (transmission is None
                            or transmission.cutoff_radius_arcsec is None)
                    else (transmission.cutoff_radius_arcsec
                          * self.star.distance))
            integrated = _gaussian_density_volume_by_distance(
                np.ascontiguousarray(cylindrical_radius_au),
                np.ascontiguousarray(height_au),
                np.ascontiguousarray(density_without_azimuth),
                np.ascontiguousarray(radial_coefficients),
                np.ascontiguousarray(lower_distance_bin.astype(np.int64)),
                np.ascontiguousarray(fraction_above),
                np.ascontiguousarray(inside_distance_grid),
                inclination_radian, np.asarray(fwhm_au),
                np.asarray(cutoff_au), np.asarray(full_transmission),
                np.ascontiguousarray(self._cosine_azimuth),
                np.ascontiguousarray(self._sine_azimuth),
                len(distance_grid_au))
            for index, instrument in enumerate(simple_instruments):
                density_volume_by_instrument[instrument] = integrated[index]

        for instrument, transmission in self.transmission_by_instrument.items():
            if instrument in density_volume_by_instrument:
                continue
            if (isinstance(transmission, TabulatedFieldOfView)
                    and _NUMBA_FOV_AVAILABLE):
                distance_grid_au = self.stellocentric_distances_au
                lower_distance_bin = np.searchsorted(
                    distance_grid_au, stellocentric_distance_au) - 1
                lower_distance_bin = np.clip(
                    lower_distance_bin, 0, len(distance_grid_au) - 2)
                inside_distance_grid = (
                    (stellocentric_distance_au >= distance_grid_au[0])
                    & (stellocentric_distance_au <= distance_grid_au[-1]))
                fraction_above = (
                    (stellocentric_distance_au
                     - distance_grid_au[lower_distance_bin])
                    / (distance_grid_au[lower_distance_bin + 1]
                       - distance_grid_au[lower_distance_bin]))
                density_volume_by_instrument[instrument] = (
                    _tabulated_density_volume_by_distance(
                        np.ascontiguousarray(cylindrical_radius_au),
                        np.ascontiguousarray(height_au),
                        np.ascontiguousarray(density_without_azimuth),
                        np.ascontiguousarray(radial_coefficients),
                        np.ascontiguousarray(
                            lower_distance_bin.astype(np.int64)),
                        np.ascontiguousarray(fraction_above),
                        np.ascontiguousarray(inside_distance_grid),
                        inclination_radian, position_angle_radian,
                        self.star.distance,
                        np.ascontiguousarray(transmission.east_arcsec),
                        np.ascontiguousarray(transmission.north_arcsec),
                        np.ascontiguousarray(transmission.transmission),
                        np.ascontiguousarray(self._cosine_azimuth),
                        np.ascontiguousarray(self._sine_azimuth),
                        len(distance_grid_au)))
                continue
            if transmission is None:
                azimuth_integral = 2.0 * np.pi
            else:
                azimuth_integral = transmission.azimuth_integral(
                    cylindrical_radius_au, height_au, inclination_radian,
                    position_angle_radian, self.star.distance,
                    self.n_azimuth)
            vertical_density_volume = (
                density_without_azimuth * azimuth_integral).ravel()
            density_volume_by_instrument[instrument] = (
                self._disk_density_volume_by_stellocentric_distance(
                    cylindrical_radius_au, stellocentric_distance_au,
                    vertical_density_volume, radial_coefficients))
        return density_volume_by_instrument

    def get_SED(self, keep_separate_fluxes=False, verbose_timing=False,
                return_emission_by_distance=False, **model_parameters):
        """Calculate the transmitted SED for every wavelength/instrument."""
        if all(value is None
               for value in self.transmission_by_instrument.values()):
            return super().get_SED(
                keep_separate_fluxes=keep_separate_fluxes,
                verbose_timing=verbose_timing,
                return_emission_by_distance=return_emission_by_distance,
                **model_parameters)

        start = time.perf_counter()
        spatial_key = self._spatial_cache_key(model_parameters)
        spatial_cache_hit = (
            spatial_key == self._spatial_cache.get('key')
            and self._spatial_cache.get('integration') is not None)
        if spatial_cache_hit:
            (cylindrical_radius_au, height_au, maximum_height_au,
             density_volume_by_instrument) = self._spatial_cache['integration']
        else:
            (cylindrical_radius_au, height_au, maximum_height_au,
             stellocentric_distance_au) = (
                self._build_spatial_integration_grid(model_parameters))
            density_volume_by_instrument = self._density_volume_by_instrument(
                cylindrical_radius_au, height_au, maximum_height_au,
                stellocentric_distance_au, model_parameters)
            self._spatial_cache['key'] = spatial_key
            self._spatial_cache['integration'] = (
                cylindrical_radius_au, height_au, maximum_height_au,
                density_volume_by_instrument)
        density_ready = time.perf_counter()

        thermal_by_distance, scattered_by_distance = (
            self.calculate_emission_by_stellocentric_distance(
                model_parameters))
        thermal_sed = np.empty_like(self.model_wavelengths_micron)
        scattered_sed = np.empty_like(self.model_wavelengths_micron)
        for instrument, density_volume in density_volume_by_instrument.items():
            wavelength_mask = self.instrument_names == instrument
            thermal_sed[wavelength_mask] = (
                thermal_by_distance[wavelength_mask] @ density_volume)
            scattered_sed[wavelength_mask] = (
                scattered_by_distance[wavelength_mask] @ density_volume)

        if 'M_tot' in model_parameters:
            grain_density_kg_m3 = (
                self.grain.grain_properties['Density'] * 1000.0)
            normalization = (
                calculate_normalization_density_jacobian_sublimation_fast(
                    self.temperature_model,
                    model_parameters['M_tot'] * cst.M_earth.value,
                    self.sizes_for_integral, cylindrical_radius_au, height_au,
                    maximum_height_au, self.scaled_vertical_coordinate,
                    grain_density_kg_m3, self.density_function,
                    model_parameters, self.size_distribution_function,
                    model_parameters))
        else:
            normalization = model_parameters['A_norm']
        self.density_normalization = normalization
        thermal_sed *= normalization
        scattered_sed *= normalization
        self.timings = {
            'transmitted_density_integration': density_ready - start,
            'radiative_transfer_and_normalization': (
                time.perf_counter() - density_ready),
            'total': time.perf_counter() - start,
            'spatial_cache_hit': spatial_cache_hit}
        if verbose_timing:
            for name, value in self.timings.items():
                logger.info(f'{name}: {value:.6f} s')

        result = ((thermal_sed, scattered_sed)
                  if keep_separate_fluxes else thermal_sed + scattered_sed)
        if return_emission_by_distance:
            return result, thermal_by_distance, scattered_by_distance
        return result
