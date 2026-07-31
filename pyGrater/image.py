"""
image.py -- exact-output optimized image engine.

This module keeps the same numerical ingredients as ``pyGrater.image.Image``.
Grain emission is supplied by the same ``Fluxes`` class in
``fluxes`` that underlies ``SED``. The speedups come from
specialized image integration while retaining that common radiative-transfer
implementation.
"""

import logging
import time

import astropy.constants as cst
import numpy as np
import scipy
import scipy.integrate
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from pyGrater.constants import (
    DEFAULT_IMAGE_PIXEL_SCALE_AU,
    MINIMUM_IMAGE_SCALED_HEIGHT,
    N_IMAGE_VERTICAL_GRID_POINTS,
    PIXEL_MAJOR_DENSITY_THRESHOLD,
)
from pyGrater.density import two_power_law_numba
from pyGrater.fluxes import Fluxes
from pyGrater.utils import (
    calculate_normalization_density_jacobian_sublimation_vfast
    as calculate_normalization_density_jacobian_sublimation,
    cylinder,
    hyperboloid_2_sheets,
)


try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _integrate_total_numba(
            dust_thermal_emission_by_distance,
            scattered_starlight_before_angular_dependence,
            scattering_phase_function_by_angle,
                               density_vals,
                               rho_idx, rho_t, rho_oob,
                               ang_idx, ang_t, ang_oob,
                               line_of_sight_path_length_au,
                               pixel_area, norm_factor):
        n_w, _ = dust_thermal_emission_by_distance.shape
        n_l, n_pix = density_vals.shape
        out = np.zeros((n_w, n_pix))
        inv_n = 1.0 / (n_l - 1.0)
        for w in prange(n_w):
            for p in range(n_pix):
                acc = 0.0
                for l in range(n_l):
                    k = l * n_pix + p
                    val = 0.0
                    if not rho_oob[k]:
                        ir = rho_idx[k]
                        tr = rho_t[k]
                        therm = ((1.0 - tr)
                                 * dust_thermal_emission_by_distance[w, ir]
                                 + tr
                                 * dust_thermal_emission_by_distance[w, ir + 1])
                        scat = 0.0
                        if not ang_oob[k]:
                            ia = ang_idx[k]
                            ta = ang_t[k]
                            distance_dependent_scattered_starlight = (
                                (1.0 - tr)
                                * scattered_starlight_before_angular_dependence[
                                    w, ir]
                                + tr
                                * scattered_starlight_before_angular_dependence[
                                    w, ir + 1])
                            scattering_angular_dependence = (
                                (1.0 - ta)
                                * scattering_phase_function_by_angle[ia]
                                + ta
                                * scattering_phase_function_by_angle[ia + 1])
                            scat = (distance_dependent_scattered_starlight
                                    * scattering_angular_dependence)
                        val = (therm + scat) * density_vals[l, p]
                    line_of_sight_integration_coefficient = (
                        0.5 if (l == 0 or l == n_l - 1) else 1.0)
                    acc += line_of_sight_integration_coefficient * val
                out[w, p] = (acc * inv_n
                             * line_of_sight_path_length_au[p]
                             * pixel_area * norm_factor)
        return out

    @njit(parallel=True, cache=True)
    def _integrate_separate_numba(
            dust_thermal_emission_by_distance,
            scattered_starlight_before_angular_dependence,
            scattering_phase_function_by_angle,
                                  density_vals,
                                  rho_idx, rho_t, rho_oob,
                                  ang_idx, ang_t, ang_oob,
                                  line_of_sight_path_length_au,
                                  pixel_area, norm_factor):
        n_w, _ = dust_thermal_emission_by_distance.shape
        n_l, n_pix = density_vals.shape
        out_sca = np.zeros((n_w, n_pix))
        out_therm = np.zeros((n_w, n_pix))
        inv_n = 1.0 / (n_l - 1.0)
        for w in prange(n_w):
            for p in range(n_pix):
                acc_sca = 0.0
                acc_therm = 0.0
                for l in range(n_l):
                    k = l * n_pix + p
                    therm = 0.0
                    scat = 0.0
                    if not rho_oob[k]:
                        ir = rho_idx[k]
                        tr = rho_t[k]
                        therm = ((1.0 - tr)
                                 * dust_thermal_emission_by_distance[w, ir]
                                 + tr
                                 * dust_thermal_emission_by_distance[w, ir + 1])
                        if not ang_oob[k]:
                            ia = ang_idx[k]
                            ta = ang_t[k]
                            distance_dependent_scattered_starlight = (
                                (1.0 - tr)
                                * scattered_starlight_before_angular_dependence[
                                    w, ir]
                                + tr
                                * scattered_starlight_before_angular_dependence[
                                    w, ir + 1])
                            scattering_angular_dependence = (
                                (1.0 - ta)
                                * scattering_phase_function_by_angle[ia]
                                + ta
                                * scattering_phase_function_by_angle[ia + 1])
                            scat = (distance_dependent_scattered_starlight
                                    * scattering_angular_dependence)
                        dens = density_vals[l, p]
                        therm *= dens
                        scat *= dens
                    line_of_sight_integration_coefficient = (
                        0.5 if (l == 0 or l == n_l - 1) else 1.0)
                    acc_sca += line_of_sight_integration_coefficient * scat
                    acc_therm += line_of_sight_integration_coefficient * therm
                scale = (inv_n * line_of_sight_path_length_au[p]
                         * pixel_area * norm_factor)
                out_sca[w, p] = acc_sca * scale
                out_therm[w, p] = acc_therm * scale
        return out_sca, out_therm

    @njit(parallel=True, cache=True)
    def _integrate_total_pixel_major_numba(
            dust_thermal_emission_by_distance,
            scattered_starlight_before_angular_dependence,
            scattering_phase_function_by_angle, density_pm,
            rho_idx_pm, rho_t_pm, rho_oob_pm,
            ang_idx_pm, ang_t_pm, ang_oob_pm,
            line_of_sight_path_length_au, pixel_area, norm_factor):
        n_w, _ = dust_thermal_emission_by_distance.shape
        n_pix, n_l = density_pm.shape
        out = np.zeros((n_w, n_pix))
        inv_n = 1.0 / (n_l - 1.0)
        for w in prange(n_w):
            for p in range(n_pix):
                acc = 0.0
                for l in range(n_l):
                    val = 0.0
                    if not rho_oob_pm[p, l]:
                        ir = rho_idx_pm[p, l]
                        tr = rho_t_pm[p, l]
                        therm = ((1.0 - tr)
                                 * dust_thermal_emission_by_distance[w, ir]
                                 + tr
                                 * dust_thermal_emission_by_distance[w, ir + 1])
                        scat = 0.0
                        if not ang_oob_pm[p, l]:
                            ia = ang_idx_pm[p, l]
                            ta = ang_t_pm[p, l]
                            distance_dependent_scattered_starlight = (
                                (1.0 - tr)
                                * scattered_starlight_before_angular_dependence[
                                    w, ir]
                                + tr
                                * scattered_starlight_before_angular_dependence[
                                    w, ir + 1])
                            scattering_angular_dependence = (
                                (1.0 - ta)
                                * scattering_phase_function_by_angle[ia]
                                + ta
                                * scattering_phase_function_by_angle[ia + 1])
                            scat = (distance_dependent_scattered_starlight
                                    * scattering_angular_dependence)
                        val = (therm + scat) * density_pm[p, l]
                    line_of_sight_integration_coefficient = (
                        0.5 if (l == 0 or l == n_l - 1) else 1.0)
                    acc += line_of_sight_integration_coefficient * val
                out[w, p] = (
                    acc * inv_n * line_of_sight_path_length_au[p]
                    * pixel_area * norm_factor)
        return out

    @njit(parallel=True, cache=True)
    def _integrate_separate_pixel_major_numba(
            dust_thermal_emission_by_distance,
            scattered_starlight_before_angular_dependence,
            scattering_phase_function_by_angle, density_pm,
            rho_idx_pm, rho_t_pm, rho_oob_pm,
            ang_idx_pm, ang_t_pm, ang_oob_pm,
            line_of_sight_path_length_au, pixel_area, norm_factor):
        n_w, _ = dust_thermal_emission_by_distance.shape
        n_pix, n_l = density_pm.shape
        out_sca = np.zeros((n_w, n_pix))
        out_therm = np.zeros((n_w, n_pix))
        inv_n = 1.0 / (n_l - 1.0)
        for w in prange(n_w):
            for p in range(n_pix):
                acc_sca = 0.0
                acc_therm = 0.0
                for l in range(n_l):
                    therm = 0.0
                    scat = 0.0
                    if not rho_oob_pm[p, l]:
                        ir = rho_idx_pm[p, l]
                        tr = rho_t_pm[p, l]
                        therm = ((1.0 - tr)
                                 * dust_thermal_emission_by_distance[w, ir]
                                 + tr
                                 * dust_thermal_emission_by_distance[w, ir + 1])
                        if not ang_oob_pm[p, l]:
                            ia = ang_idx_pm[p, l]
                            ta = ang_t_pm[p, l]
                            distance_dependent_scattered_starlight = (
                                (1.0 - tr)
                                * scattered_starlight_before_angular_dependence[
                                    w, ir]
                                + tr
                                * scattered_starlight_before_angular_dependence[
                                    w, ir + 1])
                            scattering_angular_dependence = (
                                (1.0 - ta)
                                * scattering_phase_function_by_angle[ia]
                                + ta
                                * scattering_phase_function_by_angle[ia + 1])
                            scat = (distance_dependent_scattered_starlight
                                    * scattering_angular_dependence)
                        dens = density_pm[p, l]
                        therm *= dens
                        scat *= dens
                    line_of_sight_integration_coefficient = (
                        0.5 if (l == 0 or l == n_l - 1) else 1.0)
                    acc_sca += line_of_sight_integration_coefficient * scat
                    acc_therm += line_of_sight_integration_coefficient * therm
                scale = (inv_n * line_of_sight_path_length_au[p]
                         * pixel_area * norm_factor)
                out_sca[w, p] = acc_sca * scale
                out_therm[w, p] = acc_therm * scale
        return out_sca, out_therm

    _NUMBA_IMAGE_AVAILABLE = True
except ImportError:
    _NUMBA_IMAGE_AVAILABLE = False


logger = logging.getLogger(__name__)


def _place_line_of_sight_values_on_sky(values, visible_pixel_mask,
                                       n_east_west_pixels,
                                       n_north_south_pixels):
    """Restore flattened visible pixels to North-up, East-left images.

    Geometry is calculated internally with East-West as the first array axis.
    The returned image follows the public convention: row 0 is North and
    columns increase toward West, as in a conventional astronomical image.
    """
    values = np.atleast_2d(values)
    internal_images = np.zeros(
        (values.shape[0], n_east_west_pixels * n_north_south_pixels),
        dtype=values.dtype)
    internal_images[:, visible_pixel_mask.ravel()] = values
    internal_images = internal_images.reshape(
        values.shape[0], n_east_west_pixels, n_north_south_pixels)
    return np.flip(np.transpose(internal_images, (0, 2, 1)), axis=1)


class Image:
    """Generate disk images with North up and East toward the left.

    ``PA`` is measured east of North.  Consequently PA=0 degrees places an
    axisymmetric inclined disk's major axis North-South, while PA=90 degrees
    places it East-West.
    """

    def __init__(self, grain, star, density_function,
                 size_distribution_function, scattering_phase_function,
                 wavelengths_for_calc, **kwargs):
        self.grain = grain
        self.star = star
        self.density_function = density_function
        self.size_distribution_function = size_distribution_function
        self.scattering_phase_function = scattering_phase_function

        self.radiative_transfer = Fluxes(
            grain, star, wavelengths_for_calc, size_distribution_function,
            scattering_phase_function)
        self.wavelengths_for_calc = wavelengths_for_calc
        self.stellocentric_distances_au = (
            self.radiative_transfer.stellocentric_distances_au)
        self.scattering_angles_radian = (
            self.radiative_transfer.scattering_angles_radian)
        self.timings = {}
        self._pixel_grid_cache = {}
        self._ln_cache = {}

        n_zeta = N_IMAGE_VERTICAL_GRID_POINTS
        half = (n_zeta - 1) // 2
        positive_zeta = np.geomspace(MINIMUM_IMAGE_SCALED_HEIGHT, 1.0, half)
        negative_zeta = -positive_zeta[::-1]
        self.scaled_vertical_coordinate = np.concatenate(
            (negative_zeta, [0.0], positive_zeta))

    def _build_z_grid(self, kwargs):
        r = self.stellocentric_distances_au
        _ = r, kwargs['r0'], kwargs['h0'], kwargs['beta'], kwargs['gamma']
        z_max_r = self.Z0 * np.sqrt(
            1 + (self.stellocentric_distances_au**2) / self.rmax**2)
        z_2d = (z_max_r[:, np.newaxis]
                * self.scaled_vertical_coordinate[np.newaxis, :])
        return z_2d, z_max_r

    def _resolve_pixel_scale(self, nx, ny, kwargs):
        if 'pixAU' not in kwargs and 'FOV_AU' not in kwargs:
            logger.warning(
                "Pixel size or FOV not provided, defaulting to FOV=0.5 AU")
            self.pixAU = DEFAULT_IMAGE_PIXEL_SCALE_AU
        if 'FOV_AU' in kwargs:
            self.pixAU = kwargs['FOV_AU'] / max(nx, ny)
        if 'pixAU' in kwargs:
            self.pixAU = kwargs.get('pixAU')
        return self.pixAU

    def _pixel_grid(self, nx, ny, pixAU):
        key = (nx, ny, float(pixAU))
        cached = self._pixel_grid_cache.get(key)
        if cached is not None:
            return cached
        xc = (nx - 1) / 2.
        yc = (ny - 1) / 2.
        x_grid, y_grid = np.mgrid[0:nx, 0:ny]
        x_prime = (x_grid - xc) * pixAU
        y_prime = (y_grid - yc) * pixAU
        cached = (x_prime, y_prime)
        self._pixel_grid_cache[key] = cached
        return cached

    def _line_sampling(self, nl):
        cached = self._ln_cache.get(nl)
        if cached is not None:
            return cached
        normalized_line_of_sight_coordinate = np.arange(nl) / (nl - 1.)
        self._ln_cache[nl] = normalized_line_of_sight_coordinate
        return normalized_line_of_sight_coordinate

    def _prepare_geometry(self, nx, ny, pixAU, kwargs):
        x_prime, y_prime = self._pixel_grid(nx, ny, pixAU)

        r0 = kwargs['r0']
        alphain = kwargs['alphain']
        alphaout = kwargs['alphaout']
        h0 = kwargs['h0']
        beta = kwargs['beta']
        gamma = kwargs['gamma']
        itilt = kwargs['itilt']
        PA = kwargs['PA']
        omega = kwargs['omega']

        p = 0.005
        rmax = r0 * p**(1 / alphaout)
        self.rmax = rmax
        gamma_in = alphain + beta
        gamma_out = alphaout + beta
        r_peak = (-gamma_in / gamma_out)**(
            1 / (2 * gamma_in - 2 * gamma_out)) * r0
        z_peak = (h0 * (r_peak / r0)**beta) * (
            np.log(1 / p)**(1 / gamma))
        self.Z0 = z_peak / np.sqrt(r_peak**2 / rmax**2 + 1)

        itilt_rad, omega_rad, PA_rad = np.radians([itilt, omega, PA])
        csPA, ssPA = np.cos(PA_rad), np.sin(PA_rad)
        csi, ssi = np.cos(itilt_rad), np.sin(itilt_rad)
        cso, sso = np.cos(omega_rad), np.sin(omega_rad)

        x = csPA * x_prime + ssPA * y_prime
        y = -ssPA * x_prime + csPA * y_prime

        vD = np.array([ssi * cso, -ssi * sso, csi])
        rD0 = np.stack([
            x * csi * cso + y * sso,
            -x * csi * sso + y * cso,
            -x * ssi,
        ])

        axis_c = np.array([rmax, rmax, np.sqrt(2) * self.Z0])
        axis_h = np.array([rmax, rmax, self.Z0])
        faraway = -rmax * 10.
        lmc, lpc = cylinder(axis_c, vD, rD0, faraway, csi)
        lmh, lph = hyperboloid_2_sheets(
            axis_h, vD, rD0, faraway, axis_c[0], axis_c[1])
        lbounds = np.sort([lmc, lmh, lph, lpc], axis=0)

        lmin = lbounds[2]
        lmax = lbounds[3]
        dl = lmax - lmin
        mask = (dl != 0)

        nl = kwargs.get('nl', 49)
        normalized_line_of_sight_coordinate = self._line_sampling(nl)
        line_of_sight_path_length_au = dl[mask]
        line_of_sight_distance_au = (
            normalized_line_of_sight_coordinate[:, np.newaxis]
            * line_of_sight_path_length_au[np.newaxis, :] + lmin[mask])

        height_above_disk_midplane_au = (
            rD0[2][mask][np.newaxis, :]
            + line_of_sight_distance_au * vD[2])
        sky_radius_squared_au2 = (
            x_prime[mask]**2 + y_prime[mask]**2)
        stellocentric_distance_squared_au2 = (
            sky_radius_squared_au2[np.newaxis, :]
            + line_of_sight_distance_au**2)
        stellocentric_distance_au = np.sqrt(
            stellocentric_distance_squared_au2)
        disk_plane_cylindrical_radius_au = np.sqrt(np.maximum(
            stellocentric_distance_squared_au2
            - height_above_disk_midplane_au**2, 0.0))
        scattering_angle = np.pi - np.arccos(
            np.clip(line_of_sight_distance_au
                    / stellocentric_distance_au, -1, 1))

        return {
            'normalized_line_of_sight_coordinate': (
                normalized_line_of_sight_coordinate),
            'mask': mask,
            'line_of_sight_path_length_au': line_of_sight_path_length_au,
            'disk_plane_cylindrical_radius_au': (
                disk_plane_cylindrical_radius_au),
            'stellocentric_distance_au': stellocentric_distance_au,
            'height_above_disk_midplane_au': (
                height_above_disk_midplane_au),
            'scattering_angle_radian': scattering_angle.ravel(),
            'nx': nx,
            'ny': ny,
            'pixAU': pixAU,
        }

    def _normalization(self, sizes, kwargs):
        if 'A_norm' in kwargs:
            self.norm_factor = kwargs['A_norm']
            return self.norm_factor

        distances = self.stellocentric_distances_au
        grain_density = self.grain.grain_properties['Density'] * 1000
        r_mask = distances <= self.rmax
        distances_clipped = distances[r_mask]
        z_2d, z_max_r = self._build_z_grid(kwargs)
        z_2d_clipped = z_2d[r_mask]
        z_max_r_clipped = z_max_r[r_mask]

        total_mass = kwargs['M_tot'] * cst.M_earth.value
        norm_factor = calculate_normalization_density_jacobian_sublimation(
            self.radiative_transfer.temperature_model,
            total_mass, sizes, distances_clipped, z_2d_clipped,
            z_max_r_clipped, self.scaled_vertical_coordinate, grain_density,
            self.density_function, kwargs,
            self.size_distribution_function, kwargs)
        self.norm_factor = norm_factor
        return norm_factor

    def prepare_spatial_disk(self, **kwargs):
        """Calculate wavelength-independent geometry and density once.

        Multi-composition fitters can pass the returned dictionary to
        ``get_image(prepared_spatial_disk=...)`` for every composition that
        belongs to the same ring. The arrays are read-only during imaging.
        """
        nx = kwargs.get('nx', 256)
        ny = kwargs.get('ny', 256)
        pixAU = self._resolve_pixel_scale(nx, ny, kwargs)
        geometry = self._prepare_geometry(nx, ny, pixAU, kwargs)
        disk_radius = geometry['disk_plane_cylindrical_radius_au']
        disk_height = geometry['height_above_disk_midplane_au']
        use_numba_density = (
            _NUMBA_IMAGE_AVAILABLE
            and getattr(self.density_function, '__name__', '')
            == 'two_power_law')
        if use_numba_density:
            density = two_power_law_numba(
                np.ascontiguousarray(disk_radius),
                np.ascontiguousarray(disk_height),
                kwargs['r0'], kwargs['h0'], kwargs['alphain'],
                kwargs['alphaout'], kwargs['beta'], kwargs['gamma'])
        else:
            density = self.density_function(
                disk_radius, 0.0, disk_height, kwargs)
        return {
            'geometry': geometry,
            'density': density,
            'rmax': self.rmax,
            'Z0': self.Z0,
            'nx': nx,
            'ny': ny,
            'pixAU': pixAU,
        }

    @staticmethod
    def _uniform_grid_indices_and_interpolation_fractions(grid, values):
        flat = np.asarray(values).ravel()
        oob = (flat < grid[0]) | (flat > grid[-1])
        scaled = (flat - grid[0]) / (grid[1] - grid[0])
        idx = np.floor(scaled).astype(np.int64)
        idx = np.clip(idx, 0, len(grid) - 2)
        t = scaled - idx
        t = np.where(flat == grid[-1], 1.0, t)
        return idx, t.astype(np.float64), oob

    @staticmethod
    def _geometric_grid_indices_and_interpolation_fractions(grid, values):
        flat = np.asarray(values).ravel()
        oob = (flat < grid[0]) | (flat > grid[-1])
        safe = np.maximum(flat, grid[0])
        log_step = np.log(grid[1] / grid[0])
        idx = np.floor(np.log(safe / grid[0]) / log_step).astype(np.int64)
        idx = np.clip(idx, 0, len(grid) - 2)
        t = (flat - grid[idx]) / (grid[idx + 1] - grid[idx])
        return idx, t.astype(np.float64), oob

    def get_image(self, keep_separate_fluxes=False,
                  prepared_spatial_disk=None, **kwargs):
        t_total0 = time.perf_counter()
        nx = kwargs.get('nx', 256)
        ny = kwargs.get('ny', 256)
        pixAU = self._resolve_pixel_scale(nx, ny, kwargs)

        use_numba_integrator = _NUMBA_IMAGE_AVAILABLE
        # The Numba path keeps radial illumination and angular scattering
        # separate, avoiding a much larger wavelength-distance-angle cube.

        t0 = time.perf_counter()
        if use_numba_integrator:
            (dust_thermal_emission_by_distance,
             scattered_starlight_before_angular_dependence,
             scattering_phase_function_by_angle) = (
                self.radiative_transfer.calculate_emission_for_image(kwargs))
        else:
            (dust_thermal_emission_by_distance,
             scattered_starlight_before_angular_dependence,
             scattering_phase_function_by_angle) = (
                self.radiative_transfer.calculate_emission_for_image(kwargs))
            scattered_starlight_by_distance_and_angle = (
                scattered_starlight_before_angular_dependence[:, :, None]
                * scattering_phase_function_by_angle[None, None, :])
        sizes = self.radiative_transfer.sizes_for_integral
        t_flux = time.perf_counter() - t0

        t0 = time.perf_counter()
        if prepared_spatial_disk is None:
            spatial_disk = self.prepare_spatial_disk(**kwargs)
        else:
            spatial_disk = prepared_spatial_disk
            expected = (nx, ny, float(pixAU))
            supplied = (
                spatial_disk['nx'], spatial_disk['ny'],
                float(spatial_disk['pixAU']))
            if supplied != expected:
                raise ValueError(
                    'Prepared disk image dimensions or pixel scale differ '
                    'from the requested image settings.')
        geom = spatial_disk['geometry']
        density_vals = spatial_disk['density']
        self.rmax = spatial_disk['rmax']
        self.Z0 = spatial_disk['Z0']
        t_geom = time.perf_counter() - t0

        t0 = time.perf_counter()
        norm_factor = self._normalization(sizes, kwargs)
        t_norm = time.perf_counter() - t0

        stellocentric_distance_au = geom['stellocentric_distance_au']
        t_invariant = 0.0

        n_waves = self.wavelengths_for_calc.size
        normalized_line_of_sight_coordinate = (
            geom['normalized_line_of_sight_coordinate'])
        mask = geom['mask']
        line_of_sight_path_length_au = (
            geom['line_of_sight_path_length_au'])
        pixel_area = pixAU**2

        t0 = time.perf_counter()
        if use_numba_integrator:
            distance_grid_interpolator = (
                self._geometric_grid_indices_and_interpolation_fractions)
            (stellocentric_distance_grid_index,
             stellocentric_distance_interpolation_fraction,
             outside_stellocentric_distance_grid) = distance_grid_interpolator(
                self.stellocentric_distances_au, stellocentric_distance_au)
            scattering_angle_grid_interpolator = (
                self._uniform_grid_indices_and_interpolation_fractions)
            (scattering_angle_grid_index,
             scattering_angle_interpolation_fraction,
             outside_scattering_angle_grid) = (
                scattering_angle_grid_interpolator(
                self.scattering_angles_radian,
                geom['scattering_angle_radian']))

            if not keep_separate_fluxes:
                pixel_major = density_vals.size >= PIXEL_MAJOR_DENSITY_THRESHOLD
                if pixel_major:
                    n_l, n_pix = density_vals.shape
                    shape_pm = (n_pix, n_l)
                    image_values = _integrate_total_pixel_major_numba(
                        np.ascontiguousarray(
                            dust_thermal_emission_by_distance),
                        np.ascontiguousarray(
                            scattered_starlight_before_angular_dependence),
                        np.ascontiguousarray(
                            scattering_phase_function_by_angle),
                        np.ascontiguousarray(density_vals.T),
                        np.ascontiguousarray(
                            stellocentric_distance_grid_index.reshape(
                                n_l, n_pix).T),
                        np.ascontiguousarray(
                            stellocentric_distance_interpolation_fraction.reshape(
                                n_l, n_pix).T),
                        np.ascontiguousarray(
                            outside_stellocentric_distance_grid.reshape(
                                shape_pm[::-1]).T),
                        np.ascontiguousarray(
                            scattering_angle_grid_index.reshape(n_l, n_pix).T),
                        np.ascontiguousarray(
                            scattering_angle_interpolation_fraction.reshape(
                                n_l, n_pix).T),
                        np.ascontiguousarray(
                            outside_scattering_angle_grid.reshape(
                                shape_pm[::-1]).T),
                        line_of_sight_path_length_au, pixel_area, norm_factor)
                else:
                    image_values = _integrate_total_numba(
                        np.ascontiguousarray(
                            dust_thermal_emission_by_distance),
                        np.ascontiguousarray(
                            scattered_starlight_before_angular_dependence),
                        np.ascontiguousarray(
                            scattering_phase_function_by_angle),
                        np.ascontiguousarray(density_vals),
                        stellocentric_distance_grid_index,
                        stellocentric_distance_interpolation_fraction,
                        outside_stellocentric_distance_grid,
                        scattering_angle_grid_index,
                        scattering_angle_interpolation_fraction,
                        outside_scattering_angle_grid,
                        line_of_sight_path_length_au, pixel_area, norm_factor)
                images = _place_line_of_sight_values_on_sky(
                    image_values, mask, nx, ny)
            else:
                pixel_major = density_vals.size >= PIXEL_MAJOR_DENSITY_THRESHOLD
                if pixel_major:
                    n_l, n_pix = density_vals.shape
                    image_sca_values, image_therm_values = (
                        _integrate_separate_pixel_major_numba(
                            np.ascontiguousarray(
                                dust_thermal_emission_by_distance),
                            np.ascontiguousarray(
                                scattered_starlight_before_angular_dependence),
                            np.ascontiguousarray(
                                scattering_phase_function_by_angle),
                            np.ascontiguousarray(density_vals.T),
                            np.ascontiguousarray(
                                stellocentric_distance_grid_index.reshape(
                                    n_l, n_pix).T),
                            np.ascontiguousarray(
                                stellocentric_distance_interpolation_fraction
                                .reshape(n_l, n_pix).T),
                            np.ascontiguousarray(
                                outside_stellocentric_distance_grid.reshape(
                                    n_l, n_pix).T),
                            np.ascontiguousarray(
                                scattering_angle_grid_index.reshape(
                                    n_l, n_pix).T),
                            np.ascontiguousarray(
                                scattering_angle_interpolation_fraction.reshape(
                                    n_l, n_pix).T),
                            np.ascontiguousarray(
                                outside_scattering_angle_grid.reshape(
                                    n_l, n_pix).T),
                            line_of_sight_path_length_au,
                            pixel_area, norm_factor))
                else:
                    image_sca_values, image_therm_values = (
                        _integrate_separate_numba(
                            np.ascontiguousarray(
                                dust_thermal_emission_by_distance),
                            np.ascontiguousarray(
                                scattered_starlight_before_angular_dependence),
                            np.ascontiguousarray(
                                scattering_phase_function_by_angle),
                            np.ascontiguousarray(density_vals),
                            stellocentric_distance_grid_index,
                            stellocentric_distance_interpolation_fraction,
                            outside_stellocentric_distance_grid,
                            scattering_angle_grid_index,
                            scattering_angle_interpolation_fraction,
                            outside_scattering_angle_grid,
                            line_of_sight_path_length_au,
                            pixel_area, norm_factor))
                images_sca = _place_line_of_sight_values_on_sky(
                    image_sca_values, mask, nx, ny)
                images_therm = _place_line_of_sight_values_on_sky(
                    image_therm_values, mask, nx, ny)

            t_loop = time.perf_counter() - t0
            self.timings = {
                'radiative_transfer': t_flux,
                'geometry': t_geom,
                'normalisation': t_norm,
                'dust_density': t_invariant,
                'line_of_sight_integration': t_loop,
                'total': time.perf_counter() - t_total0,
                'n_masked_px': int(mask.sum()),
                'n_pixels_total': nx * ny,
            }
            if keep_separate_fluxes:
                return images_sca, images_therm
            return images

        if not keep_separate_fluxes:
            images = np.zeros(shape=(n_waves, ny, nx))
        else:
            images_sca = np.zeros(shape=(n_waves, ny, nx))
            images_therm = np.zeros(shape=(n_waves, ny, nx))

        scalar_loop = n_waves < 4
        stellocentric_distance_flat_au = stellocentric_distance_au.ravel()
        interp_points = np.column_stack(
            [stellocentric_distance_flat_au,
             geom['scattering_angle_radian']])
        chunk = None
        if scalar_loop:
            chunk = 1
        if chunk is None:
            n_points = max(1, stellocentric_distance_flat_au.size)
            target_values = 4_000_000
            chunk = max(1, min(n_waves, target_values // n_points))

        for i0 in tqdm(range(0, n_waves, chunk),
                       desc="Optimized image optimized processing", disable=True):
            i1 = min(i0 + chunk, n_waves)
            if scalar_loop:
                dust_thermal_emission_interpolator = scipy.interpolate.interp1d(
                    self.stellocentric_distances_au,
                    dust_thermal_emission_by_distance[i0, :],
                    kind='linear', bounds_error=False, fill_value=0)
                scattered_starlight_interpolator = RegularGridInterpolator(
                    (self.stellocentric_distances_au,
                     self.scattering_angles_radian),
                    scattered_starlight_by_distance_and_angle[i0, :, :],
                    fill_value=0,
                    bounds_error=False)

                scattered_starlight_along_lines_of_sight = (
                    scattered_starlight_interpolator(interp_points).reshape(
                        disk_plane_cylindrical_radius_au.shape))
                dust_thermal_emission_along_lines_of_sight = (
                    dust_thermal_emission_interpolator(
                        stellocentric_distance_au))

                if not keep_separate_fluxes:
                    limage = (
                        scattered_starlight_along_lines_of_sight
                        + dust_thermal_emission_along_lines_of_sight
                    ) * density_vals
                    image = np.zeros([nx, ny])
                    image[mask] = (
                        scipy.integrate.trapezoid(
                            limage,
                            x=normalized_line_of_sight_coordinate, axis=0)
                        * line_of_sight_path_length_au * pixel_area)
                    images[i0, :, :] = (
                        np.flip(image.T, axis=0) * norm_factor)
                else:
                    limage_sca = (
                        scattered_starlight_along_lines_of_sight * density_vals)
                    image_sca = np.zeros([nx, ny])
                    image_sca[mask] = (
                        scipy.integrate.trapezoid(
                            limage_sca,
                            x=normalized_line_of_sight_coordinate, axis=0)
                        * line_of_sight_path_length_au * pixel_area)
                    images_sca[i0, :, :] = (
                        np.flip(image_sca.T, axis=0) * norm_factor)

                    limage_therm = (
                        dust_thermal_emission_along_lines_of_sight
                        * density_vals)
                    image_therm = np.zeros([nx, ny])
                    image_therm[mask] = (
                        scipy.integrate.trapezoid(
                            limage_therm,
                            x=normalized_line_of_sight_coordinate, axis=0)
                        * line_of_sight_path_length_au * pixel_area)
                    images_therm[i0, :, :] = (
                        np.flip(image_therm.T, axis=0) * norm_factor)
                continue

            dust_thermal_emission_interpolator = scipy.interpolate.interp1d(
                self.stellocentric_distances_au,
                dust_thermal_emission_by_distance[i0:i1, :],
                axis=1, kind='linear', bounds_error=False, fill_value=0)
            scattered_starlight_interpolator = RegularGridInterpolator(
                (self.stellocentric_distances_au,
                 self.scattering_angles_radian),
                np.moveaxis(
                    scattered_starlight_by_distance_and_angle[i0:i1, :, :],
                    0, -1),
                fill_value=0, bounds_error=False)

            scattered_starlight_along_lines_of_sight = np.moveaxis(
                scattered_starlight_interpolator(interp_points), -1, 0).reshape(
                    i1 - i0, *disk_plane_cylindrical_radius_au.shape)
            dust_thermal_emission_along_lines_of_sight = (
                dust_thermal_emission_interpolator(stellocentric_distance_au))

            if not keep_separate_fluxes:
                limage = ((scattered_starlight_along_lines_of_sight
                           + dust_thermal_emission_along_lines_of_sight)
                          * density_vals[np.newaxis, :, :])
                image_values = (
                    scipy.integrate.trapezoid(
                        limage, x=normalized_line_of_sight_coordinate, axis=1)
                    * line_of_sight_path_length_au[np.newaxis, :] * pixel_area)
                for j, values in enumerate(image_values, start=i0):
                    image = np.zeros([nx, ny])
                    image[mask] = values
                    images[j, :, :] = np.flip(image.T, axis=0) * norm_factor
            else:
                limage_sca = (
                    scattered_starlight_along_lines_of_sight
                    * density_vals[np.newaxis, :, :])
                limage_therm = (
                    dust_thermal_emission_along_lines_of_sight
                    * density_vals[np.newaxis, :, :])
                image_sca_values = (
                    scipy.integrate.trapezoid(
                        limage_sca,
                        x=normalized_line_of_sight_coordinate, axis=1)
                    * line_of_sight_path_length_au[np.newaxis, :] * pixel_area)
                image_therm_values = (
                    scipy.integrate.trapezoid(
                        limage_therm,
                        x=normalized_line_of_sight_coordinate, axis=1)
                    * line_of_sight_path_length_au[np.newaxis, :] * pixel_area)
                for j, (sca_values, therm_values) in enumerate(
                        zip(image_sca_values, image_therm_values), start=i0):
                    image_sca = np.zeros([nx, ny])
                    image_sca[mask] = sca_values
                    images_sca[j, :, :] = (
                        np.flip(image_sca.T, axis=0) * norm_factor)

                    image_therm = np.zeros([nx, ny])
                    image_therm[mask] = therm_values
                    images_therm[j, :, :] = (
                        np.flip(image_therm.T, axis=0) * norm_factor)

        t_loop = time.perf_counter() - t0
        self.timings = {
            'radiative_transfer': t_flux,
            'geometry': t_geom,
            'normalisation': t_norm,
            'dust_density': t_invariant,
            'line_of_sight_integration': t_loop,
            'total': time.perf_counter() - t_total0,
            'n_masked_px': int(mask.sum()),
            'n_pixels_total': nx * ny,
        }

        if keep_separate_fluxes:
            return images_sca, images_therm
        return images
