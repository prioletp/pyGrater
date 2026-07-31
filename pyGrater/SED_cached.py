"""Cache-aware disk-integrated SED calculations.

The public :class:`CachedSED` class preserves the numerical calculation and
API of :class:`pyGrater.SED.SED`, while allowing several grain compositions
with identical disk parameters to reuse the expensive spatial-density
integration.  A :class:`SharedSEDCache` instance is explicitly shared by the
fitters; the original ``SED`` class remains unchanged.
"""
import logging

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
import time

import astropy.constants as cst
import numpy as np

from pyGrater.SED import DEFAULT_DENSITY_CUTOFF, SED
from pyGrater.fluxes import _trapezoid_integration_coefficients
from pyGrater.utils import (

    calculate_normalization_density_jacobian_sublimation_fast,
)



logger = logging.getLogger(__name__)
_NORMALIZATION_PARAMETERS = frozenset({"A_norm", "M_tot"})


def _cacheable_value(value):
    """Return a stable, hashable representation of a model value."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return ("array", array.dtype.str, array.shape, array.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(_cacheable_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), _cacheable_value(item))
                   for key, item in value.items()))
    if isinstance(value, np.generic):
        return value.item()
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


@dataclass(frozen=True)
class SpatialGeometryState:
    """Grain-independent disk geometry and density quadrature."""

    cylindrical_radius_au: np.ndarray
    height_above_midplane_au: np.ndarray
    maximum_height_au: np.ndarray
    stellocentric_distance_au: np.ndarray
    vertical_density_volume: np.ndarray
    radial_integration_coefficients: np.ndarray


@dataclass(frozen=True)
class SpatialSEDState:
    """Geometry plus density projected onto one radiative-transfer grid."""

    geometry: SpatialGeometryState
    disk_density_volume_by_distance: np.ndarray


class SharedSEDCache:
    """Small thread-safe LRU cache shared by related ``CachedSED`` objects."""

    def __init__(self, max_entries=16):
        self.max_entries = max(1, int(max_entries))
        self._states = OrderedDict()
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_lock'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = RLock()

    def get_or_create(self, key, factory):
        """Return the cached state for ``key``, computing it once if absent."""
        with self._lock:
            state = self._states.get(key)
            if state is not None:
                self._states.move_to_end(key)
                self.hits += 1
                return state, True
            state = factory()
            self._states[key] = state
            self._states.move_to_end(key)
            self.misses += 1
            while len(self._states) > self.max_entries:
                self._states.popitem(last=False)
            return state, False

    def clear(self):
        with self._lock:
            self._states.clear()

    @property
    def size(self):
        with self._lock:
            return len(self._states)

    def info(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": self.size,
            "max_entries": self.max_entries,
        }


class CachedSED(SED):
    """``SED`` variant that reuses spatial integration across compositions."""

    def __init__(self, *args, shared_cache=None, cache_namespace=None,
                 cache_max_entries=16, spatial_grid_min_au=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_cache = (
            shared_cache if shared_cache is not None
            else SharedSEDCache(max_entries=cache_max_entries))
        self.cache_namespace = cache_namespace
        self.spatial_grid_min_au = (
            None if spatial_grid_min_au is None
            else float(spatial_grid_min_au))
        self.last_spatial_cache_hit = False
        self.spatial_group_members = None
        self.spatial_member_key = None

    def _spatial_cache_key(self, model_parameters):
        parameters = tuple(sorted(
            (str(name), _cacheable_value(value))
            for name, value in model_parameters.items()
            if name not in _NORMALIZATION_PARAMETERS))
        return (
            "geometry",
            self.cache_namespace,
            id(self.density_function),
            float(
                self.stellocentric_distances_au.min()
                if self.spatial_grid_min_au is None
                else self.spatial_grid_min_au),
            parameters,
        )

    def _calculate_spatial_geometry(self, model_parameters):
        if self.spatial_grid_min_au is None:
            (cylindrical_radius_au, height_above_midplane_au,
             maximum_height_au, stellocentric_distance_au) = (
                self._build_spatial_integration_grid(model_parameters))
        else:
            reference_radius_au = model_parameters['r0']
            inner_slope = model_parameters['alphain']
            outer_slope = model_parameters['alphaout']
            flaring_exponent = model_parameters['beta']
            vertical_exponent = model_parameters['gamma']
            scale_height_at_reference_radius_au = model_parameters['h0']
            density_cutoff = model_parameters.get(
                'p_cutoff', DEFAULT_DENSITY_CUTOFF)
            maximum_disk_radius_au = model_parameters.get('rmax', reference_radius_au * density_cutoff ** (1.0 / outer_slope))
            self.maximum_disk_radius_au = maximum_disk_radius_au
            cylindrical_radius_au = np.geomspace(
                self.spatial_grid_min_au, self.maximum_disk_radius_au,
                len(self.stellocentric_distances_au))
            effective_inner_slope = inner_slope + flaring_exponent
            effective_outer_slope = outer_slope + flaring_exponent
            peak_radius_au = (
                -effective_inner_slope / effective_outer_slope
            ) ** (1.0 / (
                2 * effective_inner_slope - 2 * effective_outer_slope
            )) * reference_radius_au
            height_at_peak_radius_au = (
                scale_height_at_reference_radius_au
                * (peak_radius_au / reference_radius_au)**flaring_exponent
                * np.log(1.0 / density_cutoff)**(
                    1.0 / vertical_exponent))
            self.vertical_grid_scale_au = (
                height_at_peak_radius_au
                / np.sqrt(
                    peak_radius_au**2 / self.maximum_disk_radius_au**2 + 1))
            (height_above_midplane_au, maximum_height_au,
             stellocentric_distance_au) = self._build_vertical_grid(
                cylindrical_radius_au)
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
        return SpatialGeometryState(
            cylindrical_radius_au=np.asarray(cylindrical_radius_au),
            height_above_midplane_au=np.asarray(height_above_midplane_au),
            maximum_height_au=np.asarray(maximum_height_au),
            stellocentric_distance_au=np.asarray(stellocentric_distance_au),
            vertical_density_volume=np.asarray(vertical_density_volume),
            radial_integration_coefficients=np.asarray(
                radial_integration_coefficients),
        )

    def _spatial_state(self, model_parameters):
        if self.spatial_group_members:
            return self._group_spatial_state(model_parameters)
        geometry_key = self._spatial_cache_key(model_parameters)
        geometry, geometry_hit = self.shared_cache.get_or_create(
            geometry_key,
            lambda: self._calculate_spatial_geometry(model_parameters))
        distance_grid = np.ascontiguousarray(
            self.stellocentric_distances_au, dtype=np.float64)
        projection_key = (
            "projection", geometry_key, distance_grid.shape,
            distance_grid.tobytes())

        def project():
            projected = self._disk_density_volume_by_stellocentric_distance(
                geometry.cylindrical_radius_au,
                geometry.stellocentric_distance_au,
                geometry.vertical_density_volume,
                geometry.radial_integration_coefficients)
            return SpatialSEDState(
                geometry=geometry,
                disk_density_volume_by_distance=np.asarray(projected))

        state, projection_hit = self.shared_cache.get_or_create(
            projection_key, project)
        self.last_spatial_cache_hit = geometry_hit
        self.last_projection_cache_hit = projection_hit
        return state

    def _group_spatial_state(self, model_parameters):
        members = tuple(self.spatial_group_members)
        member_signatures = tuple(
            (member.spatial_member_key,
             float(member.stellocentric_distances_au.min()),
             len(member.stellocentric_distances_au),
             np.ascontiguousarray(
                 member.stellocentric_distances_au,
                 dtype=np.float64).tobytes())
            for member in members)
        parameters = tuple(sorted(
            (str(name), _cacheable_value(value))
            for name, value in model_parameters.items()
            if name not in _NORMALIZATION_PARAMETERS))
        key = (
            "group", self.cache_namespace, id(self.density_function),
            member_signatures, parameters)
        states, hit = self.shared_cache.get_or_create(
            key, lambda: self._calculate_group_states(
                members, model_parameters))
        self.last_spatial_cache_hit = hit
        self.last_projection_cache_hit = hit
        return states[self.spatial_member_key]

    def _calculate_group_states(self, members, model_parameters):
        """Evaluate shared density once on the union of original radial grids."""
        reference_radius_au = model_parameters['r0']
        inner_slope = model_parameters['alphain']
        outer_slope = model_parameters['alphaout']
        flaring_exponent = model_parameters['beta']
        vertical_exponent = model_parameters['gamma']
        scale_height_at_reference_radius_au = model_parameters['h0']
        density_cutoff = model_parameters.get(
            'p_cutoff', DEFAULT_DENSITY_CUTOFF)
        maximum_disk_radius_au = (
            reference_radius_au * density_cutoff ** (1.0 / outer_slope))
        radial_grids = {
            member.spatial_member_key: np.geomspace(
                member.stellocentric_distances_au.min(),
                maximum_disk_radius_au,
                len(member.stellocentric_distances_au))
            for member in members}
        union_radius_au = np.unique(np.concatenate(
            list(radial_grids.values())))

        effective_inner_slope = inner_slope + flaring_exponent
        effective_outer_slope = outer_slope + flaring_exponent
        peak_radius_au = (
            -effective_inner_slope / effective_outer_slope
        ) ** (1.0 / (
            2 * effective_inner_slope - 2 * effective_outer_slope
        )) * reference_radius_au
        height_at_peak_radius_au = (
            scale_height_at_reference_radius_au
            * (peak_radius_au / reference_radius_au)**flaring_exponent
            * np.log(1.0 / density_cutoff)**(1.0 / vertical_exponent))
        vertical_grid_scale_au = (
            height_at_peak_radius_au
            / np.sqrt(peak_radius_au**2 / maximum_disk_radius_au**2 + 1))

        self.maximum_disk_radius_au = maximum_disk_radius_au
        self.vertical_grid_scale_au = vertical_grid_scale_au
        union_height_au, union_maximum_height_au, union_distance_au = (
            self._build_vertical_grid(union_radius_au))
        union_radius_column_au = union_radius_au[:, None]
        relative_number_density = self.density_function(
            union_radius_column_au, 0., union_height_au, model_parameters)
        density_times_cylindrical_volume = (
            relative_number_density
            * 2.0 * np.pi * union_radius_column_au
            * union_maximum_height_au[:, None])
        weighted_density_rows = (
            density_times_cylindrical_volume
            * self.vertical_integration_coefficients[None, :])

        states = {}
        for member in members:
            member.maximum_disk_radius_au = maximum_disk_radius_au
            member.vertical_grid_scale_au = vertical_grid_scale_au
            member_key = member.spatial_member_key
            cylindrical_radius_au = radial_grids[member_key]
            row_indices = np.searchsorted(
                union_radius_au, cylindrical_radius_au)
            if not np.array_equal(
                    union_radius_au[row_indices], cylindrical_radius_au):
                raise RuntimeError(
                    "Shared radial-grid union did not preserve exact rows.")
            height_above_midplane_au = union_height_au[row_indices]
            maximum_height_au = union_maximum_height_au[row_indices]
            stellocentric_distance_au = union_distance_au[row_indices]
            vertical_density_volume = (
                weighted_density_rows[row_indices].ravel())
            radial_integration_coefficients = (
                _trapezoid_integration_coefficients(
                    cylindrical_radius_au))
            projected = (
                member._disk_density_volume_by_stellocentric_distance(
                    cylindrical_radius_au, stellocentric_distance_au,
                    vertical_density_volume,
                    radial_integration_coefficients))
            geometry = SpatialGeometryState(
                cylindrical_radius_au=cylindrical_radius_au,
                height_above_midplane_au=height_above_midplane_au,
                maximum_height_au=maximum_height_au,
                stellocentric_distance_au=stellocentric_distance_au,
                vertical_density_volume=vertical_density_volume,
                radial_integration_coefficients=(
                    radial_integration_coefficients),
            )
            states[member_key] = SpatialSEDState(
                geometry=geometry,
                disk_density_volume_by_distance=np.asarray(projected))
        return states

    def get_SED(self, keep_separate_fluxes=False, verbose_timing=False,
                return_emission_by_distance=False, **model_parameters):
        """Calculate an SED while reusing matching spatial integrations."""
        timings = {}
        total_start = time.perf_counter()
        state = self._spatial_state(model_parameters)
        spatial_ready = time.perf_counter()
        timings["spatial_state"] = spatial_ready - total_start
        timings["spatial_cache_hit"] = float(self.last_spatial_cache_hit)
        timings["projection_cache_hit"] = float(
            self.last_projection_cache_hit)

        unnormalized_sed = self.calculate_disk_integrated_sed(
            model_parameters, state.disk_density_volume_by_distance,
            keep_separate_fluxes)
        emission_by_distance = None
        if return_emission_by_distance:
            emission_by_distance = (
                self.calculate_emission_by_stellocentric_distance(
                    model_parameters))
        radiative_transfer_ready = time.perf_counter()
        timings["radiative_transfer"] = (
            radiative_transfer_ready - spatial_ready)

        normalization_start = time.perf_counter()
        if "M_tot" in model_parameters:
            grain_bulk_density_kg_m3 = (
                self.grain.grain_properties["Density"] * 1000)
            total_mass_kg = model_parameters["M_tot"] * cst.M_earth.value
            density_normalization = (
                calculate_normalization_density_jacobian_sublimation_fast(
                    self.temperature_model,
                    total_mass_kg,
                    self.sizes_for_integral,
                    state.geometry.cylindrical_radius_au,
                    state.geometry.height_above_midplane_au,
                    state.geometry.maximum_height_au,
                    self.scaled_vertical_coordinate,
                    grain_bulk_density_kg_m3,
                    self.density_function,
                    model_parameters,
                    self.size_distribution_function,
                    model_parameters,
                ))
        else:
            density_normalization = model_parameters["A_norm"]
        self.density_normalization = density_normalization
        normalization_ready = time.perf_counter()
        timings["normalisation"] = normalization_ready - normalization_start
        timings["total"] = normalization_ready - total_start
        timings.update(self.radiative_transfer_timings)
        self.timings = timings

        if verbose_timing:
            logger.info("\n=== CachedSED timing breakdown ===")
            for name, duration in timings.items():
                if name in {"spatial_cache_hit", "projection_cache_hit"}:
                    continue
                percent = duration / timings["total"] * 100
                logger.info(f"  {name:30s}: {duration:8.3f} s  ({percent:5.1f}%)")
            logger.info(f"  spatial cache hit: {self.last_spatial_cache_hit}")
            logger.info(f"  projection cache hit: {self.last_projection_cache_hit}")
            logger.info("==================================\n")

        if keep_separate_fluxes:
            thermal, scattered = unnormalized_sed
            result = (
                thermal * density_normalization,
                scattered * density_normalization,
            )
            self.dust_thermal_sed, self.scattered_starlight_sed = result
        else:
            result = unnormalized_sed * density_normalization
            self.total_sed = result

        if return_emission_by_distance:
            thermal_by_distance, scattered_by_distance = emission_by_distance
            self.dust_thermal_emission_by_distance = thermal_by_distance
            self.scattered_starlight_by_distance = scattered_by_distance
            return result, thermal_by_distance, scattered_by_distance
        return result
