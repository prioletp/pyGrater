"""Instrument-dependent field-of-view transmission for ``Image``.

The underlying image calculation is unchanged.  Once the North-up,
East-left disk image has been generated, each wavelength is multiplied by its
instrument's sky transmission.  Transmission maps are cached by image shape
and pixel scale, so repeated fitting calls do not recalculate static patterns.
"""

import time

import numpy as np

from pyGrater.SED_fov import CallableFieldOfView
from pyGrater.image import Image


class ImageFOV(Image):
    """Generate optimized images after instrument-specific sky transmission.

    ``instrument_names`` must have one entry per model wavelength.
    ``transmission_by_instrument`` maps each name to a
    ``GaussianFieldOfView``, ``TabulatedFieldOfView``,
    ``CallableFieldOfView``, vectorized callable, or ``None`` for full
    transmission.
    """

    def __init__(
            self, grain, star, density_function, size_distribution_function,
            scattering_phase_function, wavelengths_for_calc,
            instrument_names, transmission_by_instrument, **kwargs):
        super().__init__(
            grain, star, density_function, size_distribution_function,
            scattering_phase_function, wavelengths_for_calc, **kwargs)
        self.instrument_names = np.asarray(instrument_names, dtype=str)
        if self.instrument_names.shape != np.shape(wavelengths_for_calc):
            raise ValueError(
                'instrument_names must have the same shape as wavelengths.')

        self.transmission_by_instrument = {}
        self._wavelength_indices_by_instrument = {}
        for instrument in np.unique(self.instrument_names):
            if instrument not in transmission_by_instrument:
                raise ValueError(f'Missing FOV transmission for {instrument}.')
            transmission = transmission_by_instrument[instrument]
            if callable(transmission) and not hasattr(
                    transmission, 'sky_transmission'):
                transmission = CallableFieldOfView(transmission)
            if (transmission is not None
                    and not hasattr(transmission, 'sky_transmission')):
                raise TypeError(
                    f'Invalid transmission for instrument {instrument}.')
            self.transmission_by_instrument[instrument] = transmission
            self._wavelength_indices_by_instrument[instrument] = np.flatnonzero(
                self.instrument_names == instrument)
        self._sky_transmission_cache = {}

    def _sky_coordinates_arcsec(self, n_rows, n_columns, pixel_scale_au):
        """Return East and North coordinates at every image pixel center."""
        center_row = (n_rows - 1.0) / 2.0
        center_column = (n_columns - 1.0) / 2.0
        row, column = np.mgrid[:n_rows, :n_columns]
        # Images are North-up and East-left.
        east_arcsec = (
            (center_column - column) * pixel_scale_au / self.star.distance)
        north_arcsec = (
            (center_row - row) * pixel_scale_au / self.star.distance)
        return east_arcsec, north_arcsec

    def _transmission_maps(self, image_shape):
        """Return one cached sky map per transmitted instrument."""
        _, n_rows, n_columns = image_shape
        cache_key = (n_rows, n_columns, float(self.pixAU))
        cached = self._sky_transmission_cache.get(cache_key)
        if cached is not None:
            return cached

        east_arcsec, north_arcsec = self._sky_coordinates_arcsec(
            n_rows, n_columns, self.pixAU)
        transmission_maps = {}
        for instrument, transmission in self.transmission_by_instrument.items():
            if transmission is not None:
                transmission_maps[instrument] = (
                    transmission.sky_transmission(east_arcsec, north_arcsec))
        self._sky_transmission_cache[cache_key] = transmission_maps
        return transmission_maps

    def _apply_transmission(self, images, transmission_maps):
        """Multiply images in place, skipping full-transmission wavelengths."""
        for instrument, transmission_map in transmission_maps.items():
            for wavelength_index in self._wavelength_indices_by_instrument[
                    instrument]:
                images[wavelength_index] *= transmission_map

    def get_image(self, keep_separate_fluxes=False, **model_parameters):
        """Return image flux per pixel after sky-plane FOV transmission."""
        image_start = time.perf_counter()
        result = super().get_image(
            keep_separate_fluxes=keep_separate_fluxes,
            **model_parameters)
        image_ready = time.perf_counter()

        reference_images = result[0] if keep_separate_fluxes else result
        transmission_start = time.perf_counter()
        transmission_maps = self._transmission_maps(reference_images.shape)
        if keep_separate_fluxes:
            for component in result:
                self._apply_transmission(component, transmission_maps)
        else:
            self._apply_transmission(result, transmission_maps)
        transmission_ready = time.perf_counter()

        self.timings['image_before_transmission'] = image_ready - image_start
        self.timings['fov_transmission'] = (
            transmission_ready - transmission_start)
        self.timings['total_with_fov'] = (
            transmission_ready - image_start)
        return result
