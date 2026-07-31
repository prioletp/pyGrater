"""Plot and export images and visibilities for one or more optimized disk models.

The public function accepts a list of ``(Image, parameter_dictionary)``
pairs. This naturally supports two rings and multiple compositions: every
component image is calculated and then added before the Fourier transform.

Baseline PA is measured east of North. Therefore ``PA=0`` is a North-South
baseline (u=0, v=B), while ``PA=90`` is East-West (u=B, v=0).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from pyGraterFit.utils.interferometry import (
    complex_visibilities_from_image,
)


def calculate_model_image_cube(image_models):
    """Return the summed ``(wavelength, north/south, east/west)`` image cube."""
    image_models = list(image_models)
    if not image_models:
        raise ValueError('Provide at least one (image_model, parameters) pair.')

    total_image = None
    wavelengths_micron = None
    pixel_scale_au = None
    distance_pc = None
    for image_model, parameters in image_models:
        component_image = np.asarray(
            image_model.get_image(**dict(parameters)), dtype=np.float64)
        component_wavelengths = np.asarray(
            image_model.wavelengths_for_calc, dtype=np.float64)
        if total_image is None:
            total_image = component_image.copy()
            wavelengths_micron = component_wavelengths
            pixel_scale_au = float(image_model.pixAU)
            distance_pc = float(image_model.star.distance)
            continue
        if component_image.shape != total_image.shape:
            raise ValueError('All component image cubes must have the same shape.')
        if not np.array_equal(component_wavelengths, wavelengths_micron):
            raise ValueError('All component wavelength arrays must be identical.')
        if float(image_model.pixAU) != pixel_scale_au:
            raise ValueError('All component images must use the same pixel scale.')
        if float(image_model.star.distance) != distance_pc:
            raise ValueError('All components must have the same stellar distance.')
        total_image += component_image
    return total_image, wavelengths_micron, pixel_scale_au, distance_pc


def calculate_visibility_grid(
        image_cube, wavelengths_micron, pixel_scale_au, distance_pc,
        baseline_lengths_m, baseline_position_angles_degree,
        unresolved_flux_jy=0.0, padding_factor=4,
        stellar_angular_diameter_mas=0.0):
    """Calculate complex visibility for every wavelength, baseline, and PA."""
    image_cube = np.asarray(image_cube, dtype=np.float64)
    wavelengths_micron = np.asarray(wavelengths_micron, dtype=np.float64)
    baseline_lengths_m = np.asarray(baseline_lengths_m, dtype=np.float64)
    baseline_position_angles_degree = np.asarray(
        baseline_position_angles_degree, dtype=np.float64)
    if image_cube.ndim != 3 or image_cube.shape[0] != len(wavelengths_micron):
        raise ValueError('image_cube must have shape (n_wavelengths, ny, nx).')
    if np.any(baseline_lengths_m < 0):
        raise ValueError('Baseline lengths cannot be negative.')

    baseline_grid, pa_grid = np.meshgrid(
        baseline_lengths_m, baseline_position_angles_degree, indexing='xy')
    pa_radian = np.radians(pa_grid)
    # OIFITS convention: u points East and v points North.
    u_m = baseline_grid * np.sin(pa_radian)
    v_m = baseline_grid * np.cos(pa_radian)
    central_flux = np.broadcast_to(
        np.asarray(unresolved_flux_jy, dtype=np.float64),
        wavelengths_micron.shape)
    visibility = np.empty(
        (len(wavelengths_micron), len(baseline_position_angles_degree),
         len(baseline_lengths_m)), dtype=np.complex128)
    for wavelength_index, wavelength_micron in enumerate(wavelengths_micron):
        visibility[wavelength_index] = complex_visibilities_from_image(
            image_cube[wavelength_index], pixel_scale_au, distance_pc,
            u_m, v_m, wavelength_micron * 1e-6,
            unresolved_flux_jy=central_flux[wavelength_index],
            stellar_angular_diameter_mas=stellar_angular_diameter_mas,
            padding_factor=padding_factor)
    return {
        'baseline_length_m': baseline_grid,
        'baseline_pa_degree': pa_grid,
        'u_m': u_m,
        'v_m': v_m,
        'complex_visibility': visibility,
    }


def save_image_cube_fits(
        path, image_cube, wavelengths_micron, pixel_scale_au, distance_pc,
        overwrite=True):
    """Save a North-up, East-left Jy/pixel model cube as FITS."""
    path = Path(path)
    image_cube = np.asarray(image_cube, dtype=np.float64)
    pixel_scale_degree = pixel_scale_au / distance_pc / 3600.0
    header = fits.Header()
    header['BUNIT'] = 'Jy/pixel'
    header['PIXAU'] = (float(pixel_scale_au), 'Pixel scale [au]')
    header['DISTPC'] = (float(distance_pc), 'Stellar distance [pc]')
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    # Array columns increase West and rows increase South.
    header['CDELT1'] = -pixel_scale_degree
    header['CDELT2'] = -pixel_scale_degree
    header['CRPIX1'] = (image_cube.shape[2] + 1.0) / 2.0
    header['CRPIX2'] = (image_cube.shape[1] + 1.0) / 2.0
    header['CRVAL1'] = 0.0
    header['CRVAL2'] = 0.0
    header['ORIENT'] = 'North up, East left'
    primary = fits.PrimaryHDU(image_cube, header=header)
    wavelength_table = fits.BinTableHDU.from_columns([
        fits.Column(name='WAVELENGTH_UM', format='D', unit='um',
                    array=np.asarray(wavelengths_micron, dtype=np.float64)),
    ], name='WAVELENGTHS')
    fits.HDUList([primary, wavelength_table]).writeto(
        path, overwrite=overwrite)
    return path


def save_visibility_fits(
        path, wavelengths_micron, visibility_grid, overwrite=True):
    """Save model complex visibilities and derived observables as FITS."""
    visibility = visibility_grid['complex_visibility']
    n_wavelength, n_pa, n_baseline = visibility.shape
    wavelength = np.broadcast_to(
        np.asarray(wavelengths_micron)[:, None, None], visibility.shape)
    baseline = np.broadcast_to(
        visibility_grid['baseline_length_m'][None, :, :], visibility.shape)
    baseline_pa = np.broadcast_to(
        visibility_grid['baseline_pa_degree'][None, :, :], visibility.shape)
    u_m = np.broadcast_to(visibility_grid['u_m'][None, :, :], visibility.shape)
    v_m = np.broadcast_to(visibility_grid['v_m'][None, :, :], visibility.shape)
    del n_wavelength, n_pa, n_baseline

    columns = [
        fits.Column(name='WAVELENGTH_UM', format='D', unit='um',
                    array=wavelength.ravel()),
        fits.Column(name='BASELINE_M', format='D', unit='m',
                    array=baseline.ravel()),
        fits.Column(name='BASELINE_PA_DEG', format='D', unit='deg',
                    array=baseline_pa.ravel()),
        fits.Column(name='U_M', format='D', unit='m', array=u_m.ravel()),
        fits.Column(name='V_M', format='D', unit='m', array=v_m.ravel()),
        fits.Column(name='VIS_REAL', format='D', array=visibility.real.ravel()),
        fits.Column(name='VIS_IMAG', format='D', array=visibility.imag.ravel()),
        fits.Column(name='VISAMP', format='D',
                    array=np.abs(visibility).ravel()),
        fits.Column(name='VIS2', format='D',
                    array=np.abs(visibility).ravel()**2),
        fits.Column(name='VISPHI_DEG', format='D', unit='deg',
                    array=np.degrees(np.angle(visibility)).ravel()),
    ]
    table = fits.BinTableHDU.from_columns(columns, name='MODEL_VISIBILITY')
    table.header['PACONV'] = 'PA east of North'
    table.header['UCONV'] = 'u positive East'
    table.header['VCONV'] = 'v positive North'
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=overwrite)
    return Path(path)


def plot_model_image_and_visibilities(
        image_cube, wavelengths_micron, pixel_scale_au, visibility_grid,
        image_wavelength_index=0, visibility_wavelength_indices=None):
    """Plot one image and visibility amplitudes at selected wavelengths."""
    if visibility_wavelength_indices is None:
        visibility_wavelength_indices = range(len(wavelengths_micron))
    image = image_cube[image_wavelength_index]
    half_width_au = image.shape[1] * pixel_scale_au / 2.0
    half_height_au = image.shape[0] * pixel_scale_au / 2.0
    figure, (image_axis, visibility_axis) = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True)
    displayed = image_axis.imshow(
        image, origin='upper', extent=(half_width_au, -half_width_au,
                                      -half_height_au, half_height_au),
        cmap='inferno')
    image_axis.set(
        xlabel='East offset [au]', ylabel='North offset [au]',
        title=f'Model image at {wavelengths_micron[image_wavelength_index]:g} um')
    figure.colorbar(displayed, ax=image_axis, label='Flux [Jy/pixel]')

    baselines = visibility_grid['baseline_length_m'][0]
    position_angles = visibility_grid['baseline_pa_degree'][:, 0]
    visibility = visibility_grid['complex_visibility']
    for wavelength_index in visibility_wavelength_indices:
        for pa_index, position_angle in enumerate(position_angles):
            visibility_axis.plot(
                baselines, np.abs(visibility[wavelength_index, pa_index]),
                label=(f'{wavelengths_micron[wavelength_index]:g} um, '
                       f'PA={position_angle:g} deg'))
    visibility_axis.set(
        xlabel='Projected baseline [m]', ylabel='Visibility amplitude',
        ylim=(-0.02, 1.02))
    visibility_axis.grid(True, alpha=0.3)
    visibility_axis.legend(fontsize='small')
    return figure


def create_model_visibility_products(
        image_models, baseline_lengths_m, baseline_position_angles_degree,
        image_fits_path=None, visibility_fits_path=None, plot_path=None,
        unresolved_flux_jy=0.0, padding_factor=4,
        stellar_angular_diameter_mas=0.0,
        image_wavelength_index=0, visibility_wavelength_indices=None):
    """Calculate, plot, and optionally save all model visibility products."""
    images, wavelengths, pixel_scale, distance = calculate_model_image_cube(
        image_models)
    visibility = calculate_visibility_grid(
        images, wavelengths, pixel_scale, distance, baseline_lengths_m,
        baseline_position_angles_degree,
        unresolved_flux_jy=unresolved_flux_jy,
        padding_factor=padding_factor,
        stellar_angular_diameter_mas=stellar_angular_diameter_mas)
    if image_fits_path is not None:
        save_image_cube_fits(
            image_fits_path, images, wavelengths, pixel_scale, distance)
    if visibility_fits_path is not None:
        save_visibility_fits(visibility_fits_path, wavelengths, visibility)
    figure = plot_model_image_and_visibilities(
        images, wavelengths, pixel_scale, visibility,
        image_wavelength_index, visibility_wavelength_indices)
    if plot_path is not None:
        figure.savefig(plot_path, dpi=180)
    return images, visibility, figure
