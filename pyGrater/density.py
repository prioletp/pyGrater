"""Dust spatial-density distributions."""
import logging

import numpy as np




logger = logging.getLogger(__name__)
try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def two_power_law_numba(cylindrical_radius_au, height_above_midplane_au,
                            r0, h0, alphain, alphaout, beta, gamma):
        """Fast compiled version of :func:`two_power_law` for image grids."""
        n_line_of_sight, n_pixels = cylindrical_radius_au.shape
        output = np.empty((n_line_of_sight, n_pixels))
        for pixel_index in prange(n_pixels):
            for line_index in range(n_line_of_sight):
                normalized_radius = (
                    cylindrical_radius_au[line_index, pixel_index] / r0)
                if normalized_radius == 0.0:
                    output[line_index, pixel_index] = 0.0
                    continue
                radial_density = 1.0 / np.sqrt(
                    normalized_radius**(-2.0 * alphain)
                    + normalized_radius**(-2.0 * alphaout))
                scale_height_au = h0 * normalized_radius**beta
                vertical_density = np.exp(-(
                    abs(height_above_midplane_au[line_index, pixel_index])
                    / scale_height_au
                )**gamma)
                output[line_index, pixel_index] = (
                    radial_density * vertical_density)
        return output

    _NUMBA_DENSITY_AVAILABLE = True
except ImportError:
    two_power_law_numba = None
    _NUMBA_DENSITY_AVAILABLE = False


def two_power_law(cylindrical_radius_au, azimuth_radian,
                  height_above_midplane_au, parameters):
    """Return the relative density of an axisymmetric two-slope disk.

    The density peaks near ``r0``.  ``alphain`` and ``alphaout`` control its
    inner and outer radial slopes; ``h0``, ``beta``, and ``gamma`` describe
    the vertical scale height and profile.  ``azimuth_radian`` is accepted for
    the common density-function API but is unused for an axisymmetric disk.
    """
    del azimuth_radian
    normalized_radius = cylindrical_radius_au / parameters['r0']
    radial_density = 1.0 / np.sqrt(
        normalized_radius**(-2.0 * parameters['alphain'])
        + normalized_radius**(-2.0 * parameters['alphaout']))
    scale_height_au = (
        parameters['h0'] * normalized_radius**parameters['beta'])
    vertical_density = np.exp(-(
        np.abs(height_above_midplane_au) / scale_height_au
    )**parameters['gamma'])
    return radial_density * vertical_density




if __name__ == '__main__':
    r0 = 1
    theta = np.linspace(0, 2*np.pi, 100)
    rho = np.linspace(0.1, 10, 120)  # Changed to positive values only
    z = np.linspace(-1,1, 140)
    r, th, z_m = np.meshgrid(rho, theta, z, indexing='ij')
    h0 = 0.1*r0
    alphain = 10.
    alphaout = -4
    gamma = 2.
    beta = 2
    density_params_dic = {'r0': r0, 'h0': h0, 'alphain': alphain, 'alphaout': alphaout,'gamma': gamma, 'beta': beta}
    
    # Calculate density at z=0 as function of radius
    density = two_power_law(r, th, z_m, density_params_dic)

    logger.info('%s %s', 'The shape is:', density.shape)

    
    #%%
    density = two_power_law(rho, theta, z, density_params_dic)

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.loglog(rho, density, color='steelblue', linewidth=2.5)
    plt.xlabel('Radial Distance ρ [AU]', fontsize=16)
    plt.ylabel('Density [arbitrary units]', fontsize=16)
    plt.title('Radial Density Profile at z=0', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    density_z = two_power_law(r0, 0., z, density_params_dic)

    plt.figure(figsize=(10, 6))
    plt.plot(z, density_z, color='steelblue', linewidth=2.5)
    plt.xlabel('Vertical Distance z [AU]', fontsize=16)
    plt.ylabel('Density [arbitrary units]', fontsize=16)
    plt.title('Vertical Density Profile at r=r0', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
# %%
