"""Normalized dust-scattering phase functions."""

import numpy as np


def isotropic(scattering_angle_radian, **parameters):
    """Return equal scattered intensity per unit solid angle."""
    del parameters
    return np.ones(scattering_angle_radian.shape) / (4.0 * np.pi)


def HenveyGreenstein(scattering_angle_radian, **parameters):
    """Return the normalized Henyey-Greenstein phase function.

    ``g=0`` is isotropic and positive ``g`` produces forward scattering.
    The historical public function name is retained even though "Henyey" is
    misspelled in it.
    """
    asymmetry_parameter = parameters['g']
    cosine_scattering_angle = np.cos(scattering_angle_radian)
    return (
        (1.0 - asymmetry_parameter**2) / (4.0 * np.pi)
        / (1.0 + asymmetry_parameter**2
           - 2.0 * asymmetry_parameter * cosine_scattering_angle)**1.5)

#%%
if __name__=='__main__':   
    import matplotlib.pyplot as plt
    phi = np.linspace(-np.pi, np.pi, 100)
    hg = HenveyGreenstein(phi, g=0.5)
    
    plt.figure(figsize=(8, 6))
    plt.plot(phi*180/np.pi, hg, label='HG g=0.5', color='steelblue', linewidth=2.5)
    plt.xlabel('Scattering Angle [degrees]', fontsize=14)
    plt.ylabel('Phase Function', fontsize=14)
    plt.title('Henyey-Greenstein Phase Function', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    # plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
# %%
