"""Dust grain-size distributions."""
import logging

import numpy as np
from scipy.integrate import quad




logger = logging.getLogger(__name__)
def normalize_power_law(size_distribution_function, parameters):
    """Numerically integrate a scalar size-distribution function."""
    minimum_grain_size_m = parameters['a_min']
    maximum_grain_size_m = parameters['a_max']
    normalization_factor = quad(
        size_distribution_function,
        minimum_grain_size_m,
        maximum_grain_size_m,
        args=parameters)[0]
    return normalization_factor


def power_law_distribution(grain_sizes_m, parameters):
    """Return a normalized ``dn/da proportional to a^-kappa`` distribution."""
    power_law_index = parameters['kappa']
    minimum_grain_size_m = parameters['a_min']
    maximum_grain_size_m = parameters['a_max']
    if power_law_index == 1:
        distribution = (
            grain_sizes_m**(-power_law_index)
            / np.log(maximum_grain_size_m / minimum_grain_size_m))
    else:
        distribution = (
            (1.0 - power_law_index) * grain_sizes_m**(-power_law_index)
            / (maximum_grain_size_m**(1.0 - power_law_index)
               - minimum_grain_size_m**(1.0 - power_law_index)))
    return distribution


#%%
if __name__ == "__main__": 
    import matplotlib.pyplot as plt
    sizes = np.logspace(-1, 3, 100)  # Example sizes from 0.1 to 100 micrometers
    power_index = 3.5  # Example power-law index
    a_min = 9.4
    a_max = 1e3
    dic = {'kappa': power_index,
    'a_min': a_min,
    'a_max': a_max}
     # Example usage
    distribution = power_law_distribution(sizes, dic)
    norm_factor = normalize_power_law(power_law_distribution, dic)
    logger.info('%s %s', 'The normalization factor is:', norm_factor)
    plt.semilogx(sizes, distribution)
    # print('Integral over full range (should be 1):', integrate_test(power_index, a_min, a_max))
    # Example integration
    # total = integrate_dn_a_r(0.1, 100, r, power_index)
    # print('Integrated dn(a,r) from 0.1 to 100:', total)
    
# %%
