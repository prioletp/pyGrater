import logging
#%%

# from pyGrater import grain_temperatures
from pyGrater import utils as utl 
from pyGrater.config.paths import DataPathConfig
from pyGrater.config.logging_config import log_banner
from scipy.stats import binned_statistic

from pathlib import Path
import yaml
import os
import numpy as np
from astropy import units 
from astropy import constants as cst
from scipy.integrate import simpson
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from pprint import pformat  # already imported

from astropy.io import fits




logger = logging.getLogger(__name__)
class Temperature:
    def __init__(self, grain, star, init_thermal_distance=True, N_temp=300, redo_therm_dist=False, talk=True):
        """
        Initialize a Temperature object for grain temperature calculations.
        
        This class computes the thermal equilibrium temperature of grains
        as a function of their size and distance from the star.
        
        Parameters
        ----------
        grain : Grain
            Grain object
        star : Star
            Star object 
        init_thermal_distance : bool, optional
            Whether to initialize thermal distance array (default: True)
        N_temp : int, optional
            Number of temperature bins (default: 300)
        redo_therm_dist : bool, optional
            Force recalculation of thermal distances (default: False)
        talk : bool, optional
            Enable verbose output (default: True)
        
        Attributes
        ----------
        grain : Grain
            Associated grain object
        star : Star
            Associated star object
        N_temp : int
            Number of temperature points
        therm_dist : ndarray
            Thermal equilibrium distances as function of grain size and temperature
            Shape: (N_sizes, N_temp)
        temp_range : ndarray
            Temperature array in Kelvin
        temperatures : ndarray
            Grain temperatures as function of size and distance (after calling get_temperature)
        
        Examples
        --------
        Create Temperature object and compute temperatures:
        
        >>> grain = Grain()
        >>> star = Star('bPic')
        >>> temp = Temperature(grain, star)
        
        Get temperatures at specific distances:
        
        >>> distances = np.linspace(10, 100, 50)  # AU
        >>> temps = temp.get_temperature(distances)
        >>> print(temps.shape)  # (N_sizes, 50)
        
        Force recalculation of thermal distances:
        
        >>> temp = Temperature(grain, star, redo_therm_dist=True)
        
        Use higher temperature resolution:
        
        >>> temp = Temperature(grain, star, N_temp=500)
        
        Plot temperature distribution:
        
        >>> temp.plot_temperatures(min_size=1e-7, max_size=1e-4)
        
        Notes
        -----
        The thermal distance array is cached to disk for reuse.
        File naming: temperatures_{composition}_Tsub{Tsub}_star{starname}.npz
        
        The equilibrium temperature is found by balancing absorbed stellar
        radiation with thermal emission from the grain.
        """
        log_banner(
            logger,
            "CREATING TEMPERATURE OBJECT for the grain temperature calculations.",
            width=70)
        self.talk = talk
        self.grain = grain
        self.star = star
        self.N_temp = N_temp
        data_path = DataPathConfig.get_data_path()

        self.path_temperature_data =  data_path/ 'temperatures'   
        self.name_of_array = Path(f'temperatures_{grain.grain_composition_name}_Tsub{grain.Tsub}_star{star.star_name}.npz')
        self.array_path = self.path_temperature_data / self.name_of_array
        
        if init_thermal_distance:
            self.therm_dist, self.temp_range = self.get_therm_dist(redo_therm_dist=redo_therm_dist)

    def get_therm_dist(self, redo_therm_dist=False):
        grain = self.grain
        star = self.star
        
        if redo_therm_dist or not os.path.exists(self.array_path):
            if self.talk:
                logger.info('Calculating the thermal distance array...')
                logger.info('%s %s', 'Creating file:', self.name_of_array)
            self.therm_dist, self.temp_range = utl.calc_therm_dist(grain.Qabs, grain.Qabs_sizes, grain.Qabs_waves,
                                                    star.waves, star.flux,
                                                    grain.Tsub, self.N_temp,
                                                    distance_to_star=star.distance,
                                                    radius_star_Rsun=star.radius, save_path=self.array_path,
                                                    talk=self.talk)
        else:  
            if self.talk:
                logger.info('%s %s', 'Loading the thermal distance array from file:', self.name_of_array)
            data = np.load(self.array_path)
            self.therm_dist = data['therm_dist']
            self.temp_range = data['temp_range']
        
        # print('The thermal distances go from', np.min(self.therm_dist), 'to', np.max(self.therm_dist), 'au')
        return self.therm_dist, self.temp_range
    def get_temperature(self, distances):
        """Get the thermal equilibrium temperature of the grain knowing 
        its position, size and composition."""
        self.T_distances = distances
        self.temperatures = np.array(utl.grain_temperatures(self.therm_dist, self.temp_range, distances, self.grain.Tsub))
        return self.temperatures

    
    def plot_temperatures(self, min_size=None, max_size=None, min_dist=None, max_dist=None):
        sizes_full = self.grain.Qabs_sizes
        distances_full = self.T_distances
        if min_size is None:
            min_size = np.min(sizes_full)
        if max_size is None:
            max_size = np.max(sizes_full)
        if min_dist is None:
            min_dist = np.min(distances_full)
        if max_dist is None:
            max_dist = np.max(distances_full)     
        
        logger.info('%s %s %s %s %s %s %s %s', 'Plotting temperatures for sizes between', min_size, 'and', max_size, 'between distances', min_dist, 'and', max_dist)
        idx_sizes =  np.argwhere((sizes_full < max_size) & (sizes_full > min_size)).flatten() 
        idx_distances = np.argwhere((distances_full < max_dist) & (distances_full > min_dist)).flatten()
        
        sizes = sizes_full[idx_sizes]
        distances = distances_full[idx_distances]
        temperatures = self.temperatures[idx_sizes,:][:,idx_distances]
        
        fig = plt.figure(figsize=(6,6),constrained_layout=False)
        gs1 = fig.add_gridspec(nrows=1, ncols=1)
        ax = fig.add_subplot(gs1[0])
        divider = make_axes_locatable(ax)
        ax.set_xlabel('Distance to the star [a.u]',fontsize=15)
        ax.set_ylabel('Grain size [µm]',fontsize=15)
        temperatures[temperatures>self.grain.Tsub] = np.inf
        plot = ax.contourf(distances,np.log10(sizes),temperatures,cmap='hot',levels=15)
        
        # Fix y-axis tick labels to show proper scientific notation
        yticks = ax.get_yticks()
        ax.set_yticklabels([f'$10^{{{int(y)}}}$' for y in yticks])
        
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cax.set_title('T [K]',fontsize=15)
        fig.colorbar(plot, cax=cax, orientation='vertical')

#%%
if __name__ == "__main__":
    import pyGrater
    # temp = Temperature(grain, star, init_thermal_distance=True, N_temp=300, redo_therm_dist=False)
    # grain.plot_Q(max_wave=None, min_wave=None, min_size=None, max_size=None)


# %%
