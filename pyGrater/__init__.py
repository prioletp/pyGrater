"""Public pyGrater API."""

from pyGrater.config.logging_config import configure_logging, setup_logger

configure_logging()

from pyGrater.add_materials import add_material
from pyGrater.add_stars import add_star
from pyGrater.config.paths import DataPathConfig
from pyGrater.fluxes import Fluxes
from pyGrater.image import Image
from pyGrater.SED import SED
from pyGrater.SED_cached import CachedSED, SharedSEDCache
from pyGrater.stargrains import Grain, Star
from pyGrater.temperatures import Temperature

def set_data_path(path, persistent=True):
    """Set the external pyGrater data directory."""
    return DataPathConfig.set_data_path(path, persistent)


def get_data_path():
    """Return the configured external pyGrater data directory."""
    return DataPathConfig.get_data_path()


__all__ = [
    "Fluxes",
    "Grain",
    "Image",
    "SED",
    "CachedSED",
    "SharedSEDCache",
    "Star",
    "Temperature",
    "add_material",
    "add_star",
    "configure_logging",
    "get_data_path",
    "set_data_path",
    "setup_logger",
]
