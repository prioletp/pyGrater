import logging
import os
import json
from pathlib import Path



logger = logging.getLogger(__name__)
class DataPathConfig:
    """Manage data folder paths for pyGrater"""
    
    CONFIG_FILE = Path.home() / '.pygrater' / 'config.json'
    ENV_VAR = 'PYGRATER_DATA_PATH'
    
    @classmethod
    def get_data_path(cls):
        """Get data path from (in order): env var, config file, or prompt user"""
        
        # 1. Check environment variable
        if cls.ENV_VAR in os.environ:
            path = Path(os.environ[cls.ENV_VAR])
            if path.exists():
                return path
        
        # 2. Check config file
        if cls.CONFIG_FILE.exists():
            with open(cls.CONFIG_FILE, 'r') as f:
                config = json.load(f)
                if 'data_path' in config:
                    path = Path(config['data_path'])
                    if path.exists():
                        return path
        
        # 3. Not configured - raise helpful error
        raise FileNotFoundError(
            f"pyGrater data folder not configured.\n\n"
            f"Please set the data path using one of:\n"
            f"1. Environment variable: export {cls.ENV_VAR}=/path/to/data\n"
            f"2. Run: import pyGrater; pyGrater.set_data_path('/path/to/data')\n"
            f"3. Run from terminal: pygrater-setup --data-path /path/to/data"
        )
    
    @classmethod
    def set_data_path(cls, path, persistent=True):
        """Set data path and optionally save to config file"""
        path = Path(path).resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"Data path does not exist: {path}")
        
        if persistent:
            cls.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump({'data_path': str(path)}, f, indent=2)
            logger.info(f"Data path saved to {cls.CONFIG_FILE}")
        else:
            os.environ[cls.ENV_VAR] = str(path)
            logger.info(f"Data path set for current session")
        
        return path