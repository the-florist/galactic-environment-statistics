"""
    Filename: parameters.py
    Author: Ericka Florio
    Created: 4th September 2025
    Description: Backwards-compatible shim. All parameters now live in the YAML
                 files under configs/ and are loaded by gal_env_stats/config.py.
                 This module is kept so that existing
                 `import gal_env_stats.parameters as pms` references continue to
                 work: it exposes the loaded configuration as module-level names
                 (pms.Omega_m, pms.num_rho, ...).
"""

from gal_env_stats.config import load_config

globals().update(vars(load_config()))
