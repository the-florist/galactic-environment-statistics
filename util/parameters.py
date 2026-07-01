"""
    Filename: parameters.py
    Author: Ericka Florio
    Created: 4th September 2025
    Description: Backwards-compatible shim. All parameters now live in the YAML
                 files under configs/ and are loaded by util/config.py. This
                 module is kept so that existing `import util.parameters as pms`
                 references continue to work: it exposes the loaded configuration
                 as module-level names (pms.Omega_m, pms.num_rho, ...).
"""

from util.config import load_config

globals().update(vars(load_config()))
