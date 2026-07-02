"""
    Filename: config.py
    Author: Ericka Florio
    Description: Load the run settings and the selected cosmology preset from the
                 YAML files in configs/, compute the derived cosmological
                 quantities, and expose everything as a single flat namespace.
                 This is the single source of truth for the parameters that were
                 previously hard-coded in parameters.py.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import yaml

# configs/ lives at the repository root, two levels above this package file
# (src/gal_env_stats/config.py -> src/ -> repo root).
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader with a corrected float resolver (see below)."""


# PyYAML's default float resolver requires a signed exponent, so scalars such as
# "1.0e14" or "1e14" wrongly parse as strings. Register a resolver that also
# accepts unsigned exponents, so scientific-notation masses load as floats.
_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
             |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
             |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
             |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
             |[-+]?\.(?:inf|Inf|INF)
             |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=_ConfigLoader)


def load_config(run_path=None, cosmology=None):
    """
        Load the run settings and the selected cosmology preset, compute the
        derived cosmological quantities, and return them as a flat namespace.

        run_path : path to the run-settings YAML (defaults to configs/run.yaml).
        cosmology : name of the cosmology preset to load, overriding the
                    'cosmology' key in the run settings (e.g. 'concordance',
                    'eds').
    """

    run_path = Path(run_path) if run_path else CONFIG_DIR / "run.yaml"
    run = _load_yaml(run_path)

    # Choose and load the cosmology preset.
    cosmology = cosmology or run["cosmology"]
    cosmo = _load_yaml(CONFIG_DIR / f"{cosmology}.yaml")

    # Pull the nested run settings into the flat names used across the code.
    sf = run["scale_factor"]
    gf = run["growth_factor"]
    dp = run["density_profile"]
    dd = run["double_distribution"]
    beta, rho, mass, gamma = dd["beta"], dd["rho"], dd["mass"], dd["gamma"]
    quant, plot = dd["quantiles"], dd["plot"]

    params = dict(
        verbose=run["verbose"],

        # Scale-factor range
        a_i=sf["a_i"], a_f=sf["a_f"], num_steps=sf["num_steps"],

        # Growth-factor program flags
        print_D_a=gf["print_D_a"],
        compare_case_1=gf["compare_case_1"],
        compare_case_2=gf["compare_case_2"],

        # Density-profile settings
        power_law_approx=dp["power_law_approx"],
        default_gamma=dp["default_gamma"],
        root_finder_precision=dp["root_finder_precision"],

        # Double-distribution flags
        enforce_positive_pdf=dd["enforce_positive_pdf"],
        transform_pdf=dd["transform_pdf"],
        normalise_pdf=dd["normalise_pdf"],

        # Double-distribution grid
        beta_min=beta["min"], beta_max=beta["max"], num_beta=beta["num"],
        beta_heuristic=beta["heuristic"],
        rho_tilde_min=rho["min"], rho_tilde_max=rho["max"], num_rho=rho["num"],
        mass_min=mass["min"], mass_max=mass["max"], num_mass=mass["num"],
        gamma_min=gamma["min"], gamma_max=gamma["max"], num_gamma=gamma["num"],
        lqr=quant["lqr"], uqr=quant["uqr"],

        # Double-distribution plot selection
        plot_dimension=plot["plot_dimension"],
        slice_in_rho=plot["slice_in_rho"],
        slice_in_beta=plot["slice_in_beta"],
        plot_statistics=plot["plot_statistics"],
        plot_untransformed_PDF=plot["plot_untransformed_PDF"],
        plot_sis=plot["plot_sis"],
        compare_pla=plot["compare_pla"],
        mode_error=plot["mode_error"],
        plot_rho_derivative=plot["plot_rho_derivative"],

        # Shared constants
        h=run["h"], n=run["n"], M_200=run["M_200"],

        # Cosmology preset (base values)
        Omega_m=cosmo["Omega_m"],
        s_8=cosmo["s_8"], m_8=cosmo["m_8"], delta_c=cosmo["delta_c"],
    )

    # Derived cosmological quantities (same expressions as the original code).
    Omega_m = params["Omega_m"]
    Omega_L = cosmo["Omega_L"] if "Omega_L" in cosmo else 1 - Omega_m
    params["Omega_L"] = Omega_L
    params["w"] = w = Omega_L / Omega_m
    params["phi"] = (Omega_m + Omega_L - 1) / Omega_m
    params["kappa"] = 1.50 * 3 * pow(w, 1 / 3) / pow(2, 2 / 3)
    params["rho_c"] = 2.78e11 * (params["h"] ** 2)

    return SimpleNamespace(**params)
