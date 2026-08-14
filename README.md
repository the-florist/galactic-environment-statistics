# Resolving the Mode: Calculations of the Galactic Outer Density Profile

[![Documentation](https://img.shields.io/badge/Docs-PDF-blue)](docs/documentation.pdf)
[![License: GPL v2](https://img.shields.io/badge/License-GPLv2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

---

## Overview

This repository contains the code and analysis accompanying the paper  
**"The Most Probable Outer Density Profile from Excursion Set
Theory"**  
by **Ericka Florio (DAMTP, University of Cambridge UK & FORTH, University of Crete GR)**
and **Vasiliki Pavlidou (FORTH and the University of Crete, Heraklion GR)**.

### Abstract
The mode of the double distribution describes the “most probable”
galactic outer density profile, and the departure of this profile from
galactic simulations can be used to infer cosmological parameters.
This python library calculates and visualises the double distribution
and the most probable profile, for a given cosmological model.

📄 **Full documentation:** [Read the PDF here](docs/documentation.pdf)

---

## Repository Structure
```
galactic-environment-statistics/
├── pyproject.toml                # Package metadata, dependencies, CLI entry point
├── configs/
│   ├── run.yaml                  # Run settings, grids, and plot options
│   ├── concordance.yaml          # Concordance ΛCDM cosmology preset
│   └── eds.yaml                  # Einstein–de Sitter cosmology preset
├── src/gal_env_stats/
│   ├── cli.py                    # Command-line interface (subcommands)
│   ├── config.py                 # Loads the YAML configuration
│   ├── parameters.py             # Exposes the loaded configuration to the code
│   ├── plotting.py               # Shared plotting / output helpers
│   ├── physics/                  # Core maths: growth, variance, collapse, distribution
│   └── analyses/                 # The three user-facing programs
│       ├── growth_factor.py      # Growth factor D(a) program
│       ├── density_profile.py    # Most probable density profile program
│       └── double_distribution/  # Double distribution program
│           ├── analysis.py       #   orchestrator (class DoubleDistribution)
│           ├── calculations.py   #   PDF grid and sample statistics
│           ├── plotting.py       #   figures for the double distribution
│           └── newton.py         #   root finder for the analytic statistics
├── tests/
│   └── test_smoke.py             # End-to-end smoke tests
├── docs/
│   └── documentation.pdf         # Full LaTeX documentation
└── README.md
```

---

## Installation

Clone the repository and install the package (editable install recommended):

```bash
git clone https://github.com/the-florist/galactic-environment-statistics.git
cd galactic-environment-statistics
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -e .                                     # add "[dev]" to also install pytest
```

## Usage

The three programs are exposed as subcommands of the `gal-env-stats` command:

```bash
gal-env-stats growth-factor         # calculate and plot the growth factor D(a)
gal-env-stats density-profile       # calculate and plot the most probable density profile
gal-env-stats double-distribution   # calculate and plot the double distribution
```

Each is equivalent to `python -m gal_env_stats <subcommand>`. Generated figures are
written to `output/plots/`.

Two global options let you change the cosmological model without editing any files:

```bash
gal-env-stats --cosmology eds growth-factor     # use a different cosmology preset
gal-env-stats --config my_run.yaml double-distribution   # use a different run-settings file
```

Run `gal-env-stats --help` for the full list of options.

## Configuration

All physical parameters and run settings live in the `configs/` directory — no code
changes are needed to alter the model:

- **`run.yaml`** — selects the cosmology preset and sets the scale-factor range, the
  double-distribution grids, and the plotting options.
- **`concordance.yaml`** / **`eds.yaml`** — cosmology presets defining `Omega_m`,
  `Omega_L`, `s_8`, `m_8`, and `delta_c`. Add your own file and point `run.yaml`
  (or `--cosmology`) at it to define a new model.

## Citation

If you use this code or analysis in your research, please cite:
```bibtex
@misc{florio2026probableouterdensityprofile,
      title={The Most Probable Outer Density Profile from Excursion Set Theory}, 
      author={Ericka Florio and Vasiliki Pavlidou},
      year={2026},
      eprint={2608.13347},
      archivePrefix={arXiv},
      primaryClass={astro-ph.CO},
      url={https://arxiv.org/abs/2608.13347}, 
}
```

## Contact

If you would like to report any bugs, feel free to report them on the repository or to contact me directly at [eaf49@cam.ac.uk](mailto:eaf49@cam.ac.uk).
