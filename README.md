# Beyond the Mean: Advancing the Analytic Outer Density Profile

[![Documentation](https://img.shields.io/badge/Docs-PDF-blue)](docs/paper.pdf)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This repository contains the code and analysis accompanying the paper  
**"Beyond the Mean: Advancing the Analytic Outer Density Profile"**  
by **Ericka Florio (DAMTP, University of Cambridge & FORTH, University of Crete)**.

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
├── growth-factor.py # Calculate and plot the growth factor
├── double-distribution.py # Calculate and plot the double distribution
├── density-profile.py # Calculate and plot the most probable density profile 
├── utils/ # Utilities
│ ├── functions.py # Functions used by the src/ programs
│ ├── parameters.py # Parameters used by the src/ programs
├── docs/
│ ├── documentation.pdf # Full LaTeX documentation
└── README.md
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/the-florist/galactic-environment-statistics.git
cd galactic-environment-statistics
pip install -r requirements.txt
```

## Citation

If you use this code or analysis in your research, please cite:
```bibtex
@misc{florio2025beyond,
  author    = {Ericka Florio},
  title     = {Beyond the Mean: Advancing the Analytic Outer Density Profile},
  year      = {2025},
  url       = {https://github.com/<your-username>/<your-repo>}
}
```

## Contact

If you would like to report any bugs, feel free to report them on the repository or to contact me directly at [eaf49@cam.ac.uk](mailto:eaf49@cam.ac.uk).

