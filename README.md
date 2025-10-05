# Beyond the Mean: Advancing the Analytic Outer Density Profile

[![Documentation](https://img.shields.io/badge/Docs-PDF-blue)](docs/paper.pdf)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> _Analytic exploration of the outer density profile beyond the mean-field approximation._

---

## Overview

This repository contains the code and analysis accompanying the paper  
**"Beyond the Mean: Advancing the Analytic Outer Density Profile"**  
by **Ericka Florio (DAMTP, University of Cambridge & FORTH, University of Crete)**.

📄 **Full documentation:** [Read the PDF here](docs/documentation.pdf)

---

## Repository Structure
```
galactic-environment-statistics/
├── src/ # Main programs
│ ├── growth-factor.py # Calculate and plot the growth factor
│ ├── double-distribution.py # Calculate and plot the double distribution
│ ├── density-profile.py # Calculate and plot the most probable density profile 
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

