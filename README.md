<div align="center">

# pyGRaTer

**Debris disk modeling and radiative transfer for optically thin media**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## Overview

pyGrater is a Python package for computing grain temperatures, scattering/emission efficiencies, spectral energy distributions (SEDs), and synthetic images of debris disks around stars.

**Key features:**
- Mie theory grain efficiency calculations (Qabs, Qsca, Qpr)
- Grain temperature equilibrium as a function of stellar type and distance
- Optimized SED and image generation for optically thin disks
- One shared `Fluxes` radiative-transfer implementation for SEDs and images
- Support for a wide range of grain compositions and stellar spectra

---

## Installation

```bash
# Standard install
pip install .

# Editable (developer) mode
pip install -e .
```

---

## Data Setup

pyGrater requires an external data directory containing optical properties, stellar catalogs, filter curves, and pre-computed efficiencies.

**Download the data:** https://osf.io/mqkyf/overview

Then configure the data path with one of the following options:

<details>
<summary><b>Option 1 — Python</b></summary>

```python
import pyGrater
pyGrater.set_data_path("/path/to/downloaded/data")
```
</details>

<details>
<summary><b>Option 2 — Environment variable</b></summary>

```bash
export PYGRATER_DATA_PATH=/path/to/downloaded/data
```
</details>

<details>
<summary><b>Option 3 — Command-line helper</b> (pip install only)</summary>

```bash
pygrater-setup --data-path /path/to/downloaded/data
pygrater-setup --show   # verify current config
```
</details>

> The persistent config is stored in `~/.pygrater/config.json`.
> If no data path is configured, pyGrater raises a `FileNotFoundError` with setup instructions.

---

## Quick Start

```python
import pyGrater

# Load a grain and a star
grain = pyGrater.Grain(composition="aC_ACAR")
star  = pyGrater.Star(star_name="bPic")

# Compute grain temperatures
temp = pyGrater.Temperature(grain, star)
```

See the `examples/` folder for full Jupyter notebook tutorials.

---

## Fitting Models

SciPy, emcee, nested-sampling, field-of-view, and interferometric fitters live
in the separate `pyGraterFit` package:

```python
from pyGraterFit import SingleRingSEDScipyFitter, SingleRingSEDMCMCFitter
```

Its README and `examples/` directory document single-component,
multi-ring/multi-composition, restartable MCMC, nested-sampling, and
interferometric workflows.

---

## Performance

The public `SED`, `Image`, and `Fluxes` classes are the fastest validated
implementations developed for repeated fitting. The first call includes Numba
compilation; the values below describe warmed repeated evaluations.

### SED timing

The SED was benchmarked using HD113766 and `c_olivine_Fe_Poor`.

| Wavelengths | Previous | Current | Additional speedup |
|---:|---:|---:|---:|
| 4 | 0.169 s | 0.127 s | 1.34x |
| 16 | 0.159 s | 0.080 s | 1.99x |
| 64 | 0.156 s | 0.136 s | 1.15x |
| 128 | 0.167 s | 0.151 s | 1.11x |
| 256 | 0.246 s | 0.206 s | 1.20x |
| 500 | 0.418 s | 0.372 s | 1.12x |

These are representative median warmed timings on the development machine;
absolute times depend on CPU, thread count, and wavelength/parameter choices.
The 500-wavelength row used 10 repeated parameter draws to reduce timing
noise.

The implementation automatically selects the most efficient thermal-emission
kernel for small and large wavelength grids. It avoids constructing
wavelength-by-distance arrays during ordinary disk-integrated SED fitting.

---

## Using Stars

### Load from the catalog

```python
from pyGrater import Star

star = Star(star_name="bPic")
print(star.temp, star.distance, star.lum)
```

### Define a star inline (no catalog entry needed)

Pass properties directly as keyword arguments:

| kwarg   | unit | required | description |
|---------|------|:--------:|-------------|
| `dist`  | pc   | ✓ | distance |
| `temp`  | K    | ✓ | effective temperature |
| `rad`   | R☉   | ✓ | stellar radius |
| `logg`  | cgs  | ✓ | surface gravity |
| `band`  | —    | ✓ | photometric band for normalisation |
| `apmag` | mag  | ✓ | apparent magnitude in that band |
| `spt`   | —    |   | spectral type *(optional)* |

```python
star = Star(dist=19.3, temp=8052, rad=1.8, logg=4.1, band="V", apmag=3.86, spt="A6V")
```

### Add a star to the catalog permanently

```python
from pyGrater.add_stars import add_star

add_star(
    star="MyStar", dist=42.0, temp=6500, rad=1.3, logg=4.1, band="V", apmag=6.2,
    mass=1.2, spt="F5V", vsini=15.0,  # optional fields — all others default to nan
)
```

```bash
# Or from the terminal:
python -m pyGrater.add_stars --star MyStar --dist 42.0 --temp 6500 \
  --rad 1.3 --logg 4.1 --band V --apmag 6.2 --spt F5V
```

> Raises `ValueError` if the star name already exists in the catalog.

---

## Adding Grain Materials

> **Step 1 — copy your optical index file(s) into:**
> ```
> data/optical_properties/
> ```
> The filenames passed to `file_par`, `file_per1`, `file_per2` are resolved relative to that folder.

**Step 2 — register the material:**

```python
from pyGrater.add_materials import add_material

# Single optical file for all orientations
add_material(nickname="my_dust", Tsub=1700, density=3.5, file_par="my_dust.txt")

# Separate files per orientation + metadata
add_material(
    nickname="my_dust", Tsub=1700, density=3.5,
    file_par="my_dust_par.txt", file_per1="my_dust_per1.txt", file_per2="my_dust_per2.txt",
    wav_min=0.2, wav_max=500.0, full_name="My custom silicate",
    formula="MgSiO3", reference="Author et al. 2025",
)
```

```bash
# Or from the terminal:
python -m pyGrater.add_materials --nickname my_dust --Tsub 1700 \
  --density 3.5 --file_par my_dust.txt --formula MgSiO3
```

| field | required | default |
|-------|:--------:|---------|
| `nickname`, `Tsub`, `density`, `file_par` | ✓ | — |
| `file_per1`, `file_per2` | | same as `file_par` |
| `weight_par/per1/per2` | | `0.333333` each |
| all metadata fields | | empty / `nan` |

---

## Notebooks

| # | Topic |
|---|-------|
| 1 | Calculating grain efficiencies Q |
| 2 | Working with stars |
| 3 | Grain temperatures |
| 4 | Flux profiles |
| 5a | Making SEDs |
| 5b | Making images |
| 6 | Phase functions |
| 7 | Adding new stars and materials |

---

## Logging

pyGrater uses the standard Python `logging` module. Importing `pyGrater`
configures a console INFO handler for the package logger, and file logging is
opt-in.

```python
import logging
import pyGrater

# Show more detailed pyGrater messages
pyGrater.configure_logging(level=logging.DEBUG)

# Also write pyGrater logs to a timestamped file
pyGrater.configure_logging(log_to_file=True, log_dir="/path/to/my/logs")
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError` for data path | Run `pygrater-setup --data-path /path` or set `PYGRATER_DATA_PATH` |
| `ValueError: Unknown star` | Check spelling; ensure the name exists in `stars_main_properties.txt` |
| Slow first run for a composition | Efficiency files are computed once and cached — subsequent runs are fast |
