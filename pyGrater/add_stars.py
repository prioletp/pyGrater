"""
add_stars.py
----------------
Utility for appending a new star to data/star_data/stars_main_properties.txt.

Column descriptions (from the file header)
-------------------------------------------
star   : star name  (no spaces)
dist   : distance (pc)
temp   : surface temperature (K)
rad    : star radius (R_sun)
mass   : star mass (M_sun)                          [optional]
logg   : star log(g) (CGS)
spt    : spectral type                              [optional]
band   : spectral band used for normalisation
apmag  : apparent magnitude in that band
vsini  : rotation velocity (km/s)                  [optional]
mdot   : stellar mass loss (10^-14 M_sun/year)      [optional]
vw     : stellar wind bulk velocity (m/s)           [optional]
tcoro  : stellar wind coronal temperature (K)       [optional]
B0     : magnetic field intensity at r0             [optional]
r0     : radius corresponding to B0 (AU)            [optional]
tilt   : tilt angle between magnetic and rotation axis (rad) [optional]
per    : global cycle period (year)                 [optional]

Required fields
---------------
star, dist, temp, rad, logg, band, apmag

All other fields default to nan / empty string.
"""
import logging

from pathlib import Path
import pyGrater


# Column order matching stars_main_properties.txt


logger = logging.getLogger(__name__)
_COLUMNS = [
    "star", "dist", "temp", "rad", "mass", "logg",
    "spt", "band", "apmag", "vsini", "mdot", "vw",
    "tcoro", "B0", "r0", "tilt", "per",
]

_NAN = "nan"


def add_star(
    star,
    dist,
    temp,
    rad,
    logg,
    band,
    apmag,
    mass=float("nan"),
    spt=_NAN,
    vsini=float("nan"),
    mdot=float("nan"),
    vw=float("nan"),
    tcoro=float("nan"),
    B0=float("nan"),
    r0=float("nan"),
    tilt=float("nan"),
    per=float("nan"),
):
    """Append a new star row to *star_properties_path*.

    Parameters
    ----------
    star_properties_path : str or Path
        Path to ``stars_main_properties.txt``.
    star : str
        Star name / identifier (no spaces).  Must be unique in the file.
    dist : float
        Distance in pc.
    temp : float
        Effective surface temperature in K.
    rad : float
        Stellar radius in R_sun.
    logg : float
        Surface gravity log(g) in CGS.
    band : str
        Photometric band used for flux normalisation (e.g. ``'V'``).
    apmag : float
        Apparent magnitude in *band*.
    mass : float, optional
        Stellar mass in M_sun (default: nan).
    spt : str, optional
        Spectral type, e.g. ``'A6V'`` (default: nan).
    vsini : float, optional
        Projected rotation velocity in km/s (default: nan).
    mdot : float, optional
        Stellar mass-loss rate in units of 1e-14 M_sun/year (default: nan).
    vw : float, optional
        Stellar wind bulk velocity in m/s (default: nan).
    tcoro : float, optional
        Coronal temperature in K (default: nan).
    B0 : float, optional
        Magnetic field intensity at *r0* (default: nan).
    r0 : float, optional
        Radius corresponding to B0, in AU (default: nan).
    tilt : float, optional
        Tilt angle between magnetic and rotation axis in rad (default: nan).
    per : float, optional
        Global cycle period in years (default: nan).

    Raises
    ------
    ValueError
        If *star* already exists in the file.
    """
    star_properties_path = pyGrater.get_data_path() / "star_data" / "stars_main_properties.txt"

    # Validate uniqueness
    if star_properties_path.exists():
        with open(star_properties_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                existing_name = stripped.split()[0]
                if existing_name == str(star).strip():
                    raise ValueError(
                        f"Star '{star}' already exists in {star_properties_path}. "
                        "Choose a different name or remove the existing entry first."
                    )

    def _fmt(v):
        """Format a value: nan for float NaN, as-is for strings."""
        if isinstance(v, float) and v != v:  # NaN check
            return "nan"
        return str(v)

    fields = [
        _fmt(star),
        _fmt(dist),
        _fmt(temp),
        _fmt(rad),
        _fmt(mass),
        _fmt(logg),
        _fmt(spt),
        _fmt(band),
        _fmt(apmag),
        _fmt(vsini),
        _fmt(mdot),
        _fmt(vw),
        _fmt(tcoro),
        _fmt(B0),
        _fmt(r0),
        _fmt(tilt),
        _fmt(per),
    ]

    new_row = "\t".join(fields)

    with open(star_properties_path, "a", encoding="utf-8") as f:
        f.write("\n" + new_row)

    logger.info(f"Star '{star}' added to {star_properties_path}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add a new star entry to pyGrater's stars_main_properties.txt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Column descriptions (from file header)
---------------------------------------
  star   : star name (no spaces) — used to look up the star in pyGrater
  dist   : distance (pc)
  temp   : surface temperature (K)
  rad    : star radius (R_sun)
  mass   : star mass (M_sun)
  logg   : star log(g) (CGS)
  spt    : spectral type (e.g. A6V)
  band   : spectral band for normalisation (e.g. V)
  apmag  : apparent magnitude in that band
  vsini  : rotation velocity (km/s)
  mdot   : stellar mass loss (10^-14 M_sun/year)
  vw     : stellar wind bulk velocity (m/s)
  tcoro  : stellar wind coronal temperature (K)
  B0     : magnetic field intensity at r0
  r0     : radius corresponding to B0 (AU)
  tilt   : tilt angle between magnetic and rotation axis (rad)
  per    : global cycle period (year)
""",
    )

    # Required (path is resolved automatically from pyGrater data config)
    parser.add_argument("--star",  required=True,               help="Star name (no spaces)")
    parser.add_argument("--dist",  required=True, type=float,   help="Distance (pc)")
    parser.add_argument("--temp",  required=True, type=float,   help="Surface temperature (K)")
    parser.add_argument("--rad",   required=True, type=float,   help="Stellar radius (R_sun)")
    parser.add_argument("--logg",  required=True, type=float,   help="Surface gravity log(g) [CGS]")
    parser.add_argument("--band",  required=True,               help="Photometric band for normalisation (e.g. V)")
    parser.add_argument("--apmag", required=True, type=float,   help="Apparent magnitude in that band")

    # Optional
    parser.add_argument("--mass",  type=float, default=float("nan"), help="Stellar mass (M_sun)  [default: nan]")
    parser.add_argument("--spt",   default="nan",                    help="Spectral type (e.g. A6V)  [default: nan]")
    parser.add_argument("--vsini", type=float, default=float("nan"), help="Rotation velocity (km/s)  [default: nan]")
    parser.add_argument("--mdot",  type=float, default=float("nan"), help="Mass loss (10^-14 M_sun/yr)  [default: nan]")
    parser.add_argument("--vw",    type=float, default=float("nan"), help="Wind bulk velocity (m/s)  [default: nan]")
    parser.add_argument("--tcoro", type=float, default=float("nan"), help="Coronal temperature (K)  [default: nan]")
    parser.add_argument("--B0",    type=float, default=float("nan"), help="Magnetic field at r0  [default: nan]")
    parser.add_argument("--r0",    type=float, default=float("nan"), help="Radius for B0 (AU)  [default: nan]")
    parser.add_argument("--tilt",  type=float, default=float("nan"), help="Magnetic/rotation axis tilt (rad)  [default: nan]")
    parser.add_argument("--per",   type=float, default=float("nan"), help="Global cycle period (year)  [default: nan]")

    args = parser.parse_args()

    add_star(
        star=args.star,
        dist=args.dist,
        temp=args.temp,
        rad=args.rad,
        logg=args.logg,
        band=args.band,
        apmag=args.apmag,
        mass=args.mass,
        spt=args.spt,
        vsini=args.vsini,
        mdot=args.mdot,
        vw=args.vw,
        tcoro=args.tcoro,
        B0=args.B0,
        r0=args.r0,
        tilt=args.tilt,
        per=args.per,
    )
