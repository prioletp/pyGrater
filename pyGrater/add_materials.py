"""
add_materials.py
--------------------
Utility for appending a new grain material to data/material_list.txt.

Required fields
---------------
nickname   : str   — short identifier used throughout pyGrater (no spaces)
Tsub       : float — sublimation temperature in K
density    : float — bulk density in g/cm³
file_par   : str   — optical-property file for the parallel orientation
                     (if this is the only file given, per1 and per2 are set
                      to the same value automatically)
file_per1  : str   — optical-property file for perpendicular orientation 1
file_per2  : str   — optical-property file for perpendicular orientation 2

Optional fields (defaults shown)
---------------------------------
wav_min        : float  — minimum wavelength in µm  (default: nan)
wav_max        : float  — maximum wavelength in µm  (default: nan)
weight_par     : float  — weight for par orientation (default: 0.333333)
weight_per1    : float  — weight for per1 orientation (default: 0.333333)
weight_per2    : float  — weight for per2 orientation (default: 0.333333)
full_name      : str    — human-readable name (default: '')
formula        : str    — chemical formula (default: '')
material_class : str    — material class (default: '')
subclass       : str    — material subclass (default: '')
group          : str    — mineral group (default: '')
reference      : str    — bibliographic reference (default: '')
web            : str    — URL (default: '')
"""
import logging

from pathlib import Path
import pyGrater


# Column order as found in material_list.txt


logger = logging.getLogger(__name__)
_COLUMNS = [
    "Nickname",
    "Wav_min[microns]",
    "Wav_max[microns]",
    "Tsub[K]",
    "Density[g/cm3]",
    "File_par",
    "File_per1",
    "File_per2",
    "Weight_par",
    "Weight_per1",
    "Weight_per2",
    "Full_Name",
    "Formula",
    "Class",
    "Subclass",
    "Group",
    "Reference",
    "Web",
]


def add_material(
    nickname,
    Tsub,
    density,
    file_par,
    file_per1=None,
    file_per2=None,
    wav_min=float("nan"),
    wav_max=float("nan"),
    weight_par=0.333333,
    weight_per1=0.333333,
    weight_per2=0.333333,
    full_name="",
    formula="",
    material_class="",
    subclass="",
    group="",
    reference="",
    web="",
):
    """Append a new material row to *material_list_path*.

    Parameters
    ----------
    material_list_path : str or Path
        Path to ``material_list.txt``.
    nickname : str
        Short identifier (no spaces).  Must be unique in the file.
    Tsub : float
        Sublimation temperature in K.
    density : float
        Bulk density in g/cm³.
    file_par : str
        Optical-property file for the parallel orientation.
        If *file_per1* and *file_per2* are omitted, all three orientations
        are set to this file.
    file_per1 : str, optional
        Optical-property file for perpendicular orientation 1.
    file_per2 : str, optional
        Optical-property file for perpendicular orientation 2.
    wav_min : float, optional
        Minimum wavelength in µm.
    wav_max : float, optional
        Maximum wavelength in µm.
    weight_par, weight_per1, weight_per2 : float, optional
        Orientation weights (each defaults to 1/3).
    full_name, formula, material_class, subclass, group, reference, web : str, optional
        Descriptive metadata fields.

    Raises
    ------
    ValueError
        If *nickname* already exists in the file.
    """
    material_list_path = pyGrater.get_data_path() / "material_list.txt"

    # Default per1/per2 to the same file as par when only one file is given
    if file_per1 is None:
        file_per1 = file_par
    if file_per2 is None:
        file_per2 = file_par

    # Validate uniqueness
    if material_list_path.exists():
        existing_text = material_list_path.read_text(encoding="utf-8")
        for line in existing_text.splitlines():
            if line.strip().startswith("#") or not line.strip():
                continue
            existing_nickname = line.split("$")[0].strip()
            if existing_nickname == nickname.strip():
                raise ValueError(
                    f"Material '{nickname}' already exists in {material_list_path}. "
                    "Choose a different nickname or remove the existing entry first."
                )

    # Format numeric fields (nan stays as 'nan')
    def _fmt_float(v):
        if v != v:  # NaN check
            return "nan"
        return f"{v:g}"

    row_fields = [
        f" {nickname} ",
        f"      {_fmt_float(wav_min)} ",
        f"      {_fmt_float(wav_max)} ",
        f"      {_fmt_float(Tsub)} ",
        f"      {_fmt_float(density)} ",
        f" {file_par} ",
        f" {file_per1} ",
        f" {file_per2} ",
        f"     {weight_par:.6f} ",
        f"     {weight_per1:.6f} ",
        f"     {weight_per2:.6f} ",
        f" {full_name} ",
        f" {formula} ",
        f" {material_class} ",
        f" {subclass} ",
        f" {group} ",
        f" {reference} ",
        f" {web}",
    ]

    new_row = "$".join(row_fields)

    with open(material_list_path, "a", encoding="utf-8") as f:
        f.write("\n" + new_row)

    logger.info(f"Material '{nickname}' added to {material_list_path}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add a new material entry to pyGrater's material_list.txt."
    )

    # Required
    parser.add_argument("--nickname",  required=True, help="Short identifier (no spaces)")
    parser.add_argument("--Tsub",      required=True, type=float, help="Sublimation temperature [K]")
    parser.add_argument("--density",   required=True, type=float, help="Bulk density [g/cm³]")
    parser.add_argument("--file_par",  required=True,
                        help="Optical-property file (par orientation). "
                             "If --file_per1/--file_per2 are omitted, this file is used for all three.")

    # Semi-optional optical files
    parser.add_argument("--file_per1", default=None, help="Optical-property file (per1 orientation)")
    parser.add_argument("--file_per2", default=None, help="Optical-property file (per2 orientation)")

    # Optional numeric
    parser.add_argument("--wav_min",     type=float, default=float("nan"), help="Min wavelength [µm]")
    parser.add_argument("--wav_max",     type=float, default=float("nan"), help="Max wavelength [µm]")
    parser.add_argument("--weight_par",  type=float, default=0.333333,     help="Weight par  (default 0.333333)")
    parser.add_argument("--weight_per1", type=float, default=0.333333,     help="Weight per1 (default 0.333333)")
    parser.add_argument("--weight_per2", type=float, default=0.333333,     help="Weight per2 (default 0.333333)")

    # Optional metadata
    parser.add_argument("--full_name",       default="", help="Human-readable material name")
    parser.add_argument("--formula",         default="", help="Chemical formula")
    parser.add_argument("--material_class",  default="", help="Material class")
    parser.add_argument("--subclass",        default="", help="Material subclass")
    parser.add_argument("--group",           default="", help="Mineral group")
    parser.add_argument("--reference",       default="", help="Bibliographic reference")
    parser.add_argument("--web",             default="", help="URL")

    args = parser.parse_args()

    add_material(
        nickname=args.nickname,
        Tsub=args.Tsub,
        density=args.density,
        file_par=args.file_par,
        file_per1=args.file_per1,
        file_per2=args.file_per2,
        wav_min=args.wav_min,
        wav_max=args.wav_max,
        weight_par=args.weight_par,
        weight_per1=args.weight_per1,
        weight_per2=args.weight_per2,
        full_name=args.full_name,
        formula=args.formula,
        material_class=args.material_class,
        subclass=args.subclass,
        group=args.group,
        reference=args.reference,
        web=args.web,
    )
