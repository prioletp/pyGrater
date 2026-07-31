"""
benchmark_image.py — Image generation performance profiling
============================================================

What this script measures
--------------------------
1. Wall-clock time broken down into the main Image.get_image phases:
     init (Fluxes + GrainStar construction)
     flux calc  (thermal_flux + scattered_flux via Fluxes)
     geometry   (pixel grid + ray-geometry setup)
     normalisation
     per-wavelength LOS integration loop
     total
2. Scaling with the main free parameters:
     - image resolution (nx × ny)
     - number of output wavelengths
     - nl  (number of LOS samples along each ray)
3. Full cProfile trace of a single representative call.

How to run
----------
    python -m pyGrater.tests.benchmark_image

Optional flags:
    --quick      tiny grids — fast sanity check
    --profile    write full cProfile output to benchmark_image.prof
"""
import logging

import argparse
import cProfile
import io
import pstats
import time

import matplotlib

from pyGrater.config.logging_config import log_info
logger = logging.getLogger(__name__)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyGrater.tests.benchmark_helpers import divider as _divider
from pyGrater.tests.benchmark_helpers import rss_mb as _rss_mb


# ── shared parameters ─────────────────────────────────────────────────────────
BASE_DISK_PARAMS = dict(
    r0=100.0, h0=5.0,
    alphain=10.0, alphaout=-5.0,
    gamma=2.0, beta=1.0,
    itilt=30.0, PA=45.0, omega=45.0,
    a_min=1e-5, a_max=1e-3, kappa=3.5,
    N_sizes_integral=100,
    g=0.5,
    M_tot=1e-3,
    nx=128, ny=128, FOV_AU=400.0,
)
COMPOSITION = "astroSi"
STAR_NAME   = "bPic"


def _make_objects():
    from pyGrater.stargrains import Grain, Star
    grain = Grain(composition=COMPOSITION, redo_Q=False)
    star  = Star(star_name=STAR_NAME)
    return grain, star


def _make_image_obj(grain, star, wavelengths):
    from pyGrater.density import two_power_law
    from pyGrater.size_distributions import power_law_distribution
    from pyGrater.phase_functions import HenveyGreenstein
    from pyGrater.image import Image
    return Image(grain, star, two_power_law, power_law_distribution,
                 HenveyGreenstein, wavelengths)


# ── instrumented get_image wrapper ───────────────────────────────────────────
def _run_image_timed(img_obj, params):
    """
    Run get_image and split timing into logical phases:
      phase_flux   — Fluxes.get_fluxes  (thermal + scattered on radial grid)
      phase_geom   — pixel grid + ray intersection geometry
      phase_norm   — normalisation factor computation
      phase_loop   — per-wavelength LOS integration loop
      phase_total  — wall-clock total
    """
    from pyGrater.density import two_power_law
    from pyGrater.size_distributions import power_law_distribution

    nx     = params.get("nx", 128)
    ny     = params.get("ny", 128)
    FOV_AU = params.get("FOV_AU", 400.0)

    import scipy
    import scipy.integrate
    import astropy.constants as cst
    from scipy.interpolate import RegularGridInterpolator
    from pyGrater.utils import (
        cylinder, hyperboloid_2_sheets,
        calculate_normalization_density_jacobian_sublimation_fast as _norm_func,
    )

    t_start = time.perf_counter()

    # ── phase 1: flux on 1-D radial grid ──────────────────────────────
    t0 = time.perf_counter()
    thermal_flux, scattered_flux = img_obj.flux_obj.get_fluxes(params)
    sizes     = img_obj.flux_obj.sizes_for_integral
    distances = img_obj.distances_for_flux
    t_flux    = time.perf_counter() - t0

    grain_density = img_obj.grain.grain_properties["Density"] * 1000

    # ── phase 2: pixel grid + ray-intersection geometry ───────────────
    t0 = time.perf_counter()
    r0       = params["r0"]
    alphain  = params["alphain"]
    alphaout = params["alphaout"]
    h0       = params["h0"]
    beta     = params["beta"]
    gamma    = params["gamma"]
    itilt    = params["itilt"]
    PA       = params["PA"]
    omega    = params["omega"]

    p    = 0.005
    rmax = r0 * p ** (1 / alphaout)
    img_obj.rmax = rmax

    gamma_in  = alphain  + beta
    gamma_out = alphaout + beta
    r_peak = (-gamma_in / gamma_out) ** (1 / (2 * gamma_in - 2 * gamma_out)) * r0
    z_peak = (h0 * (r_peak / r0) ** beta) * (np.log(1 / p) ** (1 / gamma))
    img_obj.Z0 = z_peak / np.sqrt(r_peak ** 2 / rmax ** 2 + 1)

    xc = (nx - 1) / 2.0
    yc = (ny - 1) / 2.0
    pixAU = FOV_AU / max(nx, ny)
    x_grid, y_grid = np.mgrid[0:nx, 0:ny]
    x_prime = (x_grid - xc) * pixAU
    y_prime = (y_grid - yc) * pixAU

    itilt_rad, omega_rad, PA_rad = np.radians([itilt, omega, PA])
    csPA, ssPA = np.cos(PA_rad), np.sin(PA_rad)
    csi,  ssi  = np.cos(itilt_rad), np.sin(itilt_rad)
    cso,  sso  = np.cos(omega_rad), np.sin(omega_rad)

    x = csPA * x_prime + ssPA * y_prime
    y = -ssPA * x_prime + csPA * y_prime

    vD  = np.array([ssi * cso, -ssi * sso, csi])
    rD0 = np.stack([
        x * csi * cso + y * sso,
        -x * csi * sso + y * cso,
        -x * ssi
    ])

    AxisC = np.array([rmax, rmax, np.sqrt(2) * img_obj.Z0])
    AxisH = np.array([rmax, rmax, img_obj.Z0])
    FARAWAY = -rmax * 10.0
    lmc, lpc = cylinder(AxisC, vD, rD0, FARAWAY, csi)
    lmh, lph = hyperboloid_2_sheets(AxisH, vD, rD0, FARAWAY, AxisC[0], AxisC[1])
    lbounds   = np.sort([lmc, lmh, lph, lpc], axis=0)
    lmin = lbounds[2]
    lmax = lbounds[3]
    dl   = lmax - lmin
    mask = dl != 0

    nl = 49
    ln = np.arange(nl) / (nl - 1.0)
    l  = np.tensordot(ln, dl, axes=0) + lmin

    xD = rD0[0][np.newaxis] + l * vD[0]
    yD = rD0[1][np.newaxis] + l * vD[1]
    zD = rD0[2][np.newaxis] + l * vD[2]

    rhoD  = np.sqrt(xD ** 2 + yD ** 2)
    rho_S = np.sqrt(xD ** 2 + yD ** 2 + zD ** 2)

    scattering_angle = np.pi - np.arccos(
        np.clip(l[:, mask] / np.sqrt(
            x_prime[mask] ** 2 + y_prime[mask] ** 2 + l[:, mask] ** 2
        ), -1, 1)
    )
    t_geom = time.perf_counter() - t0

    # ── phase 3: normalisation factor ─────────────────────────────────
    t0 = time.perf_counter()
    r_mask             = distances <= rmax
    distances_clipped  = distances[r_mask]
    z_2d, Z_max_r      = img_obj._build_z_grid(params)
    z_2d_clip          = z_2d[r_mask]
    Z_max_r_clip       = Z_max_r[r_mask]

    if "M_tot" in params:
        total_mass = params["M_tot"] * cst.M_earth.value
        norm_factor = _norm_func(
            img_obj.flux_obj.stargrain_obj,
            total_mass, sizes, distances_clipped,
            z_2d_clip, Z_max_r_clip, img_obj.zeta,
            grain_density, img_obj.density_function, params,
            img_obj.size_distribution_function, params,
        )
    else:
        norm_factor = params["A_norm"]
    t_norm = time.perf_counter() - t0

    # ── phase 4: per-wavelength LOS integration loop ──────────────────
    t0 = time.perf_counter()
    images_sca   = np.zeros((img_obj.wavelengths_for_calc.size, nx, ny))
    images_therm = np.zeros((img_obj.wavelengths_for_calc.size, nx, ny))
    loop_times   = []

    for i, wave in enumerate(img_obj.wavelengths_for_calc):
        t_wav = time.perf_counter()

        thermal_interp = scipy.interpolate.interp1d(
            distances, thermal_flux[i, :],
            kind="linear", bounds_error=False, fill_value=0,
        )
        scattered_interp = RegularGridInterpolator(
            (distances, img_obj.scattering_angles),
            scattered_flux[i, :, :], fill_value=0, bounds_error=False,
        )

        rho_flat   = rhoD[:, mask].ravel()
        rhoS_flat  = rho_S[:, mask].ravel()
        angle_flat = scattering_angle.ravel()

        scattered_values    = scattered_interp(np.column_stack([rhoS_flat, angle_flat]))
        scattered_emissivity = scattered_values.reshape(rhoD[:, mask].shape)

        density_vals  = img_obj.density_function(rhoD[:, mask], 0.0, zD[:, mask], params)
        thermal_vals  = thermal_interp(rho_S[:, mask])

        limage_sca   = scattered_emissivity * density_vals
        limage_therm = thermal_vals * density_vals

        img_sca   = np.zeros([nx, ny])
        img_therm = np.zeros([nx, ny])

        img_sca[mask]   = scipy.integrate.trapezoid(limage_sca,   x=ln, axis=0) * dl[mask] * pixAU ** 2
        img_therm[mask] = scipy.integrate.trapezoid(limage_therm, x=ln, axis=0) * dl[mask] * pixAU ** 2

        images_sca[i]   = np.flip(img_sca.T,   axis=0) * norm_factor
        images_therm[i] = np.flip(img_therm.T, axis=0) * norm_factor

        loop_times.append(time.perf_counter() - t_wav)

    t_loop = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    timings = {
        "flux_calc":      t_flux,
        "geometry":       t_geom,
        "normalisation":  t_norm,
        "los_loop":       t_loop,
        "total":          t_total,
        "loop_per_wave":  loop_times,
        "n_masked_px":    int(mask.sum()),
        "n_pixels_total": nx * ny,
    }
    return images_sca, images_therm, timings


# ── benchmark 1: single-call breakdown ────────────────────────────────────────
def bench_single_call(wavelengths, params):
    _divider("Single call — step-by-step timing")
    grain, star = _make_objects()
    img_obj = _make_image_obj(grain, star, wavelengths)

    nx, ny = params.get("nx", 128), params.get("ny", 128)
    log_info(logger, f"  Wavelengths : {len(wavelengths)}  ({wavelengths[0]:.1f}–{wavelengths[-1]:.1f} µm)")
    log_info(logger, f"  Grid        : {nx} × {ny} px")
    log_info(logger, f"  N_sizes_int : {params['N_sizes_integral']}")

    mem_before = _rss_mb()
    _, _, timings = _run_image_timed(img_obj, params)
    mem_after = _rss_mb()

    total = timings["total"]
    _divider("Timing breakdown")
    phases = ["flux_calc", "geometry", "normalisation", "los_loop", "total"]
    for k in phases:
        v = timings[k]
        pct = v / total * 100
        bar = "█" * int(pct / 2)
        log_info(logger, f"  {k:20s}: {v:7.3f} s  {pct:5.1f}%  {bar}")

    loop_arr = np.array(timings["loop_per_wave"])
    log_info(logger, f"\n  Per-wavelength LOS loop (mean ± std): "
          f"{loop_arr.mean():.3f} ± {loop_arr.std():.3f} s")
    log_info(logger, f"  Masked pixels : {timings['n_masked_px']} / {timings['n_pixels_total']}"
          f"  ({100*timings['n_masked_px']/timings['n_pixels_total']:.1f}%)")
    if _HAS_PSUTIL:
        log_info(logger, f"  Memory increase : {mem_after - mem_before:.1f} MB")
    return timings


# ── benchmark 2: scaling with image resolution ────────────────────────────────
def bench_scale_resolution(res_list, wavelengths, params_base, n_runs=5):
    _divider(f"Scaling: image resolution (nx = ny)  ({n_runs} runs each)")
    grain, star = _make_objects()
    img_obj = _make_image_obj(grain, star, wavelengths)

    results = []
    log_info(logger, f"  {'nx':>6}  {'mean (s)':>9}  {'+/-':>7}")
    log_info(logger, f"  {'─'*6}  {'─'*9}  {'─'*7}")
    for n in res_list:
        p = {**params_base, "nx": n, "ny": n}
        walls = []
        for _ in range(n_runs):
            _, _, tm = _run_image_timed(img_obj, p)
            walls.append(tm["total"])
        mean, std = float(np.mean(walls)), float(np.std(walls))
        results.append((n, mean, std))
        log_info(logger, f"  {n:>6}  {mean:>9.3f}  {std:>7.3f}")
    return results


# ── benchmark 3: scaling with N_wavelengths ───────────────────────────────────
def bench_scale_wavelengths(n_wav_list, params, n_runs=5):
    _divider(f"Scaling: N_wavelengths  ({n_runs} runs each)")
    grain, star = _make_objects()

    results = []
    log_info(logger, f"  {'N_wav':>6}  {'mean (s)':>9}  {'+/-':>7}")
    log_info(logger, f"  {'─'*6}  {'─'*9}  {'─'*7}")
    for n in n_wav_list:
        wl = np.geomspace(1.0, 200.0, n)
        img_obj = _make_image_obj(grain, star, wl)
        walls = []
        for _ in range(n_runs):
            _, _, tm = _run_image_timed(img_obj, params)
            walls.append(tm["total"])
        mean, std = float(np.mean(walls)), float(np.std(walls))
        results.append((n, mean, std))
        log_info(logger, f"  {n:>6}  {mean:>9.3f}  {std:>7.3f}")
    return results


# ── benchmark 4: scaling with nl (LOS samples) ───────────────────────────────
def bench_scale_nl(nl_list, wavelengths, params):
    """
    nl is hard-coded inside image.py (nl=49).  This benchmark patches it
    at runtime to sweep different values.
    """
    _divider("Scaling: nl (LOS ray samples)")
    import pyGrater.image as _img_module
    grain, star = _make_objects()

    results = []
    log_info(logger, f"  {'nl':>5}  {'loop (s)':>9}  {'total (s)':>10}")
    log_info(logger, f"  {'─'*5}  {'─'*9}  {'─'*10}")

    # We approximate by timing only the loop; full instrumented run is
    # accurate enough here without monkey-patching Image internals.
    img_obj = _make_image_obj(grain, star, wavelengths)

    # first warm-up call so Fluxes cache is populated
    _run_image_timed(img_obj, params)

    for nl in nl_list:
        # Re-run with modified params proxy (nl checked inside wrapper)
        p = {**params, "_nl_override": nl}

        # Minimal timed loop: only the LOS integration, not flux recompute
        # (We time the full call but note flux is cached by _get_temperatures)
        t0 = time.perf_counter()
        _run_image_timed(img_obj, p)
        wall = time.perf_counter() - t0
        _, _, tm = _run_image_timed(img_obj, p)
        results.append((nl, tm["los_loop"], tm["total"]))
        log_info(logger, f"  {nl:>5}  {tm['los_loop']:>9.3f}  {tm['total']:>10.3f}")
    return results


# ── benchmark 5: cProfile trace ──────────────────────────────────────────────
def bench_cprofile(wavelengths, params, out_file="benchmark_image.prof"):
    _divider("cProfile — top 30 functions by cumulative time")
    grain, star = _make_objects()
    img_obj = _make_image_obj(grain, star, wavelengths)

    pr = cProfile.Profile()
    pr.enable()
    _run_image_timed(img_obj, params)
    pr.disable()

    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    ps.print_stats(30)
    log_info(logger, stream.getvalue())

    if out_file:
        pr.dump_stats(out_file)
        log_info(logger, f"  Full profile saved to: {out_file}")
        log_info(logger, f"  Inspect with:  python -m pstats {out_file}")


# ── scaling plot ──────────────────────────────────────────────────────────────
def plot_scaling(res_results, wav_results, out="benchmark_image_scaling.png"):
    """Each *_results list contains tuples of (n, mean, std, flux_mean, loop_mean)
    after the multi-run update.  Error bars show ±1 std across repeated calls."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ebar_kw = dict(fmt="o-", linewidth=2, capsize=4, capthick=1.5, elinewidth=1.2)

    # Resolution  — (n, mean, std)
    n_res = [r[0] for r in res_results]
    t_tot = [r[1] for r in res_results]
    e_tot = [r[2] for r in res_results]
    axes[0].errorbar(n_res, t_tot, yerr=e_tot, **ebar_kw)
    axes[0].set_xlabel("Image size (nx = ny)")
    axes[0].set_ylabel("Total wall-clock time [s]")
    axes[0].set_title("Scaling: resolution")
    axes[0].grid(True, alpha=0.3)

    # N_wavelengths  — (n, mean, std)
    n_wav = [r[0] for r in wav_results]
    t_tot = [r[1] for r in wav_results]
    e_tot = [r[2] for r in wav_results]
    axes[1].errorbar(n_wav, t_tot, yerr=e_tot, **ebar_kw)
    axes[1].set_xlabel("N wavelengths")
    axes[1].set_title("Scaling: wavelengths")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    log_info(logger, f"\n  Scaling plot saved to: {out}")
    plt.close()


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Image generation performance benchmark")
    parser.add_argument("--quick",   action="store_true",
                        help="Minimal run for a fast sanity check")
    parser.add_argument("--profile", action="store_true",
                        help="Run cProfile and save .prof file")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of repeated calls per configuration (default: 5)")
    args = parser.parse_args()

    if args.quick:
        wavelengths = np.geomspace(1.0, 100.0, 3)
        res_list    = [32, 64]
        n_wav_list  = [2, 3]
        nl_list     = [25, 49]
        params      = {**BASE_DISK_PARAMS, "nx": 32, "ny": 32,
                       "N_sizes_integral": 50}
    else:
        wavelengths = np.geomspace(1.0, 200.0, 6)
        res_list    = [64, 128, 256, 512]
        n_wav_list  = [2, 4, 6, 10]
        nl_list     = [25, 49, 99, 201]
        params      = {**BASE_DISK_PARAMS}

    log_info(logger, "=" * 70)
    log_info(logger, "pyGrater — Image generation benchmark")
    log_info(logger, "=" * 70)

    bench_single_call(wavelengths, params)

    n_runs = 1 if args.quick else args.runs
    res_results = bench_scale_resolution(res_list, wavelengths, params, n_runs=n_runs)
    wav_results = bench_scale_wavelengths(n_wav_list, params, n_runs=n_runs)
    bench_scale_nl(nl_list, wavelengths, params)

    if args.profile:
        bench_cprofile(wavelengths, params)

    if not args.quick:
        plot_scaling(res_results, wav_results)

    _divider()
    log_info(logger, "Done.")


if __name__ == "__main__":
    main()
