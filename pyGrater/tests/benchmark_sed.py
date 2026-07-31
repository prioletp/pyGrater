"""
benchmark_sed.py — SED performance profiling
=============================================

What this script measures
--------------------------
1. Wall-clock time for every internal SED step (using the built-in
   ``verbose_timing`` hook in SED.get_SED).
2. Scaling with the main free parameters:
   - number of output wavelengths
   - N_sizes_integral (size quadrature resolution)
   - N_distances (radial flux grid)
3. Full cProfile trace of a single representative call, so you can
   identify the hottest Python frames.

How to run
----------
    python -m pyGrater.tests.benchmark_sed

Optional flags:
    --quick      2 wavelengths, small grids — fast sanity check
    --profile    write full cProfile output to benchmark_sed.prof
                 (open with: python -m pstats benchmark_sed.prof)
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
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import numpy as np

from pyGrater.tests.benchmark_helpers import divider as _divider
from pyGrater.tests.benchmark_helpers import rss_mb as _rss_mb


# ── shared disk / grain parameters ───────────────────────────────────────────
BASE_DISK_PARAMS = dict(
    r0=100.0, h0=5.0,
    alphain=10.0, alphaout=-5.0,
    gamma=2.0, beta=1.0,
    itilt=0.0, PA=45.0, omega=45.0,
    a_min=1e-5, a_max=1e-3, kappa=3.5,
    N_sizes_integral=100,
    g=0.0,
    M_tot=1e-3,
)
COMPOSITION  = "astroSi"
STAR_NAME    = "HD113766"


# ── helpers ───────────────────────────────────────────────────────────────────
def _make_objects(composition=COMPOSITION, star_name=STAR_NAME, N_distances=400):
    from pyGrater.stargrains import Grain, Star
    grain = Grain(composition=composition, redo_Q=False)
    star  = Star(star_name=star_name)
    return grain, star, N_distances


def _make_sed(grain, star, wavelengths, N_distances=400):
    from pyGrater.density import two_power_law
    from pyGrater.size_distributions import power_law_distribution
    from pyGrater.SED_old import SED
    return SED(grain, star, two_power_law, power_law_distribution,
               wavelengths, N_distances=N_distances)


def _run_sed(sed_obj, params, verbose_timing=True):
    """Single get_SED call; returns (thermal, scattered, timings_dict)."""
    sed_obj.get_SED(keep_separate_fluxes=True,
                    verbose_timing=verbose_timing,
                    **params)
    return sed_obj.timings          # populated by get_SED


# ── benchmark 1: step-by-step timing on a single call ────────────────────────
def bench_single_call(wavelengths, params, N_distances=400):
    _divider("Single call — step-by-step timing")
    grain, star, _ = _make_objects(N_distances=N_distances)
    sed_obj = _make_sed(grain, star, wavelengths, N_distances)

    log_info(logger, f"  Wavelengths : {len(wavelengths)}  ({wavelengths[0]:.1f}–{wavelengths[-1]:.1f} µm)")
    log_info(logger, f"  N_distances : {N_distances}")
    log_info(logger, f"  N_sizes_int : {params['N_sizes_integral']}")

    mem_before = _rss_mb()
    t0 = time.perf_counter()
    timings = _run_sed(sed_obj, params, verbose_timing=True)
    wall = time.perf_counter() - t0
    mem_after  = _rss_mb()

    total = timings.get("total", wall)
    _divider("Timing breakdown")
    # group by phase
    phases = [
        ("flux calc",   ["thermal_temperatures", "thermal_Q_interp",
                         "thermal_planck_loop", "thermal_total",
                         "scattered_Q_interp", "scattered_matmul",
                         "scattered_total"]),
        ("SED build",   ["grid_setup", "normalisation", "density",
                         "interp+zeta_int", "radial_int"]),
    ]
    for group_name, keys in phases:
        log_info(logger, f"\n  [{group_name}]")
        for k in keys:
            v = timings.get(k)
            if v is not None:
                pct = v / total * 100
                bar = "█" * int(pct / 2)
                log_info(logger, f"    {k:30s}: {v:7.3f} s  {pct:5.1f}%  {bar}")

    log_info(logger, f"\n  Wall-clock total  : {wall:.3f} s")
    if _HAS_PSUTIL:
        log_info(logger, f"  Memory increase   : {mem_after - mem_before:.1f} MB")
    return timings


# ── benchmark 2: scaling with N_wavelengths ───────────────────────────────────
def bench_scale_wavelengths(n_wav_list, params, N_distances=400, n_runs=5):
    _divider(f"Scaling: N_wavelengths  ({n_runs} runs each)")
    grain, star, _ = _make_objects()

    results = []
    log_info(logger, f"  {'N_wav':>6}  {'mean (s)':>9}  {'+/-':>7}")
    log_info(logger, f"  {'─'*6}  {'─'*9}  {'─'*7}")
    for n in n_wav_list:
        wl = np.geomspace(1.0, 300.0, n)
        sed_obj = _make_sed(grain, star, wl, N_distances)
        walls = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _run_sed(sed_obj, params, verbose_timing=False)
            walls.append(time.perf_counter() - t0)
        mean, std = float(np.mean(walls)), float(np.std(walls))
        results.append((n, mean, std))
        log_info(logger, f"  {n:>6}  {mean:>9.3f}  {std:>7.3f}")
    return results


# ── benchmark 3: scaling with N_sizes_integral ───────────────────────────────
def bench_scale_sizes(n_sizes_list, wavelengths, N_distances=400, n_runs=5):
    _divider(f"Scaling: N_sizes_integral  ({n_runs} runs each)")
    grain, star, _ = _make_objects()
    sed_obj = _make_sed(grain, star, wavelengths, N_distances)

    results = []
    log_info(logger, f"  {'N_sizes':>8}  {'mean (s)':>9}  {'+/-':>7}")
    log_info(logger, f"  {'─'*8}  {'─'*9}  {'─'*7}")
    for n in n_sizes_list:
        p = {**BASE_DISK_PARAMS, "N_sizes_integral": n}
        walls = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _run_sed(sed_obj, p, verbose_timing=False)
            walls.append(time.perf_counter() - t0)
        mean, std = float(np.mean(walls)), float(np.std(walls))
        results.append((n, mean, std))
        log_info(logger, f"  {n:>8}  {mean:>9.3f}  {std:>7.3f}")
    return results


# ── benchmark 4: scaling with N_distances ────────────────────────────────────
def bench_scale_distances(n_dist_list, wavelengths, params, n_runs=5):
    _divider(f"Scaling: N_distances (radial grid)  ({n_runs} runs each)")
    grain, star, _ = _make_objects()

    results = []
    log_info(logger, f"  {'N_dist':>7}  {'mean (s)':>9}  {'+/-':>7}")
    log_info(logger, f"  {'─'*7}  {'─'*9}  {'─'*7}")
    for nd in n_dist_list:
        sed_obj = _make_sed(grain, star, wavelengths, nd)
        walls = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _run_sed(sed_obj, params, verbose_timing=False)
            walls.append(time.perf_counter() - t0)
        mean, std = float(np.mean(walls)), float(np.std(walls))
        results.append((nd, mean, std))
        log_info(logger, f"  {nd:>7}  {mean:>9.3f}  {std:>7.3f}")
    return results


# ── benchmark 5: cProfile trace ──────────────────────────────────────────────
def bench_cprofile(wavelengths, params, N_distances=400, out_file="benchmark_sed.prof"):
    _divider("cProfile — top 30 functions by cumulative time")
    grain, star, _ = _make_objects()
    sed_obj = _make_sed(grain, star, wavelengths, N_distances)

    pr = cProfile.Profile()
    pr.enable()
    _run_sed(sed_obj, params, verbose_timing=False)
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
def plot_scaling(wav_results, size_results, dist_results, out="benchmark_sed_scaling.png"):
    """Each *_results list contains tuples of (n, mean, std, ...) after the
    multi-run update.  Error bars show ±1 std across repeated calls."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ebar_kw = dict(fmt="o-", linewidth=2, capsize=4, capthick=1.5, elinewidth=1.2)

    # N_wavelengths  — (n, mean, std)
    n_wav = [r[0] for r in wav_results]
    t_wav = [r[1] for r in wav_results]
    e_wav = [r[2] for r in wav_results]
    axes[0].errorbar(n_wav, t_wav, yerr=e_wav, **ebar_kw)
    axes[0].set_xlabel("N wavelengths")
    axes[0].set_ylabel("Total wall-clock time [s]")
    axes[0].set_title("Scaling: wavelengths")
    axes[0].grid(True, alpha=0.3)

    # N_sizes_integral  — (n, mean, std)
    n_sz = [r[0] for r in size_results]
    t_sz = [r[1] for r in size_results]
    e_sz = [r[2] for r in size_results]
    axes[1].errorbar(n_sz, t_sz, yerr=e_sz, **ebar_kw)
    axes[1].set_xlabel("N_sizes_integral")
    axes[1].set_title("Scaling: size grid")
    axes[1].grid(True, alpha=0.3)

    # N_distances  — (n, mean, std)
    n_d = [r[0] for r in dist_results]
    t_d = [r[1] for r in dist_results]
    e_d = [r[2] for r in dist_results]
    axes[2].errorbar(n_d, t_d, yerr=e_d, **ebar_kw)
    axes[2].set_xlabel("N_distances")
    axes[2].set_title("Scaling: radial grid")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    log_info(logger, f"\n  Scaling plot saved to: {out}")
    plt.close()


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SED performance benchmark")
    parser.add_argument("--quick",   action="store_true",
                        help="Minimal run for a fast sanity check")
    parser.add_argument("--profile", action="store_true",
                        help="Run cProfile and save .prof file")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of repeated calls per configuration (default: 5)")
    args = parser.parse_args()

    if args.quick:
        wavelengths   = np.geomspace(1.0, 100.0, 5)
        n_wav_list    = [3, 5]
        n_sizes_list  = [50, 100]
        n_dist_list   = [200, 400]
        params        = {**BASE_DISK_PARAMS, "N_sizes_integral": 50}
        n_dist_ref    = 200
    else:
        wavelengths   = np.geomspace(1.0, 300.0, 30)
        n_wav_list    = [5, 10, 20, 40, 80]
        n_sizes_list  = [50, 100, 200, 400, 800]
        n_dist_list   = [200, 400, 800, 1600]
        params        = {**BASE_DISK_PARAMS, "N_sizes_integral": 200}
        n_dist_ref    = 400

    log_info(logger, "=" * 70)
    log_info(logger, "pyGrater — SED benchmark")
    log_info(logger, "=" * 70)

    bench_single_call(wavelengths, params, N_distances=n_dist_ref)

    n_runs = 1 if args.quick else args.runs
    wav_results  = bench_scale_wavelengths(n_wav_list,  params,     N_distances=n_dist_ref, n_runs=n_runs)
    size_results = bench_scale_sizes(n_sizes_list,       wavelengths, N_distances=n_dist_ref, n_runs=n_runs)
    dist_results = bench_scale_distances(n_dist_list,    wavelengths, params,                n_runs=n_runs)

    if args.profile:
        bench_cprofile(wavelengths, params, N_distances=n_dist_ref)

    if not args.quick:
        plot_scaling(wav_results, size_results, dist_results)

    _divider()
    log_info(logger, "Done.")


if __name__ == "__main__":
    main()
