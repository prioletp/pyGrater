import logging
#%%


from pyGrater.config.logging_config import log_info
logger = logging.getLogger(__name__)
"""
Benchmark comparison between SED generation and image generation.

Compares:
- Execution time: SED vs Image generation
- Flux consistency: SED flux vs integrated image flux
- Memory usage
- Wavelength dependence
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import psutil
import os

from pyGrater.stargrains import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.size_distributions import power_law_distribution
from pyGrater.phase_functions import isotropic as phase_function
from pyGrater.image import Image
from pyGrater.SED_old import SED
# FOV_AU = 1
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def benchmark_sed_vs_image(wavelengths=None, nx=256, ny=256, FOV_AU=1, n_runs=3):
    """
    Compare SED and image generation with flux consistency check.
    
    Parameters
    ----------
    wavelengths : array_like, optional
        Wavelengths in microns (default: [2.0, 2.5, 3.0, 3.5, 4.0])
    nx, ny : int
        Image dimensions
    pixAU : float
        Pixel scale in AU
    n_runs : int
        Number of benchmark runs
        
    Returns
    -------
    dict
        Comparison results
    """
    # FOV_AU = max(nx, ny) * pixAU
    if wavelengths is None:
        wavelengths = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
    else:
        wavelengths = np.asarray(wavelengths)
    
    log_info(logger, "="*70)
    log_info(logger, "SED vs IMAGE GENERATION BENCHMARK")
    log_info(logger, "="*70)
    log_info(logger, f"Wavelengths: {wavelengths} µm")
    log_info(logger, f"Image size: {nx}x{ny}")
    log_info(logger, f"Field of view: {FOV_AU} AU")
    log_info(logger, f"Number of runs: {n_runs}")
    log_info(logger, )
    
    # Setup
    grain = Grain(redo_Q=False, composition='astroSi')  # Load precomputed optical efficiencies
    star = Star(star_name='HD113766')
    
    test_params = {
        'r0': 200, 'h0': 10, 'alphain': 10., 'alphaout': -6,
        'gamma': 2., 'beta': 2, 'itilt': 0., 'PA': 45., 'omega': 45.,
        'a_min': 10e-6, 'a_max': 1000e-6, 'kappa': 3.5,
        'N_sizes_integral': 200, 'g': 0.5, 'M_tot': 2.5e-10, 'FOV_AU': FOV_AU, 'nx': nx, 'ny': ny
    }
    
    # Benchmark SED generation
    log_info(logger, "Testing SED generation...")
    log_info(logger, "-" * 70)
    
    sed_gen = SED(grain, star, two_power_law, power_law_distribution, wavelengths)
    times_sed = []
    mem_sed = []
    
    for run in range(n_runs):
        mem_start = get_memory_usage()
        start = time.time()
        
        # SED.get_SED returns (thermal, scattered) when keep_separate_fluxes=True
        sed_therm, sed_sca = sed_gen.get_SED(keep_separate_fluxes=True, **test_params)

        elapsed = time.time() - start
        mem_used = get_memory_usage() - mem_start
        
        times_sed.append(elapsed)
        mem_sed.append(mem_used)
        
        log_info(logger, f"  Run {run+1}: {elapsed:.3f}s, Memory: {mem_used:.1f} MB")
    
    time_sed_avg = np.mean(times_sed)
    time_sed_std = np.std(times_sed)
    mem_sed_avg = np.mean(mem_sed)
    
    log_info(logger, f"✓ SED average: {time_sed_avg:.3f}s ± {time_sed_std:.3f}s")
    log_info(logger, f"  Memory: {mem_sed_avg:.1f} MB")
    log_info(logger, )
    
    # Benchmark Image generation
    log_info(logger, "Testing Image generation...")
    log_info(logger, "-" * 70)
    
    img_gen = Image(grain, star, two_power_law, power_law_distribution,
                    phase_function, wavelengths)
    times_img = []
    mem_img = []
    image_fluxes_sca = []
    image_fluxes_therm = []
    
    for run in range(n_runs):
        mem_start = get_memory_usage()
        start = time.time()
        pixAU = FOV_AU / max(nx, ny)
        pixel_area_AU2 = 1 #pixAU**2
        images_sca, images_therm = img_gen.get_image(
            keep_separate_fluxes=True, **test_params
        )
        log_info(logger, 'RMAX:', img_gen.rmax)

        elapsed = time.time() - start
        mem_used = get_memory_usage() - mem_start
        
        times_img.append(elapsed)
        mem_img.append(mem_used)
        
        # Calculate total flux from images (sum over all pixels)
        flux_sca = np.array([np.sum(images_sca[i])*pixel_area_AU2 for i in range(len(wavelengths))])
        flux_therm = np.array([np.sum(images_therm[i])*pixel_area_AU2 for i in range(len(wavelengths))])
        image_fluxes_sca.append(flux_sca)
        image_fluxes_therm.append(flux_therm)
        
        log_info(logger, f"  Run {run+1}: {elapsed:.3f}s, Memory: {mem_used:.1f} MB")
    
    time_img_avg = np.mean(times_img)
    time_img_std = np.std(times_img)
    mem_img_avg = np.mean(mem_img)
    
    # Average image fluxes
    avg_flux_sca = np.mean(image_fluxes_sca, axis=0)
    avg_flux_therm = np.mean(image_fluxes_therm, axis=0)
    avg_flux_total_img = avg_flux_sca + avg_flux_therm
    
    log_info(logger, f"✓ Image average: {time_img_avg:.3f}s ± {time_img_std:.3f}s")
    log_info(logger, f"  Memory: {mem_img_avg:.1f} MB")
    log_info(logger, )
    
    # Performance comparison
    speedup = time_img_avg / time_sed_avg
    
    log_info(logger, "="*70)
    log_info(logger, "PERFORMANCE COMPARISON")
    log_info(logger, "="*70)
    log_info(logger, f"SED:   {time_sed_avg:.3f}s ± {time_sed_std:.3f}s")
    log_info(logger, f"Image: {time_img_avg:.3f}s ± {time_img_std:.3f}s")
    log_info(logger, f"Ratio: Image is {speedup:.2f}x {'slower' if speedup > 1 else 'faster'} than SED")
    log_info(logger, f"Memory: SED {mem_sed_avg:.1f} MB, Image {mem_img_avg:.1f} MB")
    log_info(logger, )
    
    # Flux comparison
    log_info(logger, "="*70)
    
    log_info(logger, "FLUX CONSISTENCY CHECK")
    log_info(logger, "="*70)
    
    # Compare SED flux to integrated image flux
    sed_total = sed_therm + sed_sca
    flux_ratio = avg_flux_total_img / sed_total
    flux_diff_percent = (avg_flux_total_img - sed_total) / sed_total * 100

    # Normalization constants
    sed_norm_factor = float(np.nanmean(np.asarray(getattr(sed_gen, 'norm_factor', np.nan), dtype=float)))
    image_norm_factor = float(getattr(img_gen, 'norm_factor'))
    norm_factor_ratio = image_norm_factor / sed_norm_factor if np.isfinite(sed_norm_factor) and sed_norm_factor != 0 else np.nan

    for i, wave in enumerate(wavelengths):
        log_info(logger, f"λ = {wave:.2f} µm:")
        log_info(logger, f"  SED flux:   {sed_total[i]:.3e}")
        log_info(logger, f"  Image flux: {avg_flux_total_img[i]:.3e}")
        log_info(logger, f"  Ratio:      {flux_ratio[i]:.4f}")
        log_info(logger, f"  Difference: {flux_diff_percent[i]:+.2f}%")
    
    return {
        'wavelengths': wavelengths,
        'sed': {
            'time': time_sed_avg,
            'time_std': time_sed_std,
            'memory': mem_sed_avg,
            'flux_scattered': sed_sca,
            'flux_thermal': sed_therm,
            'flux_total': sed_total
        },
        'image': {
            'time': time_img_avg,
            'time_std': time_img_std,
            'memory': mem_img_avg,
            'flux_scattered': avg_flux_sca,
            'flux_thermal': avg_flux_therm,
            'flux_total': avg_flux_total_img,
            'images_sca': images_sca,
            'images_therm': images_therm
        },
        'comparison': {
            'speedup': speedup,
            'flux_ratio': flux_ratio,
            'flux_diff_percent': flux_diff_percent
        },
        'normalization': {
            'sed_norm_factor': sed_norm_factor,
            'image_norm_factor': image_norm_factor,
            'ratio_image_to_sed': norm_factor_ratio
        },
        'params': test_params,
        'config': {'nx': nx, 'ny': ny, 'FOV_AU': FOV_AU},
        'functions': {
            'density': two_power_law.__name__,
            'size_dist': power_law_distribution.__name__,
            'phase': phase_function.__name__
        }
    }

def plot_comparison(results):
    """Create comprehensive comparison plots."""
    
    wavelengths = results['wavelengths']
    mid_idx = len(wavelengths) // 2
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.35)
    
    # Row 1: Flux comparison (scattered, thermal, total)
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.plot(wavelengths, results['sed']['flux_total'], 'o-', linewidth=2, 
            markersize=8, label='SED Total', color='blue')
    ax1.plot(wavelengths, results['image']['flux_total'], 's--', linewidth=2,
            markersize=8, label='Image Total', color='red')
    ax1.set_xlabel('Wavelength [µm]', fontsize=13)
    ax1.set_ylabel('Total Flux', fontsize=13)
    ax1.set_title('Flux Comparison: SED vs Integrated Image', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Row 2: Scattered vs Thermal vs Total with image
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(wavelengths, results['sed']['flux_scattered'], 'o-', 
            label='SED Scattered', color='blue')
    ax2.plot(wavelengths, results['image']['flux_scattered'], 's--',
            label='Image Scattered', color='red')
    ax2.set_xlabel('Wavelength [µm]', fontsize=11)
    ax2.set_ylabel('Scattered Flux', fontsize=11)
    ax2.set_title('Scattered Light', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(wavelengths, results['sed']['flux_thermal'], 'o-',
            label='SED Thermal', color='blue')
    ax3.plot(wavelengths, results['image']['flux_thermal'], 's--',
            label='Image Thermal', color='red')
    ax3.set_xlabel('Wavelength [µm]', fontsize=11)
    ax3.set_ylabel('Thermal Flux', fontsize=11)
    ax3.set_title('Thermal Emission', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(wavelengths, results['sed']['flux_total'], 'o-', 
            label='SED Total', color='blue')
    ax4.plot(wavelengths, results['image']['flux_total'], 's--',
            label='Image Total', color='red')
    ax4.set_xlabel('Wavelength [µm]', fontsize=11)
    ax4.set_ylabel('Total Flux', fontsize=11)
    ax4.set_title('Total Flux', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Image at mid-wavelength
    ax_img = fig.add_subplot(gs[1, 3])
    total_mid = results['image']['images_sca'][mid_idx] + results['image']['images_therm'][mid_idx]
    im_mid = ax_img.imshow(total_mid, cmap='inferno', origin='lower')
    ax_img.set_title(f'Total Image\n@ {wavelengths[mid_idx]:.2f} µm', fontsize=11, fontweight='bold')
    plt.colorbar(im_mid, ax=ax_img, shrink=0.8)
    
    # Row 3: Flux ratios below fluxes
    ax5 = fig.add_subplot(gs[2, 0])
    flux_ratio_sca = results['image']['flux_scattered'] / results['sed']['flux_scattered']
    ax5.plot(wavelengths, flux_ratio_sca*100, 'o-', 
            linewidth=2, markersize=7, color='purple')
    ax5.axhline(y=100.0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Wavelength [µm]', fontsize=11)
    ax5.set_ylabel('Ratio [%]', fontsize=11)
    ax5.set_title('Scattered Flux Ratio', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[2, 1])
    flux_ratio_therm = results['image']['flux_thermal'] / results['sed']['flux_thermal']
    ax6.plot(wavelengths, flux_ratio_therm*100, 'o-',
            linewidth=2, markersize=7, color='orange')
    ax6.axhline(y=100.0, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Wavelength [µm]', fontsize=11)
    ax6.set_ylabel('Ratio [%]', fontsize=11)
    ax6.set_title('Thermal Flux Ratio', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.plot(wavelengths, results['comparison']['flux_ratio']*100, 'o-', 
            linewidth=2, markersize=7, color='green')
    ax7.axhline(y=100.0, color='k', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Wavelength [µm]', fontsize=11)
    ax7.set_ylabel('Ratio [%]', fontsize=11)
    ax7.set_title('Total Flux Ratio', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Row 4: Information boxes on left, compact
    ax_perf = fig.add_subplot(gs[3, 0:3])  # wider box (3 columns instead of 2)
    ax_perf.axis('off')

    sed_sca_total = np.sum(results['sed']['flux_scattered'])
    sed_therm_total = np.sum(results['sed']['flux_thermal'])
    sed_total = np.sum(results['sed']['flux_total'])
    img_sca_total = np.sum(results['image']['flux_scattered'])
    img_therm_total = np.sum(results['image']['flux_thermal'])
    img_total = np.sum(results['image']['flux_total'])
    
    # Build full parameter list text (all model params), compacted by rows
    params = results['params']
    param_items = []
    for k, v in params.items():
        if isinstance(v, (float, np.floating)):
            param_items.append(f"{k}={v:.4g}")
        else:
            param_items.append(f"{k}={v}")
    param_lines = [" | ".join(param_items[i:i+3]) for i in range(0, len(param_items), 3)]
    param_block = "\n".join(param_lines)

    sed_norm = results['normalization']['sed_norm_factor']
    img_norm = results['normalization']['image_norm_factor']
    norm_ratio = results['normalization']['ratio_image_to_sed']

    perf_text = (
        f"PERFORMANCE  |  SED: {results['sed']['time']:.3f}s ± {results['sed']['time_std']:.3f}s, {results['sed']['memory']:.1f} MB"
        f"  |  Image: {results['image']['time']:.3f}s ± {results['image']['time_std']:.3f}s, {results['image']['memory']:.1f} MB"
        f"  |  Speedup: {results['comparison']['speedup']:.2f}x\n"
        f"MODEL: density={results['functions']['density']} | size={results['functions']['size_dist']} | phase={results['functions']['phase']}\n"
        f"{param_block}\n"
        f"NORM: SED={sed_norm:.6e} | Image={img_norm:.6e} | Image/SED={norm_ratio:.6e}"
    )
    ax_perf.text(
        0.02, 0.5, perf_text, fontsize=8.2, verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='lightblue', alpha=0.6)
    )

    ax_stats = fig.add_subplot(gs[3, 3])  # moved to last column only
    ax_stats.axis('off')
    
    mean_ratio = np.mean(results['comparison']['flux_ratio'])
    std_ratio = np.std(results['comparison']['flux_ratio'])
    max_diff = np.max(np.abs(results['comparison']['flux_diff_percent']))
    
    stats_text = (
        f"FLUX CONSISTENCY\n"
        f"Mean={mean_ratio:.4f}\n"
        f"Std={std_ratio:.4f}\n"
        f"Max Δ={max_diff:.2f}%\n"
        f"Cfg: {results['config']['nx']}×{results['config']['ny']}"
    )
    ax_stats.text(
        0.05, 0.5, stats_text, fontsize=8.8, verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='lightyellow', alpha=0.6)
    )

    plt.suptitle('SED vs Image Generation Comparison', fontsize=16, fontweight='bold', y=0.998)
    
    return fig

if __name__ == "__main__":
    log_info(logger, "\n" + "="*70)
    log_info(logger, "SED vs IMAGE BENCHMARK SUITE")
    log_info(logger, "="*70 + "\n")
    
    # Test 1: Few wavelengths
    # print("\n### TEST 1: Few wavelengths (3) ###\n")
    # results_few = benchmark_sed_vs_image(
    #     wavelengths=np.array([2.0, 3.0, 4.0]),
    #     nx=128, ny=128, pixAU=0.003, n_runs=3
    # )
    
    # Test 2: Many wavelengths
    log_info(logger, "\n### TEST 2: Many wavelengths (50) ###\n")
    results_many = benchmark_sed_vs_image(
        wavelengths=np.linspace(2, 50, 5),
        nx=128, ny=128, n_runs=1, FOV_AU=800
    )
    
    # Generate plots
    log_info(logger, "\n" + "="*70)
    log_info(logger, "GENERATING PLOTS")
    log_info(logger, "="*70)
    
    # fig1 = plot_comparison(results_few)
    # fig1.savefig('benchmark_SEDvsimg_few.png', dpi=150, bbox_inches='tight')
    # print("✓ Saved: benchmark_SEDvsimg_few.png")
    
    fig2 = plot_comparison(results_many)
    fig2.savefig('benchmark_SEDvsimg_many.png', dpi=150, bbox_inches='tight')
    log_info(logger, "✓ Saved: benchmark_SEDvsimg_many.png")
    
    plt.show()
    
    log_info(logger, "\nBenchmark complete!")
# %%
