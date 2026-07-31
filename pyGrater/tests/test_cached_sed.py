"""Numerical-equivalence tests for the cache-aware SED backend."""

import numpy as np
import pytest

from pyGrater import CachedSED, Grain, SED, SharedSEDCache, Star
from pyGrater.density import two_power_law
from pyGrater.size_distributions import power_law_distribution


PARAMETERS = {
    "r0": 1.0,
    "h0": 0.05,
    "alphain": 10.0,
    "alphaout": -5.0,
    "gamma": 2.0,
    "beta": 1.0,
    "itilt": 0.0,
    "PA": 90.0,
    "omega": 45.0,
    "a_min": 1e-6,
    "a_max": 1e-3,
    "kappa": 3.5,
    "N_sizes_integral": 24,
    "g": 0.5,
    "A_norm": 1e30,
}


def _models(composition="astroSi"):
    wavelengths = np.array([5.0, 8.0, 12.0, 20.0, 30.0])
    star = Star(star_name="HD113766")
    grain = Grain(redo_Q=False, composition=composition)
    arguments = (
        grain, star, two_power_law, power_law_distribution, wavelengths)
    return (
        SED(*arguments, N_distances=48),
        CachedSED(
            *arguments, N_distances=48,
            shared_cache=SharedSEDCache(max_entries=8)),
    )


def test_cached_sed_matches_original_exactly():
    original, cached = _models()
    expected = original.get_SED(
        keep_separate_fluxes=True, **PARAMETERS)
    actual = cached.get_SED(
        keep_separate_fluxes=True, **PARAMETERS)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])

    expected_total = original.get_SED(**PARAMETERS)
    actual_total = cached.get_SED(**PARAMETERS)
    np.testing.assert_array_equal(actual_total, expected_total)
    assert cached.last_spatial_cache_hit
    assert cached.last_projection_cache_hit


def test_cache_key_invalidates_when_physics_changes():
    _, cached = _models()
    first = cached.get_SED(**PARAMETERS)
    changed = cached.get_SED(**{**PARAMETERS, "r0": 1.1})
    assert not cached.last_spatial_cache_hit
    assert not np.array_equal(first, changed)


def test_grouped_compositions_share_density_and_remain_exact():
    wavelengths = np.array([8.0, 12.0, 20.0])
    star = Star(star_name="HD113766")
    cache = SharedSEDCache(max_entries=4)
    pairs = []
    for name in ("c_olivine_Fe_Poor", "astroSi"):
        grain = Grain(redo_Q=False, composition=name)
        arguments = (
            grain, star, two_power_law, power_law_distribution, wavelengths)
        pairs.append((
            name,
            SED(*arguments, N_distances=40),
            CachedSED(
                *arguments, N_distances=40, shared_cache=cache,
                cache_namespace="ring"),
        ))
    members = [cached for _, _, cached in pairs]
    for name, _, cached in pairs:
        cached.spatial_group_members = members
        cached.spatial_member_key = name

    for index, (_, original, cached) in enumerate(pairs):
        expected = original.get_SED(**PARAMETERS)
        actual = cached.get_SED(**PARAMETERS)
        np.testing.assert_array_equal(actual, expected)
        assert cached.last_spatial_cache_hit is (index > 0)
    assert cache.info()["misses"] == 1
    assert cache.info()["hits"] == 1


@pytest.mark.parametrize("ring_radius_au", [0.01, 0.02, 0.05, 0.1])
def test_grouped_cache_preserves_sublimation_region_physics(
        ring_radius_au):
    """Small hot rings and tiny grains must remain composition-specific."""
    wavelengths = np.array([2.0, 3.0, 5.0, 8.0, 10.0, 15.0])
    parameters = {
        **PARAMETERS,
        "r0": ring_radius_au,
        "h0": 0.05 * ring_radius_au,
        "alphaout": -3.0,
        "a_min": 0.05e-6,
        "a_max": 1000e-6,
        "N_sizes_integral": 50,
    }
    star = Star(star_name="HD113766")
    cache = SharedSEDCache(max_entries=4)
    pairs = []
    for name in ("c_olivine_Fe_Poor", "fayalite", "astroSi"):
        grain = Grain(redo_Q=False, composition=name)
        arguments = (
            grain, star, two_power_law, power_law_distribution, wavelengths)
        pairs.append((
            name,
            SED(*arguments, N_distances=64),
            CachedSED(
                *arguments, N_distances=64, shared_cache=cache,
                cache_namespace="hot_ring"),
        ))
    members = [cached for _, _, cached in pairs]
    for name, _, cached in pairs:
        cached.spatial_group_members = members
        cached.spatial_member_key = name

    for _, original, cached in pairs:
        expected = original.get_SED(
            keep_separate_fluxes=True, **parameters)
        actual = cached.get_SED(
            keep_separate_fluxes=True, **parameters)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(
            actual[0] + actual[1], expected[0] + expected[1])

        # The inherited mass path independently applies this material's
        # sublimation temperature and must also remain unchanged.
        expected_mass = original.get_total_mass(**parameters)
        actual_mass = cached.get_total_mass(**parameters)
        np.testing.assert_allclose(
            actual_mass, expected_mass, rtol=5e-15, atol=0.0)
