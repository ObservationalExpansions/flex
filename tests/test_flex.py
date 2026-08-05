import pytest
import numpy as np

import flex

def test_flex_initialization():
    # Test valid initialization
    rscl = 1.0
    mmax = 3
    nmax = 5
    R = np.array([0.1, 0.2, 0.3])
    phi = np.array([0.0, np.pi/4, np.pi/2])
    mass = np.array([1.0, 2.0, 3.0])
    velocity = np.array([10.0, 20.0, 30.0])

    flex_instance = flex.FLEX(rscl, mmax, nmax, R, phi, mass, velocity)
    assert flex_instance.rscl == rscl
    assert flex_instance.mmax == mmax
    assert flex_instance.nmax == nmax
    np.testing.assert_array_equal(flex_instance.R, R)
    np.testing.assert_array_equal(flex_instance.phi, phi)
    np.testing.assert_array_equal(flex_instance.mass, mass)
    np.testing.assert_array_equal(flex_instance.velocity, velocity)

    # Test invalid rscl type
    with pytest.raises(ValueError):
        flex.FLEX("invalid_rscl", mmax, nmax, R, phi, mass, velocity)

    # Test negative mmax
    with pytest.raises(ValueError):
        flex.FLEX(rscl, -1, nmax, R, phi, mass, velocity)

    # Test negative nmax
    with pytest.raises(ValueError):
        flex.FLEX(rscl, mmax, -1, R, phi, mass, velocity)

    # Test mismatched R and phi lengths
    with pytest.raises(ValueError):
        flex.FLEX(rscl, mmax, nmax, R, phi[:-1], mass, velocity)

    # Test mismatched mass length
    with pytest.raises(ValueError):
        flex.FLEX(rscl, mmax, nmax, R, phi, mass[:-1], velocity)

    # Test mismatched velocity length
    with pytest.raises(ValueError):
        flex.FLEX(rscl, mmax, nmax, R, phi, mass, velocity[:-1])

    # Test nonsense inputs
    with pytest.raises(ValueError):
        flex.FLEX(rscl, mmax, nmax, True, phi, mass, velocity)
    with pytest.raises(ValueError):
        flex.FLEX(rscl, mmax, nmax, R, True, mass, velocity)
    with pytest.raises(ValueError):
        flex.FLEX(rscl, mmax, nmax, R, phi, mass='nonsense')
    with pytest.raises(ValueError):
        flex.FLEX(rscl, mmax, nmax, R, phi, velocity='nonsense')

def test_flex_version():
    assert isinstance(flex.__version__, str)

def test_flex_scalars():
    # Create a FLEX instance
    rscl = 1.0
    mmax = 2
    nmax = 10
    R = np.linspace(0.1, 5.0, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    mass = 1.0
    velocity = 1.0

    F = flex.FLEX(rscl, mmax, nmax, R, phi, mass, velocity)


def test_flex_accepts_zero_dimensional_array_scalars():
    rscl = 1.0
    mmax = 2
    nmax = 10
    R = np.linspace(0.1, 5.0, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    mass = np.array(1.0)
    velocity = np.array(2.0)

    F = flex.FLEX(rscl, mmax, nmax, R, phi, mass, velocity)

    assert F.mass == 1.0
    assert F.velocity == 2.0


@pytest.mark.parametrize(
    "R, phi, mass, velocity",
    [
        (["bad"], [0.0], 1.0, 1.0),
        (np.ones((2, 2)), np.ones((2, 2)), 1.0, 1.0),
        ([0.1], [0.0], True, 1.0),
        ([0.1], [0.0], ["bad"], 1.0),
    ],
)
def test_flex_rejects_invalid_numeric_inputs(R, phi, mass, velocity):
    with pytest.raises(ValueError):
        flex.FLEX(1.0, 0, 1, R, phi, mass, velocity)


def test_flex_total_power():
    # Create a FLEX instance
    rscl = 1.0
    mmax = 2
    nmax = 10
    R = np.linspace(0.1, 5.0, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    mass = np.random.uniform(0, 1, 100)
    velocity = np.random.uniform(0, 100, 100)

    # test the slower, careful, newaxis version
    F = flex.FLEX(rscl, mmax, nmax, R, phi, mass, velocity, newaxis=True)

    # test the faster vectorised version
    G = flex.FLEX(rscl, mmax, nmax, R, phi, mass, velocity)

    # check that both methods give the same coefficients
    np.testing.assert_allclose(F.coscoefs, G.coscoefs)
    np.testing.assert_allclose(F.sincoefs, G.sincoefs)

    # Compute total power in each harmonic
    totalm = np.linalg.norm(np.sqrt(F.coscoefs**2 + F.sincoefs**2), axis=1)

    # Check that totalm has the correct shape
    assert totalm.shape[0] == mmax + 1

    # Check that totalm values are non-negative
    assert np.all(totalm >= 0)


def test_laguerre_covariance_matches_numpy_covariance():
    R = np.array([0.2, 0.5, 0.9, 1.4])
    phi = np.array([0.1, 0.7, 1.8, 2.6])
    mass = np.array([1.0, 2.0, 0.5, 1.5])
    velocity = np.array([3.0, 1.0, 2.0, 4.0])
    F = flex.FLEX(1.0, 2, 3, R, phi, mass, velocity)

    coscovariance, sincovariance = F.laguerre_covariance()

    G_j = F._G_n(R, np.arange(F.nmax), F.rscl)
    mrange = np.arange(F.mmax + 1)
    angular_norm = F._n_m()
    cos_terms = (
        angular_norm[:, None, None]
        * np.cos(mrange[:, None, None] * phi[None, None, :])
        * G_j[None, :, :]
        * (mass * velocity)
    )
    sin_terms = (
        angular_norm[:, None, None]
        * np.sin(mrange[:, None, None] * phi[None, None, :])
        * G_j[None, :, :]
        * (mass * velocity)
    )
    expected_cos = np.array([np.cov(terms) for terms in cos_terms])
    expected_sin = np.array([np.cov(terms) for terms in sin_terms])

    assert coscovariance.shape == (3, 3, 3)
    assert sincovariance.shape == (3, 3, 3)
    np.testing.assert_allclose(coscovariance, expected_cos)
    np.testing.assert_allclose(sincovariance, expected_sin)
    assert F.coscovariance is coscovariance
    assert F.sincovariance is sincovariance


def test_laguerre_covariance_requires_two_particles():
    F = flex.FLEX(1.0, 1, 2, [0.2], [0.3])

    with pytest.raises(ValueError, match="At least two particles"):
        F.laguerre_covariance()


def test_flex_reconstruction_peak_amplitude():
    rscl = 1.0
    mmax = 2
    nmax = 6
    peak_density = 3.5

    nr = 120
    nphi = 96
    r_edges = np.linspace(0, 6, nr + 1)
    phi_edges = np.linspace(0, 2*np.pi, nphi + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])
    phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    R, phi_grid = np.meshgrid(r, phi, indexing="ij")
    dr = np.diff(r_edges)[:, np.newaxis]
    dphi = np.diff(phi_edges)[np.newaxis, :]

    density = peak_density * np.exp(-R / rscl) * (1 + 0.2 * np.cos(2 * phi_grid))
    mass = density * R * dr * dphi

    F = flex.FLEX(rscl, mmax, nmax, R.ravel(), phi_grid.ravel(), mass.ravel())
    F.laguerre_reconstruction(R.ravel(), phi_grid.ravel())

    np.testing.assert_allclose(
        np.max(F.reconstruction),
        np.max(density),
        rtol=0.05,
    )


def test_flex_reconstruction_requires_matching_shapes():
    F = flex.FLEX(
        1.0,
        0,
        1,
        np.array([0.1, 0.2, 0.3]),
        np.array([0.0, np.pi/4, np.pi/2]),
    )

    with pytest.raises(ValueError):
        F.laguerre_reconstruction(np.array([0.1, 0.2, 0.3]), np.array([0.0, np.pi/4]))
