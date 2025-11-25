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


def test_flex_total_normalisation():
    # Create a FLEX instance
    rscl = 1.0
    mmax = 0
    nmax = 1
    R = np.linspace(0.1, 5.0, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    mass = np.ones(100)

    F = flex.FLEX(rscl, mmax, nmax, R, phi, mass)

    F.laguerre_reconstruction(R, phi)

    # check the values for the norm
    #print(F.reconstruction)
