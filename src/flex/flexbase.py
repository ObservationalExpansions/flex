from numbers import Integral, Real

import numpy as np

# for the Laguerre polynomials
from scipy.special import eval_genlaguerre


def _as_1d_numeric_array(name, value):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array-like structure.") from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array-like structure.")

    return array


def _as_scalar_or_matching_array(name, value, shape):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a scalar or array-like structure.")

    if isinstance(value, Real):
        return value

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a scalar or numeric array-like structure.") from exc

    if array.ndim == 0:
        return array.item()
    if array.shape != shape:
        raise ValueError(f"{name} must be scalar or have the same shape as R.")

    return array


class FLEX:
    """
   FLEX class for calculating Laguerre basis amplitudes.

    This class provides methods for calculating Laguerre basis amplitudes based on Weinberg & Petersen (2021).

    Parameters:
        rscl (float): Scale parameter for the Laguerre basis.
        mass (array-like): Mass values for particles.
        phi (array-like): Angular phi values.
        velocity (array-like): Velocity values.
        R (array-like): Radial values.
        mmax (int): Maximum order parameter for m.
        nmax (int): Maximum order parameter for n.

    Methods:
        gamma_n(nrange, rscl): Calculate the Laguerre alpha=1 normalisation.
        G_n(R, nrange, rscl): Calculate the Laguerre basis.
        n_m(): Calculate the angular normalisation.
        laguerre_amplitudes(): Calculate Laguerre amplitudes for the given parameters.
        laguerre_reconstruction(rr, pp): Calculate Laguerre reconstruction.

    Attributes:
        rscl (float): Scale parameter for the Laguerre basis.
        mass (array-like): Mass values for particles.
        phi (array-like): Angular phi values.
        velocity (array-like): Velocity values.
        R (array-like): Radial values.
        mmax (int): Maximum order parameter for m.
        nmax (int): Maximum order parameter for n.
        coscoefs (array-like): Cosine coefficients.
        sincoefs (array-like): Sine coefficients.
        reconstruction (array-like): Laguerre reconstruction result.
    """

    def __init__(self, rscl, mmax, nmax, R, phi, mass=1., velocity=1., newaxis=False):
        """
        Initialize the LaguerreAmplitudes instance with parameters.

        Args:
            rscl (float): Scale parameter for the Laguerre basis.
            mmax (int): Maximum Fourier harmonic order.
            nmax (int): Maximum Laguerre order.
            R (array-like): Radial values.
            velocity (array-like): Velocity values.
            mass (integer or array-like): Mass values for particles.
            phi (integer or array-like): Angular phi values.

        """

        # check for input validity
        if not isinstance(rscl, Real) or isinstance(rscl, bool):
            raise ValueError("rscl must be a scalar value.")
        if not isinstance(mmax, Integral) or isinstance(mmax, bool) or mmax < 0:
            raise ValueError("mmax must be a non-negative integer.")
        if not isinstance(nmax, Integral) or isinstance(nmax, bool) or nmax < 0:
            raise ValueError("nmax must be a non-negative integer.")
        if not isinstance(R, (np.ndarray, list, tuple)):
            raise ValueError("R must be an array-like structure.")
        if not isinstance(phi, (np.ndarray, list, tuple)):
            raise ValueError("phi must be an array-like structure.")
        if not isinstance(mass, (Real, np.ndarray, list, tuple)):
            raise ValueError("mass must be a scalar or array-like structure.")
        if not isinstance(velocity, (Real, np.ndarray, list, tuple)):
            raise ValueError("velocity must be a scalar or array-like structure.")

        R = _as_1d_numeric_array("R", R)
        phi = _as_1d_numeric_array("phi", phi)
        if phi.shape != R.shape:
            raise ValueError("phi must have the same shape as R.")

        mass = _as_scalar_or_matching_array("mass", mass, R.shape)
        velocity = _as_scalar_or_matching_array("velocity", velocity, R.shape)

        self.rscl     = rscl
        self.mmax     = mmax
        self.nmax     = nmax
        self.R        = R
        self.phi      = phi
        self.mass     = mass
        self.velocity = velocity

        # run the amplitude calculation
        if newaxis:
            self.laguerre_amplitudes_newaxis()
        else:
            # default behaviour 
            self.laguerre_amplitudes()

    def _gamma_n(self, nrange, rscl):
        """
        Calculate the Laguerre alpha=1 normalisation.

        Args:
            nrange (array-like): Range of order parameters.
            rscl (float): Scale parameter for the Laguerre basis.

        Returns:
            array-like: Laguerre alpha=1 normalisation values.
        """
        return (rscl / 2.) * np.sqrt(nrange + 1.)

    def _G_n(self, R, nrange, rscl):
        """
        Calculate the Laguerre basis.

        Args:
            R (array-like): Radial values.
            nrange (array-like): Range of order parameters.
            rscl (float): Scale parameter for the Laguerre basis.

        Returns:
            array-like: Laguerre basis values.
        """
        laguerrevalues = np.array([eval_genlaguerre(n, 1, 2 * R / rscl)/self._gamma_n(n, rscl) for n in nrange])
        return np.exp(-R / rscl) * laguerrevalues

    def _n_m(self):
        """
        Calculate the angular normalisation.

        Returns:
            array-like: Angular normalisation values.
        """
        deltam0 = np.zeros(self.mmax+1)

        deltam0[0] = 1.0

        return np.power((deltam0 + 1) * np.pi / 2.,-1.)

    def laguerre_amplitudes_newaxis(self):
        """
        Calculate Laguerre amplitudes for the given parameters.

        Returns:
            tuple: Tuple containing the cosine and sine amplitudes.
        """

        G_j = self._G_n(self.R, np.arange(0, self.nmax, 1), self.rscl)

        nmvals = self._n_m()
        mrange = np.arange(0, self.mmax + 1, 1)
        cosm = nmvals[:, np.newaxis] * np.cos(mrange[:, np.newaxis] * self.phi)
        sinm = nmvals[:, np.newaxis] * np.sin(mrange[:, np.newaxis] * self.phi)

        # broadcast to sum values
        self.coscoefs = np.nansum(
            cosm[:, np.newaxis, :] * G_j[np.newaxis, :, :] * self.mass * self.velocity,
            axis=2,
        )
        self.sincoefs = np.nansum(
            sinm[:, np.newaxis, :] * G_j[np.newaxis, :, :] * self.mass * self.velocity,
            axis=2,
        )


    def laguerre_amplitudes(self):
        """
        Calculate Laguerre amplitudes for the given parameters.

        Returns:
            none. Attributes are added containing the cosine and sine amplitudes.
        """

        G_j = self._G_n(self.R, np.arange(0, self.nmax, 1), self.rscl)

        nmvals = self._n_m()
        mrange = np.arange(0, self.mmax + 1, 1)
        cosm = nmvals[:, np.newaxis] * np.cos(mrange[:, np.newaxis] * self.phi)
        sinm = nmvals[:, np.newaxis] * np.sin(mrange[:, np.newaxis] * self.phi)

        if np.isscalar(self.mass) and np.isscalar(self.velocity):
            scale = self.mass * self.velocity  # scalar
            self.coscoefs = scale * np.einsum('mn,jn->mj', cosm, G_j)
            self.sincoefs = scale * np.einsum('mn,jn->mj', sinm, G_j)   
        else:
            # vector case
            self.coscoefs = np.einsum('mn,jn,n->mj', cosm, G_j, self.mass * self.velocity)
            self.sincoefs = np.einsum('mn,jn,n->mj', sinm, G_j, self.mass * self.velocity)

    def laguerre_reconstruction(self, rr, pp):
        """
        Reconstruct a function using Laguerre amplitudes.

        Args:
            rr (array-like): Radial values.
            pp (array-like): Angular phi values.

        This method reconstructs a function using the Laguerre amplitudes calculated with the `laguerre_amplitudes` method.

        Returns:
            array-like: The reconstructed function values.
        """
        rr = _as_1d_numeric_array("rr", rr)
        pp = _as_1d_numeric_array("pp", pp)
        if pp.shape != rr.shape:
            raise ValueError("pp must have the same shape as rr.")

        G_j = self._G_n(rr, np.arange(0, self.nmax, 1), self.rscl)

        mrange = np.arange(0, self.mmax + 1, 1)
        cosm = np.cos(mrange[:, np.newaxis] * pp)
        sinm = np.sin(mrange[:, np.newaxis] * pp)

        cos_reconstruction = np.einsum('mn,mi,ni->i', self.coscoefs, cosm, G_j)
        sin_reconstruction = np.einsum('mn,mi,ni->i', self.sincoefs, sinm, G_j)

        self.reconstruction = 0.5 * (cos_reconstruction + sin_reconstruction)
