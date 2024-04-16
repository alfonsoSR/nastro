import numpy as np
from ..types import Vector


def enrke(M: Vector, e: float | Vector, tol: float = 3.0e-15) -> Vector:
    """Calculate eccentric anomaly from mean anomaly using the ENRKE method.

    source: https://doi.org/10.1051/0004-6361/202141423

    :param M: Mean anomaly [rad]
    :param e: Eccentricity of the keplerian orbit
    :param tol: Method's accuracy (default 3.0e-15)
    :return: Eccentric anomaly [rad]
    """

    if not isinstance(e, float):
        raise NotImplementedError("Vectorized eccentricity not implemented yet")

    # TODO: Vectorize loops

    PI = 3.1415926535897932385
    TWOPI = 6.2831853071795864779

    tol2s = 2.0 * tol / (e + 2.2e-16)
    al = tol / 1.0e7
    be = tol / 0.3

    # Fit angle in range (0, 2pi) if needed

    Mr = M % TWOPI
    flip = np.ones_like(Mr, dtype=float)
    Eout = np.zeros_like(Mr, dtype=float)

    for idx, Mri in enumerate(Mr):
        if Mri > PI:
            Mr[idx] = TWOPI - Mri
        else:
            flip[idx] = -1.0

    for idx, Mri in enumerate(Mr):

        if e > 0.99 and Mri < 0.0045:
            fp = 2.7 * Mri
            fpp = 0.301
            f = 0.154
            while fpp - fp > (al + be * f):
                if (f - e * np.sin(f) - Mri) > 0.0:
                    fpp = f
                else:
                    fp = f
                f = 0.5 * (fp + fpp)
            Eout[idx] = (M[idx] + flip[idx] * (Mri - f)) % TWOPI
        else:
            Eapp = Mri + 0.999999 * Mri * (PI - Mri) / (
                2.0 * Mri + e - PI + 2.4674011002723395 / (e + 2.2e-16)
            )
            fpp = e * np.sin(Eapp)
            fppp = e * np.cos(Eapp)
            fp = 1.0 - fppp
            f = Eapp - fpp - Mri
            delta = -f / fp
            fp3 = fp * fp * fp
            ffpfpp = f * fp * fpp
            f2fppp = f * f * fppp
            delta = (
                delta
                * (fp3 - 0.5 * ffpfpp + f2fppp / 3.0)
                / (fp3 - ffpfpp + 0.5 * f2fppp)
            )

            while delta * delta > fp * tol2s:
                Eapp += delta
                fp = 1.0 - e * np.cos(Eapp)
                delta = (Mri - Eapp + e * np.sin(Eapp)) / fp

            Eapp += delta
            Eout[idx] = (M[idx] + flip[idx] * (Mri - Eapp)) % TWOPI

    return Eout
