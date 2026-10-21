"""Process-local candidate; no installed DFTpy files are modified.

Differentiate the *discrete* sqrt(rho) quadratic vW energy, retaining
Nyquist modes via the spectral Hessian rather than a real first gradient.
Supports positive, spin-unpolarized density and sigma=None only.
"""
import numpy as np

def vw_stress_discrete(rho, y=1.0, energy=None, sigma=None, **kwargs):
    if sigma is not None or rho.ndim != 3 or np.any(np.asarray(rho) <= 0):
        raise ValueError('vW candidate requires sigma=None, positive scalar density')
    phi = np.sqrt(rho)
    phiG = phi.fft()
    g = rho.grid.get_reciprocal().g
    out = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            hij = (g[i] * g[j] * phiG).ifft(force_real=True)
            out[i, j] = out[j, i] = -float(y) * float(np.sum(phi * hij)) * rho.grid.dV / rho.grid.volume
    return out

def install():
    import dftpy.functional.kedf as kedf
    old = kedf.KEDFEngines_Stress['VW']
    kedf.KEDFEngines_Stress['VW'] = vw_stress_discrete
    return old
