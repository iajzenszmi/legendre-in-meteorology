import numpy as np
import math
import matplotlib.pyplot as plt

# ---------- Core Math ----------

def double_factorial_log(n: int) -> float:
    """Return n!! using log-gamma for stability."""
    if n == -1 or n == 0:
        return 1.0
    m = n // 2
    log_val = math.lgamma(2*m + 1) - m * math.log(2.0) - math.lgamma(m + 1)
    return math.exp(log_val)

def assoc_legendre_stable(l: int, m: int, mu: np.ndarray) -> np.ndarray:
    """Stable associated Legendre with Condon–Shortley phase."""
    if m < 0 or m > l:
        raise ValueError("Require 0 <= m <= l")
    mu = np.asarray(mu)
    mask = np.isclose(np.abs(mu), 1.0)

    if m == 0:
        Pmm = np.ones_like(mu)
    else:
        df = double_factorial_log(2*m - 1)
        Pmm = ((-1)**m) * df * (1.0 - mu*mu)**(m/2.0)

    if l == m:
        Pmm[mask] = 0.0 if m > 0 else ((mu[mask] >= 0)*1.0 + (mu[mask] < 0)*((-1)**l))
        return Pmm

    Pm1m = mu * (2*m + 1) * Pmm
    if l == m + 1:
        Pm1m[mask] = 0.0 if m > 0 else ((mu[mask] >= 0)*1.0 + (mu[mask] < 0)*((-1)**l))
        return Pm1m

    P_lm_minus2, P_lm_minus1 = Pmm, Pm1m
    for ell in range(m+2, l+1):
        Plm = ((2*ell - 1)*mu*P_lm_minus1 - (ell + m - 1)*P_lm_minus2) / (ell - m)
        P_lm_minus2, P_lm_minus1 = P_lm_minus1, Plm

    P_lm_minus1[mask] = 0.0 if m > 0 else ((mu[mask] >= 0)*1.0 + (mu[mask] < 0)*((-1)**l))
    return P_lm_minus1

def N_lm(l, m):
    """Fully normalized spherical harmonic constant."""
    return math.sqrt((2*l+1)/(4*math.pi) * math.factorial(l-m)/math.factorial(l+m))

def gaussian_latitudes(nlat):
    """Return Gaussian latitudes (deg) and weights."""
    mu, w = np.polynomial.legendre.leggauss(nlat)
    lat = np.rad2deg(np.arcsin(mu))
    return lat[::-1], w[::-1]  # N to S order

def Y_lm_complex(l, m, lat_g, lon_g):
    """Complex spherical harmonic Y_lm on grid."""
    lat = np.deg2rad(lat_g)
    lon = np.deg2rad(lon_g)
    mu = np.sin(lat)
    P = assoc_legendre_stable(l, abs(m), mu) * N_lm(l, abs(m))
    if m < 0:
        # Relation for negative m
        return ((-1)**m) * np.conjugate(Y_lm_complex(l, -m, lat_g, lon_g))
    return P[:, None] * np.exp(1j * m * lon[None, :])

# ---------- Transform Routines ----------

def forward_complex_SH(field, lat_g, lon_g, w_g, Lmax):
    """Forward complex spherical harmonic transform."""
    nlat, nlon = field.shape
    dlon = 2*math.pi / nlon
    a_lm = {}
    for l in range(Lmax+1):
        for m in range(-l, l+1):
            Ylm = Y_lm_complex(l, m, lat_g, lon_g)
            integrand = field * np.conjugate(Ylm)
            coef = np.sum(w_g[:, None] * integrand) * dlon
            a_lm[(l, m)] = coef
    return a_lm

def inverse_complex_SH(a_lm, lat_g, lon_g, Lmax):
    """Inverse complex spherical harmonic transform."""
    nlat, nlon = len(lat_g), len(lon_g)
    field = np.zeros((nlat, nlon), dtype=complex)
    for l in range(Lmax+1):
        for m in range(-l, l+1):
            field += a_lm[(l, m)] * Y_lm_complex(l, m, lat_g, lon_g)
    return field.real  # input was real

# ---------- Test Field ----------

def synthetic_field(lat_g, lon_g):
    LAT, LON = np.meshgrid(lat_g, lon_g, indexing='ij')
    return np.exp(-((LAT-45)**2/200) - ((LON-90)**2/500))

# ---------- Orthogonality Check ----------

def orthogonality_test(lat_g, lon_g, w_g, Lmax):
    """Return max deviation from identity in Gram matrix."""
    dlon = 2*math.pi / len(lon_g)
    basis = []
    for l in range(Lmax+1):
        for m in range(-l, l+1):
            basis.append(Y_lm_complex(l, m, lat_g, lon_g))
    nbasis = len(basis)
    G = np.zeros((nbasis, nbasis), dtype=complex)
    for i in range(nbasis):
        for j in range(nbasis):
            G[i, j] = np.sum(w_g[:, None] * basis[i] * np.conjugate(basis[j])) * dlon
    dev = np.max(np.abs(G - np.eye(nbasis)))
    return dev

# ---------- Main Execution ----------

if __name__ == "__main__":
    Lmax = 63
    nlat = 96
    nlon = 192
    lat_g, w_g = gaussian_latitudes(nlat)
    lon_g = np.linspace(0, 360, nlon, endpoint=False)

    # Create test field
    field_in = synthetic_field(lat_g, lon_g)

    # Forward & inverse transform
    a_lm = forward_complex_SH(field_in, lat_g, lon_g, w_g, Lmax)
    field_out = inverse_complex_SH(a_lm, lat_g, lon_g, Lmax)

    # Error metrics
    err_max = np.max(np.abs(field_in - field_out))
    err_rms = np.sqrt(np.mean((field_in - field_out)**2))

    print(f"Max reconstruction error: {err_max:.3e}")
    print(f"RMS reconstruction error: {err_rms:.3e}")

    # Orthogonality check
    dev = orthogonality_test(lat_g, lon_g, w_g, 5)  # test small subset for speed
    print(f"Max orthogonality deviation (Lmax=5): {dev:.3e}")

    # Plot
    plt.figure(figsize=(9,4), dpi=120)
    plt.subplot(1,2,1)
    plt.pcolormesh(lon_g, lat_g, field_in, shading='auto')
    plt.title("Original Field")
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.pcolormesh(lon_g, lat_g, field_out, shading='auto')
    plt.title("Reconstructed Field (T63 Complex SH)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
