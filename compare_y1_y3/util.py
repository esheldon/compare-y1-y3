import numpy as np
import esutil as eu


Y3_MVALS = [
    (-0.006, 0.008),
    (-0.010, 0.013),
    (-0.026, 0.009),
    (-0.032, 0.012),
]
Y1_MVAL = 0.012

Y1_CHI_FACTORS = np.array([
    675.60175456, 1039.66805889, 1374.68291108, 1699.66063854, 1987.87549653]
)
Y3_CHI_FACTORS = np.array(
    [761.98657437, 1158.2870951, 1497.06693415, 1795.64094319, 2029.11026506]
)

# mean/width
Y1_ZS_OFFSETS = np.array([
    (0.1, 1.6),
    (-1.9, 1.3),
    (0.9,  1.1),
    (-1.8, 2.2),
])
Y1_ZS_OFFSETS /= 100


def interpolate_y1_onto_y3(r3, r1, ds1):
    """
    interpolate the Y1 delta sigma measurements at the Y3 mean radii

    r3: array
        Array of Y3 radii
    r1: array
        Array of Y1 radii
    ds1: array
        Array of Y1 DeltaSigma

    Returns
    -------
    Delta Sigma interpolated to Y3 radii
    """
    ds1_interp = np.interp(
        r3, r1, ds1,
    )

    if False:
        import hickory
        plt = hickory.Plot()
        plt.plot(r1, ds1, label='y1 binning')
        plt.plot(r3, ds1_interp, label='interp')
        plt.set(
            xlabel=r'$R [\mathrm{Mpc}]$',
            ylabel=r'$\Delta\Sigma$',
        )
        plt.set_xscale('log')
        plt.set_yscale('log')
        plt.legend()
        plt.show()

    return ds1_interp


def inv_sigma_crit_eff_fast(*, zlbin, nzl, zsbin, nzs):
    """
    from Carles and Judit
    """
    c = eu.cosmology.Cosmo(omega_m=0.3)

    dzl = zlbin[1]-zlbin[0]
    dzs = zsbin[1]-zsbin[0]
    norml = np.sum(nzl*dzl)
    norms = np.sum(nzs*dzs)
    nzl = nzl/norml
    nzs = nzs/norms

    # Define meshgrid for redshifts and for Nzs
    X, Y = np.meshgrid(zlbin, zsbin)
    NZL, NZS = np.meshgrid(nzl, nzs)
    # Construct 2-D integrand
    sci_flat = c.sigmacritinv(X, Y)
    sci_re = np.reshape(sci_flat, (len(zsbin), len(zlbin)), order='C')
    integrand = NZL*NZS*sci_re
    # Do a 1-D integral over every row
    Integ = np.zeros(len(zsbin))
    for i in range(len(zsbin)):
        Integ[i] = np.trapz(integrand[i, :], zlbin)

    # Then an integral over the result
    F = np.trapz(Integ, zsbin)

    return F


def get_covdiff_inv(*, cov1, cov2):
    covdiff = cov1 - cov2
    covinv = np.linalg.inv(covdiff)
    return covinv


def fit_amp(*, d, t, covinv):
    """
    d = A * t
    A = t C^{-1} d/ [t C^{-1} t]
    """

    einv = 1/np.dot(t, np.dot(covinv, t))
    amp = np.dot(t, np.dot(covinv, d)) * einv
    amp_err = np.sqrt(einv)

    return amp, amp_err




