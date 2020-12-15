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


def inv_sigma_crit_eff_fast(*, zlbin, nzl, zsbin, nzs, cosmo_pars):
    """
    from Carles and Judit
    """
    c = eu.cosmology.Cosmo(
        omega_m=cosmo_pars['omega_m'],
        h=cosmo_pars['h'],
    )

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


def get_mean_z(*, z, nz):
    return eu.integrate.qgauss(
        z,
        z*nz,
        1000,
    )


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


def jackknife(*, data, weights):

    nchunks = data.size

    sum = (data * weights).sum()
    wsum = weights.sum()

    mn = sum/wsum

    mns = np.zeros(data.size)

    for i in range(nchunks):

        tsum = sum - data[i] * weights[i]
        twsum = wsum - weights[i]

        mns[i] = tsum/twsum

    fac = (nchunks-1)/float(nchunks)
    var = fac*(((mns - mn)**2).sum())

    err = np.sqrt(var)
    return mn, err


def jackknife_ratio(*, data1, weights1, data2, weights2, doplot=False):

    assert data1.size == data2.size
    nchunks = data1.size

    wsum1 = weights1.sum()
    wsum2 = weights2.sum()

    sum1 = (data1 * weights1).sum()
    sum2 = (data2 * weights2).sum()

    mn1 = sum1/wsum1
    mn2 = sum2/wsum2

    rat = mn1/mn2

    rats = np.zeros(data1.size)

    for i in range(nchunks):

        tsum1 = sum1 - data1[i] * weights1[i]
        tsum2 = sum2 - data2[i] * weights2[i]

        twsum1 = wsum1 - weights1[i]
        twsum2 = wsum2 - weights2[i]

        tmn1 = tsum1/twsum1
        tmn2 = tsum2/twsum2

        rats[i] = tmn1/tmn2

    fac = (nchunks-1)/float(nchunks)
    rat_var = fac*(((rat - rats)**2).sum())

    if doplot:
        import hickory
        # hickory.hist( (rats - rats.mean())*fac*np.sqrt(data1.size) + rats.mean()
        plt = hickory.Plot()
        plt.hist((rats - rats.mean())*fac + rats.mean(), bins=20)
        plt.show()

    rat_err = np.sqrt(rat_var)
    return rat, rat_err


def print_stats(*, data1, data1_err, data2, data2_err, name, doplot=False):

    weights1 = 1.0/data1_err**2
    weights2 = 1.0/data2_err**2

    m1, m1err = jackknife(data=data1, weights=weights1)
    m2, m2err = jackknife(data=data2, weights=weights2)

    print('%s means:' % name)
    print('%s1: %g +/- %g' % (name, m1, m1err))
    print('%s2: %g +/- %g' % (name, m2, m2err))

    rat, rat_err = jackknife_ratio(
        data1=data1,
        weights1=weights1,
        data2=data2,
        weights2=weights2,
        doplot=doplot,
    )
    print('%s ratio of means:  %g +/- %g' % (name, rat, rat_err))


def get_y1_nofz(*, data, lbin, sbin, sample=False):
    """
    get n(z) data for the given lens and source bin.  A random y3 source n(z)
    is used each time this code is called


    the Y1 n(z) are shifted according to the prior each time this is called

    data: dict
        Dictionary of arrays, keyed by
            y1y1: y1 lenses, y1 sources
            y1y3: y1 lenses, y3 sources
        The data for each key is
            'gammat': The gamma_t data vector for all l/s bins
            'nzl': redshift and n(z) for the lenses for all l/s bins
            'nzs': redshift and n(z) for the sources for all l/s bins
    lbin: int
        lens bin, 1 offset
    sbin: int
        source bin, 1 offst

    Returns
    -------
    dict keyed by y1y1 or y1y3 with the particular data from the
    requested lens/source bins.  Each key has entries
        'lzbin': lens z grid
        'lnofz': n(zl) on the grid
        'szbin': source z grid
        'snofz': n(zs) on z grid
    """

    lbin_name = 'bin%d' % lbin
    sbin_name = 'bin%d' % sbin

    moff = Y1_ZS_OFFSETS[sbin-1][0]
    woff = Y1_ZS_OFFSETS[sbin-1][1]
    if sample:
        z1off = np.random.normal(
            loc=moff,
            scale=woff,
        )
    else:
        z1off = moff

    zs = data['nzs']['z_mid'] + z1off

    w, = np.where(zs > 0)
    zs = zs[w]
    nofzs = data['nzs'][sbin_name][w]

    zdata = {
        'lzbin': data['nzl']['z_mid'],
        'lnofz': data['nzl'][lbin_name],
        'szbin': zs,
        'snofz': nofzs,
    }
    return zdata


def get_y3_nofz(*, data, lbin, sbin, sample=False):
    """
    get n(z) data for the given lens and source bin.  A random y3 source n(z)
    is used each time this code is called


    the Y1 n(z) are shifted according to the prior each time this is called

    data: dict
        Dictionary of arrays, keyed by
            y1y1: y1 lenses, y1 sources
            y1y3: y1 lenses, y3 sources
        The data for each key is
            'gammat': The gamma_t data vector for all l/s bins
            'nzl': redshift and n(z) for the lenses for all l/s bins
            'nzs': redshift and n(z) for the sources for all l/s bins
    lbin: int
        lens bin, 1 offset
    sbin: int
        source bin, 1 offst

    Returns
    -------
    dict keyed by y1y1 or y1y3 with the particular data from the
    requested lens/source bins.  Each key has entries
        'lzbin': lens z grid
        'lnofz': n(zl) on the grid
        'szbin': source z grid
        'snofz': n(zs) on z grid
    """

    if sample:
        nsamp = len(data['nzs_samples'])
        i = np.random.randint(nsamp)
        nzs = data['nzs_samples'][i]
    else:
        nzs = data['nzs']

    lbin_name = 'bin%d' % lbin
    sbin_name = 'bin%d' % sbin

    zdata = {
        'lzbin': data['nzl']['z_mid'],
        'lnofz': data['nzl'][lbin_name],
        'szbin': nzs['z_mid'],
        'snofz': nzs[sbin_name],
    }
    return zdata


def get_oneplusm(*, sbin, source_type, sample=False, use_y1_m=False):
    y3_oneplusm = 1 + Y3_MVALS[sbin-1][0]
    if source_type == 'y3':
        oneplusm = y3_oneplusm
    else:
        if use_y1_m:
            oneplusm = 1 + Y1_MVAL
        else:
            oneplusm = y3_oneplusm
            if sample:
                oneplusm = (
                    oneplusm + np.random.normal(scale=Y3_MVALS[sbin-1][1])
                )

    return oneplusm


def add_rescaled_data(
    *, data, cosmo_pars, source_type,
    sample=False, use_y1_m=False,
):

    npts = data['gammat']['value'].size

    data['r'] = np.zeros(npts)
    data['ds'] = np.zeros(npts)
    data['dscov'] = np.zeros((npts, npts))

    for lbin in range(1, 5+1):
        for sbin in range(1, 4+1):

            if source_type == 'y1':
                zdata = get_y1_nofz(
                    data=data, lbin=lbin, sbin=sbin, sample=sample,
                )
            else:
                zdata = get_y3_nofz(
                    data=data, lbin=lbin, sbin=sbin, sample=sample,
                )

            siginv = inv_sigma_crit_eff_fast(
                zlbin=zdata['lzbin'],
                nzl=zdata['lnofz'],
                zsbin=zdata['szbin'],
                nzs=zdata['snofz'],
                cosmo_pars=cosmo_pars,
            )

            data['zl_mean'] = get_mean_z(
                z=zdata['lzbin'],
                nz=zdata['lnofz'],
            )

            oneplusm = get_oneplusm(
                sbin=sbin, source_type=source_type,
                sample=sample, use_y1_m=use_y1_m,
            )

            w, = np.where(
                (data['gammat']['bin1'] == lbin) &
                (data['gammat']['bin2'] == sbin)
            )
            fac = 1.0/oneplusm/siginv

            ds = data['gammat']['value'][w] * fac
            rad = np.deg2rad(data['gammat']['ang'][w]/60)
            r = rad * data['chi_factors'][lbin-1]

            data['r'][w] = r

            imin, imax = w[0], w[-1]+1
            cov = (
                data['gammat_cov'][imin:imax, imin:imax] * fac**2
            )

            data['ds'][w] = ds
            data['dscov'][imin:imax, imin:imax] = cov
