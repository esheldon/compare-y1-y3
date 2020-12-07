import fitsio
import hickory
import numpy as np
import esutil as eu

# public data vectors for Y1
y1lenses_y1sources = '2pt_NG_mcal_1110.fits'

# y1lenses_y3sources = '/home/esheldon/git/xcorr/runs/Y3_mastercat___UNBLIND___final_v1.0_DO_NOT_USE_FOR_2PT/zslim_som/zs_som/redmagic_y1/zllim_y1/lens_w_True/njk_150/thbin_2.50_250_20/bslop_0/source_only_close_to_lens_True_nside4/measurement/gt_boosted_twopointfile.fits'  # noqa

# we use the unboosted data vectors for comparision with Y1 unboosted data
y1lenses_y3sources_noboost = '/home/esheldon/git/xcorr/runs/Y3_mastercat___UNBLIND___final_v1.0_DO_NOT_USE_FOR_2PT/zslim_som/zs_som/redmagic_y1/zllim_y1/lens_w_True/njk_150/thbin_2.50_250_20/bslop_0/source_only_close_to_lens_True_nside4/measurement/gt_twopointfile.fits'  # noqa

# we need to get the n(zl) from here, the new run did not have the correct
# values in it
# also has gammat for y3y3
y3sources = '2pt_NG_final_2ptunblind_11_13_20_wnz.fits'

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
# Y3_CHI_FACTORS = np.array(
#     [761.98657437, 1158.2870951, 1497.06693415, 1795.64094319, 2029.11026506]
# )

# mean/width
Y1_ZS_OFFSETS = np.array([
    (0.1, 1.6),
    (-1.9, 1.3),
    (0.9,  1.1),
    (-1.8, 2.2),
])
Y1_ZS_OFFSETS /= 100


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntrial', type=int, default=1000)
    parser.add_argument('--no-m-priors', action='store_true')
    parser.add_argument('--scatter-y1-m', action='store_true')
    parser.add_argument('--plot-dsig', action='store_true')
    return parser.parse_args()


def read_data():
    """
    read data for both y1 and y3 sources.  A random y3 source n(z)
    is read from disk each time this code is run

    Returns
    --------
    data: dict
        Dictionary of arrays, keyed by
            y1y1: y1 lenses, y1 sources
            y1y3: y1 lenses, y3 sources
        The data for each key is
            'gammat': The gamma_t data vector for all l/s bins
            'nzl': redshift and n(z) for the lenses for all l/s bins
            'nzs': redshift and n(z) for the sources for all l/s bins
    """
    i = np.random.randint(1000)
    sname_y1y3 = 'nz_source_realisation_%d' % i

    data = {
        'y1y1': {
            'gammat': fitsio.read(y1lenses_y1sources, ext='gammat', lower=True),  # noqa
            'nzl': fitsio.read(y1lenses_y1sources, ext='nz_lens', lower=True),  # noqa
            'nzs': fitsio.read(y1lenses_y1sources, ext='nz_source', lower=True),  # noqa
        },
        'y1y3': {
            'gammat':  fitsio.read(y1lenses_y3sources_noboost, ext='gammat', lower=True),  # noqa
            'nzl': fitsio.read(y1lenses_y1sources, ext='nz_lens', lower=True),  # noqa
            'nzs': fitsio.read(y3sources, ext=sname_y1y3, lower=True),  # noqa
        }
    }

    return data


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


def get_nofz(*, data, lbin, sbin):
    """
    get n(z) data for the given lens and source bin.  the Y1
    n(z) are shifted according to the prior each time this
    is called

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
    z1off = np.random.normal(
        loc=moff,
        scale=woff,
    )

    y1_zs = data['y1y1']['nzs']['z_mid'] + z1off

    w, = np.where(y1_zs > 0)
    y1_zs = y1_zs[w]
    y1_nofzs = data['y1y1']['nzs'][sbin_name][w]

    zdata = {
        'y1y1': {
            'lzbin': data['y1y1']['nzl']['z_mid'],
            'lnofz': data['y1y1']['nzl'][lbin_name],
            'szbin': y1_zs,
            'snofz': y1_nofzs,
        },
        'y1y3': {
            'lzbin': data['y1y3']['nzl']['z_mid'],
            'lnofz': data['y1y3']['nzl'][lbin_name],
            'szbin': data['y1y3']['nzs']['z_mid'],
            'snofz': data['y1y3']['nzs'][sbin_name],
        }
    }
    return zdata


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


def plot_bin(*, plt, data, lbin, sbin, args, dolabel=False):
    """

    Calculate the fractional difference Y1/Y3-1 as a function of radius and
    plot.   The m and y1 n(z) are sampled..  The input is expected to have a
    random realization of Y3 source n(z).


    Parameters
    ----------
    plt: the plot object
        Curves will be added to the object
    data: the data dict
        See read_data() for the contents
    lbin/sbin: int
        lens and source bin indexes, 1 offset
    args: parsed args
        See get_args()
    dolabel: bool
        If True, add a label

    Returns
    -------
    frac:  float
        The overall mean fractional difference
    """
    zdata = get_nofz(data=data, lbin=lbin, sbin=sbin)

    siginv_y1y1 = inv_sigma_crit_eff_fast(
        zlbin=zdata['y1y1']['lzbin'],
        nzl=zdata['y1y1']['lnofz'],
        zsbin=zdata['y1y1']['szbin'],
        nzs=zdata['y1y1']['snofz'],
    )
    siginv_y1y3 = inv_sigma_crit_eff_fast(
        zlbin=zdata['y1y3']['lzbin'],
        nzl=zdata['y1y3']['lnofz'],
        zsbin=zdata['y1y3']['szbin'],
        nzs=zdata['y1y3']['snofz'],
    )

    print('siginv_y1y1:', siginv_y1y1)
    print('siginv_y1y3:', siginv_y1y3)

    wy1y1, = np.where(
        (data['y1y1']['gammat']['bin1'] == lbin) &
        (data['y1y1']['gammat']['bin2'] == sbin)
    )
    wy1y3, = np.where(
        (data['y1y3']['gammat']['bin1'] == lbin) &
        (data['y1y3']['gammat']['bin2'] == sbin)
    )

    gt_y1y1 = data['y1y1']['gammat']['value'][wy1y1]
    gt_y1y3 = data['y1y3']['gammat']['value'][wy1y3]

    if args.scatter_y1_m:
        y3_oneplusm = 1 + Y3_MVALS[sbin-1][0]
        y1_oneplusm = y3_oneplusm + np.random.normal(scale=Y3_MVALS[sbin-1][1])
    elif not args.no_m_priors:
        y3_oneplusm = 1 + Y3_MVALS[sbin-1][0]
        y1_oneplusm = (1 + Y1_MVAL)
    else:
        y3_oneplusm = 1 + Y3_MVALS[sbin-1][0]
        y1_oneplusm = y3_oneplusm

    gt_y1y1 /= y1_oneplusm
    gt_y1y3 /= y3_oneplusm

    ds_y1y1 = gt_y1y1/siginv_y1y1
    ds_y1y3 = gt_y1y3/siginv_y1y3

    rad_y1y1 = np.deg2rad(data['y1y1']['gammat']['ang'][wy1y1]/60)
    r_y1y1 = rad_y1y1*Y1_CHI_FACTORS[lbin-1]

    rad_y1y3 = np.deg2rad(data['y1y3']['gammat']['ang'][wy1y3]/60)
    r_y1y3 = rad_y1y3*Y1_CHI_FACTORS[lbin-1]

    # interpolation might not be needed, same binning
    ds_y1y1_interp = interpolate_y1_onto_y3(r_y1y3, r_y1y1, ds_y1y1)
    frac = ds_y1y1_interp/ds_y1y3 - 1

    print('#'*70)
    print(sbin, lbin)

    alpha = 0.01
    ls = 'solid'
    plt.axhline(0, color='gray')

    if args.plot_dsig:
        plt.curve(
            r_y1y3, ds_y1y1_interp*r_y1y3,
            color='blue', linestyle=ls, alpha=alpha,
        )
        plt.curve(
            r_y1y3, ds_y1y3*r_y1y3,
            color='red', linestyle=ls, alpha=alpha,
        )
        if dolabel:
            plt.curve(
                r_y1y3 - 1.e9, ds_y1y1_interp*r_y1y3,
                color='blue', linestyle=ls, label='Y1 sources',
            )
            plt.curve(
                r_y1y3 - 1.e9, ds_y1y3*r_y1y3,
                color='red', linestyle=ls, label='Y3 sources',
            )

            plt.legend()

    else:
        plt.curve(r_y1y3, frac, color='blue', linestyle=ls, alpha=alpha)

    return frac


def main():
    """
    calculate the fractional difference over all radii for realizations
    of the N(z)
    """
    args = get_args()
    xlabel = r'$R [\mathrm{Mpc}]$'
    xlim = 0.4, 200

    if args.plot_dsig:
        ylabel = r'$r \times \Delta\Sigma$'
        ylim = (0, 60)
    else:
        ylabel = r'$\Delta\Sigma^{\mathrm{Y1}}/\Delta\Sigma^{\mathrm{Y3}} - 1$'  # noqa
        ylim = (-1, 3)

    sbins = list(range(1, 4+1))
    lbins = list(range(1, 4+1))

    dlists = {}
    for sbin in sbins:
        for lbin in lbins:
            key = 'S%d-L%d' % (sbin, lbin)
            dlists[key] = []

    for sbin in sbins:

        tab = hickory.Table(
            figsize=(8, 6),
            nrows=2, ncols=2,
        )
        tab.suptitle('Y1 lenses, different sources')
        tab[0, 0].set(
            xlim=xlim,
            ylim=ylim,
            ylabel=ylabel,
        )
        tab[0, 1].set(
            xlim=xlim,
            ylim=ylim,
        )

        tab[1, 0].set(
            xlim=xlim,
            ylim=ylim,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        tab[1, 1].set(
            xlim=xlim,
            ylim=ylim,
            xlabel=xlabel,
        )

        for plt in tab.axes:
            plt.set_xscale('log')

        tab[0, 0].ntext(0.1, 0.9, 'sbin %d, lbin 1' % sbin)
        tab[0, 1].ntext(0.1, 0.9, 'sbin %d, lbin 2' % sbin)
        tab[1, 0].ntext(0.1, 0.9, 'sbin %d, lbin 3' % sbin)
        tab[1, 1].ntext(0.1, 0.9, 'sbin %d, lbin 4' % sbin)

        for trial in range(args.ntrial):
            data = read_data()

            if trial == 0:
                dolabel = True
            else:
                dolabel = False

            res1 = plot_bin(
                args=args,
                plt=tab[0, 0], lbin=1, sbin=sbin,
                data=data,
                dolabel=dolabel,
            )
            res2 = plot_bin(
                args=args,
                plt=tab[0, 1], lbin=2, sbin=sbin,
                data=data,
            )
            res3 = plot_bin(
                args=args,
                plt=tab[1, 0], lbin=3, sbin=sbin,
                data=data,
            )
            res4 = plot_bin(
                args=args,
                plt=tab[1, 1], lbin=4, sbin=sbin,
                data=data,
            )
            dlists['S%d-L%d' % (sbin, 1)] += [res1]
            dlists['S%d-L%d' % (sbin, 2)] += [res2]
            dlists['S%d-L%d' % (sbin, 3)] += [res3]
            dlists['S%d-L%d' % (sbin, 4)] += [res4]

        pltname = (
            'fracdiff-y1-y3-sbin%d-y1lenses-'
            'diffsources-noboost-sample.png' % sbin
        )
        if args.no_m_priors:
            pltname = pltname.replace('.png', '-nompriors.png')
        if args.plot_dsig:
            pltname = pltname.replace('.png', '-dsig.png')

        print('writing:', pltname)
        tab.savefig(pltname, dpi=150)

    means = []
    errs = []
    for sbin in sbins:
        for lbin in lbins:
            key = 'S%d-L%d' % (sbin, lbin)

            arr = eu.numpy_util.combine_arrlist(dlists[key])
            mean = arr.mean()
            err = arr.std()/np.sqrt(arr.size)

            means.append(mean)
            errs.append(err)

    means = np.array(means)
    errs = np.array(errs)

    meanfrac, meanfrac_err = eu.stat.wmom(means, 1.0/errs**2, calcerr=True)

    print('overall mean: %g +/- %g' % (meanfrac, meanfrac_err))


if __name__ == '__main__':
    main()
