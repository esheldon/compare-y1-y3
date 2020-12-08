import fitsio
import hickory
import numpy as np
import esutil as eu

y1lenses_y1sources = '2pt_NG_mcal_1110.fits'

# y1lenses_y3sources = '/home/esheldon/git/xcorr/runs/Y3_mastercat___UNBLIND___final_v1.0_DO_NOT_USE_FOR_2PT/zslim_som/zs_som/redmagic_y1/zllim_y1/lens_w_True/njk_150/thbin_2.50_250_20/bslop_0/source_only_close_to_lens_True_nside4/measurement/gt_boosted_twopointfile.fits'  # noqa
y1lenses_y3sources_noboost = '/home/esheldon/git/xcorr/runs/Y3_mastercat___UNBLIND___final_v1.0_DO_NOT_USE_FOR_2PT/zslim_som/zs_som/redmagic_y1/zllim_y1/lens_w_True/njk_150/thbin_2.50_250_20/bslop_0/source_only_close_to_lens_True_nside4/measurement/gt_twopointfile.fits'  # noqa

# also has gammat for y3y3
y3sources = '2pt_NG_final_2ptunblind_11_13_20_wnz.fits'
# '2pt_NG_final_2ptunblind_11_13_20_wnz.fits'

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


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-m-priors', action='store_true')
    parser.add_argument('--sample', action='store_true', help='sample n(z)')
    return parser.parse_args()


def read_data(args):

    if args.sample:
        i = np.random.randint(1000)
        sname_y1y3 = 'nz_source_realisation_%d' % i
    else:
        sname_y1y3 = 'nz_source'

    data = {
        'y1y1': {
            'gammat': fitsio.read(y1lenses_y1sources, ext='gammat', lower=True),  # noqa
            'nzl': fitsio.read(y1lenses_y1sources, ext='nz_lens', lower=True),  # noqa
            'nzs': fitsio.read(y1lenses_y1sources, ext='nz_source', lower=True),  # noqa
        },
        'y1y3': {
            # 'gammat':  fitsio.read(y1lenses_y3sources, ext='gammat', lower=True),  # noqa
            'gammat':  fitsio.read(y1lenses_y3sources_noboost, ext='gammat', lower=True),  # noqa
            # these are wrong, actually same binning as Y1 was used
            # 'nzl': fitsio.read(y1lenses_y3sources, ext='nz_lens', lower=True),  # noqa
            'nzl': fitsio.read(y1lenses_y1sources, ext='nz_lens', lower=True),  # noqa
            # 'nzs': fitsio.read(y1lenses_y3sources, ext='nz_source', lower=True),  # noqa
            'nzs': fitsio.read(y3sources, ext=sname_y1y3, lower=True),  # noqa
        }
    }

    return data


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


def interpolate_y1_onto_y3(r3, r1, ds1):
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

    lbin_name = 'bin%d' % lbin
    sbin_name = 'bin%d' % sbin

    zdata = {
        'y1y1': {
            'lzbin': data['y1y1']['nzl']['z_mid'],
            'lnofz': data['y1y1']['nzl'][lbin_name],
            'szbin': data['y1y1']['nzs']['z_mid'],
            'snofz': data['y1y1']['nzs'][sbin_name],
        },
        'y1y3': {
            'lzbin': data['y1y3']['nzl']['z_mid'],
            'lnofz': data['y1y3']['nzl'][lbin_name],
            'szbin': data['y1y3']['nzs']['z_mid'],
            'snofz': data['y1y3']['nzs'][sbin_name],
        }
    }
    return zdata


def plot_bin(*, plt, data, lbin, sbin, label, args):

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

    if not args.no_m_priors:
        y1_oneplusm = (1 + Y1_MVAL)
        y3_oneplusm = 1 + Y3_MVALS[sbin-1][0]
        gt_y1y1 /= y1_oneplusm
        gt_y1y3 /= y3_oneplusm

    ds_y1y1 = gt_y1y1/siginv_y1y1
    ds_y1y3 = gt_y1y3/siginv_y1y3
    # ds_y1y1 = data['y1y1']['gammat']['value'][wy1y1]/siginv_y1y1/y1_oneplusm
    # ds_y1y3 = data['y1y3']['gammat']['value'][wy1y3]/siginv_y1y3/y3_oneplusm

    # used y1 lens redshift bins
    rad_y1y1 = np.deg2rad(data['y1y1']['gammat']['ang'][wy1y1]/60)
    r_y1y1 = rad_y1y1*Y1_CHI_FACTORS[lbin-1]

    # y1 lens z binning was used
    rad_y1y3 = np.deg2rad(data['y1y3']['gammat']['ang'][wy1y3]/60)
    r_y1y3 = rad_y1y3*Y1_CHI_FACTORS[lbin-1]
    print('compare radii')
    eu.stat.print_stats(r_y1y1 - r_y1y3)

    # interpolation might not be needed, same binning
    ds_y1y1_interp = interpolate_y1_onto_y3(r_y1y3, r_y1y1, ds_y1y1)
    frac = ds_y1y1_interp/ds_y1y3 - 1
    # frac = ds_y1y1/ds_y1y3 - 1
    # frac = gt_y1y1/gt_y1y3 - 1

    meanfrac = frac.mean()
    meanfrac_err = frac.std()/np.sqrt(frac.size)

    wr, = np.where(r_y1y3 < 10)
    medfrac_lo = np.median(frac[wr])
    meanfrac_lo = frac[wr].mean()
    meanfrac_err_lo = frac[wr].std()/np.sqrt(wr.size)

    wr, = np.where(r_y1y3 > 10)
    # medfrac_hi = np.median(frac[wr])
    medfrac_hi = np.median(frac[wr])
    meanfrac_hi = frac[wr].mean()
    meanfrac_err_hi = frac[wr].std()/np.sqrt(wr.size)

    print('lor median fracdiff:', medfrac_lo)
    print('lor mean fracdiff: %g +/- %g' % (meanfrac_lo, meanfrac_err_lo))
    print('hir median fracdiff:', medfrac_hi)
    print('hir mean fracdiff: %g +/- %g' % (meanfrac_hi, meanfrac_err_hi))

    # ply = np.poly1d(np.polyfit(y3['ang'][w3], frac, 1))
    ply = np.poly1d(np.polyfit(r_y1y3, frac, 1))
    # pred = ply(y3['ang'][w3])
    pred = ply(r_y1y3)

    if sbin == 2 and lbin == 3:
        nsig = 3
    else:
        nsig = 4

    _, scatter = eu.stat.sigma_clip(frac - pred, nsig=nsig)
    print('#'*70)
    print(sbin, lbin, scatter)

    # plt.plot(
    #     y1['ang'][w1],
    #     y1['value'][w1],
    #     label='Y1',
    # )
    # plt.plot(
    #     y3['ang'][w3],
    #     y3['value'][w3],
    #     label='Y3',
    # )
    plt.axhline(0, color='gray')
    plt.errorbar(
        # y3['ang'][w3],
        r_y1y3,
        frac,
        scatter + frac*0,
    )
    plt.curve(
        # y3['ang'][w3],
        r_y1y3,
        pred,
        color='darkgreen',
    )
    plt.ntext(
        0.1, 0.9,
        label
    )
    plt.ntext(
        0.1, 0.8,
        # r'$r < 50$ mean %.2f +/- %.2f' % (meanfrac_lo, meanfrac_err_lo)  # noqa
        r'$r < 10$ mean %.2f +/- %.2f' % (meanfrac_lo, meanfrac_err_lo)  # noqa
    )
    plt.ntext(
        0.1, 0.7,
        # r'$r > 50$ mean %.2f +/- %.2f' % (meanfrac_hi, meanfrac_err_hi)  # noqa
        r'$r > 10$ mean %.2f +/- %.2f' % (meanfrac_hi, meanfrac_err_hi)  # noqa
    )

    return (
        meanfrac_lo, meanfrac_err_lo,
        meanfrac_hi, meanfrac_err_hi,
        meanfrac, meanfrac_err,
    )
    # return medfrac_lo, meanfrac_err_lo, medfrac_hi, meanfrac_err_hi


def main():
    args = get_args()
    xlabel = r'$R [\mathrm{Mpc}]$'
    ylabel = r'$\Delta\Sigma^{\mathrm{Y1}}/\Delta\Sigma^{\mathrm{Y3}} - 1$'  # noqa
    data = read_data(args)
    xlim = 0.4, 200
    ylim = (-1, 3)

    meanfrac = []
    meanfrac_err = []
    meanlo = []
    meanlo_err = []
    meanhi = []
    meanhi_err = []

    for sbin in [1, 2, 3, 4]:

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

        tab[0, 0].set_xscale('log')
        tab[0, 1].set_xscale('log')
        tab[1, 0].set_xscale('log')
        tab[1, 1].set_xscale('log')

        res1 = plot_bin(
            args=args,
            plt=tab[0, 0], lbin=1, sbin=sbin,
            data=data,
            label='sbin %d, lbin 1' % sbin,
        )
        res2 = plot_bin(
            args=args,
            plt=tab[0, 1], lbin=2, sbin=sbin,
            data=data,
            label='sbin %d, lbin 2' % sbin,
        )
        res3 = plot_bin(
            args=args,
            plt=tab[1, 0], lbin=3, sbin=sbin,
            data=data,
            label='sbin %d, lbin 3' % sbin,
        )
        res4 = plot_bin(
            args=args,
            plt=tab[1, 1], lbin=4, sbin=sbin,
            data=data,
            label='sbin %d, lbin 4' % sbin,
        )

        meanlo += [res1[0], res2[0], res3[0], res4[0]]
        meanlo_err += [res1[1], res2[1], res3[1], res4[1]]

        meanhi += [res1[2], res2[2], res3[2], res4[2]]
        meanhi_err += [res1[3], res2[3], res3[3], res4[3]]

        meanfrac += [res1[4], res2[4], res3[4], res4[4]]
        meanfrac_err += [res1[5], res2[5], res3[5], res4[5]]

        pltname = (
            'fracdiff-y1-y3-sbin%d-y1lenses-diffsources-noboost.png' % sbin
        )
        if args.no_m_priors:
            pltname = pltname.replace('.png', '-nompriors.png')

        print('writing:', pltname)
        tab.savefig(pltname, dpi=150)

    meanfrac = np.array(meanfrac)
    meanfrac_err = np.array(meanfrac_err)

    meanlo = np.array(meanlo)
    meanlo_err = np.array(meanlo_err)
    meanhi = np.array(meanhi)
    meanhi_err = np.array(meanhi_err)

    meanlo, meanlo_err = eu.stat.wmom(meanlo, 1.0/meanlo_err**2, calcerr=True)
    meanhi, meanhi_err = eu.stat.wmom(meanhi, 1.0/meanhi_err**2, calcerr=True)
    meanfrac, meanfrac_err = eu.stat.wmom(
        meanfrac, 1.0/meanfrac_err**2, calcerr=True,
    )

    print('overall mean: %g +/- %g' % (meanfrac, meanfrac_err))
    print('overall meanlo: %g +/- %g' % (meanlo, meanlo_err))
    print('overall meanhi: %g +/- %g' % (meanhi, meanhi_err))


if __name__ == '__main__':
    main()
