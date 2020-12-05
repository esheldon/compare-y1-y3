import fitsio
import hickory
import numpy as np
import esutil as eu

y1lenses_y3sources = '/home/esheldon/git/xcorr/runs/Y3_mastercat___UNBLIND___final_v1.0_DO_NOT_USE_FOR_2PT/zslim_som/zs_som/redmagic_y1/zllim_y1/lens_w_True/njk_150/thbin_2.50_250_20/bslop_0/source_only_close_to_lens_True_nside4/measurement/gt_boosted_twopointfile.fits'  # noqa

y3lenses_y3sources = '/home/esheldon/git/xcorr/runs/Y3_mastercat___UNBLIND___final_v1.0_DO_NOT_USE_FOR_2PT/zslim_som/zs_som/redmagic_x40randoms/zllim_y3/lens_w_True/njk_150/thbin_2.50_250_20/bslop_0/source_only_close_to_lens_True_nside4/measurement/gt_boosted_twopointfile.fits'  # noqa

OMEGA_M = 0.3

Y3_MVALS = [
    -0.006,
    -0.010,
    -0.026,
    -0.032,
]
Y1_MVAL = 0.012

Y1_CHI_FACTORS = np.array([
    675.60175456, 1039.66805889, 1374.68291108, 1699.66063854, 1987.87549653]
)
Y3_CHI_FACTORS = np.array(
    [761.98657437, 1158.2870951, 1497.06693415, 1795.64094319, 2029.11026506]
)


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


def read_data():
    y1 = fitsio.read(y1lenses_y3sources, ext=2, lower=True)
    y3 = fitsio.read(y3lenses_y3sources, ext=2, lower=True)

    nzl1 = fitsio.read('2pt_NG_mcal_1110.fits', ext='nz_lens', lower=True)

    nzs3 = fitsio.read('2pt_NG_final_2ptunblind_11_13_20_wnz.fits',
                       ext='nz_source', lower=True)

    nzl3 = fitsio.read('2pt_NG_final_2ptunblind_11_13_20_wnz.fits',
                       ext='nz_lens', lower=True)

    return y1, y3, nzs3, nzl1, nzl3


def inv_sigma_crit_eff_fast(*, zlbin, nzl, zsbin, nzs):
    """
    from Carles and Judit
    """
    c = eu.cosmology.Cosmo(omega_m=OMEGA_M)

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


def get_nofz(*, nzs3, nzl1, nzl3, lbin, sbin):

    lbin_name = 'bin%d' % lbin
    sbin_name = 'bin%d' % sbin

    l1zbin = nzl1['z_mid']
    l1nofz = nzl1[lbin_name]
    l3zbin = nzl3['z_mid']
    l3nofz = nzl3[lbin_name]

    s3zbin = nzs3['z_mid']
    s3nofz = nzs3[sbin_name]

    return s3zbin, s3nofz, l1zbin, l1nofz, l3zbin, l3nofz


def plot_bin(*, plt, y1, y3, nzs3, nzl1, nzl3, lbin, sbin, label):

    s3zbin, s3nofz, l1zbin, l1nofz, l3zbin, l3nofz = get_nofz(
        nzs3=nzs3, nzl1=nzl1, nzl3=nzl3, lbin=lbin, sbin=sbin,
    )

    y3_oneplusm = 1 + Y3_MVALS[sbin-1]

    siginv1 = inv_sigma_crit_eff_fast(
        zlbin=l1zbin, nzl=l1nofz,
        zsbin=s3zbin, nzs=s3nofz,
    )
    siginv3 = inv_sigma_crit_eff_fast(
        zlbin=l3zbin, nzl=l3nofz,
        zsbin=s3zbin, nzs=s3nofz,
    )
    print('siginv1:', siginv1)
    print('siginv3:', siginv3)

    plt.axhline(0, color='black')
    w1, = np.where(
        (y1['bin1'] == lbin) &
        (y1['bin2'] == sbin)
    )
    w3, = np.where(
        (y3['bin1'] == lbin) &
        (y3['bin2'] == sbin)
    )

    ds1 = y1['value'][w1]/siginv1/y3_oneplusm
    ds3 = y3['value'][w3]/siginv3/y3_oneplusm

    # did y1 lenses get put into y3 z bins?  If so, these should
    # actually be the same
    r1 = np.deg2rad(y1['ang'][w1]/60)*Y1_CHI_FACTORS[lbin-1]
    r3 = np.deg2rad(y3['ang'][w3]/60)*Y3_CHI_FACTORS[lbin-1]
    ds1_interp = interpolate_y1_onto_y3(r3, r1, ds1)
    # stop

    frac = ds1_interp/ds3 - 1

    # wr, = np.where(y3['ang'][w3] < 50)
    wr, = np.where(r3 < 10)
    medfrac = np.median(frac[wr])
    meanfrac_lo = frac[wr].mean()
    meanfrac_err_lo = frac[wr].std()/np.sqrt(wr.size)

    # wr, = np.where(y3['ang'][w3] > 50)
    wr, = np.where(r3 > 10)
    # medfrac_hi = np.median(frac[wr])
    meanfrac_hi = frac[wr].mean()
    meanfrac_err_hi = frac[wr].std()/np.sqrt(wr.size)

    print('median fracdiff:', medfrac)
    print('lor mean fracdiff: %g +/- %g' % (meanfrac_lo, meanfrac_err_lo))
    print('hir mean fracdiff: %g +/- %g' % (meanfrac_hi, meanfrac_err_hi))

    # ply = np.poly1d(np.polyfit(y3['ang'][w3], frac, 1))
    ply = np.poly1d(np.polyfit(r3, frac, 1))
    # pred = ply(y3['ang'][w3])
    pred = ply(r3)

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
    plt.errorbar(
        # y3['ang'][w3],
        r3,
        frac,
        scatter + frac*0,
    )
    plt.curve(
        # y3['ang'][w3],
        r3,
        pred,
        color='black',
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

    return meanfrac_lo, meanfrac_err_lo, meanfrac_hi, meanfrac_err_hi


def main():
    # xlabel='theta (arcmin)'
    xlabel = r'$R [\mathrm{Mpc}]$'
    ylab = r'$\Delta\Sigma^{\mathrm{Y1}}/\Delta\Sigma^{\mathrm{Y3}} - 1$'  # noqa
    y1, y3, nzs3, nzl1, nzl3 = read_data()

    xlim = 0.4, 200
    ylim = (-1, 3)

    meanlo = []
    meanlo_err = []
    meanhi = []
    meanhi_err = []

    for sbin in [1, 2, 3, 4]:

        # xlim = 1, 400
        tab = hickory.Table(
            figsize=(8, 6),
            nrows=2, ncols=2,
        )
        tab.suptitle('Y3 Sources, Different Lenses')
        tab[0, 0].set(
            xlim=xlim,
            ylim=ylim,
            ylabel=ylab,
        )
        tab[0, 1].set(
            xlim=xlim,
            ylim=ylim,
        )

        tab[1, 0].set(
            xlim=xlim,
            ylim=ylim,
            xlabel=xlabel,
            ylabel=ylab,
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
            plt=tab[0, 0], y1=y1, y3=y3, lbin=1, sbin=sbin,
            nzs3=nzs3, nzl1=nzl1, nzl3=nzl3,
            label='sbin %d, lbin 1' % sbin,
        )
        res2 = plot_bin(
            plt=tab[0, 1], y1=y1, y3=y3, lbin=2, sbin=sbin,
            nzs3=nzs3, nzl1=nzl1, nzl3=nzl3,
            label='sbin %d, lbin 2' % sbin,
        )
        res3 = plot_bin(
            plt=tab[1, 0], y1=y1, y3=y3, lbin=3, sbin=sbin,
            nzs3=nzs3, nzl1=nzl1, nzl3=nzl3,
            label='sbin %d, lbin 3' % sbin,
        )
        res4 = plot_bin(
            plt=tab[1, 1], y1=y1, y3=y3, lbin=4, sbin=sbin,
            nzs3=nzs3, nzl1=nzl1, nzl3=nzl3,
            label='sbin %d, lbin 4' % sbin,
        )

        meanlo += [res1[0], res2[0], res3[0], res4[0]]
        meanlo_err += [res1[1], res2[1], res3[1], res4[1]]

        meanhi += [res1[2], res2[2], res3[2], res4[2]]
        meanhi_err += [res1[3], res2[3], res3[3], res4[3]]

        # plt.set_yscale('log')
        # tab[0, 0].legend()
        # tab[0, 1].legend()
        # tab[0, 0].legend()
        # tab[1, 0].legend()
        # tab[1, 1].legend()
        pltname = 'fracdiff-y1-y3-sbin%d-y3sources.png' % sbin
        print('writing:', pltname)
        tab.savefig(pltname, dpi=150)
    # tab.show()

    meanlo = np.array(meanlo)
    meanlo_err = np.array(meanlo_err)
    meanhi = np.array(meanhi)
    meanhi_err = np.array(meanhi_err)

    meanlo, meanlo_err = eu.stat.wmom(meanlo, 1.0/meanlo_err**2, calcerr=True)
    meanhi, meanhi_err = eu.stat.wmom(meanhi, 1.0/meanhi_err**2, calcerr=True)

    print('overall meanlo: %g +/- %g' % (meanlo, meanlo_err))
    print('overall meanhi: %g +/- %g' % (meanhi, meanhi_err))


if __name__ == '__main__':
    main()
