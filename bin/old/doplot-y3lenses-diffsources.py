import fitsio
import hickory
import numpy as np
import esutil as eu

y1lenses_y3sources = '/home/esheldon/git/xcorr/runs/Y3_mastercat___UNBLIND___final_v1.0_DO_NOT_USE_FOR_2PT/zslim_som/zs_som/redmagic_y1/zllim_y1/lens_w_True/njk_150/thbin_2.50_250_20/bslop_0/source_only_close_to_lens_True_nside4/measurement/gt_boosted_twopointfile.fits'  # noqa

y3lenses_y1sources = '/home/esheldon/git/xcorr/runs/y1sources/Y3_mastercat___UNBLIND___final_v1.0_DO_NOT_USE_FOR_2PT/zslim_y1/zs_bpz/redmagic_x40randoms_year1footprint/zllim_y3/lens_w_True/njk_150/thbin_2.50_250_20/bslop_0/source_only_close_to_lens_True_nside4/measurement/gt_boosted_twopointfile.fits'  # noqa

y3lenses_y3sources = '/home/esheldon/git/xcorr/runs/Y3_mastercat___UNBLIND___final_v1.0_DO_NOT_USE_FOR_2PT/zslim_som/zs_som/redmagic_x40randoms/zllim_y3/lens_w_True/njk_150/thbin_2.50_250_20/bslop_0/source_only_close_to_lens_True_nside4/measurement/gt_boosted_twopointfile.fits'  # noqa


Y3_MVALS = [
    -0.006,
    -0.010,
    -0.026,
    -0.032,
]
Y1_MVAL = 0.012


def read_data():
    y1 = fitsio.read(y3lenses_y1sources, ext=2, lower=True)
    y3 = fitsio.read(y3lenses_y3sources, ext=2, lower=True)

    nzs1 = fitsio.read('2pt_NG_mcal_1110.fits', ext='nz_source', lower=True)
    nzs3 = fitsio.read('2pt_NG_final_2ptunblind_11_13_20_wnz.fits',
                       ext='nz_source', lower=True)
    nzl3 = fitsio.read('2pt_NG_final_2ptunblind_11_13_20_wnz.fits',
                       ext='nz_lens', lower=True)
    return y1, y3, nzs1, nzs3, nzl3


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


def get_nofz(*, nzs1, nzs3, nzl3, lbin, sbin):

    lbin_name = 'bin%d' % lbin
    sbin_name = 'bin%d' % sbin

    l3zbin = nzl3['z_mid']
    l3nofz = nzl3[lbin_name]

    s1zbin = nzs1['z_mid']
    s1nofz = nzs1[sbin_name]
    s3zbin = nzs3['z_mid']
    s3nofz = nzs3[sbin_name]

    return s1zbin, s1nofz, s3zbin, s3nofz, l3zbin, l3nofz


def plot_bin(*, plt, y1, y3, nzs1, nzs3, nzl3, lbin, sbin, label):

    s1zbin, s1nofz, s3zbin, s3nofz, l3zbin, l3nofz = get_nofz(
        nzs1=nzs1, nzs3=nzs3, nzl3=nzl3, lbin=lbin, sbin=sbin,
    )

    y3_oneplusm = 1 + Y3_MVALS[sbin-1]

    siginv1 = inv_sigma_crit_eff_fast(
        zlbin=l3zbin, nzl=l3nofz,
        zsbin=s1zbin, nzs=s1nofz,
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

    ds1 = y1['value'][w1]/siginv1/(1 + Y1_MVAL)
    ds3 = y3['value'][w3]/siginv3/y3_oneplusm
    frac = ds1/ds3 - 1
    # frac = ds3/ds1 - 1

    wr, = np.where(y3['ang'][w3] < 50)
    medfrac = np.median(frac[wr])
    meanfrac_lo = frac[wr].mean()
    meanfrac_err_lo = frac[wr].std()/np.sqrt(wr.size)

    wr, = np.where(y3['ang'][w3] > 50)
    # medfrac_hi = np.median(frac[wr])
    meanfrac_hi = frac[wr].mean()
    meanfrac_err_hi = frac[wr].std()/np.sqrt(wr.size)

    print('median fracdiff:', medfrac)
    print('lor mean fracdiff: %g +/- %g' % (meanfrac_lo, meanfrac_err_lo))
    print('hir mean fracdiff: %g +/- %g' % (meanfrac_hi, meanfrac_err_hi))

    ply = np.poly1d(np.polyfit(y3['ang'][w3], frac, 1))
    pred = ply(y3['ang'][w3])

    _, scatter = eu.stat.sigma_clip(frac - pred)

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
        y3['ang'][w3],
        frac,
        scatter + frac*0,
    )
    plt.curve(y3['ang'][w3], pred, color='black')
    plt.ntext(
        0.1, 0.9,
        label
    )
    plt.ntext(
        0.1, 0.8,
        r'$r < 50$ mean %.2f +/- %.2f' % (meanfrac_lo, meanfrac_err_lo)  # noqa
    )
    plt.ntext(
        0.1, 0.7,
        r'$r > 50$ mean %.2f +/- %.2f' % (meanfrac_hi, meanfrac_err_hi)  # noqa
    )

    return meanfrac_lo, meanfrac_err_lo, meanfrac_hi, meanfrac_err_hi


def main():
    ylab = r'$\Delta\Sigma^{\mathrm{Y1}}/\Delta\Sigma^{\mathrm{Y3}} - 1$'  # noqa
    y1, y3, nzs1, nzs3, nzl3 = read_data()

    meanlo = []
    meanlo_err = []
    meanhi = []
    meanhi_err = []

    for sbin in [1, 2, 3, 4]:

        xlim = 1, 400
        tab = hickory.Table(
            figsize=(8, 6),
            nrows=2, ncols=2,
        )
        tab.suptitle('Y3 lenses')
        tab[0, 0].set(
            xlim=xlim,
            ylim=(-1, 3),
            ylabel=ylab,
        )
        tab[0, 1].set(
            xlim=xlim,
            ylim=(-1, 3),
        )

        tab[1, 0].set(
            xlim=xlim,
            ylim=(-1, 3),
            xlabel='theta (arcmin)',
            ylabel=ylab,
        )
        tab[1, 1].set(
            xlim=xlim,
            ylim=(-1, 3),
            xlabel='theta (arcmin)',
        )

        tab[0, 0].set_xscale('log')
        tab[0, 1].set_xscale('log')
        tab[1, 0].set_xscale('log')
        tab[1, 1].set_xscale('log')

        res1 = plot_bin(
            plt=tab[0, 0], y1=y1, y3=y3, lbin=1, sbin=sbin,
            nzs1=nzs1, nzs3=nzs3, nzl3=nzl3,
            label='sbin %d, lbin 1' % sbin,
        )
        res2 = plot_bin(
            plt=tab[0, 1], y1=y1, y3=y3, lbin=2, sbin=sbin,
            nzs1=nzs1, nzs3=nzs3, nzl3=nzl3,
            label='sbin %d, lbin 2' % sbin,
        )
        res3 = plot_bin(
            plt=tab[1, 0], y1=y1, y3=y3, lbin=3, sbin=sbin,
            nzs1=nzs1, nzs3=nzs3, nzl3=nzl3,
            label='sbin %d, lbin 3' % sbin,
        )
        res4 = plot_bin(
            plt=tab[1, 1], y1=y1, y3=y3, lbin=4, sbin=sbin,
            nzs1=nzs1, nzs3=nzs3, nzl3=nzl3,
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
        pltname = 'fracdiff-y1-y3-sbin%d-y3lenses.png' % sbin
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
