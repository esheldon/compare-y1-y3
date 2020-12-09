#!/usr/bin/env python

import os
import fitsio
import hickory
import numpy as np
import esutil as eu
from compare_y1_y3.util import (
    Y3_MVALS,
    Y1_MVAL,
    Y1_CHI_FACTORS,
    Y1_ZS_OFFSETS,
    interpolate_y1_onto_y3,
    inv_sigma_crit_eff_fast,
)

y1lenses_y1sources = '2pt_NG_mcal_1110.fits'

y3lenses_y1sources = '/home/esheldon/git/xcorr/runs/y1sources/Y3_mastercat___UNBLIND___final_v1.0_DO_NOT_USE_FOR_2PT/zslim_y1/zs_bpz/redmagic_x40randoms_year1footprint/zllim_y3/lens_w_True/njk_150/thbin_2.50_250_20/bslop_0/source_only_close_to_lens_True_nside4/measurement/gt_boosted_twopointfile.fits'  # noqa

y3lenses_y3sources = '/home/esheldon/git/xcorr/runs/Y3_mastercat___UNBLIND___final_v1.0_DO_NOT_USE_FOR_2PT/zslim_som/zs_som/redmagic_x40randoms/zllim_y3/lens_w_True/njk_150/thbin_2.50_250_20/bslop_0/source_only_close_to_lens_True_nside4/measurement/gt_boosted_twopointfile.fits'  # noqa


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntrial', type=int, default=1000)
    parser.add_argument('--use-y1-m-prior', action='store_true')
    parser.add_argument('--plot-dsig', action='store_true')
    return parser.parse_args()


def read_data():
    """
    read data for both y1 and y3 sources.

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
    sname_y1y3 = 'nz_source_realisation_%d'

    with fitsio.FITS(y3sources, lower=True) as fits:
        nzs_samples = []
        for i in range(1000):
            sname = sname_y1y3 % i
            tmp = fits[sname][:]
            nzs_samples.append(tmp)

    data = {
        'y3y1': {
            'gammat': fitsio.read(y3lenses_y1sources, ext='gammat', lower=True),  # noqa
            'nzl': fitsio.read(y3lenses_y3sources, ext='nz_lens', lower=True),  # noqa
            'nzs': fitsio.read(y1lenses_y1sources, ext='nz_source', lower=True),  # noqa
        },
        'y3y3': {
            # 'gammat':  fitsio.read(y1lenses_y3sources_noboost, ext='gammat', lower=True),  # noqa
            'gammat':  fitsio.read(y3lenses_y3sources, ext='gammat', lower=True),  # noqa
            'nzl': fitsio.read(y3lenses_y3sources, ext='nz_lens', lower=True),  # noqa
            'nzs_samples': nzs_samples,
        }
    }

    return data


def get_nofz(*, data, lbin, sbin):
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

    i = np.random.randint(1000)
    y1y3_nzs = data['y3y3']['nzs_samples'][i]

    lbin_name = 'bin%d' % lbin
    sbin_name = 'bin%d' % sbin

    moff = Y1_ZS_OFFSETS[sbin-1][0]
    woff = Y1_ZS_OFFSETS[sbin-1][1]
    z1off = np.random.normal(
        loc=moff,
        scale=woff,
    )

    y1_zs = data['y3y1']['nzs']['z_mid'] + z1off

    w, = np.where(y1_zs > 0)
    y1_zs = y1_zs[w]
    y1_nofzs = data['y3y1']['nzs'][sbin_name][w]

    zdata = {
        'y3y1': {
            'lzbin': data['y3y1']['nzl']['z_mid'],
            'lnofz': data['y3y1']['nzl'][lbin_name],
            'szbin': y1_zs,
            'snofz': y1_nofzs,
        },
        'y3y3': {
            'lzbin': data['y1y3']['nzl']['z_mid'],
            'lnofz': data['y1y3']['nzl'][lbin_name],
            'szbin': y1y3_nzs['z_mid'],
            'snofz': y1y3_nzs[sbin_name],
            # 'szbin': data['y1y3']['nzs']['z_mid'],
            # 'snofz': data['y1y3']['nzs'][sbin_name],
        }
    }
    return zdata


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

    if args.use_y1_m_prior:
        y3_oneplusm = 1 + Y3_MVALS[sbin-1][0]
        y1_oneplusm = (1 + Y1_MVAL)
    else:
        y3_oneplusm = 1 + Y3_MVALS[sbin-1][0]
        y1_oneplusm = y3_oneplusm + np.random.normal(scale=Y3_MVALS[sbin-1][1])

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
