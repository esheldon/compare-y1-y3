import numpy as np
import dsfit
from .util import get_cosmo_pars_from_cc


def fit_nfw_lin(*, rng, data, sbin, lbin, plt, cc):
    """
    cc is a colossus.cosmology.cosmology.setCosmology('planck18')
    """

    cosmo_pars = get_cosmo_pars_from_cc(cc)

    w, = np.where(
        (data['gammat']['bin1'] == lbin) &
        (data['gammat']['bin2'] == sbin)
    )
    imin, imax = w[0], w[-1]+1

    r = data['r'][w]
    dsig = data['ds'][w]
    dsigcov = data['dscov'][imin:imax, imin:imax]

    w, = np.where(r < 100)
    r = r[w]
    dsig = dsig[w]
    dsigcov = dsigcov[
        w[0]: w[-1]+1,
        w[0]: w[-1]+1,
    ]

    fitter = dsfit.NFWBiasFitter(
        z=data['zl_mean'][lbin-1],
        r=r,
        cosmo_pars=cosmo_pars,
    )

    B_bounds = [0.1, 2.0]
    c_bounds = [0.1, 5.0]

    r200_guess = 0.3*rng.uniform(low=0.9, high=1.1)
    c_guess = rng.uniform(low=c_bounds[0], high=c_bounds[1])
    B_guess = rng.uniform(low=B_bounds[0], high=B_bounds[1])
    guess = np.array([r200_guess, c_guess, B_guess])

    res = fitter.fit(
        dsig=dsig,
        dsigcov=dsigcov,
        guess=guess,
        c_bounds=c_bounds,
        B_bounds=B_bounds,
    )

    zlmean = data['zl_mean'][lbin-1]

    Dgrowth = cc.growthFactorUnnormalized(zlmean)
    Dgrowth0 = cc.growthFactorUnnormalized(0.0)
    growth_factor = Dgrowth/Dgrowth0

    # bfactor = np.sqrt(
    #     cosmo_pars['omega_m'] * cosmo_pars['sigma_8']**2 * growth_factor**2
    # )
    bfactor = (
        cc.Om(zlmean) * cc.sigma(8.0, zlmean)**2 * growth_factor**2
    )
    res['b'] = res['B'] / bfactor
    res['b_err'] = res['B_err'] / bfactor

    res['growth_factor'] = growth_factor
    res['bfactor'] = bfactor

    print('-'*70)
    print('sbin %d, lbin %d' % (sbin, lbin))
    print('r200: %(r200)g +/- %(r200_err)g' % res)
    print('c: %(c)g +/- %(c_err)g' % res)
    print('B: %(B)g +/- %(B_err)g' % res)
    print('growth factor:', growth_factor)
    print('b: %(b)g +/- %(b_err)g' % res)
    print('m200: %(m200)g +/- %(m200_err)g' % res)

    dsfit.fit.plot(
        r=r,
        z=data['zl_mean'][lbin-1],
        r200=res['r200'],
        c=res['c'],
        B=res['B'],
        dsig=dsig,
        dsigcov=dsigcov,
        # xlim=(0.25, 65),
        # xlim=(0.25, 185),
        # ylim=(0.025, 45),
        plt=plt,
    )
    x = 0.075
    plt.ntext(x, 0.25, 'sbin: %d, lbin: %d' % (sbin, lbin))
    plt.ntext(x, 0.20, r'm200: %(m200).3g $\pm$ %(m200_err).3g' % res)
    plt.ntext(x, 0.15, r'B: %(B).3g $\pm$ %(B_err).3g' % res)
    plt.ntext(x, 0.10, r'c: %(c).3g $\pm$ %(c_err).3g' % res)

    return res


def fit_nfw_lin_old(*, rng, data, sbin, lbin, plt, cosmo_pars):
    """
    cc is a colossus.cosmology.cosmology.setCosmology('planck18')
    """

    w, = np.where(
        (data['gammat']['bin1'] == lbin) &
        (data['gammat']['bin2'] == sbin)
    )
    imin, imax = w[0], w[-1]+1

    r = data['r'][w]
    dsig = data['ds'][w]
    dsigcov = data['dscov'][imin:imax, imin:imax]

    fitter = dsfit.NFWBiasFitter(
        z=data['zl_mean'],
        r=r,
        cosmo_pars=cosmo_pars,
    )

    B_bounds = [0.1, 2.0]
    c_bounds = [0.1, 5.0]

    r200_guess = 0.3*rng.uniform(low=0.9, high=1.1)
    c_guess = rng.uniform(low=c_bounds[0], high=c_bounds[1])
    B_guess = rng.uniform(low=B_bounds[0], high=B_bounds[1])
    guess = np.array([r200_guess, c_guess, B_guess])

    res = fitter.fit(
        dsig=dsig,
        dsigcov=dsigcov,
        guess=guess,
        c_bounds=c_bounds,
        B_bounds=B_bounds,
    )

    print('-'*70)
    print('sbin %d, lbin %d' % (sbin, lbin))
    print('r200: %(r200)g +/- %(r200_err)g' % res)
    print('c: %(c)g +/- %(c_err)g' % res)
    print('B: %(B)g +/- %(B_err)g' % res)
    print('m200: %(m200)g +/- %(m200_err)g' % res)

    dsfit.fit.plot(
        r=r,
        z=data['zl_mean'],
        r200=res['r200'],
        c=res['c'],
        B=res['B'],
        dsig=dsig,
        dsigcov=dsigcov,
        # xlim=(0.25, 65),
        # xlim=(0.25, 185),
        # ylim=(0.025, 45),
        plt=plt,
    )
    x = 0.075
    plt.ntext(x, 0.25, 'sbin: %d, lbin: %d' % (sbin, lbin))
    plt.ntext(x, 0.20, r'm200: %(m200).3g $\pm$ %(m200_err).3g' % res)
    plt.ntext(x, 0.15, r'B: %(B).3g $\pm$ %(B_err).3g' % res)
    plt.ntext(x, 0.10, r'c: %(c).3g $\pm$ %(c_err).3g' % res)

    return res
