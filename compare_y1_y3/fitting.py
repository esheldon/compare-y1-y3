import numpy as np
import dsfit


def fit_nfw_lin(*, rng, data, sbin, lbin, plt, cosmo):
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
        cosmo=cosmo,
    )

    b_bounds = [0.1, 3.0]
    c_bounds = [0.1, 20.0]

    r200_guess = 0.3*rng.uniform(low=0.9, high=1.1)
    c_guess = rng.uniform(low=c_bounds[0], high=c_bounds[1])
    b_guess = rng.uniform(low=b_bounds[0], high=b_bounds[1])
    guess = np.array([r200_guess, c_guess, b_guess])

    res = fitter.fit(
        dsig=dsig,
        dsigcov=dsigcov,
        guess=guess,
        c_bounds=c_bounds,
        b_bounds=b_bounds,
    )

    print('-'*70)
    print('sbin %d, lbin %d' % (sbin, lbin))
    print('r200: %(r200)g +/- %(r200_err)g' % res)
    print('c: %(c)g +/- %(c_err)g' % res)
    print('b: %(b)g +/- %(b_err)g' % res)
    print('m200: %(m200)g +/- %(m200_err)g' % res)

    dsfit.fit.plot(
        r=r,
        z=data['zl_mean'][lbin-1],
        r200=res['r200'],
        c=res['c'],
        b=res['b'],
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
    plt.ntext(x, 0.15, r'b: %(b).3g $\pm$ %(b_err).3g' % res)
    plt.ntext(x, 0.10, r'c: %(c).3g $\pm$ %(c_err).3g' % res)

    return res
