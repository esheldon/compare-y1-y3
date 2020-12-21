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

    lm200_guess = rng.uniform(low=11.5, high=14.0)
    b_guess = rng.uniform(low=0.2, high=3.0)
    guess = np.array([lm200_guess, b_guess])

    res = fitter.fit(
        dsig=dsig,
        dsigcov=dsigcov,
        guess=guess,
    )

    print('-'*70)
    print('sbin %d, lbin %d' % (sbin, lbin))
    print('b: %(b)g +/- %(b_err)g' % res)
    print('lm200: %(lm200)g +/- %(lm200_err)g' % res)

    dsfit.fit.plot(
        r=r,
        z=data['zl_mean'][lbin-1],
        lm200=res['lm200'],
        b=res['b'],
        dsig=dsig,
        dsigcov=dsigcov,
        plt=plt,
    )
    x = 0.075
    plt.ntext(x, 0.25, 'sbin: %d, lbin: %d' % (sbin, lbin))
    plt.ntext(
        x, 0.20, r'log$_{10}$(m200): %(lm200).3g $\pm$ %(lm200_err).3g' % res
    )
    plt.ntext(x, 0.15, r'b: %(b).3g $\pm$ %(b_err).3g' % res)

    return res
