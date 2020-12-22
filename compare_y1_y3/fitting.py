import numpy as np
import dsfit


def fit_nfw_lin(*, rng, data, sbin, lbin, plt, cosmo):
    """
    cc is a colossus.cosmology.cosmology.setCosmology('planck18')
    """

    xlim = (0.25, 185)
    ylim = (0.025, 45)
    resid_axis_kw = {
        'xlim': xlim,
        'ylim': (-5.3, 5.3),
        # 'ylim': (-1, 1),
    }

    if lbin == 1:
        no_resid_yticklabels = False
        resid_axis_kw['ylabel'] = r'$\Delta$'
    else:
        no_resid_yticklabels = True

    if sbin == 4:
        no_resid_xticklabels = False
    else:
        no_resid_xticklabels = True

    if sbin == 4:
        resid_axis_kw['xlabel'] = r"$r$ [Mpc]"

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

    dsfit.fit.plot_residuals(
        r=r,
        dsig=dsig,
        dsigcov=dsigcov,
        z=data['zl_mean'][lbin-1],
        lm200=res['lm200'],
        b=res['b'],
        xlim=xlim, ylim=ylim, resid_axis_kw=resid_axis_kw,
        no_resid_xticklabels=no_resid_xticklabels,
        no_resid_yticklabels=no_resid_yticklabels,
        plt=plt,
    )
    x = 0.075
    plt.ntext(x, 0.25, 'sbin: %d, lbin: %d' % (sbin, lbin))
    plt.ntext(
        x, 0.20, r'log$_{10}$(m200): %(lm200).3g $\pm$ %(lm200_err).3g' % res
    )
    plt.ntext(x, 0.15, r'b: %(b).3g $\pm$ %(b_err).3g' % res)

    return res


def make_table():
    import hickory
    tab = hickory.Table(
        nrows=4,
        ncols=4,
        constrained_layout=False,
        sharex=True, sharey=True,
        figsize=(16, 14),
        gridspec_kw={'wspace': 0.0, 'hspace': 0.0},
    )
    # xlabel=r"$r$ [Mpc]"
    ylabel = r"$\Delta\Sigma ~ [\mathrm{M}_{\odot} \mathrm{pc}^{-2}]$"
    for i, j in ((0, 0), (1, 0), (2, 0), (3, 0)):
        tab[i, j].set(ylabel=ylabel)
    # for i, j in ((3, 0), (3, 1), (3, 2), (3, 3)):
    #     tab[i, j].set(ylabel=ylabel)

    for ax in tab.axes:
        # ax.set(
        #     xlabel=r"$r$ [Mpc]",
        #     ylabel=r"$\Delta\Sigma ~ [\mathrm{M}_{\odot} \mathrm{pc}^{-2}]$",
        #     xlim=(0.25, 185),
        #     ylim=(0.025, 45),
        # )
        ax.set_xscale('log')
        ax.set_yscale('log')

    return tab
