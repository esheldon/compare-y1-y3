import numpy as np
import cosmology
import time
c=cosmology.Cosmo(omega_m=0.3)

def inv_sigma_crit_eff(zlbin, zsbin, nzl, nzs):
    t1 = time.time()
    dzl = zlbin[1]-zlbin[0] 
    dzs = zsbin[1]-zsbin[0]
    norml = np.sum(nzl*dzl)
    norms = np.sum(nzs*dzs)
    nzl = nzl/norml
    nzs = nzs/norms
    isc = 0.
    norm = 0.
    for i in range(len(zlbin)):
        for j in range(len(zsbin)):
            isc += nzl[i]*dzl*nzs[j]*dzs*c.sigmacritinv(zlbin[i], zsbin[j])
    #print '%.2f sec' %(time.time() - t1)
    #print 'I1:', isc
    return isc

def inv_sigma_crit_eff_fast(zlbin, zsbin, nzl, nzs):
    t1 = time.time()
    dzl = zlbin[1]-zlbin[0] 
    dzs = zsbin[1]-zsbin[0]
    norml = np.sum(nzl*dzl)
    norms = np.sum(nzs*dzs)
    nzl = nzl/norml
    nzs = nzs/norms
    isc = 0.
    norm = 0.

    # Define meshgrid for redshifts and for Nzs
    X,Y = np.meshgrid(zlbin, zsbin)
    NZL, NZS = np.meshgrid(nzl, nzs)
    # Construct 2-D integrand
    sci_flat = c.sigmacritinv(X,Y)
    sci_re = np.reshape(sci_flat, (len(zsbin),len(zlbin)), order='C')
    integrand = NZL*NZS*sci_re
    # Do a 1-D integral over every row
    I = np.zeros(len(zsbin))
    for i in range(len(zsbin)):
        I[i] = np.trapz(integrand[i,:], zlbin)

    # Then an integral over the result
    F = np.trapz(I, zsbin)

    #print '%.2f sec' %(time.time() - t1)
    #print 'I2:', F
    return F

def inv_sigma_crit_eff_fast_om(omega_m, zlbin, zsbin, nzl, nzs):
    c=cosmology.Cosmo(omega_m)
    t1 = time.time()
    dzl = zlbin[1]-zlbin[0] 
    dzs = zsbin[1]-zsbin[0]
    norml = np.sum(nzl*dzl)
    norms = np.sum(nzs*dzs)
    nzl = nzl/norml
    nzs = nzs/norms
    isc = 0.
    norm = 0.

    # Define meshgrid for redshifts and for Nzs
    X,Y = np.meshgrid(zlbin, zsbin)
    NZL, NZS = np.meshgrid(nzl, nzs)
    # Construct 2-D integrand
    sci_flat = c.sigmacritinv(X,Y)
    sci_re = np.reshape(sci_flat, (len(zsbin),len(zlbin)), order='C')
    integrand = NZL*NZS*sci_re
    # Do a 1-D integral over every row
    I = np.zeros(len(zsbin))
    for i in range(len(zsbin)):
        I[i] = np.trapz(integrand[i,:], zlbin)

    # Then an integral over the result
    F = np.trapz(I, zsbin)

    #print '%.2f sec' %(time.time() - t1)
    #print 'I2:', F
    return F


#Definition of the shear ratio theory, calls the double integral module
def shear_ratio_theory(z,nzl,nzs1,nzs2):
       isc1 = inv_sigma_crit_eff_fast(z, z, nzl, nzs1)
       isc2 = inv_sigma_crit_eff_fast(z, z, nzl, nzs2)
       return isc1/isc2

if __name__ == '__main__':
	path = '../cats/njk_100/'
	z, nzl = np.loadtxt('%s/nz_lens'%path,unpack=True)
	z, nzs1 = np.loadtxt('%s/nz_source1'%path,unpack=True)
	z, nzs2 = np.loadtxt('%s/nz_source2'%path,unpack=True)

	nzs2_samples = np.load('%s/shearratio_nz_samples.npy'%path).T

	#print 1./shear_ratio_theory(z,nzl,nzs1,nzs2)
	#print 1./shear_ratio_theory(z,nzl,nzs1,nzs2_samples[10])
	ratios = np.zeros(7001)
	#The first element of this array is the truth value -- no bias
	ratios[0] = 1./shear_ratio_theory(z,nzl,nzs1,nzs2)
	for i in range(7000):
		ratios[i+1] = 1./shear_ratio_theory(z,nzl,nzs1,nzs2_samples[i])
	np.savetxt('theory_ratios', ratios)
	#print ratios.mean(), ratios.std()
