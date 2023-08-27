import numpy as np
import json, glob, os, emcee, datetime, configparser
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
import scipy.stats as ss

def NanAverage(data,weights=None, error=False, axis=0):
    '''
    (Weighted) Average that ignore nan values for an arbitrary axis
    '''
    if weights is None:
        return np.nanmean(data, axis=axis)
    else:
        # Mask NaNs
        ma = np.ma.MaskedArray(data, mask=np.isnan(data))
        avg = np.ma.average(ma, weights=weights, axis=axis)
        if error:
            ma = np.ma.MaskedArray(weights, mask=np.isnan(data))
            err = np.ma.sum(ma, axis=axis) ** (-0.5) # This only works if its a 1-D array so it can't work for Source averaging
            return avg, err
        else:
            return avg

def AverageBySource(data,weight=None):
    '''
    (Weighted) Average parameters across the epoch-axis, resulting in a single value for each source.
    '''
    return NanAverage(data,weights=weight, axis=1)

def AverageByEpoch(data,weight=None, error=False):
    '''
    (Weighted) Average parameters across the source-axis, resulting in a single value for each epoch. The default is uniform weighting.
    '''
    return NanAverage(data,weights=weight, error=error, axis=0)

def MedianBySource(data):
    '''
    Median of parameters across the epoch-axis, resulting in a single value for each source.
    '''
    return np.nanmedian(data,axis=1)


def StdevBySource(data):
    '''
    Sample standard deviation across the epoch-axis, resulting in a single value for each source.
    '''
    return np.nanstd(data,ddof=1,axis=1)

def BootstrapPositions(data, n_bootstrap=1000):
    '''
    Boostrap the results to get the best-fit offsets and the errors from the Confidence intervals
    '''

    # Signal-to-noise Ratio
    average_data = AverageBySource(data)
    snr = np.array(average_data[:,4])

    # Get the bootstrapped values + errors

    bootstrap_index  = np.random.randint(0, data.shape[1], size=(n_bootstrap, data.shape[1]))
    bootstrap_data   = np.swapaxes(np.swapaxes(data,0,1)[bootstrap_index,:,:], 1, 2) # We need to swap around the axes for indexing purposes because we want to replace epochs
    bootstrap_data   = np.nanstd(bootstrap_data, ddof=1, axis = 2)          

    # Need to figure out a way to not bootstrap the NaN value because this breaks for small number of epochs

    value, err_minus, err_plus = CI(bootstrap_data, axis = 0)

    # Break up the output
    ra     = value[:,-2]
    ra_err = np.amax([err_minus[:,-2], err_plus[:,-2]], axis=0)
    dec     = value[:,-1]
    dec_err = np.amax([err_minus[:,-1], err_plus[:,-1]], axis=0)

    # Return the values for fitting
    return snr, ra, ra_err, dec, dec_err 


def FilterData(data,snr_low=0.0,snr_high=1e10, phase_offset_thresh=1e10):
    '''
    Filter the data catalog by snr and angular offset from the phase center:
        Defaults (Include all sources):
            snr_low  = 0.0
            snr_high = 1e10
            phase_offset_thresh = 1e10 [deg]
    '''
    average_data = AverageBySource(data)
    snr, offset = average_data[:,4], average_data[:,7]

    snr_index  = np.where((snr > snr_low) & (snr < snr_high))[0]
    po_index = np.where(offset < phase_offset_thresh)[0]

    if len(snr_index) >= len(po_index):
        return data[snr_index[np.isin(snr_index,po_index)],:,:]
    else:
        return data[po_index[np.isin(po_index,snr_index)],:,:]

#Decompose the Offsets from the total seperation -- Credit: Gregory Sivakoff
def DecomposeDeltaCoord(ref_RA, ref_Dec, RAs, Decs):
    '''
    Take RAs and Decs as inputs and calculate offset parameters:
        Total Offset  [deg]
        Postion Angle [deg]
        Offset in RA  [deg]
        Offset in Dec [deg]

    Credit: Gregory R. Sivakoff
    '''
    c1 = np.expand_dims(SkyCoord(ref_RA * u.deg, ref_Dec * u.deg, frame='icrs'), -1)
    c2 = SkyCoord(RAs * u.deg,   Decs * u.deg,   frame='icrs')

    seps = c1.separation(c2)
    PAs  = c1.position_angle(c2)

    offsetsRA  = -seps*np.cos(PAs.rad+np.pi/2)
    offsetsDec =  seps*np.sin(PAs.rad+np.pi/2)

    return(seps.deg, PAs.deg, offsetsRA.deg, offsetsDec.deg)

def AveragePositions(RAs,Decs,avg_type,weight=None):
    '''
    Calculate the average position of a set of RAs and Decs
    Credit: Gregory R. Sivakoff
    '''
    if avg_type == 'source':
        axis = 1
    elif avg_type == 'epoch':
        axis = 0
    else:
        raise TypeError("Invalid Average, enter either: 'source' or 'epoch'")

    #Initial centering
    CentreDec = NanAverage(Decs,axis = axis,weights=weight)
    CentreRA = NanAverage(RAs*np.cos(Decs*np.pi/180.),axis=axis,weights=weight)/np.cos(CentreDec*np.pi/180.)

    #Calculate offsets
    seps, PAs, offsetRA, offsetDec = DecomposeDeltaCoord(CentreRA, CentreDec, RAs, Decs)

    #Find the average corrections in the RA and DEC directions
    avgCorrectionRA  = NanAverage(offsetRA, axis = axis, weights = weight)
    avgCorrectionDec = NanAverage(offsetDec, axis = axis, weights = weight)

    return CentreRA + avgCorrectionRA, CentreDec + avgCorrectionDec

def AppendValues(data,index_dict, append_data, append_indexes):
    '''
    Append a subset of data to the larger data catalog
    '''

    append_data = np.expand_dims(append_data, -1)
    data = np.concatenate((data, append_data), axis=len(data.shape) - 1)
    for k,append_index in enumerate(append_indexes):
        index_dict[append_index] = data.shape[-1] - (len(append_indexes) - k)

    return data, index_dict


def stdfit(snr,A,B):
    '''
    Fit function to quantify the (relative) astrometric precision:
        The parameter A is the signal-to-noise dependency
        The parameter B is the systematic limit (independant of signal-to-noise)
    '''
    return np.sqrt(A**2 / (snr**2) + B**2)

def CI(data, prob = 68.27, axis=0): #Confidence interval solver
    prob = (100. - prob)/2.
    lower = prob
    upper = 100. - prob
    pctL=np.percentile(data, lower, axis=axis)
    pctU=np.percentile(data, upper, axis=axis)
    value = np.nanmedian(data, axis=axis)
    err_minus = value - pctL
    err_plus =  pctU - value

    #Return an array consisting of the value, err_minus, err_plus
    return [value, err_minus,err_plus]

def plot_chains(sampler):
    '''
    Function to plot the sampler chain evolution and posterior distribution
    '''

    figa = plt.figure(figsize=(30,5))
    for k,j in enumerate(range(sampler.flatchain.shape[1])):
        plt.subplot(1,sampler.flatchain.shape[1],k+1)
        plt.xlabel('Parameter {}'.format(k + 1), fontsize=20)
        plt.ylabel('Counts', fontsize=20)
        patches = plt.hist(sampler.flatchain[:,j],bins=100)
    figa.subplots_adjust(hspace=.5)
    plt.savefig('../plots/MCMC_Distribution.png')
    plt.clf()
    plt.close()

    figb = plt.figure(figsize=(30,5))
    for k,j in enumerate(range(sampler.flatchain.shape[1])):
        plt.subplot(1,sampler.flatchain.shape[1],k+1)
        plt.plot(sampler.chain[:,:,j].T)
        plt.ylabel('Parameter {}'.format(k + 1), fontsize=20)
        plt.xlabel('Step #', fontsize=20)
    figb.subplots_adjust(hspace=.5)
    plt.savefig('../plots/MCMC_chains.png')
    plt.clf()
    plt.close()

def plot_autocorr_time(sampler):
    '''
    Routine to plot the autocorrelation time following documentation
    on the emceee website
    '''

    def next_pow_two(n):
        i = 1
        while i < n:
            i = i << 1
        return i

    # Plot autocorrelation length -- Taken from emcee documaentation 
    def autocorr_func_1d(x, norm=True):
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError("invalid dimensions for 1D autocorrelation function")
        n = next_pow_two(len(x))

        # Compute the FFT and then (from that) the auto-correlation function
        f = np.fft.fft(x - np.mean(x), n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
        acf /= 4 * n

        # Optionally normalize
        if norm:
            acf /= acf[0]

        return acf

    # Automated windowing procedure following Sokal (1989)
    def auto_window(taus, c):
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    # Following the suggestion from Goodman & Weare (2010)
    def autocorr_gw2010(y, c=5.0):
        f = autocorr_func_1d(np.mean(y, axis=0))
        taus = 2.0 * np.cumsum(f) - 1.0
        window = auto_window(taus, c)
        return taus[window]
        
    # Autocorrelation calculation
    def autocorr_new(y, c=5.0):
        f = np.zeros(y.shape[1])
        for yy in y:
            f += autocorr_func_1d(yy)
        f /= len(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = auto_window(taus, c)
        return taus[window]
  
    for k in range(sampler.flatchain.shape[1]):
        chain = sampler.get_chain()[:, :, k].T

        # Compute the estimators for a few different chain lengths
        N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
        gw2010 = np.empty(len(N))
        new = np.empty(len(N))
        for i, n in enumerate(N):
            gw2010[i] = autocorr_gw2010(chain[:, :n])
            new[i] = autocorr_new(chain[:, :n])
        
        # Plot the comparisons
        plt.loglog(N, gw2010, "o-", label="G&W 2010")
        plt.loglog(N, new, "o-", label="new")
        ylim = plt.gca().get_ylim()
        plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
        plt.ylim(ylim)
        plt.xlabel("number of samples, $N$")
        plt.ylabel(r"$\tau$ estimates")
        plt.legend(fontsize=14)
        plt.savefig('../plots/parameter{}_autocorr.png'.format(k))
        plt.clf()
        plt.close()


def MCMC(snr, std, std_err, prefix, iteration, cfg):
    '''
    Simple Markov-Chain Monte Carlo routine to solve for the best fit parameters:
        To upweight high signal-to-noise detections we are fitting in log-space
    '''

    #std = np.log(std)
    #std_err = abs(std_err/std)

    def log_prior(p): #log prior
        A,B=p[0],p[1]
        prior = 0.0
        prior += ss.uniform.logpdf(A,loc=0.0,scale=1.0)
        prior += ss.uniform.logpdf(B,loc=0.0,scale=1.0)
        if np.isnan(prior):
            return(-np.inf)
        return prior

    def lp(p,snr,std, std_err): #log likelihood
        A,B = p[0], p[1]
        model = stdfit(snr,A,B)
        #model = np.log(stdfit(snr,A,B))
        logprob = -np.nansum((model - std) ** 2 / (2 * std_err ** 2) + np.log(np.sqrt(2 * np.pi * std_err ** 2)))
        #logprob = -np.nansum((model - std) ** 2 / (2 * std_err ** 2) + np.log(np.sqrt(2 * np.pi * std_err ** 2 * std ** 2)))
        return logprob + log_prior(p)

    #Define the initial Parameters
    pGuess = np.array([0.0,0.0])
    nDim = len(pGuess)
    nWalkers = 10 * nDim
    nBurn=1000
    nSamp=5000

    pRand = np.array([1.0 * np.random.rand(nWalkers),
                      1.0 * np.random.rand(nWalkers)])

    p0 = np.zeros((nWalkers,nDim))
    for k in range(nDim):
        p0[:,k] = pRand[k] + pGuess[k]

    #Run the burn-in iterations
    sampler = emcee.EnsembleSampler(nWalkers, nDim,lp,args=[snr,std,std_err])
    result = sampler.run_mcmc(p0,nBurn)
    pos,prob,state=result.coords,result.log_prob,result.random_state

    #Run the Sampling iterations
    sampler.reset()
    result = sampler.run_mcmc(pos,nSamp)
    pos,prob,state=result.coords,result.log_prob,result.random_state

    plot_autocorr_time(sampler)
    plot_chains(sampler)

    #Extract fit parameters
    A = CI(sampler.flatchain[:,0])
    B = CI(sampler.flatchain[:,1])
    results_dict = {'A':A, 'B':B}

    with open('../results/{}-fit-{}_{}.json'.format(prefix,iteration,cfg['SOURCE']['name']),'w') as jfile:
        json.dump(results_dict,jfile)

    return results_dict

def PlotResults(snr, pos, pos_err, global_snr, global_pos, global_pos_err, dType, fit, name, cfg):
    '''
    Plotting script to show the astrometric fits
    '''

    # Initialize plotting parameters
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_facecolor('white')

    # Initialize a dummy x-axis to plot the model value
    xMod = np.linspace(snr.min(),snr.max(),10000)
    ubound   = stdfit(xMod,fit['A'][0] - fit['A'][1],fit['B'][0] - fit['B'][1])
    lbound   = stdfit(xMod,fit['A'][0] + fit['A'][2],fit['B'][0] + fit['B'][2])
    best_fit = stdfit(xMod,fit['A'][0],fit['B'][0])

    # Calculate the chi-square for both populations
    chi_data   = np.sum((stdfit(snr,fit['A'][0],fit['B'][0]) - pos) ** 2 / pos_err ** 2)
    dof_data   = len(snr) - 2
    chi_global = np.sum((stdfit(global_snr,fit['A'][0],fit['B'][0]) - global_pos) ** 2 / global_pos_err ** 2)
    dof_global = len(global_snr) - 2 

    # Get the fits
    A_fit = fit['A'][0]
    A_err = np.amax((fit['A'][1],fit['A'][2]))

    B_fit = fit['B'][0]
    B_err = np.amax((fit['B'][1],fit['B'][2]))

    ax.errorbar(snr, pos, yerr = pos_err, mec='k',fmt='o',mfc='C0', ecolor='k', label='$\chi^2=${:.1f}/{}'.format(chi_data,dof_data), zorder=100)
    ax.errorbar(global_snr, global_pos, yerr = global_pos_err, mec='k',fmt='o',mfc='w', ecolor='k', label='$\chi^2=${:.1f}/{}'.format(chi_global,dof_global), zorder=0, alpha=0.25)
    ax.plot(xMod,best_fit,ls='--', lw=5, color='red',zorder=1000, label=r'Astrometry fit')
    ax.fill_between(xMod, lbound, ubound, color='k', alpha=0.25, zorder = 1000)
    ax.axhline(0.1, color='k', ls=':', zorder=1000, label='10% of the synthesized beam')

    # Plot the data + MCMC model
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('{} - Astrometric Error (beams)'.format(dType),fontfamily='serif',fontsize=15)
    ax.set_xlabel('Source-Averaged Signal-to-Noise Ratio',fontfamily='serif',fontsize=15)
    ax.legend(framealpha=1.0, fontsize=15, prop={'family':'serif'}).set_zorder(10000000)
    ax.set_ylim(1e-3,0.3)

    # Make the plot look pretty
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='in',length=5,top=True,right=True)
    ax.tick_params(axis='both', which='major', direction='in',length=10,top=True,right=True, labelsize=12)

    # Save the plot
    plt.savefig('../plots/{}_{}_{}.png'.format(dType,name,cfg['SOURCE']['name']))
    plt.clf()
    plt.close()

def PlotChi(snr, pos, pos_err, global_snr, global_pos, global_pos_err, dType, fit, name, cfg):
    '''
    Plotting script for chi vs. snr 
    '''

    # Initialize plotting parameters
    fig, ax = plt.subplots(figsize=(10,8))
    fig.patch.set_facecolor('white')

    # Initialize a dummy x-axis to plot the model value
    xMod = np.linspace(snr.min(),snr.max(),10000)
    ubound   = stdfit(xMod,fit['A'][0] - fit['A'][1],fit['B'][0] - fit['B'][1])
    lbound   = stdfit(xMod,fit['A'][0] + fit['A'][2],fit['B'][0] + fit['B'][2])
    best_fit = stdfit(xMod,fit['A'][0],fit['B'][0])

    # Calculate the chi-square for both populations
    chi_data   = (stdfit(snr,fit['A'][0],fit['B'][0]) - pos) / pos_err
    chi_global = (stdfit(global_snr,fit['A'][0],fit['B'][0]) - global_pos) / global_pos_err

    ax.errorbar(snr, chi_data, yerr = 0.0, mec='k',fmt='o',mfc='C0', ecolor='k', zorder=100)
    ax.errorbar(global_snr, chi_global, yerr = 0.0, mec='k',fmt='o',mfc='w', ecolor='k', zorder=0, alpha=0.25)
    ax.axhline(0.0, color='k', ls=':', zorder=1000)

    # Plot the data + MCMC model
    ax.set_xscale('log')
    ax.set_ylabel('{} -- $\chi$'.format(dType),fontfamily='serif',fontsize=12)
    ax.set_xlabel('Source-Averaged Signal-to-Noise Ratio', fontfamily='serif', fontsize=12)

    # Make the plot look pretty
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='in',length=5,top=True,right=True)
    ax.tick_params(axis='both', which='major', direction='in',length=10,top=True,right=True)

    # Save the plot
    plt.savefig('../plots/{}_{}_chi_{}.png'.format(dType,name,cfg['SOURCE']['name']))
    plt.clf()
    plt.close()
