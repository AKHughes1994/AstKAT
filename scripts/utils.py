import json
import glob 
import emcee
import configparser
import time
import subprocess
import bdsf
import os

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import scipy.stats as ss

from astropy.io import fits
from astropy.coordinates import SkyCoord,match_coordinates_sky
from astropy.time import Time

def chi2(data, fit):
    '''
    Simple Chi-square solver to test fit quality
    '''

    # Separate the data
    snr        = data[0]
    std        = data[1]
    std_err = data[2]

    # Decompose fit paramaters 
    A = fit['A'][0]
    B = fit['B'][0]
    std_mod = stdfit(snr, A, B)

    # Calculate the chi-square for both populations
    chi   = (std - std_mod) / std_err
    dof   = len(snr) - 2

    return [chi, np.sum(chi ** 2), dof]
    

def load_json(fname, numpize=True):
    with open(fname, 'r') as j:
            dictionary = json.load(j)

    if numpize:
        for key in dictionary:
            dictionary[key] = np.array(dictionary[key])

    return dictionary

def msg(txt):
    stamp = time.strftime(' %Y-%m-%d %H:%M:%S | ')
    print(stamp+txt)

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
        raise ValueError("Cannot covert {} to a bool".format(s))

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

def SolveForEpochCorrection(data, fit_ra, fit_dec):
    '''
    Take the input catalog and the weights and solve for a global epoch astrometric offset
    This will correct for any (relative) systematic astrometric errros
    '''

    # Initial Uncorrected MCMC iteration
    ra                 = data[:,:,0]
    dec               = data[:,:,1]
    snr               = data[:,:,4]
    ra_extent    = data[:,:,5]
    dec_extent  = data[:,:,6]

    # Solve for the average positions    
    avg_ra, avg_dec = AveragePositions(ra, dec,'source',weight=snr)

    # Solve for per epoch offsets from the average position
    _, _, delta_ra, delta_dec = DecomposeDeltaCoord(avg_ra, avg_dec, ra, dec)

    # Convert to be in units of beams
    delta_ra    /= ra_extent
    delta_dec /= dec_extent

    # Append offsets to data array 
    data = AppendValues(data, delta_ra)
    data = AppendValues(data, delta_dec)

    # Reshape array for to simplify matrix operations 
    weight_ra = np.tile(weight_ra[:, np.newaxis, np.newaxis], (1, data.shape[1], data.shape[2]))
    weight_dec = np.tile(weight_dec[:, np.newaxis, np.newaxis], (1, data.shape[1], data.shape[2]))

    # Solve for per epoch offsets
    ra_corr, ra_corr_err      = (AverageByEpoch(filter_data, weight = weight_ra, error=True))[:,-2]
    dec_corr, dec_corr_err = (AverageByEpoch(filter_data, weight = weight_ra, error=True))[:,-1]

    return [ra_corr, ra_corr_err , dec_corr, dec_corr_err]


def BootstrapPositions(data, n_bootstrap=500):
    '''
    Boostrap the results to get the best-fit offsets and the errors from the Confidence intervals
    '''

    # Initial Uncorrected MCMC iteration
    ra                 = data[:,:,0]
    dec               = data[:,:,1]
    snr               = data[:,:,4]
    ra_extent    = data[:,:,5]
    dec_extent  = data[:,:,6]

    # Solve for the average positions    
    avg_ra, avg_dec = AveragePositions(ra, dec,'source',weight=snr)

    # Solve for per epoch offsets from the average position
    _, _, delta_ra, delta_dec = DecomposeDeltaCoord(avg_ra, avg_dec, ra, dec)

    # Convert to be in units of beams
    delta_ra    /= ra_extent
    delta_dec /= dec_extent

    # Append offsets to data array 
    data = AppendValues(data, delta_ra)
    data = AppendValues(data, delta_dec)

    # Get the bootstrapped values + errors
    bootstrap_index  = np.random.randint(0, data.shape[1], size=(n_bootstrap, data.shape[1]))

    # We need to swap around the axes for indexing purposes because we want to replace epochs
    bootstrap_data   = np.swapaxes(np.swapaxes(data,0,1)[bootstrap_index,:,:], 1, 2) 
    bootstrap_data   = np.nanstd(bootstrap_data, ddof=1, axis = 2)         

    # Need to figure out a way to not bootstrap the NaN value because this breaks for small number of epochs
    value, err_minus, err_plus = CI(bootstrap_data, axis = 0)

    # Break up the output
    ra     = value[:,-2]
    ra_err = np.amax([err_minus[:,-2], err_plus[:,-2]], axis=0)
    dec     = value[:,-1]
    dec_err = np.amax([err_minus[:,-1], err_plus[:,-1]], axis=0)

    # Signal-to-noise Ratio
    average_data = AverageBySource(data)
    snr = np.array(average_data[:,4])

    # Return the values for fitting
    return np.array([snr, ra, ra_err, dec, dec_err])


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

def AppendValues(data, append_data):
    '''
    Append a subset of data to the larger data catalog
    '''

    append_data = np.expand_dims(append_data, -1)
    data = np.concatenate((data, append_data), axis=len(data.shape) - 1)

    return data


def stdfit(snr,A,B):
    '''
    Fit function to quantify the (relative) astrometric precision:
        The parameter A is the signal-to-noise dependency
        The parameter B is the systematic limit (independant of signal-to-noise)
    '''
    return np.sqrt(A ** 2 / (snr**2) + B**2)

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

def plot_chains(sampler, plot_prefix):
    '''
    Function to plot the sampler chain evolution and posterior distribution
    '''

    params = ['A', 'B']

    figa = plt.figure(figsize=(30,5))
    for k,j in enumerate(range(sampler.flatchain.shape[1])):
        param = params[k]
        plt.subplot(1,sampler.flatchain.shape[1],k+1)
        plt.xlabel('Parameter {param}', fontsize=20)
        plt.ylabel('Counts', fontsize=20)
        patches = plt.hist(sampler.flatchain[:,j],bins=100)
    figa.subplots_adjust(hspace=.5)
    plt.savefig(f'{plot_prefix}_MCMC_Distribution.png')
    plt.clf()
    plt.close()

    figb = plt.figure(figsize=(30,5))
    for k,j in enumerate(range(sampler.flatchain.shape[1])):
        param = params[k]
        plt.subplot(1,sampler.flatchain.shape[1],k+1)
        plt.plot(sampler.chain[:,:,j].T)
        plt.ylabel(f'Parameter {param}', fontsize=20)
        plt.xlabel('Step number', fontsize=20)
    figb.subplots_adjust(hspace=.5)
    plt.savefig(f'{plot_prefix}_MCMC_chains.png')
    plt.clf()
    plt.close()

def plot_autocorr_time(sampler, plot_prefix):
    '''
    Routine to plot the autocorrelation time following documentation
    on the emceee website
    '''

    params = ['A', 'B']

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

        param = params[k]

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
        plt.xlabel("Number of samples, $N$")
        plt.ylabel(r"$\tau$ estimates")
        plt.legend(fontsize=14)
        plt.savefig(f'{plot_prefix}_{param}_autocorr.png')
        plt.clf()
        plt.close()


def MCMC(data, plot_prefix):
    '''
    Simple Markov-Chain Monte Carlo routine to solve for the best fit parameters:
        To upweight high signal-to-noise detections we are fitting in log-space
    '''

    # Break data into relevant components
    snr        = data[0]
    std        = data[1]
    std_err = data[2]

    def log_prior(p): #log prior
        A,B=p[0],p[1]
        prior = 0.0
        prior += ss.uniform.logpdf(A,loc=0.0,scale=2.0)
        prior += ss.uniform.logpdf(B,loc=0.0,scale=2.0)
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

    plot_autocorr_time(sampler, plot_prefix)
    plot_chains(sampler, plot_prefix)

    #Extract fit parameters
    A = CI(sampler.flatchain[:,0])
    B = CI(sampler.flatchain[:,1])

    fit = {'A':A, 'B':B}

    # Calculate chi statistics
    chi = chi2(data, fit)   
    fit['chi']   = chi[0] .tolist()
    fit['chi2'] = chi[1]
    fit['dof']  =  chi[2]

    return fit

def PlotResults(data, fit, plot_prefix):
    '''
    Plotting script to show the astrometric fits
    '''

    # Separate the data
    snr        = data[0]
    std        = data[1]
    std_err = data[2]

    # Initialize plotting parameters
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_facecolor('white')

    # Decompose the the fit
    A = fit['A'][0]
    B = fit['B'][0]
    plot_fit = stdfit(np.sort(snr), A, B)

    #ax.errorbar(global_snr, global_pos, yerr = global_pos_err, mec='k',fmt='o',mfc='w', ecolor='k', label='$\chi^2=${:.1f}/{}'.format(chi_global,dof_global), zorder=0, alpha=0.25)
    ax.plot(sorted(snr),plot_fit,ls='--', lw=2, color='r',zorder=1000, label=r'Best fit')
    ax.errorbar(snr, std, yerr = std_err, mec='k', fmt='o', mfc='C0', ecolor='k', zorder=100)
    ax.axhline(0.1, color='k', ls=':', zorder=1000, label='10% of Beam')

    # Plot the data + MCMC model
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Astrometric error (beams)',fontfamily='serif',fontsize=15)
    ax.set_xlabel('Source-averaged Signal-to-Noise ratio', fontfamily='serif', fontsize=15)
    ax.legend(framealpha=1.0, fontsize=15, prop={'family':'serif'}).set_zorder(10000000)

    # Make the plot look pretty
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='in',length=5,top=True,right=True)
    ax.tick_params(axis='both', which='major', direction='in',length=10,top=True,right=True, labelsize=12)

    # Save the plot
    plt.savefig(f'{plot_prefix}_MCMC.png')
    plt.clf()
    plt.close()
