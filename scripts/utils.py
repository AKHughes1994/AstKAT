import json
import glob 
import emcee
import configparser
import time
import photutils
import subprocess
import bdsf
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u
import scipy.stats as ss


from matplotlib.patches import Circle, Ellipse
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.io import fits
from astropy.coordinates import (SkyCoord, match_coordinates_sky)
from astropy.time import Time
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.modeling.functional_models import Gaussian2D
from astropy.table import QTable

from scipy.special import gamma
from scipy.signal import fftconvolve
from scipy.stats import (rayleigh, truncnorm, expon)

from photutils.datasets import (make_gaussian_sources_image, make_noise_image)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Simple functions for Source Extracting/Filtering/Matching/Output
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def FilterData(data, snr_low=0.0, snr_high=1e10, phase_offset_thresh=1e10):
    '''
    Filter the data catalog by snr and angular offset from the phase center:
        Defaults (Include all sources):
            snr_low  = 0.0
            snr_high = 1e10
            phase_offset_thresh = 1e10 [deg]
    '''
    med_data = MedianBySource(data)
    snr, offset = med_data[:,4], med_data[:,5]

    snr_index  = np.where((snr > snr_low) & (snr < snr_high))[0]
    po_index = np.where(offset < phase_offset_thresh)[0]

    if len(snr_index) >= len(po_index):
        return data[snr_index[np.isin(snr_index,po_index)],:,:]
    else:
        return data[po_index[np.isin(po_index,snr_index)],:,:]


def load_json(fname, numpize=True):
    '''
    As you may have guessed, 
    this function loads a json file

    Options:

        fname: Name of the json file
        numpize: Conver all the keys to numpy arrays
    
    Note: 
        Only use numpize if the highest level keys
        are all lists
    '''

    with open(fname, 'r') as j:
            dictionary = json.load(j)

    if numpize:
        for key in dictionary:
            if type(dictionary[key]) is list:
                dictionary[key] = np.array(dictionary[key])
    return dictionary



def msg(txt, skip_line=False):
    '''
    Simple function to print a time-stamped
    message containing [txt] to the terminal
    '''
    
    if skip_line:
        print('\n')
    stamp = time.strftime(' %Y-%m-%d %H:%M:%S | ')
    print(stamp+txt)



def str_to_bool(s):
    '''
    The config.ini file converts bools to strings
    when initialized, this convert the strings "True"/"False"
    to their boolean counterparts
    '''

    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
        raise ValueError("Cannot covert {} to a bool".format(s))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Simple Functions to average by source or epoch 
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def AverageBySource(data, weight=None):
    '''
    (Weighted) Average parameters across the epoch-axis, resulting in a single value for each source.
    '''
    return NanAverage(data, weights = weight, axis=1)

def MedianBySource(data):
    '''
    Median of parameters across the epoch-axis, resulting in a single value for each source.
    '''
    return np.nanmedian(data,axis=1)

def AverageByEpoch(data,weight=None, error=False):
    '''
    (Weighted) Average parameters across the source-axis, resulting in a single value for each epoch. The default is uniform weighting.
    '''
    return NanAverage(data, weights=weight, error=error, axis=0)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Functions to solve for the separations, bootstrap positions, and handle NaNs
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def NanAverage(data, weights=None, error=False, axis=0):
    '''
    (Weighted) Average that ignore nan values for an arbitrary axis
    '''

    if weights is None:

        # No weights (arithmetic mean)
        avg = np.nanmean(data, axis = axis)
        return avg

    else:
        # Mask NaNs
        mask = np.isnan(data)
        ma = np.ma.MaskedArray(data, mask = mask)
        avg = np.ma.average(ma, weights = weights, axis = axis)

        if error:
            ma = np.ma.MaskedArray(weights, mask = mask)
            err = np.ma.sum(ma, axis = axis) ** (-0.5)
            return np.array([avg, err])

        else:
            return avg

def AveragePositions(RAs, Decs, axis = 2, weight = None):

    '''
    Calculate the average position of a set of RAs and Decs
    Modified from Original Version by Gregory R. Sivakoff
    '''
    
    # Initial centering
    CentreDec = NanAverage(Decs, axis = axis, weights = weight[0])
    CentreRA  = NanAverage(RAs * np.cos(Decs * np.pi / 180.), axis = axis, weights = weight[1]) / np.cos(CentreDec * np.pi / 180.)

    # Calculate offsets
    seps, PAs, offsetRA, offsetDec = DecomposeDeltaCoord(CentreRA, CentreDec, RAs, Decs)

    # Find the average corrections in the RA and DEC directions
    avgCorrectionRA  = NanAverage(offsetRA, axis = axis, weights = weight[0])
    avgCorrectionDec = NanAverage(offsetDec, axis = axis, weights = weight[1])

    return CentreRA + avgCorrectionRA, CentreDec + avgCorrectionDec

def DecomposeDeltaCoord(ref_RA, ref_Dec, RAs, Decs):
    '''
    Take RAs and Decs as inputs and calculate offset parameters:
        Total Offset         [deg]
        Offset Postion Angle [rad]
        Offset in RA         [deg]
        Offset in Dec        [deg]

    Modified from Original Version by Gregory R. Sivakoff
    '''

    while ref_RA.ndim < RAs.ndim:
        ref_RA  = ref_RA[:, None]
        ref_Dec = ref_Dec[:, None]

    reference = SkyCoord(ref_RA * u.deg, ref_Dec * u.deg, frame='icrs')
    sources   = SkyCoord(RAs * u.deg, Decs * u.deg, frame='icrs')

    seps = sources.separation(reference)
    PAs  = sources.position_angle(reference)

    offsetsRA  = -seps * np.cos(PAs.rad + np.pi / 2)
    offsetsDec = +seps * np.sin(PAs.rad + np.pi / 2)

    return(seps.deg, PAs.rad, offsetsRA.deg, offsetsDec.deg)

def GetNormalizedSeparations(data, obs_properties, epoch_corrections = False, n_bootstrap = 10000):
    '''
    These will solve for the separations,
    normalized by the distance to the 68% CI along the
    particular position angle separating the reference 
    and source positions
    '''

    # Get the synthesized beam shape parameters
    bmaj = obs_properties['bmaj_deg'] # FWHM of the major axis (deg)
    bmin = obs_properties['bmin_deg'] # FWHM of the minor axis (deg)
    bpa  = obs_properties['bpa_rad']  # Position angle in rad

    # Convert the FWHM to the semi-major and semi-minor axis out to the 68% CI
    semi_major = 1.5096 * bmaj /  (8 * np.log(2)) ** 0.5
    semi_minor = 1.5096 * bmin /  (8 * np.log(2)) ** 0.5

    # Solve for the distances from the center of the ellipse to the 68% Confidence level along RA/DEC
    ra_extent  = semi_major * semi_minor / (semi_minor ** 2 * np.sin(bpa) ** 2 + semi_major ** 2 * np.cos(bpa) ** 2) ** 0.5
    dec_extent = semi_major * semi_minor / (semi_minor ** 2 * np.cos(bpa) ** 2 + semi_major ** 2 * np.sin(bpa) ** 2) ** 0.5

    # Split out the relevant parameters 
    ra  = data[:,:,0]
    dec = data[:,:,1]   

    # Apply epoch corrections if required
    epoch_cor_ra  = [0,0]
    epoch_cor_dec = [0,0]
    if type(epoch_corrections) is np.ndarray:
        msg('MCMC: Applying Epoch Corrections to Measured Positions')
        epoch_cor_ra  = epoch_corrections[0,:]
        epoch_cor_dec = epoch_corrections[1,:]
    
    # Bootstrap positions -- If epoch corrections apply random scatter
    ra  = ra  + epoch_cor_ra[0] 
    dec = dec + epoch_cor_dec[0]

    # Get the weights for each RA and Dec 
    ra_weights  = ra_extent  ** (-2) #/ ra_extent  ** (-2)
    dec_weights = dec_extent ** (-2) #/ dec_extent ** (-2)
    
    # Solve for the average positions -- weight by (inverse) RA/Dec extent -- i.e smaller beams are weighted higher
    avg_ra, avg_dec = AveragePositions(ra, dec, axis = 1, weight = [ra_weights, dec_weights])
  
    # Generate boostrap indexes
    index = np.random.randint(0, data.shape[1], size=(n_bootstrap, data.shape[1]))

    # Solve for separation and position angle angle and the transformed angle,
    sep, pa, _, _ = DecomposeDeltaCoord(avg_ra, avg_dec, ra[:, index] + 1.0 * epoch_cor_ra[1]  * np.random.randn(*index.shape), dec[:, index] + 1.0 * epoch_cor_dec[1] * np.random.randn(*index.shape))
    
    trans_pa = pa - bpa[index]
      
    # Solve for the distance between the center of the ellipse and the 68% CI along the line separation the source and the reference
    x68 = (semi_major[index] ** (-2) + np.tan(trans_pa) ** 2 * semi_minor[index] ** (-2)) ** (-0.5)
    r68 = abs(x68) * (1 + np.tan(trans_pa) ** 2) ** 0.5
    
    # Normalize the separations
    sep /= r68 

    # For for the sigma value for each source and each boostrap interation
    # The sigma equation is a de-biased estimator of sigma fro a Rayleigh
    # Wikipedia is pretty good for this: https://en.wikipedia.org/wiki/Rayleigh_distribution
    N = np.count_nonzero(~np.isnan(sep), axis = 2)
    sigma = (1. / (2 * N) * np.nansum(sep ** 2, axis = 2)) ** 0.5 * gamma(N) * N ** 0.5 / gamma(N + 0.5)

    # Get the values from the 68% Confidence interval of the
    sigma_sep, sigma_err_minus, sigma_err_plus = ConfidenceInterval(sigma, axis = 1)


    fig, ax = plt.subplots()
    index = 10
    plt.hist(sigma[index], bins = 75)
    plt.axvline(sigma_sep[index], zorder=100, ls='--', c='k')
    plt.axvline(sigma_sep[index] + sigma_err_plus[index],  zorder=100, ls=':', c='k')
    plt.axvline(sigma_sep[index] - sigma_err_minus[index], zorder=100, ls=':', c='k')
    plt.xlim(0.07,0.12)
    plt.savefig('test.png')
    plt.close()
    
    # Conservatively adopt the maximum error (although these should be very symetric)
    msg('MCMC: Median ratio of bootstrap errors = {:.2f} (Should be close to 1.0 as we assume symmetric Errors)'.format(np.median(sigma_err_minus / sigma_err_plus)))
    sigma_sep_err = np.mean((sigma_err_minus, sigma_err_plus), axis = 0)

    # Calculate the average S/N for each source
    med_data = MedianBySource(data)
    src_snr = np.array(med_data[:,4])
    
    return np.array([src_snr, sigma_sep, sigma_sep_err])


def SolveForEpochCorrection(data, obs_properties, fit):
    '''
    Take the input catalog and the weights and solve for a global epoch astrometric offset
    This will correct for any (relative) systematic astrometric errors
    '''

    # Get the synthesized beam shape parameters
    bmaj = obs_properties['bmaj_deg'] # FWHM of the major axis (deg)
    bmin = obs_properties['bmin_deg'] # FWHM of the minor axis (deg)
    bpa  = obs_properties['bpa_rad']  # Position angle in rad

    # Convert the FWHM to the semi-major and semi-minor axis out to the 68% CI
    semi_major = 1.5096 * bmaj /  (8 * np.log(2)) ** 0.5
    semi_minor = 1.5096 * bmin /  (8 * np.log(2)) ** 0.5

    # Solve for the distances from the center of the ellipse to the 68% Confidence level along RA/DEC
    ra_extent  = semi_major * semi_minor / (semi_minor ** 2 * np.sin(bpa) ** 2 + semi_major ** 2 * np.cos(bpa) ** 2) ** 0.5
    dec_extent = semi_major * semi_minor / (semi_minor ** 2 * np.cos(bpa) ** 2 + semi_major ** 2 * np.sin(bpa) ** 2) ** 0.5

    # Split out the relevant parameters 
    ra  = data[:,:,0]
    dec = data[:,:,1]
    snr = data[:,:,4]

    # Get the Weights to calculate the average
    ra_weights  = ra_extent  ** (-2) #/ ra_extent  ** (-2)
    dec_weights = dec_extent ** (-2) #/ dec_extent ** (-2)

    # Solve for the average positions 
    avg_ra, avg_dec = AveragePositions(ra, dec, axis = 1, weight = [ra_weights, ra_weights])
    
    # Solve for separation and position angle angle and the transformed angle
    sep, pa, sep_ra, sep_dec = DecomposeDeltaCoord(avg_ra, avg_dec, ra, dec)
    trans_pa = pa - bpa

    # Solve for the distance between the center of the ellipse and the 68% CI along the line separation the source and the reference
    x68 = (semi_major ** (-2) + np.tan(trans_pa) ** 2 * semi_minor ** (-2)) ** (-0.5) 
    r68 = abs(x68) * (1 + np.tan(trans_pa) ** 2) ** 0.5

    # Calculate the weights and the ra/dec per-epoch corrections   
    error_sep = AstrometricError(snr, fit['A'][0], fit['B'][0]) * r68

    ra_offset  = AverageByEpoch(sep_ra, weight = error_sep ** (-2), error=True)
    dec_offset = AverageByEpoch(sep_dec, weight = error_sep ** (-2), error=True)

    # These are the extents in RA and Dec from position to the 68% CI along the vector connecting the positions 
    error_ra  = abs(error_sep * np.cos(pa + np.pi/2))
    error_dec = abs(error_sep * np.sin(pa + np.pi/2))


    return np.array([ra_offset, dec_offset])
   

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Functions to fit the data and extract the Astrometric precision
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ConfidenceInterval(data, prob = 68, axis=0):

    '''
    Simple function to calculate the confidence interval along an array axis
    '''
    
    # Upper and lower probability range
    prob = (100. - prob)/2.
    lower = prob
    upper = 100. - prob

    # Percentiles
    pctL=np.percentile(data, lower, axis=axis)
    pctU=np.percentile(data, upper, axis=axis)
    value = np.nanmedian(data, axis=axis)

    # (+)/(-) Errors
    err_minus = value - pctL
    err_plus =  pctU - value

    # Return an array consisting of the value, err_minus, err_plus
    return [value, err_minus,err_plus]

def ChiSquare(data, fit):
    '''
    Simple Chi-square solver to test fit quality
    '''

    # Separate the data
    snr        = data[0]
    std        = data[1]
    std_err    = data[2]

    # Decompose fit paramaters 
    A = fit['A'][0]
    B = fit['B'][0]
    std_mod = AstrometricError(snr, A, B)

    # Calculate the chi-square for both populations
    chi   = (std - std_mod) / std_err
    dof   = len(snr) - 2

    return [np.sum(chi ** 2), dof]


def AstrometricError(snr, A, B):
    '''
    Fit function to quantify the (relative) astrometric precision:
        The parameter A is the signal-to-noise dependency
        The parameter B is the systematic limit (independant of signal-to-noise)
    '''

    return np.sqrt(A ** 2 / (snr ** 2) + B ** 2)


def MCMC(data, plot_prefix):
    '''
    Simple Markov-Chain Monte Carlo routine to solve for the best fit parameters:
        Inputs:
            data: Should have three 1-D arrays containing the S/N, astrometric variability (sigma), and error on (sigma)
            plot_prefix: String that will be appended to the front of any naming
        Outputs:
            fit: a dictionary containing the fit parameters with errors, as well as chi-square stats to measure the goodness of fit 
    '''

    # Log-prior
    def log_prior(p): 
        A, B =p[0], p[1]
        prior = 0.0
        prior += ss.uniform.logpdf(A,loc=0.0,scale=10.0)
        prior += ss.uniform.logpdf(B,loc=0.0,scale=10.0)
        
        if np.isnan(prior):
            return(-np.inf)
        
        return prior

    # Log-likelihood
    def lp(p, data): #log likelihood

        snr     = data[0]
        std     = data[1]
        std_err = data[2]

        A, B = p[0], p[1]
        model = AstrometricError(snr, A, B)

        # sys_var is a nuiscance parameter -- a cheap way to increase errors due to unmodeled S/N variation in the sources

        # If sys_var is negative the walker needs to learn its place
        if log_prior(p) == -np.inf:
            return -np.inf

        else:
            logprob = (model - std) ** 2 / (2 * std_err ** 2) + np.log((2 * np.pi * std_err ** 2) ** 0.5)            
            return -np.nansum(logprob) + log_prior(p)

    #Define the initial Parameters
    pGuess = np.array([0.0, 0.0])
    nDim = len(pGuess)
    nWalkers = 10 * nDim
    nBurn=1000
    nSamp=500

    pRand = np.array([2.0 * np.random.rand(nWalkers),
                      2.0 * np.random.rand(nWalkers)])

    p0 = np.zeros((nWalkers,nDim))
    for k in range(nDim):
        p0[:,k] = pRand[k] + pGuess[k]

    # Run the burn-in iterations
    sampler = emcee.EnsembleSampler(nWalkers, nDim, lp, args=[data])
    result = sampler.run_mcmc(p0,nBurn)
    pos,prob,state=result.coords,result.log_prob,result.random_state

    # Run the Sampling iterations
    sampler.reset()
    result = sampler.run_mcmc(pos,nSamp)
    pos,prob,state=result.coords,result.log_prob,result.random_state

    # Extract fit parameters
    A   = ConfidenceInterval(sampler.flatchain[:,0])
    B   = ConfidenceInterval(sampler.flatchain[:,1])

    fit = {'A': A, 'B': B}

    # Plot the sampler
    PlotChainsMCMC(sampler, list(fit.keys()), plot_prefix)
    PlotAutocorrTimeMCMC(sampler, list(fit.keys()), plot_prefix)

    # Calculate chi statistics
    quality_stats = ChiSquare(data, fit)   
    fit['chi2'] = quality_stats[0]
    fit['dof']  = quality_stats[1]

    return fit

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Functions to plot the fit + data
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def PlotResults(data0, fit0, data = None, fit = None, plot_prefix = 'Fit', ylim = False, xlim = False):
    '''
    Plotting script to show the astrometric fits
    '''

    # Separate the data
    if data is not None:
        snr        = data[0]
        std        = data[1]
        std_err    = data[2]

    snr0        = data0[0]
    std0        = data0[1]
    std0_err    = data0[2]

    # Initialize plotting parameters
    fig, ax = plt.subplots(figsize=(8,5))
    fig.patch.set_facecolor('white')

    # Decompose the the fit
    if data is not None:
        plot_fit  = AstrometricError(np.sort(snr),  fit['A'][0], fit['B'][0])
    plot_fit0 = AstrometricError(np.sort(snr0), fit0['A'][0], fit0['B'][0])

    # Plot corrected
    if data is not None:
        ax.errorbar(snr, std, yerr = std_err, mec='k', fmt='o', mfc='C0', ecolor='k', zorder=100, label = 'Corrected')
        ax.plot(sorted(snr), plot_fit, ls='-', lw=1, color='k', zorder=1000)

    # Plot uncorrected
    ax.plot(sorted(snr0), plot_fit0, ls='--', lw=1, color='k', zorder=10)
    ax.errorbar(snr0, std0, yerr = std0_err, mec='k', fmt='o', mfc='white', ecolor='k', zorder=1, alpha=0.5, label = 'Uncorrected')

    # Show 10% of the synthesized beam limit
    ax.axhline(0.1 * (8 * np.log(2)) ** 0.5 / 1.5096, color='k', ls=':', zorder=10000, label='10% of Idealized PSF (FWHM)')

    # Plot the data + MCMC model
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Relative Astrometric Error ($\hat{\sigma}_r$)',fontfamily='serif',fontsize=15)
    ax.set_xlabel('(S/N)$_{med}$', fontfamily='serif', fontsize=15)
    ax.legend(framealpha=1.0, fontsize=15, prop={'family':'serif'}).set_zorder(10000000)

    # Make the plot look pretty
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='in',length=5,top=True,right=True)
    ax.tick_params(axis='both', which='major', direction='in',length=10,top=True,right=True, labelsize=12)

    if ylim:    
        ax.set_ylim(ylim[0], ylim[1])

    if xlim:    
        ax.set_xlim(xlim[0], xlim[1])

    # Save the plot
    plt.savefig(f'{plot_prefix}_fit_MCMC.png')
    plt.clf()
    plt.close()

def PlotChainsMCMC(sampler, params, plot_prefix):
    '''
    Function to plot the sampler chain evolution and posterior distribution
    '''

    figa = plt.figure(figsize=(30,5))
    for k,j in enumerate(range(sampler.flatchain.shape[1])):
        param = params[k]
        plt.subplot(1,sampler.flatchain.shape[1],k+1)
        plt.xlabel(f'Parameter {param}', fontsize=20)
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

def PlotAutocorrTimeMCMC(sampler, params, plot_prefix):
    '''
    Routine to plot the autocorrelation time following documentation
    on the emcee website
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
        plt.savefig(f'{plot_prefix}_{param}_MCMC_autocorr.png')
        plt.clf()
        plt.close()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Debugging scripts (the bottom of the barallel)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def TestRayleighEstimate(scale = 0.05):

    # Hello
    x = np.linspace(0,0.5, 150)
    y = x / scale ** 2 * np.exp(-x ** 2 / (2 * scale ** 2))

    fig, ax = plt.subplots()
    plt.plot(x, y, ls = ':', c = 'k', lw = 2)
    plt.plot(x, rayleigh.pdf(x, scale = scale), ls = '--', c = 'C0', lw = 1)
    plt.savefig('test.png')
    
    # Draw 100 samples from a random Rayleigh distributions
    sample = rayleigh.rvs(scale = scale, size = 100)
    
    # For each bootstraped separation "light curve" count the number of non-NaNs that will "count" as samples
    N = len(sample)

    # Now measure the scale factors (i.e., Gaussian STDev equivalent) 
    # Wiki is pretty good for this: https://en.wikipedia.org/wiki/Rayleigh_distribution    
    sigma_bias = (1. / (2 * N) * np.sum(sample ** 2)) ** 0.5 # Biased estimator
    sigma = sigma_bias * gamma(N) * N ** 0.5 / gamma(N + 0.5) # De-bias correction    


def plot_source(pos, avg_pos, r68, obs_properties):
    
    # Iterate through the images for the test_index source
    #for k in range(len(pos)):
    for k in range(1):
        try:
            # Load in the image parameters
            imname = obs_properties['image_identifier'][k]
            image = '../images/{}'.format(imname)

            # Load the image and get image parameters
            hdu = fits.open(image)[0]
            bmaj = hdu.header['BMAJ']
            bmin = hdu.header['BMIN']
            bpa  = hdu.header['BPA'] * np.pi / 180.0
            date = hdu.header['DATE-OBS']
            pixel_size = abs(hdu.header['CDELT2'])

            # Calculate the relevant scales
            semi_major = 1.5096 * bmaj /  (8 * np.log(2)) ** 0.5
            semi_minor = 1.5096 * bmin /  (8 * np.log(2)) ** 0.5

            # Solve for the distances from the center of the ellipse to the 68% Confidence level along RA/DEC
            ra_extent  = semi_major * semi_minor / (semi_minor ** 2 * np.sin(bpa) ** 2 + semi_major ** 2 * np.cos(bpa) ** 2) ** 0.5
            dec_extent = semi_major * semi_minor / (semi_minor ** 2 * np.cos(bpa) ** 2 + semi_major ** 2 * np.sin(bpa) ** 2) ** 0.5

            # Correct the header file deleting unecessary parameters
            hdu.header['NAXIS'] = 2
            hdu.data = hdu.data[0,0,:,:]
            del_array = ['NAXIS3', 'CRPIX3', 'CRVAL3', 'CDELT3', 'CUNIT3', 'CTYPE3',
                 'NAXIS4', 'CRPIX4', 'CRVAL4', 'CDELT4', 'CUNIT4', 'CTYPE4',
                 'PC1_3', 'PC1_4', 'PC2_3', 'PC2_4',
                 'PC3_1', 'PC3_2', 'PC3_3', 'PC3_4',
                 'PC4_1', 'PC4_2', 'PC4_3', 'PC4_4']
            for del_name in del_array:
                try: 
                    del hdu.header[del_name]
                except:
                    pass

            # Get the WCS header
            wcs = WCS(hdu.header)
            xpos, ypos = wcs.wcs_world2pix(avg_pos.ra, avg_pos.dec, 1)
  
            # Make the cutout
            cutout = Cutout2D(hdu.data, position=(xpos,ypos), size=(15,15), wcs=wcs)
    
            # Put the cutout image in the FITS HDU
            hdu.data = cutout.data
    
            # Update the FITS header with the cutout WCS
            hdu.header.update(cutout.wcs.to_header())
            wcs = WCS(hdu.header)

            # MPL Formatting opinions
            mpl.rcParams.update(mpl.rcParamsDefault)
            mpl.rcParams['font.size'] = 15
            mpl.rcParams['font.family'] = 'serif'
    
            mpl.rcParams['ytick.right'] = True
            mpl.rcParams['ytick.direction'] = 'in'
            mpl.rcParams['ytick.minor.visible'] = True
            mpl.rcParams['ytick.labelsize'] = 'large'
            mpl.rcParams['ytick.major.width'] = 1.5
            mpl.rcParams['ytick.minor.width'] = 1.5
            mpl.rcParams['ytick.major.size'] = 6.0
            mpl.rcParams['ytick.minor.size'] = 3.0

            mpl.rcParams['xtick.top'] = True
            mpl.rcParams['xtick.direction'] = 'in'
            mpl.rcParams['xtick.labelsize'] = 'large'
            mpl.rcParams['xtick.minor.visible'] = True
            mpl.rcParams['xtick.major.width'] = 1.5
            mpl.rcParams['xtick.minor.width'] = 1.5
            mpl.rcParams['xtick.major.size'] = 6.0
            mpl.rcParams['xtick.minor.size'] = 3.0

            mpl.rcParams['axes.linewidth'] = 1.5

            # Get separation and position angle
            sep = pos[k].separation(avg_pos).deg
            pa  = pos[k].position_angle(avg_pos).rad

            # Initialize the plot
            fig = plt.figure(figsize=(25,25))
            fig.set_facecolor('white')
            ax = fig.add_axes([0.15, 0.1, 0.8, 0.8], projection=wcs) 

            # Make the image
            im = ax.imshow(hdu.data)

            # Labels and some Formatting
            ax.set_xlabel('Right Ascension (J2000)')
            ax.set_ylabel('Declination (J2000)')
            cbar = plt.colorbar(im, label="Pixel Flux Density (mJy / beam)", shrink=0.8)

            # Plot the average position
            x, y = wcs.wcs_world2pix(avg_pos.ra, avg_pos.dec, 0)
            circ = Circle((x,y), radius = 0.1, color='k', zorder=100000000)
            ax.add_patch(circ)

            # Plot the fit position with the 68% ellipse and the extent
            x, y = wcs.wcs_world2pix(pos[k].ra, pos[k].dec, 0)
            circ = Circle((x,y), radius = 0.1, color='r', zorder=100000000)
            ax.add_patch(circ)

            aperture = photutils.aperture.EllipticalAperture((x,y), 
                    semi_major / pixel_size,  
                    semi_minor / pixel_size, 
                    theta = bpa + np.pi / 2)
            ap_patches = aperture.plot(lw = 5, zorder=10, color = 'red', ls = '-')

            # Plot the separation vectors
            ax.plot([x, x - r68[k] / pixel_size * np.sin(pa)],
                 [y, y + r68[k] / pixel_size * np.cos(pa)],
                 color = 'purple', lw = '5', zorder=1e10)

            ax.plot([x, x - r68[k] / pixel_size * np.sin(pa)],
                 [y, y],
                 color = 'yellow', lw = '5', zorder=1e10)

            ax.plot([x, x],
                 [y, y + r68[k] / pixel_size * np.cos(pa)],
                 color = 'blue', lw = '5', zorder=1e10)

            ax.plot([x, x - sep / pixel_size * np.sin(pa)],
                 [y, y + sep / pixel_size * np.cos(pa)],
                 color = 'green', lw = '5', zorder=1e10)

            # Plot the extents in the RA and Declination directions
            ax.plot([x,x], [y + dec_extent/ pixel_size, y - dec_extent/ pixel_size], color="k", lw = '2', ls='-', zorder=100)
            ax.plot([x - ra_extent/pixel_size,x + ra_extent/ pixel_size], [y, y], color="k", lw='2', ls='-', zorder=100)

            # Plot the extents along the major and minor beam axis
            ax.plot([x + semi_major / pixel_size * np.sin(bpa), x - semi_major / pixel_size * np.sin(bpa)],
                 [y - semi_major / pixel_size * np.cos(bpa), y + semi_major / pixel_size * np.cos(bpa)],
                 color = 'k', lw = '3', ls=':', zorder=1000)

            ax.plot([x + semi_minor / pixel_size * np.sin(bpa + np.pi / 2), x - semi_minor / pixel_size * np.sin(bpa + np.pi / 2)],
                 [y - semi_minor / pixel_size * np.cos(bpa + np.pi / 2 ), y + semi_minor / pixel_size * np.cos(bpa + np.pi / 2)],
                 color = 'k', lw = '3', ls=':', zorder=1000)
        
            # Include the beam shape in the image
            #aperture = photutils.aperture.EllipticalAperture((5,5), 
            #        bmaj * 0.5 / pixel_size,  
            #        bmin * 0.5 / pixel_size, 
            #        theta = bpa + np.pi / 2)
            #ap_patches = aperture.plot(lw=5, zorder=10000, facecolor = 'white')
    
            plotfile = image.split('.fits')[0].split('images/')[-1] + '.png'
            plt.savefig(f'../plots/{plotfile}')
            plt.clf()
            plt.close()

        except ValueError:
            pass
