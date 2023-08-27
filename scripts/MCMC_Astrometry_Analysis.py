import numpy as np
import json, glob, emcee, configparser
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord  # High-level coordinates
import astropy.units as u
from astropy.time import Time
from utils import AverageBySource, AverageByEpoch, StdevBySource
from utils import AveragePositions, DecomposeDeltaCoord, FilterData
from utils import AppendValues, MCMC, PlotResults, stdfit, CI, NanAverage, BootstrapPositions, PlotChi

#Read in the configuration file
cfg = configparser.ConfigParser()
cfg.read('config.ini')

convergence_threshold = float(cfg['MCMC']['convergence_threshold'])
n_bootstrap           = int(cfg['MCMC']['n_bootstrap'])
phase_offset_limit    = float(cfg['MCMC']['phase_offset_limit'])
min_snr               = float(cfg['MCMC']['min_snr'])
max_snr               = float(cfg['MCMC']['max_snr'])

#Load in the index dictionary
with open('../files/index_dict_{}.json'.format(cfg['SOURCE']['name']), 'r') as jfile:
    index_dict = json.load(jfile)

#Load in the data arrays
catalog = np.load('../files/field_catalog_{}_arr.npy'.format(cfg['SOURCE']['name']))

########################################################
# Initial Indexes:                                     #  
# 0 -- Right Acension (deg)                            #
# 1 -- Declination (deg)                               #    
# 2 -- Source Flux density (Jy/beam)                   #
# 3 -- RMS (Jy/beam)                                   #   
# 4 -- signal-to-noise ratio                           #
# 5 -- Extent of beam in RA                            #
# 6 -- Extent of beam in Dec                           #
# 7 -- Separataion from phase (i.e. pointing centre)   #
########################################################

# Initialize the arrays
RAs       = catalog[:,:,0]
Decs      = catalog[:,:,1]
snr       = catalog[:,:,4]
ra_beams  = catalog[:,:,5]
dec_beams = catalog[:,:,6]

# Calculate the unweighted offsets [converting from deg to arcsec]
average_RAs, average_Decs = AveragePositions(RAs,Decs,'source',weight=snr)
seps, PAs, delta_RAs, delta_Decs = DecomposeDeltaCoord(average_RAs, average_Decs, RAs, Decs)


# Append the offsets in units of arcseconds 
catalog, index_dict = AppendValues(catalog, index_dict, delta_RAs  * 3600.0, ['delta_ra'])
catalog, index_dict = AppendValues(catalog, index_dict, delta_Decs * 3600.0, ['delta_dec'])

# Append the offsets in units of beams 
catalog, index_dict = AppendValues(catalog, index_dict, delta_RAs/ra_beams , ['delta_ra_beams'])
catalog, index_dict = AppendValues(catalog, index_dict, delta_Decs/dec_beams,['delta_dec_beams'])

# Use bootstrapping to solve for an error on astrometric variability
global_snr, global_stdev_RAs_beams, global_stdev_RAs_beams_err, global_stdev_Decs_beams, global_stdev_Decs_beams_err = BootstrapPositions(catalog, n_bootstrap=n_bootstrap)
filter_data  = FilterData(catalog, snr_low = min_snr, snr_high = max_snr, phase_offset_thresh = phase_offset_limit)
snr, stdev_RAs_beams, stdev_RAs_beams_err, stdev_Decs_beams, stdev_Decs_beams_err = BootstrapPositions(filter_data, n_bootstrap=n_bootstrap)

# Solve MCMC only using the filtered data
print('\nSolving Uncorrected MCMC')
fit_RAs  = MCMC(snr, stdev_RAs_beams, stdev_RAs_beams_err, 'RA', 'uncorrected', cfg)
fit_Decs = MCMC(snr, stdev_Decs_beams, stdev_Decs_beams_err, 'Dec', 'uncorrected', cfg)

print('RA:  A = {:.3f}(-{:.3f})(+{:.3f}), B = {:.4f}(-{:.4f})(+{:.4f})'.format(fit_RAs['A'][0], fit_RAs['A'][1], fit_RAs['A'][2], fit_RAs['B'][0], fit_RAs['B'][1], fit_RAs['B'][2]))
print('Dec: A = {:.3f}(-{:.3f})(+{:.3f}), B = {:.4f}(-{:.4f})(+{:.4f})'.format(fit_Decs['A'][0], fit_Decs['A'][1], fit_Decs['A'][2], fit_Decs['B'][0], fit_Decs['B'][1], fit_Decs['B'][2]))

# Plot the results without an epoch-dependant correction
PlotResults(snr, stdev_RAs_beams, stdev_RAs_beams_err, global_snr, global_stdev_RAs_beams, global_stdev_RAs_beams_err, 'RA' , fit_RAs,  'uncorrected',  cfg)
PlotResults(snr, stdev_Decs_beams, stdev_Decs_beams_err, global_snr, global_stdev_Decs_beams, global_stdev_Decs_beams_err, 'Dec', fit_Decs,  'uncorrected',  cfg) 
PlotChi(snr, stdev_RAs_beams, stdev_RAs_beams_err, global_snr, global_stdev_RAs_beams, global_stdev_RAs_beams_err, 'RA' , fit_RAs,  'uncorrected',  cfg)
PlotChi(snr, stdev_Decs_beams, stdev_Decs_beams_err, global_snr, global_stdev_Decs_beams, global_stdev_Decs_beams_err, 'Dec', fit_Decs,  'uncorrected',  cfg) 

'''
# Solve for and iteratively improve the solutions for the per-epoch global offsets until some threshold value of improvement:
loop = True
iteration = 1
good_iteration = 0
while loop:

    # Move the past iteration stdevs to "_old" version to run sigma testing
    stdev_RAs_beams_old, stdev_Decs_beams_old = stdev_RAs_beams.copy(), stdev_Decs_beams.copy()

    print('\nCorrected MCMC Iteration #',iteration)
    weight_ra  = stdfit(snr, fit_RAs['A'][0], fit_RAs['B'][0]) ** (-2)
    weight_dec = stdfit(snr, fit_Decs['A'][0], fit_Decs['B'][0]) ** (-2)

    # Make the weights the same dimension as the data -- This will help exclude the NaNs during averaging calculations
    weight_ra  = np.tile(weight_ra[:, np.newaxis, np.newaxis], (1, filter_data.shape[1], filter_data.shape[2]))
    weight_dec = np.tile(weight_dec[:, np.newaxis, np.newaxis], (1, filter_data.shape[1], filter_data.shape[2]))

    # Solve for the average offset on a per-epoch basis; i.e., an offset that affects all sources in an epoch
    ra_epoch_corr, ra_epoch_corr_err = AverageByEpoch(filter_data, weight = weight_ra, error=True)
    ra_epoch_corr = np.array(ra_epoch_corr[:,10])
    ra_epoch_corr_err = ra_epoch_corr_err[:,10]
    ra_corr_offset  = catalog[:,:,10] - ra_epoch_corr
    catalog, index_dict = AppendValues(catalog, index_dict, ra_corr_offset, ['delta_ra_beams_corr_{}'.format(iteration)])

    dec_epoch_corr, dec_epoch_corr_err  = AverageByEpoch(filter_data, weight = weight_dec, error=True)
    dec_epoch_corr =  np.array(dec_epoch_corr[:,11])
    dec_epoch_corr_err = dec_epoch_corr_err[:,11]
    dec_corr_offset  = catalog[:,:,11] - dec_epoch_corr
    catalog, index_dict = AppendValues(catalog, index_dict, dec_corr_offset, ['delta_dec_beams_corr_{}'.format(iteration)])

    # Filter data (Optional) and calculate the averages/stds over all the epochs
    filter_data  = FilterData(catalog, snr_low = min_snr, snr_high = max_snr, phase_offset_thresh = phase_offset_limit)

    snr, stdev_RAs_beams, stdev_RAs_beams_err, stdev_Decs_beams, stdev_Decs_beams_err = BootstrapPositions(filter_data, n_bootstrap=n_bootstrap)

    # Run Newest MCMC iteration
    fit_RAs  = MCMC(snr, stdev_RAs_beams, stdev_RAs_beams_err, 'RA', 'corrected', cfg)
    fit_Decs = MCMC(snr, stdev_Decs_beams, stdev_Decs_beams_err, 'Dec', 'corrected', cfg)

    # Calculate the new sigma value and compute our threshold term (sigma), i.e., the average change in offset,
    sigma_ra  = np.mean(abs(stdev_RAs_beams - stdev_RAs_beams_old)/stdev_RAs_beams_err) 
    sigma_dec = np.mean(abs(stdev_Decs_beams - stdev_Decs_beams_old)/stdev_Decs_beams_err)

    # Here we establish a criteria for convergence, for both ra and dec the average change in the astrometric precision (on a per-source basis) 
    # has to be less that 10% of the error for three consecutive iterations 

    if sigma_ra < convergence_threshold and sigma_dec < convergence_threshold:
        good_iteration += 1 
        if good_iteration == 3:
            loop=False
    else:
        good_iteration = 0

    print('Kill Loop after 3 good iterations, current number of good iterations = {}, current sigma_ra = {:.5f}, current sigma_dec = {:.5f}'.format(good_iteration, sigma_ra, sigma_dec))
    print('RA:  A = {:.3f}(-{:.3f})(+{:.3f}), B = {:.4f}(-{:.4f})(+{:.4f})'.format(fit_RAs['A'][0], fit_RAs['A'][1], fit_RAs['A'][2], fit_RAs['B'][0], fit_RAs['B'][1], fit_RAs['B'][2]))
    print('Dec: A = {:.3f}(-{:.3f})(+{:.3f}), B = {:.4f}(-{:.4f})(+{:.4f})'.format(fit_Decs['A'][0], fit_Decs['A'][1], fit_Decs['A'][2], fit_Decs['B'][0], fit_Decs['B'][1], fit_Decs['B'][2]))
    iteration += 1

print('\nConvergence Threshold Reached -- Making Final plots')
# Get final data 
global_snr, global_stdev_RAs_beams, global_stdev_RAs_beams_err, global_stdev_Decs_beams, global_stdev_Decs_beams_err = BootstrapPositions(catalog, n_bootstrap=n_bootstrap)
filter_data  = FilterData(catalog,snr_low = min_snr, snr_high = max_snr, phase_offset_thresh = phase_offset_limit)
snr, stdev_RAs_beams, stdev_RAs_beams_err, stdev_Decs_beams, stdev_Decs_beams_err = BootstrapPositions(filter_data, n_bootstrap=n_bootstrap)

# Plot the results with the epoch-dependant correction
PlotResults(snr, stdev_RAs_beams, stdev_RAs_beams_err, global_snr, global_stdev_RAs_beams, global_stdev_RAs_beams_err, 'RA' , fit_RAs,  'corrected',  cfg)
PlotResults(snr, stdev_Decs_beams, stdev_Decs_beams_err, global_snr, global_stdev_Decs_beams, global_stdev_Decs_beams_err, 'Dec', fit_Decs,  'corrected',  cfg) 
PlotChi(snr, stdev_RAs_beams, stdev_RAs_beams_err, global_snr, global_stdev_RAs_beams, global_stdev_RAs_beams_err, 'RA' , fit_RAs,  'corrected',  cfg)
PlotChi(snr, stdev_Decs_beams, stdev_Decs_beams_err, global_snr, global_stdev_Decs_beams, global_stdev_Decs_beams_err, 'Dec', fit_Decs,  'corrected',  cfg) 

ra_epoch_corr = ra_epoch_corr.flatten()
ra_epoch_corr_err = ra_epoch_corr_err.flatten()
ra_extent = 3600.0 * AverageByEpoch(filter_data, weight = None)[:,5]

dec_epoch_corr = dec_epoch_corr.flatten()
dec_epoch_corr_err = dec_epoch_corr_err.flatten()
dec_extent = 3600.0 * AverageByEpoch(filter_data, weight = None)[:,6]

# Save the per-epoch corrections
np.save('../results/RA_epoch_corr_beams_{}'.format(cfg['SOURCE']['name']), [ra_epoch_corr, ra_epoch_corr_err])
np.save('../results/RA_epoch_corr_asec_{}'.format(cfg['SOURCE']['name']), [ra_epoch_corr * ra_extent, ra_epoch_corr_err * ra_extent])
np.save('../results/Dec_epoch_corr_beams_{}'.format(cfg['SOURCE']['name']), [dec_epoch_corr, dec_epoch_corr_err])
np.save('../results/Dec_epoch_corr_asec_{}'.format(cfg['SOURCE']['name']), [dec_epoch_corr * dec_extent, dec_epoch_corr_err * dec_extent])

# Save the catalog + index array
np.save('../results/field_catalog_{}_arr'.format(cfg['SOURCE']['name']), np.array(catalog))	
with open('../results/index_dict.json','w') as jfile:
    json.dump(index_dict,jfile)
'''
