import numpy as np
import matplotlib.pyplot as plt
import json, os, glob, configparser, sys
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord,match_coordinates_sky

# Read in the configuration file
cfg = configparser.ConfigParser()
cfg.read('../scripts/config.ini')



# Load date-time array
date_times = np.load('../files/date-times.npy')

# Use first epoch as reference
ref_index = 0

with open('../files/fieldsources_{}_{}.json'.format(date_times[ref_index],cfg['SOURCE']['name']), 'r') as jfile:
    reference = json.load(jfile)
referenceCoords = SkyCoord(reference['ra'],reference['dec'],frame='icrs',unit='deg')

############################################################ 
# Initialize the numpy array that will contain the data    #
# Array will have dimensions (n_sources, n_epochs, n_vars) #
# Initially n_vars will be:                                #  
# 0 -- Right Acension (deg)                                #
# 1 -- Declination (deg)                                   #    
# 2 -- Source Flux density (Jy/beam)                       #
# 3 -- RMS (Jy/beam)                                       #   
# 4 -- signal-to-noise  ratio                              #
# 5 -- Extent of beam in RA                                #
# 6 -- Extent of beam in Dec                               #
# 7 -- Separataion from phase (i.e. pointing centre)       #
############################################################

n_sources = len(reference['ra'])
n_epochs  = len(date_times)
n_vars    = 8
catalog   = np.empty((n_sources,n_epochs,n_vars)) * np.nan

# Append the reference data
catalog[:,ref_index,0] = referenceCoords.ra.value
catalog[:,ref_index,1] = referenceCoords.dec.value
catalog[:,ref_index,2] = reference['peak']
catalog[:,ref_index,3] = reference['rms']
catalog[:,ref_index,4] = np.array(reference['peak'])/np.array(reference['rms'])
catalog[:,ref_index,5] = reference['ra_beam']
catalog[:,ref_index,6] = reference['dec_beam']

# For matching remove reference epoch from arrays being iterated through
epoch_indexes = np.arange(len(date_times)) #indices for storing in the catalog 
epoch_indexes = np.delete(epoch_indexes, ref_index)
epochs = np.delete(date_times, ref_index)

# Iterate through the images
for epoch_index, epoch in zip(epoch_indexes[:], epochs[:]):
    print('Matching: ', epoch)

    # Define the field source coords
    with open('../files/fieldsources_{}_{}.json'.format(epoch,cfg['SOURCE']['name']), 'r') as jfile:
        field = json.load(jfile)

    for key in field.keys():
        field[key] = np.array(field[key])

    # Define a SkyCoord object containing all field source positions
    fieldCoords = SkyCoord(field['ra'],field['dec'],frame='icrs',unit='deg')
    
    #Initialize 2-D array containing the epoch parameters (n_sources,n_vars)
    epoch_catalog = np.empty((len(field['ra']), n_vars))  * np.nan
    epoch_catalog[:,0] = fieldCoords.ra.value
    epoch_catalog[:,1] = fieldCoords.dec.value
    epoch_catalog[:,2] = field['peak']
    epoch_catalog[:,3] = field['rms']
    epoch_catalog[:,4] = field['peak']/field['rms']
    epoch_catalog[:,5] = field['ra_beam']
    epoch_catalog[:,6] = field['dec_beam']

    ############################################################
    # Note the duplicate trimming is a very convoluted routine #
    # I was unable to escape the doom that is for loops !!!!!! #
    # If you are reading this and have a better idea please !! #
    # Email me at hughes1@ualberta.ca !!!!!!!!!!!!!!!!!!!!!!!! #
    ############################################################ 

    # Match the source in the current image to the reference
    match = match_coordinates_sky(fieldCoords,referenceCoords)
    offsets = match[1].value * 3600.0
    sources = match[0]

    # Sort the indexes according to the reference catalog source numbers; so it goes [0,0,0,1,1,2,2,...etc.]
    sorted_index = np.argsort(sources)
    offsets = offsets[sorted_index]
    sources = sources[sorted_index]
    epoch_catalog = epoch_catalog[sorted_index,:]
    
    # Get the position for duplicates + number of duplicates, sort the offsets so the first element in a chain of duplicates is the smallest offset (e.g., [0,0,0], [1,2,3])
    arr, inds, counts = np.unique(sources, return_index=True, return_inverse=False, return_counts=True)

    for ind, count in zip(inds,counts):
        sorted_index = ind + np.argsort(offsets[ind:ind + count])
        offsets[ind:ind + count] = offsets[sorted_index]
        epoch_catalog[ind:ind + count, :] = epoch_catalog[sorted_index, :]
    
    # Trim the duplicates so the arrays only have 1 source per reference sources (with the smallest offset from the reference)
    sources, inds = np.unique(sources,return_index=True)
    offsets = offsets[inds]
    epoch_catalog = epoch_catalog[inds,:]

    # Check for matches that are within the match threshold (default is 5 arcseconds)
    good_matches = np.where(offsets < match_threshold)
    catalog[sources[good_matches],epoch_index, :] = epoch_catalog[good_matches,:]

#############################
# Remove transient sources  #
#############################

# For each source count the number of epochs with NaN values, the max flux, and min flux, and median SNR
n_nans       = np.count_nonzero(np.isnan(catalog[:,:,4]), axis=1)
max_flux     = np.nanmax(catalog[:,:, 2], axis=1)
min_flux     = np.nanmin(catalog[:,:, 2], axis=1)
median_snr   = np.nanmedian(catalog[:,:, 4], axis=1)

# Remove sources with too many np.nans (i.e., include sources that have NaN values in less than epoch_fraction% of the epochs also ensure a minimum number of epochs)
# Also only include sources where the maximum and minimum values are separated by a factor < variability limit:
epoch_threshold   = np.amin((epoch_fraction * len(date_times), epoch_min))

good_sources = np.where((n_nans < epoch_threshold) & (max_flux < variability_limit * min_flux))# & (median_snr > snr_threshold))
catalog = catalog[good_sources[0],:,:]

###########################################
# Solve for offsets from the phase center #
###########################################

# Phase offsets -- record the position of the nans and temporarily make them zeros 
nan_locs = np.isnan(catalog)
nan_locs[:,:,7] = nan_locs[:,:,0]
catalog = np.nan_to_num(catalog)

# Calculate the phase offsets with respect to the pointing center
phase_center = SkyCoord("{} {}".format(cfg['POSITIONS']['phase_center_ra'],cfg['POSITIONS']['phase_center_dec']), frame='icrs')
c0 = SkyCoord(ra = catalog[:,:,0]*u.deg, dec = catalog[:,:,1]*u.deg, frame='icrs')
catalog[:,:,7] = c0.separation(phase_center).deg

#Repopulate with nans
catalog[nan_locs] = np.nan

#########################################
# Save numpy array and index dictionary #
#########################################

#Save the index dictionary and catalog array
index_dict = {'ra': 0, 'dec': 1, 'peak_flux': 2, 'rms': 3, 'snr': 4, 'ra_beam': '5', 'dec_beam':'6', 'phase_offset':'7'}
with open('../files/index_dict_{}.json'.format(cfg['SOURCE']['name']),'w') as jfile:
    json.dump(index_dict,jfile)

np.save('../files/field_catalog_{}_arr'.format(cfg['SOURCE']['name']),catalog)
