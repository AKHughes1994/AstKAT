from utils import *


def main():

    # Initialize variables pointing to the relevant directories
    scripts = os.getcwd()
    files      = scripts.replace('scripts', 'files')
    images = scripts.replace('scripts', 'images')
    results = scripts.replace('scripts', 'results')
    plots    = scripts.replace('scripts', 'plots')

    # Read in the configuration file
    cfg = configparser.ConfigParser()
    cfg.read(f'{scripts}/config.ini')
    
    # Load in config file parameters
    target_name = str(cfg['TARGET']['name'])
    phase_center_ra = str(cfg['POSITIONS']['phase_center_ra'])
    phase_center_dec = str(cfg['POSITIONS']['phase_center_dec'])
    match_threshold = float(cfg['THRESHOLDS']['match_threshold'])
    epoch_min = float(cfg['THRESHOLDS']['epoch_min'])
    epoch_fraction = float(cfg['THRESHOLDS']['epoch_fraction'])
    variability_threshold = float(cfg['THRESHOLDS']['variability_threshold'])
    snr_threshold = float(cfg['THRESHOLDS']['snr_threshold'])
    ref_index = int(cfg['THRESHOLDS']['ref_index'])
    
    # SkyCoord objecy contain source coordinates
    source_coords = SkyCoord(f'{phase_center_ra} {phase_center_dec}', frame='icrs')
    
    # Load in observation parameters 
    obs_properties = load_json(f'{results}/observation_properties.json')
    obs_isots = obs_properties['obs_isot']

    # Initialize the reference epoch
    reference = load_json(f'{files}/field_{obs_isots[ref_index]}_{target_name}.json')
    reference_coords = SkyCoord(reference['ra_deg'] * u.deg,reference['dec_deg'] * u.deg,frame='icrs') 
    reference['snr'] = np.array(reference['peak_Jy'])/np.array(reference['rms_Jy'])

    ##################################################################
    # Initialize the numpy array that will contain the data          
    # Array will have dimensions (n_sources, n_epochs, n_vars) 
    # Initially n_vars will be:                                                            
    # 0 -- Right Acension (deg)                                                      
    # 1 -- Declination (deg)                                                 
    # 2 -- Source Flux density (Jy/beam)                                       
    # 3 -- RMS (Jy/beam)                                                   
    # 4 -- signal-to-noise  ratio                                                                
    # 5 -- Separation from phase centre (deg)             
    #################################################################

    # Initialize catalog array with the appropraite dimensions
    n_sources = len(reference['ra_deg'])
    n_epochs  = len(obs_isots)
    n_vars    = 6
    catalog   = np.empty((n_sources,n_epochs,n_vars)) * np.nan

    # Truncated sources outside the desired PB cutoff (units of deg)
    phase_center_coords = SkyCoord(f'{phase_center_ra} {phase_center_dec}', frame='icrs')
    phase_center_offset  = reference_coords.separation(phase_center_coords).deg

    # Append the reference data to the array
    catalog[:,ref_index,0] = reference['ra_deg'] 
    catalog[:,ref_index,1] = reference['dec_deg'] 
    catalog[:,ref_index,2] = reference['peak_Jy']
    catalog[:,ref_index,3] = reference['rms_Jy']
    catalog[:,ref_index,4] = reference['snr'] 
    catalog[:,ref_index,5] = phase_center_offset

    # For matching remove reference epoch from arrays being iterated through
    epoch_indexes = np.arange(n_epochs) 
    epoch_indexes = np.delete(epoch_indexes, ref_index)

    # Iterate through the images
    for  epoch_index  in epoch_indexes[:]:
        
        obs_isot = obs_isots[epoch_index]
        msg(f'Matching: {obs_isot}')

        # Get relevant observation paramters
        bmaj      = obs_properties['bmaj_deg'][epoch_index]
        field       = load_json(f'{files}/field_{obs_isot}_{target_name}.json')
        field['snr'] = np.array(field['peak_Jy'])/np.array(field['rms_Jy'])
        beam_threshold = match_threshold * bmaj

        # Define a SkyCoord object containing all field source positions
        field_coords = SkyCoord(field['ra_deg'] * u.deg, field['dec_deg'] * u.deg, frame='icrs')
        phase_center_offset  = field_coords.separation(phase_center_coords).deg

        #Initialize 2-D array containing the epoch parameters (n_sources,n_vars)
        epoch_catalog = np.empty((len(field['ra_deg']), n_vars))  * np.nan
        epoch_catalog[:,0] = field['ra_deg']
        epoch_catalog[:,1] = field['dec_deg']
        epoch_catalog[:,2] = field['peak_Jy']
        epoch_catalog[:,3] = field['rms_Jy']
        epoch_catalog[:,4] = field['snr']
        epoch_catalog[:,5] = phase_center_offset

        # Match the source in the current image to the reference
        match        = match_coordinates_sky(field_coords, reference_coords)
        match_offset = match[1].value 
        match_index  = match[0]

        # Get good matches and update epoch catalog
        good_matches = np.where(match_offset < beam_threshold)[0]

        # Check if there one-to-many matches and remove duplicates
        unique_matches, unique_counts = np.unique(good_matches, return_counts = True)
        if len(good_matches) != len(np.unique(good_matches)):
            msg(f'There exist some one-to-many matches')

        good_matches = good_matches[unique_counts == 1]
        catalog[match_index[good_matches], epoch_index, :] = epoch_catalog[good_matches,:]

    # For each source count the number of epochs with NaN values, the max flux, and min flux, and median SNR
    n_nans   = np.count_nonzero(np.isnan(catalog[:,:,2]), axis = 1)
    max_flux = np.nanmax(catalog[:,:, 2], axis=1)
    min_flux = np.nanmin(catalog[:,:, 2], axis=1)
    
    # Remove sources that do not exist within a minimum number of epochs or are highly variable
    epoch_threshold   = np.amin((epoch_fraction * n_epochs, epoch_min))
    good_sources = np.where((n_nans <= epoch_threshold) & (max_flux < variability_threshold * min_flux))
    catalog = catalog[good_sources[0],:,:]


    # Output the minimum source snr
    min_snr = np.nanmedian(catalog[:,:,4], axis = 1)
    msg('Matching: Minimum Source S/N ratio {:.1f} for {} Sources'.format(np.amin(min_snr), catalog.shape[0]))

    # The array and a json dictionary  containing the index mapping for the array
    index_dict = {'ra': 0, 'dec': 1, 'peak_flux': 2, 'rms': 3, 'snr': 4, 'phase_offset':5}
    
    with open(f'{results}/index_dict_{target_name}.json', 'w') as j:
        j.write(json.dumps(index_dict, indent=4))

    # Save the data to a numpy array
    np.save(f'{results}/field_catalog_{target_name}', catalog)


if __name__ in "__main__":
    main()
