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
    
    # Initialize relevant parameters from the Config Files
    target_name       = str(cfg['TARGET']['name'])
    target_ra             = str(cfg['POSITIONS']['target_ra'])
    target_dec           = str(cfg['POSITIONS']['target_dec'])
    flux_threshold     = float(cfg['THRESHOLDS']['flux_threshold'])
    size_threshold     = float(cfg['THRESHOLDS']['size_threshold'])
    target_threshold = float(cfg['THRESHOLDS']['target_threshold'])

    # SkyCoord objecy contain source coordinates
    source_coords = SkyCoord(f'{target_ra} {target_dec}', frame='icrs')
    
    # Load in observation parameters 
    obs_properties=  load_json(f'{results}/observation_properties.json')

    # Iterate through the files -- separating target and field sources
    for k,obs_isot in enumerate(obs_properties['obs_isot'][:]):

        msg(f'Filtering: {obs_isot}')

        # Open PyBDSF fits catalog output
        fname = glob.glob(f'{files}/total*{obs_isot}*.fits')[0] #get filename
        im = fits.open(fname)

        # Initialize dictionaries to store  revelant parameters
        target = {'ra_deg' : [], 'dec_deg': [], 'peak_Jy':[], 'rms_Jy':[], 'ra_extent_deg':[], 'dec_extent_deg':[]}
        field    = {'ra_deg' : [], 'dec_deg': [], 'peak_Jy':[], 'rms_Jy':[], 'ra_extent_deg':[], 'dec_extent_deg':[]}

        # Extract data from FITS files
        ra                 = im[1].data['RA']
        dec              = im[1].data['DEC']
        source_flux = im[1].data['Total_flux']              # Flux of the source (Jy)
        rms_err       = im[1].data['isl_rms']                  # Get RMS value from background RMS-map local to the source
        peak_flux    = im[1].data['Peak_flux']             #  Flux Density of the peak (Jy/beam)
        island_flux  = im[1].data['Isl_Total_flux']       # Total Flux in the island (Jy)
        major           = im[1].data['Maj_img_plane']   # Major Axis of the Source (deg)
        minor           = im[1].data['Min_img_plane']   # Minor Axis of the Source (deg)
        stype           = im[1].data['S_Code']
        ra_extent    = np.ones(len(ra)) * obs_properties['ra_extent_deg'][k]
        dec_extent  = np.ones(len(dec)) * obs_properties['dec_extent_deg'][k]

        # Make SkyCoord object from coordinates
        allCoords = SkyCoord(ra = ra * u.deg, dec = dec * u.deg, frame='icrs')
        separations = allCoords.separation(source_coords)

        # Find what indexes are target vs.field sources indexes -- Within [target_threshold] arcsec
        target_index = np.where(separations.arcsec < target_threshold) 

        ##################################################
        # Find the Field sources, enforcing point-source conditions                     #
        #   Condition 1: (Default) Peak flux within 25% of the  island_flux            #
        #   Condition 2: (Default) Source shape within 25% of the beam shape   #
        #   Condition 3: It is not the target source                                                   #
        ## ###############################################

        # Define conditions     
        bmaj = obs_properties['bmaj_deg'][k]
        bmin = obs_properties['bmin_deg'][k]

        flux_condition   = abs(peak_flux/island_flux - 1.0) 
        bmaj_condition = abs(major/bmaj - 1.0)
        bmin_condition = abs(minor/bmin - 1.0)

        # Point sources
        field_index = np.where((flux_condition < flux_threshold) & (bmaj_condition < size_threshold) & (bmin_condition < size_threshold))[0] 

        # Remove target(s) from the field indexes
        field_index = np.setdiff1d(field_index,target_index) 

        # Fill target dictionary
        target['ra_deg'], target['dec_deg'], target['peak_Jy'], target['rms_Jy'], target['ra_extent_deg'], target['dec_extent_deg'] = ra[target_index].tolist(), dec[target_index].tolist(), peak_flux[target_index].tolist(), rms_err[target_index].tolist(), ra_extent[target_index].tolist(), dec_extent[target_index].tolist()

        # Fill field source dictionary
        field['ra_deg'], field['dec_deg'], field['peak_Jy'], field['rms_Jy'], field['ra_extent_deg'], field['dec_extent_deg'] = ra[field_index].tolist(), dec[field_index].tolist(), peak_flux[field_index].tolist(), rms_err[field_index].tolist(), ra_extent[field_index].tolist(), dec_extent[field_index].tolist()


        # Save the individual epoch dictionaries -- Field sources
        with open(f'{files}/field_{obs_isot}_{target_name}.json', 'w') as j:
            j.write(json.dumps(field, indent=4))

        # Target sources
        with open(f'{files}/target_{obs_isot}_{target_name}.json','w') as j:
            j.write(json.dumps(target, indent=4))

if __name__ in "__main__":
    main()
