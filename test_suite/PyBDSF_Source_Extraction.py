from utils import *

def main():

    # Initialize variables pointing to the relevant directories
    scripts = os.getcwd()
    files  = scripts.replace('test_suite', 'files')
    images = scripts.replace('test_suite', 'images')
    results = scripts.replace('test_suite', 'results')
    plots   = scripts.replace('test_suite', 'plots')
    
    for directory in [files, results, plots]:
        if not os.path.exists(directory):
            msg(f'Extracting: Initializing Directory {directory}')
            os.makedirs(directory)

    # Read in the configuration file
    cfg = configparser.ConfigParser()
    cfg.read(f'{scripts}/config.ini')
    
    # Intialize PyBDSF Parameters
    target_name = str(cfg['TARGET']['name'])
    fix_to_beam = str_to_bool(cfg['PYBDSF']['fix_to_beam'])
    snr_threshold =  float(cfg['PYBDSF']['snr_threshold'])
    output_file_type   = str(cfg['PYBDSF']['output_file_type'])

    # Initialize arrays
    image_identifiers = []
    obs_isots         = []
    bmajs             = []
    bmins             = []
    bpas              = []
    freqs             = []

    # Iterate through the images extracting the relevant parameters
    image_names = sorted(glob.glob(f'{images}/*.fits'))
    for image_name in image_names:

        # Save identifier
        image_identifier =image_name.split('images/')[-1]
    
        # Load in the header and pull out relevant properties
        header = fits.getheader(image_name)
        obs_isot               = header['DATE-OBS']
        bmaj                   = header['BMAJ']
        bmin                   = header['BMIN']
        bpa                    = header['BPA']
        freq                   = header['CRVAL3'] / 1e9

        # Confine the bpa to the first two quadrants [-pi/2, pi/2]
        if bpa > 180.0: 
            bpa -= 360.0
        if bpa > 90.0: 
            bpa -= 180.0

        if bpa < -180.0:
            bpa += 360.0
        if bpa < -90.0:
            bpa += 180.0

        # Calculate the extents (this is a place holder) -- Need to figure out best spatial scale
        a = bmaj 
        b = bmin 
        p = bpa

        # Append to lists
        image_identifiers.append(image_identifier)
        obs_isots.append(obs_isot)
        bmajs.append(bmaj)
        bmins.append(bmin)
        bpas.append(bpa)
        freqs.append(freq)

    # Sort in time
    sorted_index = np.argsort(obs_isots)

    # Initialize the monitoring dictionary and save
    obs_properties = {
                                'image_identifier': image_identifiers,
                                'obs_isot': obs_isots,
                                'bmaj_deg': bmajs,  
                                'bmin_deg': bmins,
                                'bpa_rad': np.radians(bpas),
                                'bpa_deg': bpas,
                                'freq_GHz': freqs,
                              }

    for key in obs_properties.keys():
        obs_properties[key] = (np.array(obs_properties[key])[sorted_index]).tolist()

    with open(f'{results}/observation_properties.json', 'w') as j:
        j.write(json.dumps(obs_properties, indent=4))

    msg('Extracting: Running PyBDSF')    

    # Run each image through PyBDSF
    for image_name, obs_isot in zip(image_names[:],obs_isots[:]): #All images
        msg(f'Extracting: {obs_isot}')

        # Check if its flat noise images
        if '_Flat_' in image_name:
            rms_map = False
            rms_value = 1.0
            thresh_isl = 1.0
            minpix_isl = 3.0
        else:
            rms_map = None
            rms_value = None
            thresh_isl = 3.0
            minpix_isl = None

        # PyBDSF run
        img = bdsf.process_image(image_name,
                            thresh_pix = snr_threshold,
                            thresh_isl = thresh_isl,
                            fix_to_beam = fix_to_beam, 
                            group_by_isl = True,
                            rms_map = rms_map,
                            rms_value = rms_value,
                            minpix_isl = minpix_isl)
                            #rms_box=(300,100))
                            #trim_box=(7750,8250,7750,8250))

        # Save FITS catalog
        img.write_catalog(format='fits', 
                                    catalog_type=output_file_type, 
                                    outfile = f'{files}/total_{obs_isot}_{target_name}.fits', 
                                    clobber=True)

if __name__ in "__main__":
    main()
