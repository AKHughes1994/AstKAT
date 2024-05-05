from utils import *

def main():

    # Initialize variables pointing to the relevant directories
    scripts = os.getcwd()
    files      = scripts.replace('scripts', 'files')
    images = scripts.replace('scripts', 'images')
    results = scripts.replace('scripts', 'results')
    plots    = scripts.replace('scripts', 'plots')
    
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
    ra_extents        = []
    dec_extents       = []

    # Iterate through the images extracting the relevant parameters
    image_names = sorted(glob.glob(f'{images}/*image.fits'))
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

        # Calculate the extents (this is a place holder) -- Need to figure out best spatial scale
        a = bmaj 
        b = bmin 
        p = np.radians(bpa)
        ra_extent   =  np.sqrt(a  ** 2 * np.sin(p) ** 2  + b ** 2 * np.cos(p) ** 2) 
        dec_extent = np.sqrt(a ** 2 * np.cos(p) ** 2  + b ** 2 * np.sin(p) ** 2)

        #a = 0.5 * bmaj 
        #b = 0.5 * bmin 
        #p = np.radians(bpa)
        #dec_extent   =  a * b / np.sqrt(a  ** 2 * np.sin(p) ** 2  + b ** 2 * np.cos(p) ** 2)    
        #ra_extent     = a * b / np.sqrt(a ** 2 * np.cos(p) ** 2  + b ** 2 * np.sin(p) ** 2)

        # Append to lists
        image_identifiers.append(image_identifier)
        obs_isots.append(obs_isot)
        bmajs.append(bmaj)
        bmins.append(bmin)
        bpas.append(bpa)
        freqs.append(freq)
        ra_extents.append(ra_extent)
        dec_extents.append(dec_extent)
    
    # Sort in time
    sorted_index = np.argsort(obs_isots)

    # Initialize the monitoring dictionary and save
    obs_properties = {
                                'image_identifier': image_identifiers,
                                'obs_isot': obs_isots,
                                'bmaj_deg': bmajs,  
                                'bmin_deg': bmins,
                                'bpa_deg': bpas,
                                'freq_GHz': freqs,
                                'ra_extent_deg': ra_extents,
                                'dec_extent_deg': dec_extents,
                              }

    for key in obs_properties.keys():
        obs_properties[key] = (np.array(obs_properties[key])[sorted_index]).tolist()

    with open(f'{results}/observation_properties.json', 'w') as j:
        j.write(json.dumps(obs_properties, indent=4))

    msg('Extracting: Running PyBDSF')    

    # Run each image through PyBDSF
    for image_name, obs_isot in zip(image_names[-1:],obs_isots[-1:]): #All images

        # PyBDSF run
        img = bdsf.process_image(image_name,
                                                  thresh_pix=snr_threshold,
                                                  fix_to_beam=fix_to_beam, 
                                                  group_by_isl = True,
                                                  #minpix_isl=5.0,  
                                                  rms_box=(300,100)) #trim_box=(7750,8250,7750,8250))

        # Save FITS catalog
        img.write_catalog(format='fits', 
                                    catalog_type=output_file_type, 
                                    outfile = f'{files}/total_{obs_isot}_{target_name}.fits', 
                                    clobber=True)

if __name__ in "__main__":
    main()
