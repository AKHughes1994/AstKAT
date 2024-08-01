from utils import *

def main():


    # Initialize variables containing the paths to the relevant directories
    scripts = os.getcwd()
    files  = scripts.replace('test_suite', 'files')
    images = scripts.replace('test_suite', 'images')
    results = scripts.replace('test_suite', 'results')
    plots   = scripts.replace('test_suite', 'plots')
    
    # Load the obs_properties
    obs_properties = load_json(f'{results}/observation_properties.json', numpize = True)

    # Read in the configuration file
    cfg = configparser.ConfigParser()
    cfg.read(f'{scripts}/config.ini')

    # Load in Target name
    target_name = str(cfg['TARGET']['name'])
    
    # Catalog filtering parameters
    min_snr = float(cfg['MCMC']['min_snr'])
    max_snr = float(cfg['MCMC']['max_snr'])
    phase_offset_limit = float(cfg['MCMC']['phase_offset_limit'])

    # MCMC fitting parameters
    convergence_threshold = float(cfg['MCMC']['convergence_threshold'])
    n_bootstrap = int(cfg['MCMC']['n_bootstrap'])
    n_iteration = int(cfg['MCMC']['n_iteration'])

    # Load in the numpy array containg the full catalog
    catalog = np.load(f'{results}/field_catalog_{target_name}.npy')

    # Truncate the array according to the filters
    catalog  = FilterData(catalog, 
                        snr_low = min_snr, 
                        snr_high = max_snr, 
                        phase_offset_thresh = phase_offset_limit)


    msg(f'MCMC: Number of good sources = {catalog.shape[0]}')

    # Get the seperation vs. S/N data for MCMC fitting    
    sep_data = GetNormalizedSeparations(catalog, obs_properties, epoch_corrections = False, n_bootstrap = n_bootstrap) 

    # Run MCMC fitting + Plot the results
    msg(f'MCMC: Running the first (Simulated) MCMC fit')
    fit = MCMC(sep_data, plot_prefix = f'{plots}/Uncorrected')

    msg(f'MCMC: Error parameters are A = {fit["A"][0]:.2f} and B = {fit["B"][0]:.3f} with a chi2(dof) = {fit["chi2"]/fit["dof"]:.2f}({fit["dof"]})\n')
    print(fit)

    msg('MCMC: Plotting Fit')
    PlotResults(sep_data, fit, plot_prefix = f'{plots}/Uncorrected')

if __name__ in "__main__":
    main()
