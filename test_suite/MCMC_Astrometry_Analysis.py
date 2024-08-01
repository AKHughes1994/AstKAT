from utils import *

def main():


    # Initialize variables containing the paths to the relevant directories
    scripts = os.getcwd()
    files = scripts.replace('scripts', 'files')    
    images = scripts.replace('scripts', 'images')
    results = scripts.replace('scripts', 'results')
    plots = scripts.replace('scripts', 'plots')
    
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
    sep_data_unc = GetNormalizedSeparations(catalog, obs_properties, epoch_corrections = False, n_bootstrap = n_bootstrap) 
    
    # Run MCMC fitting + Plot the results
    msg(f'MCMC: Running the first (Uncorrected) MCMC iteration, Interation 0')
    fit0 = MCMC(sep_data_unc, plot_prefix = f'{plots}/Uncorrected')
    msg(f'MCMC: Error parameters are A = {fit0["A"][0]:.2f} and B = {fit0["B"][0]:.3f} with a chi2(dof) = {fit0["chi2"]/fit0["dof"]:.2f}({fit0["dof"]})\n')

    # Solve for the Per-epoch Corrections and corrected separations
    epoch_corrections = SolveForEpochCorrection(catalog, obs_properties, fit0)
    sep_data = GetNormalizedSeparations(catalog, obs_properties, epoch_corrections = epoch_corrections, n_bootstrap = n_bootstrap) 

    # Solve for an epoch corrected astrometric error
    msg(f'MCMC: Running subsequent (Corrected) MCMC iterations, Interation 1')
    fit = MCMC(sep_data,  plot_prefix = f'{plots}/Corrected')
    msg(f'MCMC: Error parameters are A = {fit["A"][0]:.2f} and B = {fit["B"][0]:.3f} with a chi2(dof) = {fit["chi2"]/fit["dof"]:.2f}({fit["dof"]})\n')

    # Iterate until convergence
    good_epoch = 0  
    i = 2
    msg(f'MCMC: Covergence threshold of delta = {convergence_threshold:.1f}')
    while good_epoch < n_iteration:

        # Solve for an epoch corrected astrometric error   
        sep_data0 = np.copy(sep_data)   
        msg(f'MCMC: Running subsequent (Corrected) MCMC iterations, Interation {i}')    
        epoch_corrections = SolveForEpochCorrection(catalog, obs_properties, fit)
        sep_data = GetNormalizedSeparations(catalog, obs_properties, epoch_corrections = epoch_corrections, n_bootstrap = n_bootstrap) 
        fit = MCMC(sep_data, plot_prefix = f'{plots}/Corrected')
        msg(f'MCMC: Error Parameters are A = {fit["A"][0]:.2f} and B = {fit["B"][0]:.3f} with a chi2(dof) = {fit["chi2"]/fit["dof"]:.2f}({fit["dof"]})')

        # Caclulate convergence parmeter
        delta = np.mean(abs((sep_data[:,1] - sep_data0[:,1])) / np.sqrt(sep_data[:,2] ** 2 + sep_data0[:,2] ** 2))
        
        # Check if good or bad iteration
        if delta < convergence_threshold: 
            good_epoch += 1
            msg(f'MCMC: Good Iteration, delta = {delta:.4f}, Consecutive good iterations = {good_epoch}')
        else:
            good_epoch = 0
            msg(f'MCMC: Bad Iteration, delta = {delta:.4f}, Resetting good iteration counter')
        
        # Iteration counter
        print(' ')
        i += 1

    msg('MCMC: Convergence reached Plotting and Saving fits')
    PlotResults(sep_data_unc, fit0, sep_data, fit, plot_prefix = f'{plots}/{target_name}', ylim = (0.005,0.3), xlim = (4.5,330))   
    
    # Save the fits and epoch_corrections to observation properties dictionary
    for key in obs_properties:
        if type(obs_properties[key]) is np.ndarray:
            obs_properties[key] = obs_properties[key].tolist()

    obs_properties['Uncorrected_fit'] = fit0
    obs_properties['Corrected_fit']   = fit
    obs_properties['Epoch_RA_corr_deg']   = epoch_corrections[0,:].tolist()
    obs_properties['Epoch_Dec_corr_deg']  = epoch_corrections[1,:].tolist()

    with open(f'{results}/observation_properties.json', 'w') as j:
        json.dump(obs_properties, j, indent = 4)


if __name__ in "__main__":
    main()
