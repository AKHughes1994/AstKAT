from utils import *

def main():

    # Initialize variables pointing to the relevant directories
    scripts = os.getcwd()
    files = scripts.replace('scripts', 'files')    
    images = scripts.replace('scripts', 'images')
    results = scripts.replace('scripts', 'results')
    plots = scripts.replace('scripts', 'plots')

    # Read in the configuration file
    cfg = configparser.ConfigParser()
    cfg.read(f'{scripts}/config.ini')

    # Initialize MCMC variables specified in config file 
    target_name = str(cfg['TARGET']['name'])
    convergence_threshold = float(cfg['MCMC']['convergence_threshold'])
    n_bootstrap = int(cfg['MCMC']['n_bootstrap'])
    phase_offset_limit = float(cfg['MCMC']['phase_offset_limit'])
    min_snr = float(cfg['MCMC']['min_snr'])
    max_snr = float(cfg['MCMC']['max_snr'])

    msg(f'MCMC: Convergence threshold chosen to be {convergence_threshold}')

    # Load in the data catalog
    catalog = np.load(f'{files}/field_catalog_{target_name}.npy')
    total_std = BootstrapPositions(catalog, n_bootstrap=n_bootstrap)
   
    # Apply filters and re-boostrap the data -- source for fitting
    filter_catalog  = FilterData(catalog, snr_low = min_snr, snr_high = max_snr, phase_offset_thresh = phase_offset_limit)
    filter_std         = BootstrapPositions(filter_catalog, n_bootstrap=n_bootstrap)

    # Run MCMC to get the fits
    msg(f'MCMC: Intial uncorrected MCMC fit')

    # ~~~~~~~~~~ #
    # Righ Acension  #
    # ~~~~~~~~~~ #

    plot_prefix = f'{plots}/RA_uncorrected'
    uncorrected_fit_ra   =  MCMC(filter_std[[0,1,2]], plot_prefix)
    A, B, chi2, dof = uncorrected_fit_ra['A'], uncorrected_fit_ra['B'], uncorrected_fit_ra['chi2'], uncorrected_fit_ra['dof']
    PlotResults(filter_std[[0,1,2]], uncorrected_fit_ra, plot_prefix)

    msg(f'MCMC: Uncorrected RA fit - A = {A[0]:.3f}(-{A[1]:.3f})(+{A[2]:.3f}) and B = {B[0]:.3f}(-{B[1]:.3f})(+{B[2]:.3f}), chi2 = {chi2:.1f}/{dof:.0f}')

    # ~~~~~~~~ #
    # Declination  #
    # ~~~~~~~~ #

    plot_prefix = f'{plots}/Dec_uncorrected'
    uncorrected_fit_dec   =  MCMC(filter_std[[0,3,4]], plot_prefix)
    A, B, chi2, dof = uncorrected_fit_dec['A'], uncorrected_fit_dec['B'], uncorrected_fit_dec['chi2'], uncorrected_fit_dec['dof']
    PlotResults(filter_std[[0,3,4]], uncorrected_fit_dec, plot_prefix)

    msg(f'MCMC: Uncorrected Dec fit - A = {A[0]:.3f}(-{A[1]:.3f})(+{A[2]:.3f}) and B = {B[0]:.3f}(-{B[1]:.3f})(+{B[2]:.3f}), chi2 = {chi2:.1f}/{dof:.0f}')

    # Solve for per-epoch corrections
    corrections = SolveForEpochCorrection(filter_catalog, uncorrected_fit_ra, uncorrected_fit_dec)

    # Append fits to observation_properties
    obs_properties =  load_json(f'{results}/observation_properties.json', numpize = False)
    obs_properties['Uncorrected_RA_fit'] = uncorrected_fit_ra
    obs_properties['Uncorrected_Dec_fit'] = uncorrected_fit_dec
    obs_properties['RA_epoch_correction_deg'] = (np.array(corrections[0]) * np.array(obs_properties['ra_extent_deg'])).tolist()
    obs_properties['RA_epoch_correction_err'] =(np.array(corrections[1]) * np.array(obs_properties['ra_extent_deg'])).tolist()
    obs_properties['Dec_epoch_correction_deg'] = (np.array(corrections[2]) * np.array(obs_properties['dec_extent_deg'])).tolist()
    obs_properties['Dec_epoch_correction_err'] = (np.array(corrections[3]) * np.array(obs_properties['dec_extent_deg'])).tolist() 

    with open(f'{results}/observation_properties.json', 'w') as j:
        j.write(json.dumps(obs_properties, indent=4))

if __name__ in "__main__":
    main()
