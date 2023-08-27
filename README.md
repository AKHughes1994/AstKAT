# AstKAT
Astrometry routine for radio interferometric data (Originally Designed for MeerKAT)<br>

To run the program, all that is needed is to put your images in `images/` and then `cd` into the `scripts/` directory and run the shell script `run_all_scripts.sh`<br>

# Config Options
## [Source]
name = Just the name of the source (used for making plots)<br>
## [POSITIONS]
source_ra/dec = the known position of your source <br>
phase_center_ra/dec = the phase pointing center of the instrument (often offset from the source position by ~15 asec)<br>

## [PYBDSF]
snr_threshold = minimum signal-to-nouse (snr) ratio for PyBDSF to identify the source (Default = 4.0)<br>
fix_to_beam = decided whether PyBDSF fits all of the Gaussian components to be the shape of the beam (Default = True)<br>
output_file_type = type of output catalog (Default='srl') look at PyBDSF documentation for more details<br>

## [THRESHOLDS]
variability_threshold = maximum separation between the min/max flux before the source is rejected; i.e., max_flux <  variability_threshold * min_flux (Default = 2.0)<br>
flux_threshold = maximum factor separating the island and peak fluxes before the source is rejected; i.e., abs(peak_flux/island_flux - 1.0) <  flux_threshold (Default = 0.25)<br>
size_threshold = maximum factor separating the component size and beam size before the source is rejected; i.e., abs(comp_size/beam_size - 1.0) <  size_threshold (Default = 0.25)<br>
target_threshold = maximum distance in asecs that PyBDSF component is for it to be identified as the source (Default 20 arcseconds)<br>
epoch_min = minimum number of epochs for a source to be included in the catalog (Default = 100; **NOTE** for a small number of epochs, the bootstrapping will fail for some sources unless epoch_min is set to 0) <br>
epoch_fraction = maximum fraction of epochs a source can not be included in the catalog; n_epochs > epoch_fraction * number of epochs (Default = 0.25)<br>
match_threshold = maximum distance in asecs for a component to be matched (Default 5 arcseconds)<br>
snr_threshold = minimum signal to noise (Default 4.0)<br>

## [MCMC]
convergence_threshold = Convergence parameter for MCMC fitting, defined as the average difference between the current and last iteration in units of standard deviation (default = 0.1)<br>
n_bootstrap           = number of bootstrap iterations (Default = 500)<br>
phase_offset_limit    = maximum distance from phase center for a source to be used in MCMC fitting (Default = 0.3 -- The inner ~50% of the MeerKAT L-band beam)<br>
min_snr               = minimum signal to noise for a source to be used in MCMC fitting (Default = 5.0)<br>
max_snr               = maximum signal to noise for a source to be used in MCMC fitting (Default=1e10 -- No Maximum)<br>
