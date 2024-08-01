# AstKAT

--- 

### What is this?

A routine to empirically determine the astrometric error for the time domain with radio interferometric data (Originally Designed for MeerKAT observation). See the file `explanation.pdf' for the motivation behind the routine and a description. 

---
### The Controlling Parameters are in the config.ini file

#### [TARGET]

* name — Name of the source/target (used for naming of output files plots)

#### [POSITIONS]
* target_ra — Right ascension of target: in astropy SkyCoord format "hms"
* target_dec — Declination of target: in astropy SkyCoord format "dms"
* phase_center_ra — Right ascension of phase/pointing centre: in astropy SkyCoord format "hms"
* phase_center_dec — Declination of of phase/pointing centre: in astropy SkyCoord format "dms"

#### [PYBDSF]
* snr_threshold — minimum signal-to-nouse (SNR) ratio for PyBDSF to identify the source (Default = 3.5). Make it ~0.5 less than the desired minimum source SNR
* fix_to_beam — decided whether PyBDSF fits all of the Gaussian components to be the shape of the PSF (Default = True)
* output_file_type — a type of output catalogue (Default='srl'), `source list' see https://pybdsf.readthedocs.io/en/latest/ for more information

##### [THRESHOLDS]
**variability_threshold** = maximum separation between the min/max flux before the source is rejected; i.e., max_flux <  variability_threshold * min_flux (Default = 2.0)
**flux_threshold** = maximum factor separating the island and peak fluxes before the source is rejected; i.e., abs(peak_flux/island_flux - 1.0) <  flux_threshold (Default = 0.25)
**size_threshold** = maximum factor separating the component size and beam size before the source is rejected; i.e., abs(comp_size/beam_size - 1.0) <  size_threshold (Default = 0.25)
**target_threshold** = maximum distance in asecs that PyBDSF component is for it to be identified as the source (Default 20 arcseconds)
**epoch_min** = minimum number of epochs for a source to be included in the catalog (Default = 100; **NOTE** for a small number of epochs, the bootstrapping will fail for some sources unless epoch_min is set to 0) 
**epoch_fraction** = maximum fraction of epochs a source can not be included in the catalog; n_epochs > epoch_fraction * number of epochs (Default = 0.25)
**match_threshold** = maximum distance in asecs for a component to be matched (Default 5 arcseconds)
**snr_threshold** = minimum signal to noise (Default 4.0)

## [MCMC]
**convergence_threshold** = Convergence parameter for MCMC fitting, defined as the average difference between the current and last iteration in units of standard deviation (default = 0.1)
**n_bootstrap**           = number of bootstrap iterations (Default = 500)
**phase_offset_limit**    = maximum distance from phase center for a source to be used in MCMC fitting (Default = 0.3 -- The inner ~50% of the MeerKAT L-band beam)
**min_snr**               = minimum signal to noise for a source to be used in MCMC fitting (Default = 5.0)
**max_snr**               = maximum signal to noise for a source to be used in MCMC fitting (Default=1e10 -- No Maximum)
