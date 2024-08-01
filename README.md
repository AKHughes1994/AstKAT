# AstKAT

--- 

### What is this?

This routine empirically determines the astrometric error for the time domain with radio interferometric data (Originally Designed for MeerKAT observation). See the file `explanation.pdf' for the motivation behind the routine and a description. 

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

#### [THRESHOLDS]
* variability_threshold — maximum separation between the min/max flux before the source is rejected (Default = 2.0)
* flux_threshold — maximum (fractional) deviation between the island and peak fluxes for a source to be considered a point source (Default = 0.1)
* size_threshold — maximum (fractional) deviation between the source shape and PSF shape for a source to be considered a point source (Default = 0.1)
* target_threshold — the maximum distance in arcseconds for PyBDSF source to be identified as the target (Default 20 arcseconds)
* epoch_min —  minimum absolute number of epochs that a source is detected for it to be included in the catalogue; i.e., if a source is missing from > epoch_min number of observations, it is excluded (Default = 100; NOTE: for a small number of epochs, the bootstrapping will fail for some sources * unless epoch_min is set to 0) 
* epoch_fraction  — minimum relative number of epochs a source for a source to include in the catalogue; i.e., if a source is missing from > epoch_fraction * (total number of observations), it is excluded (Default = 0.25)
* match_threshold — the maximum distance; this is expressed in units of fractional PSF; i.e., point sources are matched if they are within match_threshold * (FWHM BMAJ) (Default=0.33)
* snr_threshold — minimum (median) signal to noise of a source (Default = 4.0)
* ref_index — reference index to perform catalogue matching (Default = 0, the first eepoch)


## [MCMC]
* convergence_threshold — Convergence parameter for MCMC fitting, defined as the average difference between the current and last iteration in units of standard deviation (default = 0.1)
* n_bootstrap — number of bootstrap iterations (Default = 5000; lower this if your computer is running out of memory)
* phase_offset_limit    — maximum distance from phase centre for a source to be used in MCMC fitting (Default = 0.3 -- The inner ~25% of the MeerKAT L-band beam)
* n_iteration           — number of concurrent good iterations before the routine stops (Default = 5)
* min_snr               — minimum signal to noise for a source to be used in MCMC fitting (Default = 0.0 -- No minimum)
* max_snr               — maximum signal to noise for a source to be used in MCMC fitting (Default=1e10 -- No Maximum)

---
### Test suite

The test suite includes two extra scripts to simulate images:

* ``Simulate_Images_Flat_Noise.py'' — this will make images and scatter the positions, enforcing the assumed astrometric error; this is just to make sure things work.
* ``Simulate_Images_Flat_Noise.py'' — this will make images with (correlated) Gaussian noise, applying a user-defined $B$ offset (EXPERIMENTAL as of 2024 July 1)
