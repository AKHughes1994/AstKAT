#!/bin/bash

# Clear old results/plots directory
rm ../plots/* ../results/*

# Run each image through PyBDSF
python3 PyBDSF_Source_Extraction.py

# Filter sources from PyBDSF catalogs
python3 PyBDSF_Source_Filtering.py

# Match the sources across all epochs
python3 PyBDSF_Source_Matching.py

# Run the MCMC Astrometric Analysis
python3 MCMC_Astrometry_Analysis.py
