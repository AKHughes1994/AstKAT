#!/usr/bin/env bash
# AstKAT end-to-end simulation: images, pipeline, then a moving target recovered with target.py.
# Nothing varies between epochs except the noise realisation and the beam: constant field rms and target flux density.
set -euo pipefail          # stop at the first failing step

# Run in an empty directory, with astkat installed in the active Python environment (see README)

NAME=sim                   # project name
MODE=noise             # noise, flat or catalogue (see the simulate calls below)
IMAGES=images              # where simulate.py writes the epoch images
N_EPOCHS=100               # number of epoch images
N_TARGET=20                # epochs the target is simulated in
A=1.0                      # flat and catalogue: injected SNR term of the position error at SNR 1, beam units
BETA=1.0                   # flat and catalogue: injected SNR exponent, error A SNR^-BETA (1 = textbook 1 / SNR)
SNR_BREAK=15               # flat and catalogue: injected break SNR of the SNR term
SNR_GAMMA=0                # flat and catalogue: slope -BETA - SNR_GAMMA below SNR_BREAK (0 = no break)
B=0.01                     # injected position floor, fraction of the beam sigma per axis (every mode) ...
B_ABSOLUTE=false           # ... or, true, B in arcsec on both axes; the fit then uses an absolute B too
K=1.3                      # flat and catalogue: noise on each peak, in rms
FLUX_SYS=0.05              # every mode: spread of the per-epoch flux scale g_e, shared by all sources (f_sys)
FLUX_FRAC=0.02             # every mode: fractional error on each source's peak in each epoch (f_frac)
FLUX="--flux-sys $FLUX_SYS --flux-frac $FLUX_FRAC"
OUTLIER_FRACTION=0.002     # flat and catalogue: chance each position is an outlier (~100 of 500 sources x 100 epochs)
OUTLIER_SCALE=6            # flat and catalogue: outlier width over the inlier width
RMS_VARIATION=0.1          # noise mode: each epoch's rms is normal about the field rms with this fractional sigma
                           # (0.1 keeps 99.7% of epochs within 0.7-1.3); flat and catalogue keep a unit rms
# Seed of every simulate call: an integer, so runs repeat; random, a fresh seed per call (-R, printed and saved
# with the truth); or default, simulate.py's fixed default seed
#SEED_VALUE=random
SEED_VALUE=3329369676
SEED=$(case "$SEED_VALUE" in default) ;; random) echo -R ;; *) echo "--seed $SEED_VALUE" ;; esac)
FLOOR="-b $B $([ "$B_ABSOLUTE" = true ] && echo --b-absolute || true)"

# Fresh run: remove previous products
rm -rf astkat_files pybdsf_files results plots logs "$IMAGES" sim_target

# 1. Epoch images, one mode
case "$MODE" in
    # Realistic images: beam-correlated noise; the SNR error comes from PyBDSF's fitting, B is injected
    noise)     astkat simulate --mode noise -i "$N_EPOCHS" $FLOOR $FLUX --rms-variation "$RMS_VARIATION" \
                   -o "$IMAGES" $SEED ;;
    # Noise-free images with every error injected (A, B, k, f_sys, f_frac); tests extraction, matching and the fit
    flat)      astkat simulate --mode flat -i "$N_EPOCHS" -a "$A" --beta "$BETA" $FLOOR -k "$K" \
                   $FLUX --snr-break "$SNR_BREAK" --snr-gamma "$SNR_GAMMA" \
                   --rms-variation "$RMS_VARIATION" --outlier-fraction "$OUTLIER_FRACTION" \
                   --outlier-scale "$OUTLIER_SCALE" -o "$IMAGES" $SEED ;;
    # The model's own data written straight to the matched catalogue; tests the fit alone
    catalogue) astkat simulate --mode catalogue -i "$N_EPOCHS" -a "$A" --beta "$BETA" $FLOOR -k "$K" \
                   $FLUX --snr-break "$SNR_BREAK" --snr-gamma "$SNR_GAMMA" \
                   --rms-variation "$RMS_VARIATION" --outlier-fraction "$OUTLIER_FRACTION" \
                   --outlier-scale "$OUTLIER_SCALE" -o "$IMAGES" $SEED ;;
    *)         echo "Unknown MODE $MODE (noise, flat or catalogue)"; exit 1 ;;
esac

# 2. Project set-up; marked simulated, so no aberration correction. Noise and flat build the stacked deep image,
#    the matching reference; catalogue copies its catalogue into the project instead
astkat init -i "$IMAGES" -n "$NAME" --simulated

# No outliers injected (noise mode, or a zero outlier fraction): fit without the outlier mixture
if [ "$MODE" = noise ] || awk "BEGIN { exit !($OUTLIER_FRACTION == 0) }"; then
    sed -i 's/enable_outlier_mixture: *true/enable_outlier_mixture: false/' astkat_files/astkat.yaml
fi

# Flat and catalogue: a flux term not injected is not fitted. Noise mode fits both, as PyBDSF's own peak
# errors (beam-dependent biases shared by an epoch's sources, errors growing with flux) can show up in them.
if [ "$MODE" != noise ]; then
    if awk "BEGIN { exit !($FLUX_SYS == 0) }"; then
        sed -i 's/fit_flux_systematic: *true/fit_flux_systematic: false/' astkat_files/astkat.yaml
    fi
    if awk "BEGIN { exit !($FLUX_FRAC == 0) }"; then
        sed -i 's/fit_flux_fractional: *true/fit_flux_fractional: false/' astkat_files/astkat.yaml
    fi
fi

# Flat and catalogue: the SNR term is the simplest form that holds the injected one: fixed (beta 1, no break),
# power (beta free, no break) or broken (beta 1, a break). With both injected no form fits exactly; broken is kept.
if [ "$MODE" != noise ]; then
    if awk "BEGIN { exit !($SNR_GAMMA == 0 && $BETA == 1) }"; then
        TERM_FIT=fixed
    elif awk "BEGIN { exit !($SNR_GAMMA == 0) }"; then
        TERM_FIT=power
    else
        TERM_FIT=broken
        awk "BEGIN { exit !($BETA != 1) }" && echo "BETA and SNR_GAMMA both injected: no SNR term fits exactly; fitting broken"
    fi
    sed -i "s/snr_term: *[a-z]*/snr_term: $TERM_FIT/" astkat_files/astkat.yaml
fi

# Flat and catalogue: no injected floor, no fitted floor. Noise mode fits it, as PyBDSF can add its own
if [ "$MODE" != noise ] && awk "BEGIN { exit !($B == 0) }"; then
    sed -i 's/enable_systematic_floor: *true/enable_systematic_floor: false/' astkat_files/astkat.yaml
fi

# An absolute injected floor is fitted as an absolute floor
if [ "$B_ABSOLUTE" = true ]; then
    sed -i 's/systematic_floor_as_fraction: *true/systematic_floor_as_fraction: false/' astkat_files/astkat.yaml
fi

# 3. Source finding, point-source selection and cross-matching; a catalogue simulation is already matched
if [ "$MODE" != catalogue ]; then
    astkat extract
    astkat filter
    astkat match
fi

# 4. Joint fit (-y replaces any existing fit), then translations for every epoch
astkat fit -y
astkat offsets         # compares with the injected translations automatically

# 5. Moving target with a constant flux density, then recovered with target.py
astkat simulate --mode target -i "$N_TARGET" --v0 3 --decel 0.04 --target-flux-scatter 0 $SEED
astkat target sim_target/target_trajectory.txt

# Recovered: results/target_target_trajectory.csv; truth: sim_target/target_trajectory_truth.csv
