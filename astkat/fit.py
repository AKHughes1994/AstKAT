"""
AstKAT joint astrometric fit.

Fits every position measurement of every source at once,

    x_je = mu_j + tau_e + noise_je,

where mu_j is the true position of source j, tau_e the translation of epoch
e, and the noise has a beam-shaped covariance set by the source's
signal-to-noise ratio plus a systematic floor, with an outlier component.
Positions are relative to the phase centre in the MeerKAT frame; the
translations sum to zero over epochs.

Usage
    python fit.py        # asks before replacing an existing fit
    python fit.py -y     # re-fit without asking
    python fit.py -n     # keep the existing fit and only redo the figures
"""

import os

# PyTensor's elementwise fusion can miscompile the mixture likelihood under Numba on some
# machines. Turning it off costs little speed, and must happen before PyTensor is imported.
if 'optimizer_excluding=fusion' not in os.environ.get('PYTENSOR_FLAGS', ''):
    os.environ['PYTENSOR_FLAGS'] = ','.join(filter(None, [os.environ.get('PYTENSOR_FLAGS'),
                                                          'optimizer_excluding=fusion']))

import argparse
import importlib.util
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import bottleneck as bn
import arviz as az
import pymc as pm
import pytensor.tensor as pt
import yaml
from astropy.io import fits
from astropy.time import Time
from pytensor.graph.basic import Variable
from scipy.stats import chi2 as chi2_distribution, norm

try:
    from .utils import (msg, title, parse_phase_center, column_index, CATALOG_COLUMNS, S_CODES, AMBIGUOUS,
                        epoch_frequencies, frequency_bands, DEFAULT_SEED,
                        apply_aberration_correction_from_config, load_epoch_times, is_simulated,
                        aberration_scale, observatory_location, SIMULATION_TRUTH_FILE, ARCSEC_PER_RAD,
                        tangent_plane_arcsec)
except ImportError:
    from utils import (msg, title, parse_phase_center, column_index, CATALOG_COLUMNS, S_CODES, AMBIGUOUS,
                       epoch_frequencies, frequency_bands, DEFAULT_SEED,
                       apply_aberration_correction_from_config, load_epoch_times, is_simulated,
                       aberration_scale, observatory_location, SIMULATION_TRUTH_FILE, ARCSEC_PER_RAD,
                       tangent_plane_arcsec)


CONFIG_FILE = Path('astkat_files') / 'astkat.yaml'
RESULTS_DIR = Path('results')
RESULTS_FILE = RESULTS_DIR / 'fit_results.pkl'
TRACE_FILE = RESULTS_DIR / 'fit_trace.nc'
TRANSLATIONS_FILE = RESULTS_DIR / 'fit_translations.npz'
POSITIONS_FILE = RESULTS_DIR / 'fit_source_positions.npz'

MIN_EPOCHS_PER_SOURCE = 5        # sources seen in fewer epochs carry no information on the noise
LEFT_OUT_SNR_EDGES = (0.0, 5.0, 10.0, 30.0, 100.0, 1000.0, np.inf)   # detection SNR bins of the C or M log
# Free shape parameters of each SNR term (MCMC.snr_term).
SNR_SHAPES = {'fixed': (), 'power': ('beta',), 'broken': ('snr_break', 'snr_gamma')}
# Sampled scalars as named in the log; FLUX_SCALARS belong to the flux density likelihood.
SCALAR_NAMES = {'sigma_tau': 'sigma_tau', 'A': 'A', 'beta': 'beta', 'log_snr_break': 'SNR_b', 'snr_gamma': 'gamma',
                'B': 'B', 'alpha': 'alpha', 'outlier_fraction_unit': 'f_out', 'log_outlier_scale': 'lambda_out',
                'peak_noise': 'k', 'flux_systematic': 'f_sys', 'flux_fractional': 'f_frac'}
FLUX_SCALARS = ('peak_noise', 'flux_systematic', 'flux_fractional')
SCALAR_PARAMETERS = ('A', 'beta', 'snr_break', 'snr_gamma', 'B', 'alpha', 'outlier_fraction', 'outlier_scale',
                     'sigma_tau', 'peak_noise', 'flux_systematic',
                     'flux_fractional')
CENSOR_MARGIN = 0.5               # least gap between PyBDSF's threshold and the censoring threshold, in rms
FLUX_OFFSET_FACTOR = 3.0         # a fitted flux this factor from its source's median detected peak is flagged
SUMMARY_TEXT_FILE = 'fit_summary.txt'
SNR_BIN_DEX = 0.1                # width of the SNR bins for the residual-versus-SNR diagnostic
MIN_PER_SNR_BIN = 20             # sparser bins are merged with their higher-SNR neighbour
HIST_BIN_DEX = 0.25              # broader SNR bins for the residual histograms ...
MIN_PER_HIST_BIN = 100           # ... each holding at least this many measurements
DRAW_CHUNK = 50                  # posterior draws processed together; bounds memory to ~ 50 x n_obs x 2 values
BOOTSTRAP_DRAWS = 200            # resamples of the sources in each SNR bin for its scatter's error
FEW_SOURCES = 5                  # SNR bins with fewer sources are flagged in the log
MORPHOLOGY_GROUPS = ('S', 'C or M')   # PyBDSF S_Code: one Gaussian alone in its island, or not
AFFINE_TERMS = ('scale', 'rotation', 'stretch', 'shear')
MIN_SOURCES_AFFINE = 6           # inlier sources an epoch needs for its affine fit
ELLIPSE_DISTORTION = 0.2         # largest distortion shown in the deformation-ellipse figure
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


# ============================================================================
# DATA
# ============================================================================

@dataclass
class FitData:
    """
    Everything the model needs, one entry per valid position measurement.

    Attributes
    ----------
    x : ndarray, shape (n_obs, 2)
        Position (RA, Dec) relative to the phase centre, arcsec,
        orthographic projection.
    snr : ndarray, shape (n_obs,)
        Signal-to-noise ratio: the source's median detected peak flux density
        over the measurement's rms; after a flux density fit, the posterior
        median true peak flux density over that rms (see fitted_snr).
    source : ndarray of int, shape (n_obs,)
        Source index, 0 .. n_sources - 1.
    epoch : ndarray of int, shape (n_obs,)
        Epoch index, 0 .. n_epochs - 1.
    var_min, var_maj : ndarray, shape (n_epochs,)
        Beam variance along the minor and major axes, arcsec^2 (sigma = FWHM / 2.355).
    u_min, u_maj : ndarray, shape (n_epochs, 2)
        Unit vectors (RA, Dec) along the beam's minor and major axes.
    source_median : ndarray, shape (n_sources, 2)
        Median position of each source, arcsec; starting point for mu.
    source_median_err : ndarray, shape (n_sources, 2)
        Error on that median per axis, arcsec: 1.2533 x 1.4826 MAD / sqrt(n).
    peak, rms : ndarray, shape (n_obs,)
        Measured peak flux density and local rms of each detection, Jy.
    source_flux_median : ndarray, shape (n_sources,)
        Median detected peak flux density of each source, Jy; biased high near the threshold.
    epoch_rms : ndarray, shape (n_epochs,)
        Median rms of every catalogued detection in each epoch, Jy; the noise for non-detections.
    missed_source, missed_epoch : ndarray of int, shape (n_missed,)
        Source and epoch of every non-detection: a fitted source absent from a fitted epoch.
    s_code : ndarray of int, shape (n_obs,)
        PyBDSF S_Code of each detection, as an index into utils.S_CODES.
    pixel_arcsec : float
        Image pixel size, arcsec (largest of the epochs).
    catalog_sources, catalog_epochs : ndarray of int
        Indices of the fitted sources and epochs in the matched catalogue.
    """
    x: np.ndarray
    snr: np.ndarray
    source: np.ndarray
    epoch: np.ndarray
    var_min: np.ndarray
    var_maj: np.ndarray
    u_min: np.ndarray
    u_maj: np.ndarray
    source_median: np.ndarray
    pixel_arcsec: float
    catalog_sources: np.ndarray
    catalog_epochs: np.ndarray
    source_median_err: np.ndarray = None
    peak: np.ndarray = None
    rms: np.ndarray = None
    source_flux_median: np.ndarray = None
    epoch_rms: np.ndarray = None
    missed_source: np.ndarray = None
    missed_epoch: np.ndarray = None
    s_code: np.ndarray = None

    @property
    def n_obs(self):
        return len(self.snr)

    @property
    def n_sources(self):
        return len(self.source_median)

    @property
    def n_epochs(self):
        return len(self.var_min)


def load_catalog(config, results_dir=RESULTS_DIR):
    """
    Matched catalogue of the project, corrected for differential aberration.

    The correction runs on the full catalogue, before any cut, while its
    epochs still match observation_properties.json one to one.

    Returns
    -------
    catalog : ndarray, shape (n_sources, n_epochs, n_columns)
    aberration_info : dict
        Record of the correction (see utils.apply_aberration_correction_from_config).
    """
    path = Path(results_dir) / f"matched_catalog_{config['BOOKKEEPING']['project_name']}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run matching.py first")
    with np.load(path) as npz:
        catalog, columns = npz['data'], [str(c) for c in npz['columns']]
    if tuple(columns) != CATALOG_COLUMNS:
        raise ValueError(f"{path} has columns {columns}; expected {list(CATALOG_COLUMNS)}")
    msg(f"Loaded {path}: {catalog.shape[0]} sources x {catalog.shape[1]} epochs")

    catalog, aberration_info = apply_aberration_correction_from_config(catalog, config, results_dir)
    return catalog, aberration_info


def measured_peak_noise(config, results_dir=RESULTS_DIR):
    """The peak noise k matching.py measured, saved with the matched catalogue; where the fit starts its own."""
    with np.load(Path(results_dir) / f"matched_catalog_{config['BOOKKEEPING']['project_name']}.npz") as npz:
        if 'peak_noise' not in npz:
            raise KeyError("The matched catalogue has no measured peak noise; run matching.py again")
        return float(npz['peak_noise'])


def select_epochs(catalog, mcmc):
    """
    Epochs to fit: the dominant frequency band and, optionally, no anomalous noise.

    MCMC.enable_frequency_filtering keeps epochs within
    MCMC.frequency_match_tolerance (fractional) of the most common frequency.
    MCMC.enable_rms_filtering drops epochs whose median source rms lies more
    than MCMC.rms_outlier_sigma_threshold robust standard deviations (1.4826
    MAD) from the median over epochs.

    Returns
    -------
    ndarray of bool, shape (n_epochs,)
    """
    n_epochs = catalog.shape[1]
    keep = np.ones(n_epochs, dtype=bool)

    if mcmc.get('enable_frequency_filtering', True):
        # Positions shift with frequency (beam size, source structure), so one band only.
        freq = epoch_frequencies(catalog)
        tolerance = float(mcmc.get('frequency_match_tolerance', 0.1))
        in_band = frequency_bands(freq, tolerance) == 0
        if not in_band.all():
            keep &= in_band
            msg(f"Frequency cut: {int(in_band.sum())} of {n_epochs} epochs in the dominant band, within "
                f"{tolerance * 100:.0f}% of {np.median(freq[in_band]):.0f} Hz")

    if mcmc.get('enable_rms_filtering', False):
        rms = bn.nanmedian(catalog[:, :, column_index('rms_Jy')], axis=0)
        centre = bn.nanmedian(rms)
        # 1.4826 MAD is the standard deviation for Gaussian scatter, robust to the outliers we seek.
        spread = 1.4826 * bn.nanmedian(np.abs(rms - centre))
        clip = float(mcmc.get('rms_outlier_sigma_threshold', 3.0))
        normal = np.abs(rms - centre) <= clip * spread
        keep &= normal
        msg(f"Noise cut: {int((~normal).sum())} epochs with median rms more than {clip:g} sigma "
            f"from {centre:.2e} Jy removed")
    return keep


def select_sources(catalog, mcmc, left_out):
    """
    Sources to fit: within MCMC.max_phase_offset_deg of the phase centre and seen
    in at least MIN_EPOCHS_PER_SOURCE of the selected epochs, not counting the
    detections in left_out (n_sources, n_epochs).

    Returns
    -------
    ndarray of bool, shape (n_sources,)
    """
    # The median phase offset is used because a source's offset barely changes between epochs.
    offset = bn.nanmedian(catalog[:, :, column_index('phase_offset_deg')], axis=1)
    radius = float(mcmc.get('max_phase_offset_deg', 0.45))
    inside = offset <= radius
    n_seen = np.sum(np.isfinite(catalog[:, :, column_index('ra_deg')]) & ~left_out, axis=1)
    enough = n_seen >= MIN_EPOCHS_PER_SOURCE
    msg(f"Source cuts: {int(inside.sum())} within {radius} deg of the phase centre, "
        f"{int((inside & enough).sum())} of those seen in >= {MIN_EPOCHS_PER_SOURCE} epochs")
    return inside & enough


def write_epoch_mask(keep, results_dir=RESULTS_DIR):
    """
    Record which catalogue epochs the fit uses, in results/fit_epoch_mask.json.

    Entry i refers to epoch i of the matched catalogue and entry i of
    observation_properties.json, so later steps can map fitted epochs back to
    images. Written on every run.
    """
    with open(Path(results_dir) / 'fit_epoch_mask.json', 'w') as f:
        json.dump({'n_epochs_input': int(keep.size),
                   'n_epochs_retained': int(keep.sum()),
                   'epoch_mask': [bool(k) for k in keep],
                   'retained_indices': [int(i) for i in np.nonzero(keep)[0]]}, f, indent=2)


def pixel_size_arcsec(config, epochs, results_dir=RESULTS_DIR):
    """
    Largest image pixel size of the fitted epochs, arcsec.

    Read from observation_properties.json (cdelt1_deg, cdelt2_deg), or from
    the first image header when those fields are absent.
    """
    with open(Path(results_dir) / 'observation_properties.json', 'r') as f:
        props = json.load(f)
    # The prior on each true position is one pixel wide, far looser than the data; it only
    # keeps the sampler near the right place.
    if 'cdelt1_deg' in props:
        sizes = [max(props['cdelt1_deg'][e], props['cdelt2_deg'][e]) for e in epochs]
        return float(max(sizes)) * 3600.0

    image = sorted(Path(config['BOOKKEEPING']['image_dir']).glob('*.fits'))[0]
    header = fits.getheader(image)
    msg(f"Pixel size read from {image.name}")
    return max(abs(header['CDELT1']), abs(header['CDELT2'])) * 3600.0


def epoch_beams(catalog):
    """
    Beam of each epoch in the model's form.

    Every source in an epoch carries that epoch's restoring beam, so the
    median over sources is the beam itself.

    Returns
    -------
    var_min, var_maj : ndarray, shape (n_epochs,)
        Variances along the minor and major axes, arcsec^2.
    u_min, u_maj : ndarray, shape (n_epochs, 2)
        Unit vectors (RA, Dec) along the minor and major axes; BPA is
        East of North, so the major axis points along (sin BPA, cos BPA).
    """
    # The BPA is the same for every source in an epoch, so a plain median is safe here.
    bmaj = bn.nanmedian(catalog[:, :, column_index('bmaj_deg')], axis=0) * 3600.0
    bmin = bn.nanmedian(catalog[:, :, column_index('bmin_deg')], axis=0) * 3600.0
    bpa = np.radians(bn.nanmedian(catalog[:, :, column_index('bpa_deg')], axis=0))
    u_maj = np.column_stack([np.sin(bpa), np.cos(bpa)])
    u_min = np.column_stack([-np.cos(bpa), np.sin(bpa)])
    return (bmin * FWHM_TO_SIGMA) ** 2, (bmaj * FWHM_TO_SIGMA) ** 2, u_min, u_maj


def log_left_out(catalog, left_out, edges=LEFT_OUT_SNR_EDGES):
    """Log how many detections MCMC.single_gaussian_only leaves out, overall and in bins of detection SNR."""
    detected = np.isfinite(catalog[:, :, column_index('ra_deg')])
    snr = catalog[:, :, column_index('snr')]
    n_out, n_all = int(left_out.sum()), int(detected.sum())
    msg(f"Single Gaussians only: {n_out} of {n_all} detections left out as C or M ({n_out / n_all:.4f}, "
        f"{100 * n_out / n_all:.2f}%), from {int(left_out.any(axis=1).sum())} of {catalog.shape[0]} sources")
    all_counts = np.histogram(snr[detected], bins=edges)[0]
    out_counts = np.histogram(snr[left_out], bins=edges)[0]
    labels = ([f"<{edges[1]:g}"] + [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[1:-2], edges[2:-1])]
              + [f">{edges[-2]:g}"])
    msg(f"  {'Detection SNR':<16}" + "".join(f"{label:>9}" for label in labels))
    msg(f"  {'Detections':<16}" + "".join(f"{n:>9}" for n in all_counts))
    msg(f"  {'Left out':<16}" + "".join(f"{n:>9}" for n in out_counts))
    msg(f"  {'Percent':<16}" + "".join(f"{100 * o / n:>8.2f}%" if n else f"{'.':>9}"
                                       for o, n in zip(out_counts, all_counts)))


def prepare_data(config, results_dir=RESULTS_DIR):
    """
    Load, correct, cut and flatten the matched catalogue for the model.

    Returns
    -------
    data : FitData
    aberration_info : dict
    """
    mcmc = config.get('MCMC', {}) or {}

    # Aberration is removed first, while catalogue epochs still map one to one onto
    # observation_properties.json; every later step sees corrected positions.
    catalog, aberration_info = load_catalog(config, results_dir)

    # Epoch cuts come before source cuts, so the epoch count per source is
    # counted over the epochs actually fitted.
    keep_epochs = select_epochs(catalog, mcmc)
    write_epoch_mask(keep_epochs, results_dir)
    catalog_epochs = np.nonzero(keep_epochs)[0]
    catalog = catalog[:, keep_epochs, :]

    # Each epoch's noise from all of its detections, before the source cuts; non-detections use it.
    epoch_rms = bn.nanmedian(catalog[:, :, column_index('rms_Jy')], axis=0)

    # C and M detections (blended, or several Gaussians) have displaced centroids at every SNR. Left out,
    # they are not observed rather than missed, so the censoring never reads them as faint.
    code = catalog[:, :, column_index('s_code')]
    left_out = (np.isfinite(code) & (code != S_CODES.index('S')) & (code != AMBIGUOUS) if mcmc['single_gaussian_only']
                else np.zeros(code.shape, dtype=bool))

    keep_sources = select_sources(catalog, mcmc, left_out)
    catalog_sources = np.nonzero(keep_sources)[0]
    catalog, left_out = catalog[keep_sources], left_out[keep_sources]
    if left_out.any():
        log_left_out(catalog, left_out)
    if catalog.shape[0] == 0:
        raise ValueError("No sources left to fit after the cuts")

    # One row per detection. A fitted source absent from a fitted epoch is a non-detection: it lies
    # inside the image and the phase-offset cut, so it was missed because it fell below the threshold.
    ra, dec = catalog[:, :, column_index('ra_deg')], catalog[:, :, column_index('dec_deg')]
    peak, rms = catalog[:, :, column_index('peak_flux_Jy')], catalog[:, :, column_index('rms_Jy')]
    detected = np.isfinite(ra) & np.isfinite(dec)
    valid = detected & ~left_out & np.isfinite(peak) & np.isfinite(rms)
    source, epoch = np.nonzero(valid)
    # Ambiguous matches (S_Code A) are not observed: no position, and not a miss either.
    ambiguous = catalog[:, :, column_index('s_code')] == AMBIGUOUS
    missed_source, missed_epoch = np.nonzero(~detected & ~ambiguous)

    # Projection about the phase centre. Its small distortion (< 20 mas at 0.45 deg) is the
    # same in every epoch, so the fitted true positions absorb it.
    x = tangent_plane_arcsec(ra[valid], dec[valid], parse_phase_center(config['BOOKKEEPING']['phase_ctr']))

    # Median positions, after aberration correction, are the centres of the priors on mu.
    source_median = np.array([np.median(x[source == j], axis=0) for j in range(catalog.shape[0])])
    # Their errors: 1.4826 MAD is a robust standard deviation, and a median's error is 1.2533 sigma / sqrt(n).
    source_median_err = np.array([1.2533 * 1.4826 * np.median(np.abs(x[source == j] - source_median[j]), axis=0)
                                  / np.sqrt(np.sum(source == j)) for j in range(catalog.shape[0])])

    # Each source's median detected peak over each measurement's rms: the SNR of MCMC.snr_from_median_flux,
    # and otherwise the reference the fitted flux densities are compared with.
    flux_median = bn.nanmedian(np.where(valid, peak, np.nan), axis=1)

    # Beam variances (sigma = FWHM / 2.355); the noise model scales them by (A / SNR)^2.
    var_min, var_maj, u_min, u_maj = epoch_beams(catalog)

    data = FitData(x=x, snr=flux_median[source] / rms[valid], source=source, epoch=epoch,
                   var_min=var_min, var_maj=var_maj, u_min=u_min, u_maj=u_maj,
                   source_median=source_median,
                   pixel_arcsec=pixel_size_arcsec(config, catalog_epochs, results_dir),
                   catalog_sources=catalog_sources, catalog_epochs=catalog_epochs,
                   source_median_err=source_median_err, peak=peak[valid], rms=rms[valid],
                   source_flux_median=flux_median,
                   epoch_rms=epoch_rms, missed_source=missed_source, missed_epoch=missed_epoch,
                   s_code=catalog[:, :, column_index('s_code')][valid].astype(int))
    log_data_summary(data, int((detected & left_out).sum()), int(ambiguous.sum()),
                     int((detected & ~left_out & ~valid).sum()))
    return data, aberration_info


def log_data_summary(data, n_left_out, n_ambiguous, n_unusable):
    """
    Log the size and ranges of the data going into the fit.

    Every source-epoch pair is accounted for: fitted detections, detections
    left out (C or M, n_left_out), ambiguous matches (A, n_ambiguous) and
    detections without a peak or rms (n_unusable), all not observed, and
    non-detections.
    """
    per_source = np.bincount(data.source, minlength=data.n_sources)
    per_epoch = np.bincount(data.epoch, minlength=data.n_epochs)
    missed = np.bincount(data.missed_source, minlength=data.n_sources)
    radius = np.hypot(*data.source_median.T) / 3600.0
    err = np.max(data.source_median_err, axis=1) * 1e3
    pairs = data.n_sources * data.n_epochs

    def share(n):
        return f"{n} ({n / pairs * 100:.1f}%)"

    msg(f"Fit data: {data.n_sources} sources x {data.n_epochs} epochs = {pairs} source-epoch pairs")
    msg(f"  Fitted: {share(data.n_obs)} detections, {2 * data.n_obs} coordinates")
    if n_left_out:
        msg(f"  Left out (C or M): {share(n_left_out)}, not observed")
    if n_ambiguous:
        msg(f"  Ambiguous matches (A): {share(n_ambiguous)}, not observed")
    if n_unusable:
        msg(f"  No peak or rms: {share(n_unusable)}, not observed")
    msg(f"  Not detected: {share(len(data.missed_source))}, censored in the flux fit; "
        f"{int((missed > 0).sum())} sources missed at least once")
    msg(f"  Epochs per source: {per_source.min()} to {per_source.max()} (median {np.median(per_source):.0f})")
    msg(f"  Sources per epoch: {per_epoch.min()} to {per_epoch.max()} (median {np.median(per_epoch):.0f})")
    msg(f"  SNR (median detected peak): {data.snr.min():.1f} to {data.snr.max():.1f} "
        f"(median {np.median(data.snr):.1f})")
    msg(f"  Beam sigma: minor {np.sqrt(data.var_min).min():.2f}-{np.sqrt(data.var_min).max():.2f} arcsec, "
        f"major {np.sqrt(data.var_maj).min():.2f}-{np.sqrt(data.var_maj).max():.2f} arcsec")
    msg(f"  Distance from phase centre: up to {radius.max():.3f} deg; pixel size {data.pixel_arcsec:.3f} arcsec")
    msg(f"  Error on the median positions: {err.min():.1f} to {err.max():.1f} mas (median {np.median(err):.1f})")
    msg(f"  Epoch rms: {np.nanmin(data.epoch_rms) * 1e6:.1f} to {np.nanmax(data.epoch_rms) * 1e6:.1f} uJy")


# ============================================================================
# MODEL
# ============================================================================
#
#   x_je = mu_j + tau_e + r_je
#
# The residual r_je is split along the epoch's beam axes. Along each axis its
# variance is
#
#   sigma^2 = s(SNR)^2 v' + B^2        (absolute B)
#   sigma^2 = [s(SNR)^2 + B^2] v'      (fractional B)
#
# with v' = v_axis^alpha (v_min v_maj)^((1 - alpha) / 2). alpha = 1 gives noise
# shaped like the beam, alpha = 0 isotropic noise at the geometric-mean beam size.
# The SNR term s, set by MCMC.snr_term, is A at SNR_0 times a shape:
#
#   fixed   s = A SNR_0 / SNR, SNR_0 = 1                       the textbook 1 / SNR
#   power   s = A (SNR / SNR_0)^-beta
#   broken  s = A (SNR_0 / SNR) [(1 + SNR_b / SNR) / (1 + SNR_b / SNR_0)]^gamma
#
# PyBDSF's errors fall more slowly than 1 / SNR at low SNR and as 1 / SNR above a
# few tens. broken follows that, slope -1 - gamma below SNR_b and -1 above, so B
# is the true floor; power is one slope for all SNRs, which pulls B low. With
# SNR_0 = MCMC.snr_pivot, A is nearly independent of the shape parameters.
# A fraction f of measurements belongs to an outlier component whose standard
# deviation is lambda times larger; membership is shared by both axes.

def model_options(mcmc):
    """
    Model switches from the MCMC section of astkat.yaml.

    Returns
    -------
    dict
        snr_term (fixed, power or broken), snr_pivot (SNR_0: 1 for fixed,
        else snr_pivot), fit_b (enable_systematic_floor),
        b_fractional (systematic_floor_as_fraction), fit_alpha
        (fit_beam_shape_exponent), outliers (enable_outlier_mixture),
        fit_tau (fit_translations), fit_flux (not snr_from_median_flux),
        fit_fsys (fit_flux_systematic, with fit_flux), fit_ffrac
        (fit_flux_fractional, with fit_flux),
        inflation (error_inflation as given: False, True or a factor) and
        error_inflation, the factor on every error; it is 1 while the model
        is built and sampled and is set after the fit (see compute_diagnostics).
    """
    snr_term = mcmc['snr_term']
    if snr_term not in SNR_SHAPES:
        raise ValueError(f"MCMC.snr_term must be one of {', '.join(SNR_SHAPES)}, not {snr_term!r}")
    inflation = mcmc['error_inflation']
    if not isinstance(inflation, bool) and not (isinstance(inflation, (int, float)) and inflation >= 1):
        raise ValueError(f"MCMC.error_inflation must be False, True or a number of at least 1, not {inflation!r}")
    return {
        'fit_flux': not bool(mcmc['snr_from_median_flux']),
        'fit_fsys': not bool(mcmc['snr_from_median_flux']) and bool(mcmc['fit_flux_systematic']),
        'fit_ffrac': not bool(mcmc['snr_from_median_flux']) and bool(mcmc['fit_flux_fractional']),
        'fit_tau': bool(mcmc.get('fit_translations', True)),
        'snr_term': snr_term,
        'snr_pivot': 1.0 if snr_term == 'fixed' else float(mcmc['snr_pivot']),
        'fit_b': bool(mcmc.get('enable_systematic_floor', True)),
        'b_fractional': bool(mcmc.get('systematic_floor_as_fraction', True)),
        'fit_alpha': bool(mcmc.get('fit_beam_shape_exponent', True)),
        'outliers': bool(mcmc.get('enable_outlier_mixture', True)),
        'inflation': inflation,
        'error_inflation': 1.0,
    }


def prior_bounds(data, priors, opts):
    """
    Lower and upper bounds of the bounded noise and outlier parameters that are fitted.

    Read from the PRIORS section of astkat.yaml as [min, max] pairs. An
    absolute B with a maximum of null is bounded by a tenth of the largest
    beam FWHM. The flux terms have unbounded priors and no entry.

    Returns
    -------
    dict
        Parameter name -> (lower, upper).
    """
    def pair(name):
        return tuple(float(v) for v in priors[name])

    bounds = {'A': pair('A')}
    for name in SNR_SHAPES[opts['snr_term']]:
        bounds[name] = pair(name)
    if opts['fit_b'] and opts['b_fractional']:
        bounds['B'] = pair('B')
    elif opts['fit_b']:
        low, high = priors['B_absolute']
        bounds['B'] = (float(low), float(high) if high is not None
                       else 0.1 * np.sqrt(data.var_maj.max()) / FWHM_TO_SIGMA)
    if opts['fit_alpha']:
        bounds['alpha'] = pair('alpha')
    if opts['outliers']:
        bounds['outlier_fraction'] = pair('outlier_fraction')
        bounds['outlier_scale'] = pair('outlier_scale')
    return bounds


def build_priors(data, priors, opts):
    """
    Free parameters and their priors. Call inside an active pm.Model.

    Parameters
    ----------
    data : FitData
    priors : dict
        PRIORS section of astkat.yaml.
    opts : dict
        From model_options, with 'bounds' from prior_bounds.

    Returns
    -------
    dict of tensors
        mu (n_sources, 2) and tau (2, n_epochs) in arcsec; A, beta,
        snr_break, snr_gamma, B, alpha, f and lam as scalars (fixed values
        where a switch or the SNR term leaves them out); flux
        (n_sources,), the true peak flux densities in Jy, or None; snr
        (n_obs,), the SNR of every measurement the noise model uses.
    """
    # Gaussian prior on each true position about its median position, PRIORS.position_scale times the
    # error on the median wide and never narrower than one pixel, so it adds almost no information.
    delta = pm.Normal('delta', mu=0.0, sigma=position_prior_width(data, priors), shape=(data.n_sources, 2))
    mu = pm.Deterministic('mu', pt.as_tensor(data.source_median) + delta)

    # Epoch translations sum to zero over epochs, which fixes the frame to the mean epoch. Without
    # them, epoch-to-epoch shifts are absorbed by the noise model, a deliberately conservative fit.
    if opts['fit_tau']:
        sigma_tau = pm.HalfNormal('sigma_tau', sigma=float(priors['sigma_tau']))
        tau = pm.ZeroSumNormal('tau', sigma=sigma_tau, shape=(2, data.n_epochs))
    else:
        tau = pt.zeros((2, data.n_epochs))

    # Bounded terms start at the middle of their range; alpha starts at 1, noise shaped like the beam.
    bounds = opts['bounds']
    A = pm.Uniform('A', *bounds['A'])
    # SNR shape: beta starts at 1 and gamma at 0, the textbook 1 / SNR; the break is log-uniform.
    shape = {'beta': pt.constant(1.0), 'snr_break': pt.constant(1.0), 'snr_gamma': pt.constant(0.0)}
    if opts['snr_term'] == 'power':
        shape['beta'] = pm.Uniform('beta', *bounds['beta'], initval=1.0)
    elif opts['snr_term'] == 'broken':
        log_break = pm.Uniform('log_snr_break', *np.log(bounds['snr_break']))
        shape['snr_break'] = pm.Deterministic('snr_break', pt.exp(log_break))
        shape['snr_gamma'] = pm.Uniform('snr_gamma', *bounds['snr_gamma'], initval=0.0)
    B = pm.Uniform('B', *bounds['B']) if opts['fit_b'] else pt.constant(0.0)
    alpha = pm.Uniform('alpha', *bounds['alpha'], initval=1.0) if opts['fit_alpha'] else pt.constant(1.0)

    if opts['outliers']:
        # A Beta(a, b) stretched over [min, max]: zero density at no outliers for a > 1, which keeps chains away
        # from f = 0 where the outlier scale is unconstrained, and a maximum below 0.5 rules out the swapped
        # solution where the wide component holds the bulk.
        low, high = bounds['outlier_fraction']
        a, b = (float(v) for v in priors['outlier_fraction_beta'])
        unit = pm.Beta('outlier_fraction_unit', alpha=a, beta=b, initval=(0.02 - low) / (high - low))
        f = pm.Deterministic('outlier_fraction', low + (high - low) * unit)
        # Log-uniform, so each factor in width is equally likely and very wide outliers are discouraged.
        low, high = bounds['outlier_scale']
        log_lam = pm.Uniform('log_outlier_scale', lower=np.log(low), upper=np.log(high))
        lam = pm.Deterministic('outlier_scale', pt.exp(log_lam))
    else:
        f, lam = pt.constant(0.0), pt.constant(1.0)

    # The flux terms have smooth priors without hard bounds, so no chain can run off to a bound and stick there.
    # True peak flux densities: log-normal about each source's median detected value, PRIORS.source_flux wide in
    # ln S, so the peaks set S wherever they constrain it.
    if opts['fit_flux']:
        median = data.source_flux_median
        flux = pm.LogNormal('source_flux', mu=np.log(median), sigma=float(priors['source_flux']),
                            shape=data.n_sources, initval=median)
        # The fitted peak is noisier than the rms alone (position fitted too, correlated noise): log-normal about
        # 1, starting at the k matching.py measured.
        k = pm.LogNormal('peak_noise', mu=0.0, sigma=float(priors['peak_noise']), initval=opts['peak_noise_start'])
        # Each epoch's flux-scale (gain) error g_e multiplies its peaks and its rms alike, shared by every
        # source, so the SNR is S (1 + g_e) / rms. The scales sum to zero: the flux densities set the overall
        # scale. Their spread f_sys is half-normal. Each g_e is measured by its epoch's many sources, so the
        # scales are sampled directly (centred); scaled unit variables would tie f_sys to every g_e at once.
        if opts['fit_fsys']:
            fsys = pm.HalfNormal('flux_systematic', sigma=float(priors['flux_systematic']), initval=0.02)
            gain = 1.0 + pm.ZeroSumNormal('flux_scale', sigma=fsys, shape=data.n_epochs)
        else:
            gain = pt.ones(data.n_epochs)
        snr = flux[data.source] * gain[data.epoch] / data.rms
        # A smaller fractional error on each peak that the epoch's sources do not share (primary beam,
        # direction-dependent calibration), added in quadrature to the peak noise; half-normal.
        ffrac = (pm.HalfNormal('flux_fractional', sigma=float(priors['flux_fractional']), initval=0.01)
                 if opts['fit_ffrac'] else pt.constant(0.0))
    else:
        flux, snr, k, gain, ffrac = None, data.snr, None, None, None

    return {'mu': mu, 'tau': tau, 'A': A, **shape, 'B': B, 'alpha': alpha, 'f': f, 'lam': lam,
            'flux': flux, 'snr': snr, 'k': k, 'gain': gain, 'ffrac': ffrac}


def position_prior_width(data, priors):
    """
    Standard deviation of the prior on each true position, arcsec, shape (n_sources, 2).

    PRIORS.position_scale times the error on the median position, floored at one pixel.
    """
    return np.maximum(float(priors['position_scale']) * data.source_median_err, data.pixel_arcsec)


def noise_variances(data, p, opts):
    """
    Inlier variance of every measurement along the minor and major beam axes, arcsec^2.

    Works on model tensors and on plain numbers from posterior draws alike.

    Parameters
    ----------
    data : FitData
    p : dict
        From build_priors, or floats A, B, alpha and the SNR shape
        parameters (see snr_sigma); its 'snr', when
        present, replaces the median-flux SNR stored in data.
    opts : dict
        From model_options. Both variances are scaled by its error_inflation
        squared: 1 in the fit, the conservative factor afterwards.

    Returns
    -------
    (var_min, var_maj), tensors of shape (n_obs,)
    """
    v_min, v_maj = beam_variances(data, p['alpha'], opts['fit_alpha'])
    snr = p['snr'] if p.get('snr') is not None else data.snr
    snr_term = snr_sigma(p['A'], snr, p, opts) ** 2
    if opts['b_fractional']:
        var_min, var_maj = (snr_term + p['B'] ** 2) * v_min, (snr_term + p['B'] ** 2) * v_maj
    else:
        var_min, var_maj = snr_term * v_min + p['B'] ** 2, snr_term * v_maj + p['B'] ** 2
    inflation = opts['error_inflation'] ** 2
    return inflation * var_min, inflation * var_maj


def snr_sigma(A, snr, p, opts):
    """
    SNR term of the position error per axis, in beam sigmas: A at SNR_0, shaped by opts['snr_term'].

    Works on model tensors and on arrays of posterior draws alike; p holds the
    shape parameters the form needs (beta; snr_break and snr_gamma).
    """
    pivot = opts['snr_pivot']
    if opts['snr_term'] == 'power':
        return A * (snr / pivot) ** -p['beta']
    if opts['snr_term'] == 'broken':
        return A * (pivot / snr) * ((1.0 + p['snr_break'] / snr) / (1.0 + p['snr_break'] / pivot)) ** p['snr_gamma']
    return A * pivot / snr


def beam_variances(data, alpha, fit_alpha):
    """
    Beam variance of every measurement along its minor and major axes, reshaped by alpha, arcsec^2.

    v' = v_axis^alpha (v_min v_maj)^((1 - alpha) / 2): the scale the noise
    model multiplies by A. alpha may be a model tensor or a number; with
    fit_alpha off the beam variances are returned unchanged.
    """
    v_min, v_maj = data.var_min[data.epoch], data.var_maj[data.epoch]
    if fit_alpha:
        geometric = np.sqrt(v_min * v_maj)
        v_min = _power(v_min, alpha) * _power(geometric, 1.0 - alpha)
        v_maj = _power(v_maj, alpha) * _power(geometric, 1.0 - alpha)
    return v_min, v_maj


def _power(base, exponent):
    """base ** exponent for a constant array base and a float or model-tensor exponent."""
    return pt.as_tensor(base) ** exponent if isinstance(exponent, Variable) else base ** exponent


def beam_frame_residuals(data, p):
    """Residuals x - mu - tau projected on each epoch's minor and major beam axes, tensors (n_obs,)."""
    r = pt.as_tensor(data.x) - p['mu'][data.source] - p['tau'].T[data.epoch]
    r_min = pt.sum(r * data.u_min[data.epoch], axis=1)
    r_maj = pt.sum(r * data.u_maj[data.epoch], axis=1)
    return r_min, r_maj


def add_likelihood(data, p, opts):
    """
    Add the log-likelihood of all measurements to the active pm.Model.

    Each measurement is a 2-D normal along its beam axes, mixed with an
    outlier component of the same shape and lam times the width.
    """
    r_min, r_maj = beam_frame_residuals(data, p)
    v_min, v_maj = noise_variances(data, p, opts)
    chi2 = r_min ** 2 / v_min + r_maj ** 2 / v_maj
    log_norm = -0.5 * pt.log(v_min * v_maj) - np.log(2.0 * np.pi)

    log_inlier = log_norm - 0.5 * chi2
    if opts['outliers']:
        lam2 = p['lam'] ** 2
        log_outlier = log_norm - pt.log(lam2) - 0.5 * chi2 / lam2
        log_like = pt.logaddexp(pt.log1p(-p['f']) + log_inlier, pt.log(p['f']) + log_outlier)
    else:
        log_like = log_inlier
    pm.Potential('log_likelihood', pt.sum(log_like))
    if opts['fit_flux']:
        add_flux_likelihood(data, p, opts)


def add_flux_likelihood(data, p, opts):
    """
    Add the log-likelihood of the peak flux densities, interval-censored between t and T.

    PyBDSF decides detection on its brightest pixel, not on the fitted peak,
    so only peaks at or above T = MCMC.flux_censor_snr rms, at least
    CENSOR_MARGIN above PyBDSF's threshold where it finds essentially every
    source, are used as measured. Below T each epoch is censored, with
    detection approximated as a cut at t = MCMC.detection_censor_snr rms on
    the fitted peak (PyBDSF's threshold less the brightest pixel's excess
    over the fitted peak). With S' = S (1 + g_e), the true value times the
    epoch's flux scale, and sigma_p = sqrt((k rms)^2 + (f_frac S')^2):
        measured, p >= T rms      N(p; S', sigma_p)
        detected, p < T rms       Phi((T rms - S') / sigma_p) - Phi((t rms - S') / sigma_p)
        not detected              Phi((t rms - S') / sigma_p)
    so a faint detection bounds S from below as well as above. Positions of
    every detection still enter the astrometric likelihood. A chance
    MCMC.detection_floor of either censored outcome for any other reason
    (a failed fit, a lost match) keeps one odd epoch from dragging S.
    """
    threshold, floor = opts['detection']
    detect = opts['detect_threshold']
    flux, k, gain, ffrac = p['flux'], p['k'], p['gain'], p['ffrac']
    measured = data.peak >= threshold * data.rms
    true = flux[data.source[measured]] * gain[data.epoch[measured]]
    sigma = pt.sqrt((k * data.rms[measured]) ** 2 + (ffrac * true) ** 2)
    pm.Potential('flux_detected', pt.sum(-0.5 * ((data.peak[measured] - true) / sigma) ** 2 - pt.log(sigma)))

    def upper_tail(z):
        """1 - Phi(z), accurate far into the tail."""
        return 0.5 * pt.erfc(z / np.sqrt(2.0))

    # Faint detections: between t and T in that epoch. The difference of tails is taken on the side away from
    # 1, so it stays accurate when S lies far below t.
    rms = data.rms[~measured]
    true = flux[data.source[~measured]] * gain[data.epoch[~measured]]
    sigma = pt.sqrt((k * rms) ** 2 + (ffrac * true) ** 2)
    z_low, z_high = (detect * rms - true) / sigma, (threshold * rms - true) / sigma
    between = pt.switch(z_low > 0, upper_tail(z_low) - upper_tail(z_high), upper_tail(-z_high) - upper_tail(-z_low))
    log_between = pt.log(pt.maximum(between, 1e-300))
    pm.Potential('flux_faint', pt.sum(pt.logaddexp(np.log1p(-floor) + log_between, np.log(floor))))

    # Non-detections: below t in that epoch.
    rms = data.epoch_rms[data.missed_epoch]
    true = flux[data.missed_source] * gain[data.missed_epoch]
    z = (detect * rms - true) / pt.sqrt((k * rms) ** 2 + (ffrac * true) ** 2)
    log_below = pm.logcdf(pm.Normal.dist(0.0, 1.0), z)
    pm.Potential('flux_missed', pt.sum(pt.logaddexp(np.log1p(-floor) + log_below, np.log(floor))))


def build_model(data, config):
    """
    The joint model for the prepared data.

    Returns
    -------
    model : pm.Model
    opts : dict
        Switches used (see model_options), plus 'bounds', the prior bounds
        of the bounded parameters (see prior_bounds).
    """
    mcmc = config.get('MCMC', {}) or {}
    opts = model_options(mcmc)
    opts['bounds'] = prior_bounds(data, config['PRIORS'], opts)
    opts['detection'] = (float(mcmc['flux_censor_snr']), float(mcmc['detection_floor']))
    opts['detect_threshold'] = float(mcmc['detection_censor_snr'])
    if opts['fit_flux']:
        opts['peak_noise_start'] = measured_peak_noise(config)
    if opts['fit_flux'] and opts['detect_threshold'] >= opts['detection'][0]:
        raise ValueError("MCMC.detection_censor_snr must lie below MCMC.flux_censor_snr")
    if opts['fit_flux'] and opts['detection'][0] < float(config['PYBDSF']['snr_threshold']) + CENSOR_MARGIN:
        raise ValueError(f"MCMC.flux_censor_snr must lie at least {CENSOR_MARGIN} above PYBDSF.snr_threshold, "
                         f"so PyBDSF detects every source whose peak passes it")
    with pm.Model() as model:
        params = build_priors(data, config['PRIORS'], opts)
        add_likelihood(data, params, opts)

    log_model_summary(model, data, opts)
    return model, opts


def log_model_summary(model, data, opts):
    """
    Log the free parameters of each likelihood, its degrees of freedom and the model switches.

    Degrees of freedom are data minus free parameters: two coordinates per
    fitted detection for the astrometry, and the measured peaks (at or above
    the censoring threshold) for the flux densities.
    """
    point = model.initial_point()
    sizes = {rv.name: int(point[model.rvs_to_values[rv].name].size) for rv in model.free_RVs}
    n_pos, n_tau, n_flux = sizes['delta'], sizes.get('tau', 0), sizes.get('source_flux', 0)
    n_scale = sizes.get('flux_scale', 0)
    astro = [SCALAR_NAMES[n] for n in sizes
             if n not in ('delta', 'tau', 'source_flux', 'flux_scale') + FLUX_SCALARS]
    flux = [SCALAR_NAMES[n] for n in sizes if n in FLUX_SCALARS]
    n_coords, n_astro = 2 * data.n_obs, n_pos + n_tau + len(astro)

    msg(f"Model: {sum(sizes.values())} free parameters")
    msg(f"  Astrometric: {n_pos} true-position coordinates ({data.n_sources} x 2), "
        + (f"{n_tau} translations ({data.n_epochs - 1} x 2, summing to zero), " if n_tau else "no translations, ")
        + f"{len(astro)} scalars ({', '.join(astro)})")
    msg(f"  SNR term {opts['snr_term']}, A at SNR {opts['snr_pivot']:g}; "
        f"B {'off' if not opts['fit_b'] else ('fractional' if opts['b_fractional'] else 'absolute (arcsec)')}; "
        f"alpha {'fitted' if opts['fit_alpha'] else 'fixed at 1'}; "
        f"outlier mixture {'on' if opts['outliers'] else 'off'}")
    if not opts['fit_flux']:
        msg("  Flux densities: median detected values (MCMC.snr_from_median_flux)")
        msg(f"  Degrees of freedom: {n_coords} - {n_astro} = {n_coords - n_astro}")
        return
    threshold, floor = opts['detection']
    n_measured = int(np.sum(data.peak >= threshold * data.rms))
    n_faint = data.n_obs - n_measured
    n_flux_free = n_flux + n_scale + len(flux)
    msg(f"  Flux: {n_flux} true peaks, "
        + (f"{n_scale} epoch flux scales ({data.n_epochs} summing to zero), " if n_scale else "")
        + f"{len(flux)} scalars ({', '.join(flux)})")
    msg(f"    {n_measured} peaks at or above {threshold:g} rms measured; {n_faint} detections censored between "
        f"{opts['detect_threshold']:g} and {threshold:g} rms; {len(data.missed_source)} non-detections censored "
        f"below {opts['detect_threshold']:g} rms; floor {floor:g}")
    msg(f"  Degrees of freedom: astrometric {n_coords} - {n_astro} = {n_coords - n_astro}; "
        f"flux {n_measured} - {n_flux_free} = {n_measured - n_flux_free}")


# ============================================================================
# SAMPLING
# ============================================================================

def sample_posterior(model, config):
    """
    Draw from the posterior with NUTS.

    Settings from the MCMC section of astkat.yaml: mcmc_samples,
    mcmc_warmup_samples, mcmc_chains, mcmc_cores, mcmc_acceptance_rate and
    nuts_sampler ('nutpie' by default, 'pymc', or 'numpyro'). If nutpie is not
    installed the fit falls back to PyMC's own sampler. The seed is
    BOOKKEEPING.seed.

    Returns
    -------
    idata : xarray.DataTree
    seconds : float
        Wall time of the sampling.
    sampler : str
        The NUTS implementation used.
    """
    mcmc = config.get('MCMC', {}) or {}
    chains = int(mcmc.get('mcmc_chains', 4))
    cores = int(mcmc.get('mcmc_cores') or max(1, (os.cpu_count() or 2) - 1))
    sampler = mcmc.get('nuts_sampler', 'nutpie')
    if sampler == 'nutpie' and importlib.util.find_spec('nutpie') is None:
        msg("WARNING: nutpie is not installed (pip install nutpie); using PyMC's sampler")
        sampler = 'pymc'
    msg(f"Sampling: {chains} chains x {mcmc.get('mcmc_samples', 1000)} draws "
        f"(+{mcmc.get('mcmc_warmup_samples', 1000)} tuning), {sampler} NUTS")

    start = time.perf_counter()
    with model:
        idata = pm.sample(draws=int(mcmc.get('mcmc_samples', 1000)),
                          tune=int(mcmc.get('mcmc_warmup_samples', 1000)),
                          chains=chains, cores=min(cores, chains),
                          target_accept=float(mcmc.get('mcmc_acceptance_rate', 0.95)),
                          random_seed=int(config['BOOKKEEPING'].get('seed', DEFAULT_SEED)),
                          nuts_sampler=sampler)
    seconds = time.perf_counter() - start
    msg(f"Sampling took {seconds / 60:.1f} min")
    return idata, seconds, sampler


def check_bounds(idata, bounds, fraction=0.01, margin=0.02):
    """
    Parameters whose posterior presses against a bound of its prior.

    A parameter is flagged when more than `fraction` of its draws lie within
    `margin` of its prior range of a bound; the bound is then shaping the
    result. Lower bounds of zero are physical limits (no such term) and are
    not checked.

    Returns
    -------
    dict
        Flagged 'name upper' or 'name lower' -> fraction of draws near that bound.
    """
    flagged = {}
    for name, (low, high) in bounds.items():
        if name not in idata.posterior:
            continue
        values, width = idata.posterior[name].values, margin * (high - low)
        ends = [('upper', high, values > high - width)] + ([('lower', low, values < low + width)] if low != 0 else [])
        for end, bound, near in ends:
            share = float(np.mean(near))
            if share > fraction:
                flagged[f'{name} {end}'] = share
                msg(f"  WARNING: {share * 100:.1f}% of {name} draws lie within {margin * 100:.0f}% of its "
                    f"{end} bound {bound:g}; the prior is shaping the result (PRIORS in astkat.yaml)")
    if not flagged:
        msg("  No parameter presses against a bound of its prior")
    return flagged


def check_flux_offsets(idata, data, factor=FLUX_OFFSET_FACTOR):
    """
    Sources whose fitted flux density lies far from their median detected value.

    Detection biases the median detected value high only near the
    threshold, so a posterior median more than `factor` below it or above it
    marks a source whose peaks the model cannot reconcile, often a blend.

    Returns
    -------
    dict
        'low' and 'high': number of sources whose posterior median lies more
        than `factor` below or above their median detected value.
    """
    if 'source_flux' not in idata.posterior:
        return {}
    ratio = np.median(idata.posterior['source_flux'].values.reshape(-1, data.n_sources), axis=0) \
        / data.source_flux_median
    counts = {'low': int(np.sum(ratio < 1.0 / factor)), 'high': int(np.sum(ratio > factor))}
    for end, n in counts.items():
        if n:
            msg(f"  WARNING: {n} of {data.n_sources} fitted source flux densities lie more than a factor {factor:g} "
                f"{'below' if end == 'low' else 'above'} their median detected value")
    if not any(counts.values()):
        msg(f"  Every fitted source flux density lies within a factor {factor:g} of its median detected value")
    return counts


def check_convergence(idata):
    """
    Convergence diagnostics, logged with a warning where they fail.

    Flags R-hat above 1.01, bulk effective sample size below 400, and any
    divergent transition. R-hat is also checked over all true positions and
    epoch translations.

    Returns
    -------
    dict
        Per-parameter r_hat and ess_bulk, worst R-hat over mu and tau, and the
        number of divergences.
    """
    names = [n for n in SCALAR_PARAMETERS if n in idata.posterior]
    table = az.summary(idata, var_names=names, kind='diagnostics')
    # Worst R-hat over every element of the vector parameters.
    rhat_bulk = float(az.summary(idata, var_names=[v for v in ('mu', 'tau') if v in idata.posterior],
                                 kind='diagnostics')['r_hat'].max())
    divergences = int(idata.sample_stats['diverging'].sum())

    flux_names = [v for v in ('source_flux', 'flux_scale') if v in idata.posterior]
    rhat_flux = (float(az.summary(idata, var_names=flux_names, kind='diagnostics')['r_hat'].max())
                 if flux_names else None)
    report = {'parameters': {n: {'r_hat': float(table.loc[n, 'r_hat']),
                                 'ess_bulk': float(table.loc[n, 'ess_bulk'])} for n in names},
              'max_r_hat_positions_and_translations': rhat_bulk,
              'max_r_hat_flux': rhat_flux,
              'divergences': divergences}
    for n, d in report['parameters'].items():
        flag = "  WARNING" if d['r_hat'] > 1.01 or d['ess_bulk'] < 400 else ""
        msg(f"  {n:<17} R-hat {d['r_hat']:.3f}, ESS {d['ess_bulk']:.0f}{flag}")
    msg(f"  Worst R-hat over true positions and translations: {rhat_bulk:.3f}"
        + ("  WARNING" if rhat_bulk > 1.01 else ""))
    if rhat_flux is not None:
        msg(f"  Worst R-hat over source flux densities and epoch flux scales: {rhat_flux:.3f}" + ("  WARNING" if rhat_flux > 1.01 else ""))
    msg(f"  Divergent transitions: {divergences}" + ("  WARNING" if divergences else ""))
    # Chains settled on different solutions, most likely the inlier and outlier components swapped.
    split = [n for n in ('outlier_fraction', 'A', 'B') if n in report['parameters']
             and report['parameters'][n]['r_hat'] > 1.1]
    if split:
        per_chain = idata.posterior[split[0]].median(dim='draw').values
        msg(f"  WARNING: chains disagree on {', '.join(split)} (R-hat > 1.1), possibly swapped mixture "
            f"components; chain medians of {split[0]}: {', '.join(f'{v:.3g}' for v in per_chain)}. "
            f"Do not use this fit")
    report['chains_disagree'] = split
    return report


# ============================================================================
# POSTERIOR SUMMARIES
# ============================================================================

def parameter_summary(idata):
    """Median, 68% and 95% intervals, mean and standard deviation of each scalar parameter."""
    out = {}
    for n in SCALAR_PARAMETERS:
        if n in idata.posterior:
            v = idata.posterior[n].values.ravel()
            p2, p16, p50, p84, p97 = np.percentile(v, [2.5, 16, 50, 84, 97.5])
            out[n] = {'median': float(p50), 'p16': float(p16), 'p84': float(p84),
                      'p2.5': float(p2), 'p97.5': float(p97),
                      'mean': float(v.mean()), 'sd': float(v.std())}
            msg(f"  {n:<17} {p50:.4g} (+{p84 - p50:.2g} / -{p50 - p16:.2g})")
    return out


def thinned_posterior(idata, data, n_draws):
    """
    Posterior draws thinned to about n_draws in total, spread evenly over each chain, chains stacked.

    Returns
    -------
    post : xarray Dataset
        The thinned posterior.
    shape : tuple
        (n_chains, n_kept), the shape the stacked draws come from.
    mu : ndarray, shape (n_draws, n_sources, 2)
    tau : ndarray, shape (n_draws, 2, n_epochs); zeros without fitted translations
    scalars : dict
        A, beta, snr_break, snr_gamma, B, alpha, outlier_fraction and outlier_scale, shape (n_draws,),
        fixed values where off.
    flux : ndarray, shape (n_draws, n_sources), or None without a flux density fit
    gain : ndarray, shape (n_draws, n_epochs)
        1 + g_e, each epoch's flux scale; ones without fitted flux scales.
    """
    post = idata.posterior
    step = max(1, int(np.ceil(post.sizes['chain'] * post.sizes['draw'] / max(n_draws, 1))))
    post = post.isel(draw=slice(None, None, step))
    shape = (post.sizes['chain'], post.sizes['draw'])

    def stack(values):
        return values.reshape(shape[0] * shape[1], *values.shape[2:])

    tau = post['tau'].values if 'tau' in post else np.zeros(shape + (2, data.n_epochs))
    defaults = {'A': 0.0, 'beta': 1.0, 'snr_break': 1.0, 'snr_gamma': 0.0, 'B': 0.0, 'alpha': 1.0, 'outlier_fraction': 0.0, 'outlier_scale': 1.0}
    scalars = {k: stack(post[k].values if k in post else np.full(shape, v)) for k, v in defaults.items()}
    flux = stack(post['source_flux'].values) if 'source_flux' in post else None
    gain = (1.0 + stack(post['flux_scale'].values) if 'flux_scale' in post
            else np.ones((shape[0] * shape[1], data.n_epochs)))
    return post, shape, stack(post['mu'].values), stack(tau), scalars, flux, gain


def draw_chunks(data, opts, mu, tau, scalars, flux, gain):
    """
    Residuals and model variances for successive chunks of DRAW_CHUNK posterior draws.

    Each draw's own true flux densities and epoch flux scales set the SNR, so their uncertainty enters too.

    Yields
    ------
    chunk : slice
        The draws in this chunk.
    r : ndarray, shape (n_chunk, n_obs, 2)
        Residuals x - mu - tau in (RA, Dec), arcsec.
    v_min, v_maj : ndarray, shape (n_chunk, n_obs)
        Inlier variances along the minor and major beam axes, arcsec^2.
    p : dict
        The chunk's noise-model parameters, each of shape (n_chunk, 1).
    """
    for start in range(0, len(mu), DRAW_CHUNK):
        chunk = slice(start, start + DRAW_CHUNK)
        p = {k: scalars[k][chunk, None] for k in ('A', 'beta', 'snr_break', 'snr_gamma', 'B', 'alpha')}
        p['snr'] = (flux[chunk][:, data.source] * gain[chunk][:, data.epoch] / data.rms if flux is not None
                    else None)
        r = data.x[None] - mu[chunk][:, data.source] - tau[chunk].transpose(0, 2, 1)[:, data.epoch]
        v_min, v_maj = noise_variances(data, p, opts)
        yield chunk, r, v_min, v_maj, p


def pointwise_terms(data, opts, idata, n_draws):
    """
    Per-measurement log-likelihood and outlier probability over thinned posterior draws.

    Parameters
    ----------
    data : FitData
    opts : dict
        From model_options.
    idata : xarray.DataTree
    n_draws : int
        Approximate number of draws to use in total, spread evenly over each chain.

    Returns
    -------
    log_like : ndarray, shape (n_chains, n_kept_draws, n_obs)
    p_outlier : ndarray, shape (n_obs,)
        Posterior probability that each measurement belongs to the outlier component.
    thinned : dict
        The thinned draws of each scalar parameter, shape (n_chains, n_kept_draws),
        matching log_like.
    """
    post, shape, mu, tau, scalars, flux, gain = thinned_posterior(idata, data, n_draws)
    log_like = np.empty((len(mu), data.n_obs))
    p_outlier = np.zeros(data.n_obs)
    for chunk, r, v_min, v_maj, _ in draw_chunks(data, opts, mu, tau, scalars, flux, gain):
        r_min = np.sum(r * data.u_min[data.epoch], axis=2)
        r_maj = np.sum(r * data.u_maj[data.epoch], axis=2)
        chi2 = r_min ** 2 / v_min + r_maj ** 2 / v_maj
        log_norm = -0.5 * np.log(v_min * v_maj) - np.log(2.0 * np.pi)
        log_inlier = log_norm - 0.5 * chi2
        if opts['outliers']:
            f = scalars['outlier_fraction'][chunk, None]
            lam2 = scalars['outlier_scale'][chunk, None] ** 2
            outlier = np.log(f) + log_norm - np.log(lam2) - 0.5 * chi2 / lam2
            log_like[chunk] = np.logaddexp(np.log1p(-f) + log_inlier, outlier)
            p_outlier += np.exp(outlier - log_like[chunk]).sum(axis=0)
        else:
            log_like[chunk] = log_inlier
    thinned = {k: post[k].values for k in SCALAR_PARAMETERS if k in post}
    return log_like.reshape(shape + (data.n_obs,)), p_outlier / len(mu), thinned


def realised_scatter(data, opts, idata, n_draws, snr_bin, inlier):
    """
    Binned inlier scatter of the residuals about each posterior draw, with that draw's model prediction.

    Residuals are taken about each draw's own true positions and translations,
    not about their posterior medians. A draw differs from the medians by the
    posterior uncertainty, which restores exactly the noise the fitted values
    absorb, so the realised scatter matches the model scatter without any
    correction, and every parameter's uncertainty (mu, tau, A, the SNR shape, B, alpha
    and the flux densities) is carried by the draws. Both are in beam units
    with the draw's own alpha. The whitened residuals, each over the draw's
    own inlier sigma, are taken the same way.

    Returns
    -------
    observed, predicted : ndarray, shape (n_kept_draws, n_bins, 2)
        Realised rms of the inlier residuals and the model's rms for the same
        measurements, along the minor and major axes, one row per draw.
    whitened : dict
        Per draw and axis, the rms of the whitened inlier residuals overall
        (std, (n_kept_draws, 2)) and per bin (axes, (n_kept_draws, n_bins, 2)),
        and their Wasserstein distance to N(0, 1) (w1, (n_kept_draws,)); the
        whitened residuals of every measurement about the first draw (sample,
        (n_obs, 2)), and that draw's residuals in arcsec, inlier variances and
        beam variances, each (n_obs, 2) (draw: residual, variance, beam).
    """
    _, _, mu, tau, scalars, flux, gain = thinned_posterior(idata, data, n_draws)
    n_bins = int(snr_bin.max()) + 1
    # Averaging matrix: column b holds 1 / n_b for each inlier in bin b, so (values @ mean_in_bin) is the bin mean.
    mean_in_bin = np.zeros((data.n_obs, n_bins))
    mean_in_bin[np.nonzero(inlier)[0], snr_bin[inlier]] = 1.0
    mean_in_bin /= np.maximum(mean_in_bin.sum(axis=0), 1.0)

    observed = np.empty((len(mu), n_bins, 2))
    predicted = np.empty((len(mu), n_bins, 2))
    whitened = {'std': np.empty((len(mu), 2)), 'axes': np.empty((len(mu), n_bins, 2)), 'w1': np.empty(len(mu))}
    for chunk, r, v_min, v_maj, p in draw_chunks(data, opts, mu, tau, scalars, flux, gain):
        b_min, b_maj = beam_variances(data, p['alpha'], opts['fit_alpha'])
        z, residual = np.empty(r.shape), np.empty(r.shape)
        for k, (u, v, b) in enumerate(((data.u_min, v_min, b_min), (data.u_maj, v_maj, b_maj))):
            residual[..., k] = np.sum(r * u[data.epoch], axis=2)
            observed[chunk, :, k] = np.sqrt((residual[..., k] ** 2 / b) @ mean_in_bin)
            predicted[chunk, :, k] = np.sqrt((v / b) @ mean_in_bin)
            z[..., k] = residual[..., k] / np.sqrt(v)
            whitened['axes'][chunk, :, k] = np.sqrt(z[..., k] ** 2 @ mean_in_bin)
        whitened['std'][chunk] = np.sqrt(np.mean(z[:, inlier] ** 2, axis=1))
        whitened['w1'][chunk] = [wasserstein_to_normal(z_draw[inlier]) for z_draw in z]
        if chunk.start == 0:
            whitened['sample'] = z[0]
            beam = np.broadcast_to(np.stack([b_min, b_maj], axis=-1), r.shape)
            whitened['draw'] = {'residual': residual[0], 'variance': np.stack([v_min[0], v_maj[0]], axis=-1),
                                'beam': beam[0]}
    return observed, predicted, whitened


def loo_summary(log_like, thinned):
    """
    PSIS leave-one-out estimate of predictive accuracy, for comparing model choices.

    Higher elpd_loo is better. Pareto k above 0.7 marks measurements the
    estimate cannot handle reliably. The thinned draws let arviz account for
    the correlation between successive draws.

    Parameters
    ----------
    log_like : ndarray, shape (n_chains, n_kept_draws, n_obs)
    thinned : dict
        Matching draws of the scalar parameters (see pointwise_terms).
    """
    loo = az.loo(az.from_dict({'posterior': thinned, 'log_likelihood': {'positions': log_like}}), pointwise=True)
    out = {'elpd_loo': float(loo.elpd), 'se': float(loo.se), 'p_loo': float(loo.p),
           'n_pareto_k_above_0.7': int(np.sum(np.asarray(loo.pareto_k) > 0.7))}
    msg(f"  PSIS-LOO: elpd {out['elpd_loo']:.1f} +/- {out['se']:.1f}, p_loo {out['p_loo']:.1f}, "
        f"{out['n_pareto_k_above_0.7']} measurements with Pareto k > 0.7")
    return out


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def flux_products(idata):
    """
    Posterior summaries of the true peak flux densities, Jy.

    Returns
    -------
    dict or None
        median, sd, p16, p84, each (n_sources,); None without a flux density fit.
    """
    if 'source_flux' not in idata.posterior:
        return None
    draws = idata.posterior['source_flux'].values.reshape(-1, idata.posterior['source_flux'].shape[-1])
    p16, median, p84 = np.percentile(draws, [16, 50, 84], axis=0)
    return {'median': median, 'sd': draws.std(axis=0, ddof=1), 'p16': p16, 'p84': p84}


def flux_scale_products(idata, data, meta, injected):
    """
    Posterior summaries of the epoch flux scales g_e, and their comparison with a simulation's.

    Each epoch's peaks and rms are 1 + g_e times their true values; the
    scales sum to zero over the fitted epochs. A simulation's injected
    scales are rescaled to that frame, (1 + g) / mean(1 + g) - 1, before
    comparing.

    Returns
    -------
    dict or None
        median, sd, p16, p84, each (n_epochs,), catalog_epochs and the epoch
        identifiers of meta; with injected scales also truth, chi =
        (median - truth) / sd, and the rms of median - truth and the standard
        deviation of chi. None without fitted flux scales.
    """
    if 'flux_scale' not in idata.posterior:
        return None
    draws = idata.posterior['flux_scale'].values.reshape(-1, data.n_epochs)
    p16, median, p84 = np.percentile(draws, [16, 50, 84], axis=0)
    out = {'median': median, 'sd': draws.std(axis=0, ddof=1), 'p16': p16, 'p84': p84,
           'catalog_epochs': data.catalog_epochs, **meta}
    msg(f"  Flux scales: {data.n_epochs} epochs, spread of the medians {np.std(median) * 100:.2f}%, "
        f"median error {np.median(out['sd']) * 100:.2f}%")
    if injected is not None:
        truth = (1.0 + injected) / np.mean(1.0 + injected) - 1.0
        out.update({'truth': truth, 'chi': (median - truth) / out['sd'],
                    'rms_error': float(np.sqrt(np.mean((median - truth) ** 2)))})
        out['chi_sd'] = float(np.std(out['chi']))
        msg(f"    Against the injected scales: rms difference {out['rms_error'] * 100:.2f}%, "
            f"chi standard deviation {out['chi_sd']:.2f}")
    return out


def fitted_snr(data, flux, scale):
    """SNR of every measurement from the posterior median true peak flux densities, epoch flux scales and rms."""
    gain = 1.0 + (scale['median'][data.epoch] if scale is not None else 0.0)
    return flux['median'][data.source] * gain / data.rms


def posterior_point(idata, n_epochs):
    """
    Posterior medians of mu (n_sources, 2), tau (2, n_epochs) and the scalar parameters, for diagnostics.

    tau is zero when the fit has no translations.
    """
    post = idata.posterior
    mu = post['mu'].values.reshape(-1, *post['mu'].shape[2:])
    tau = (np.median(post['tau'].values.reshape(-1, *post['tau'].shape[2:]), axis=0) if 'tau' in post
           else np.zeros((2, n_epochs)))
    point = {'mu': np.median(mu, axis=0), 'tau': tau}
    for name, default in (('A', 0.0), ('beta', 1.0), ('snr_break', 1.0), ('snr_gamma', 0.0), ('B', 0.0),
                          ('alpha', 1.0)):
        point[name] = float(np.median(post[name].values)) if name in post else default
    return point


def snr_bins(snr, width_dex=SNR_BIN_DEX, min_count=MIN_PER_SNR_BIN):
    """
    Log-spaced SNR bins covering the full range, each holding at least min_count measurements.

    Returns
    -------
    index : ndarray of int, shape (n_obs,)
        Bin of each measurement.
    edges : ndarray
        Bin edges.
    """
    fine = 10 ** np.arange(np.log10(snr.min()), np.log10(snr.max()) + width_dex, width_dex)
    fine[-1] = max(fine[-1], snr.max() * (1 + 1e-9))
    edges, count = [fine[0]], 0
    for upper, n in zip(fine[1:], np.histogram(snr, bins=fine)[0]):
        count += n
        if count >= min_count:
            edges.append(upper)
            count = 0
    if count or len(edges) == 1:
        edges = edges[:-1] + [fine[-1]] if len(edges) > 1 else [fine[0], fine[-1]]
    edges = np.array(edges)
    index = np.clip(np.searchsorted(edges, snr, side='right') - 1, 0, len(edges) - 2)
    return index, edges


def beam_axis_residuals(data, point):
    """
    Residuals x - mu - tau along the minor and major beam axes at a posterior point.

    Returns
    -------
    ndarray, shape (n_obs, 2)
        Residuals (minor, major), arcsec.
    """
    r = data.x - point['mu'][data.source] - point['tau'].T[data.epoch]
    return np.column_stack([np.sum(r * data.u_min[data.epoch], axis=1),
                            np.sum(r * data.u_maj[data.epoch], axis=1)])


def wasserstein_to_normal(z):
    """
    Wasserstein-1 distance between a sample and N(0, 1).

    0 for a perfect match. Unlike Kolmogorov-Smirnov p-values it stays
    informative at large sample sizes, where any tiny mismatch is "significant".
    """
    z = np.sort(np.asarray(z).ravel())
    return float(np.mean(np.abs(z - norm.ppf((np.arange(z.size) + 0.5) / z.size))))


def model_curve(data, opts, draws, n_draws, n_points=200):
    """
    The fitted error model in beam units as a smooth function of SNR, for every posterior draw.

    In beam units the predicted scatter per axis is sqrt(s(SNR)^2 + B^2),
    s from snr_sigma, for every measurement when B is fractional. An
    absolute B is B / sqrt(v') along each axis and differs between axes and
    epochs; the floor drawn is B sqrt((1 / v'_min + 1 / v'_maj) / 2), the
    two axes averaged as squares as the binned scatter averages them, taken
    at the median over epochs, a representative floor.

    Returns
    -------
    dict
        snr (n_points,); rms (n_draws, n_points); floor (n_draws,);
        representative, True when the floor is for the median beam only;
        beam_line, a tenth of the beam FWHM in beam units.
    """
    snr = np.logspace(np.log10(data.snr.min()), np.log10(data.snr.max()), n_points)
    step = max(1, draws['A'].size // max(n_draws, 1))
    A = np.ravel(draws['A'])[::step]
    B, alpha = (np.ravel(draws[name])[::step] if name in draws else np.full(A.size, default)
                for name, default in (('B', 0.0), ('alpha', 1.0)))
    shape = {name: np.ravel(draws[name])[::step, None] for name in SNR_SHAPES[opts['snr_term']]}

    snr_term = snr_sigma(A[:, None], snr, shape, opts) ** 2
    if opts['b_fractional']:
        floor = B
    else:
        # Each draw's reshaped beam variances per epoch, shape (n_draws, n_epochs).
        a = alpha[:, None]
        geometric = np.sqrt(data.var_min * data.var_maj)
        v_min = data.var_min ** a * geometric ** (1 - a)
        v_maj = data.var_maj ** a * geometric ** (1 - a)
        floor = B * np.median(np.sqrt(0.5 * (1 / v_min + 1 / v_maj)), axis=1)
    return {'snr': snr, 'rms': np.sqrt(snr_term + floor[:, None] ** 2), 'floor': floor,
            'representative': not opts['b_fractional'], 'beam_line': 0.1 / FWHM_TO_SIGMA}


def set_error_inflation(opts, bins):
    """
    Choose the factor on every error after the fit (MCMC.error_inflation) and store it in opts.

    With True it is the smallest factor that puts every SNR bin's realised
    scatter at or below the model: the largest binned ratio of realised to
    predicted scatter (median over posterior draws), and at least 1. With
    fitted flux densities only bins of fitted SNR at or above
    MCMC.flux_censor_snr count, so a few faint bins, whose SNRs rest on
    censored peaks alone, cannot set it. A number
    is used as given; False leaves 1. The fit and its diagnostics are
    unchanged; noise_variances applies the factor everywhere afterwards,
    so offsets.py and target.py inherit it through the saved options.

    Returns
    -------
    dict
        factor, source ('off', 'data' or 'config') and snr, the median SNR
        of the bin that set it (None unless from the data).
    """
    setting = opts['inflation']
    if setting is False:
        out = {'factor': 1.0, 'source': 'off', 'snr': None}
    elif setting is True:
        ratio = np.array(bins['scatter_ratio'], dtype=float)
        if opts['fit_flux'] and np.any(bins['snr_median'] >= opts['detection'][0]):
            ratio = np.where(bins['snr_median'] >= opts['detection'][0], ratio, -np.inf)
        k = int(np.argmax(ratio))
        out = {'factor': float(max(1.0, bins['scatter_ratio'][k])), 'source': 'data',
               'snr': float(bins['snr_median'][k])}
        msg(f"Errors inflated x{out['factor']:.3f}: the largest binned realised over predicted scatter, "
            f"at SNR {out['snr']:.3g} (MCMC.error_inflation)")
    else:
        out = {'factor': float(setting), 'source': 'config', 'snr': None}
        msg(f"Errors inflated x{out['factor']:.3f}, as set in MCMC.error_inflation")
    opts['error_inflation'] = out['factor']
    return out


def bootstrap_scatter_error(values, source, masks, rng, n_boot=BOOTSTRAP_DRAWS):
    """
    Fractional error of each bin's rms scatter, from resampling its sources.

    A bin's measurements are not independent draws: a few sources, each with
    many epochs, can dominate it, and one clean or blended source more or
    less moves its scatter far more than sqrt(n) suggests. Each bootstrap
    resample redraws the bin's sources with replacement, keeping all of each
    one's measurements. Never below the Gaussian 1 / sqrt(2n) per axis; a bin
    of one source, with nothing to resample, gets an error equal to its scatter.

    Parameters
    ----------
    values : ndarray, shape (n_obs, 2)
        Scaled residuals along each axis.
    masks : list of bool arrays, one per bin

    Returns
    -------
    combined : ndarray, shape (n_bins,)
        Fractional error of the scatter of both axes together.
    axes : ndarray, shape (n_bins, 2)
    n_sources : ndarray of int, shape (n_bins,)
    """
    combined, axes, n_sources = np.ones(len(masks)), np.ones((len(masks), 2)), np.zeros(len(masks), dtype=int)
    for b, m in enumerate(masks):
        ids, member = np.unique(source[m], return_inverse=True)
        n_sources[b] = ids.size
        if ids.size < 2:
            continue
        # Each source's sum of squares and count, so a resample is a weighted sum over sources.
        squares = np.stack([np.bincount(member, values[m, k] ** 2) for k in range(2)], axis=1)
        counts = np.bincount(member).astype(float)
        picks = rng.integers(0, ids.size, (n_boot, ids.size))
        n = counts[picks].sum(axis=1)
        sum_axes = squares[picks].sum(axis=1)                               # (n_boot, 2)
        rms_axes = np.sqrt(sum_axes / n[:, None])
        rms = np.sqrt(sum_axes.sum(axis=1) / (2 * n))
        point_axes = np.sqrt(squares.sum(axis=0) / counts.sum())
        point = np.sqrt(squares.sum() / (2 * counts.sum()))
        gaussian = 1.0 / np.sqrt(2.0 * counts.sum())
        combined[b] = max(np.std(rms) / point, gaussian / np.sqrt(2.0))
        axes[b] = np.maximum(np.std(rms_axes, axis=0) / point_axes, gaussian)
    return combined, axes, n_sources


def compute_diagnostics(data, opts, point, p_outlier, snr_bin, edges, realised, draws, n_draws):
    """
    Numbers behind the diagnostic figures, in beam units.

    Residuals along each beam axis are divided by that axis's beam scale
    sqrt(v'), so A applies to every measurement alike whatever its beam. The
    binned comparison with the model uses realised, the scatter of the
    residuals about each posterior draw (see realised_scatter), and so do
    the whitened residuals, which for inliers should follow N(0, 1) on both
    axes (a mismatch between the axes points at the beam shape of the noise,
    alpha); the per-measurement residuals and the axis ratio are taken at
    the posterior median.

    Returns
    -------
    dict
        residual (n_obs, 2) in arcsec (minor, major); scaled (n_obs, 2) and
        radial (n_obs,) in beam units; whitened (n_obs, 2), about one
        posterior draw, and draw, that draw's residuals and variances (see
        realised_scatter); inlier mask (p_outlier < 0.5); whitened inlier rms per
        axis and Wasserstein distance to N(0, 1), medians over draws;
        snr_bins with, per bin, the median
        SNR, inlier and source counts, the median over draws of the realised
        scatter (both axes together and per axis) with its error from
        resampling the bin's sources (see bootstrap_scatter_error), the
        16th/50th/84th percentiles over draws of the model scatter (both axes
        together and per axis), chi = (realised - model) / standard error and
        the fraction model / realised - 1 in per cent with its error, both
        medians over draws, chi2 and dof (one per bin), and the whitened
        rms per axis, median over draws, with its standard error; axes, the observed
        minor/major scatter ratio in arcsec with its error, the model's
        ratio and the ratio of the beam itself; and the model curve.
    """
    residual = beam_axis_residuals(data, point)
    observed_draws, predicted, white = realised
    whitened = white['sample']
    b_min, b_maj = beam_variances(data, point['alpha'], opts['fit_alpha'])
    scaled = residual / np.sqrt(np.column_stack([b_min, b_maj]))
    inlier = p_outlier < 0.5
    w1 = float(np.median(white['w1']))
    std_minor, std_major = np.median(white['std'], axis=0)
    msg(f"  Whitened inlier residuals about the posterior draws: rms {std_minor:.3f} (minor axis), "
        f"{std_major:.3f} (major axis), Wasserstein distance to N(0, 1) {w1:.4f}")

    n_bins = len(edges) - 1
    in_bin = [(snr_bin == b) & inlier for b in range(n_bins)]
    n_inliers = np.array([m.sum() for m in in_bin])
    # Realised and model scatter for both axes together, per draw; the points are their medians over draws.
    observed_combined = np.sqrt(np.mean(observed_draws ** 2, axis=2))
    combined = np.sqrt(np.mean(predicted ** 2, axis=2))
    observed = np.median(observed_combined, axis=0)
    observed_axes = np.median(observed_draws, axis=0)
    whitened_axes = np.median(white['axes'], axis=0)
    # Errors on the scatter come from resampling each bin's sources, not from its measurement count.
    rng = np.random.default_rng(DEFAULT_SEED)
    rel, rel_axes, n_sources = bootstrap_scatter_error(scaled, data.source, in_bin, rng)
    _, rel_whitened, _ = bootstrap_scatter_error(whitened, data.source, in_bin, rng)
    bins = {'edges': edges, 'snr_median': np.array([np.median(data.snr[snr_bin == b]) for b in range(n_bins)]),
            'n_inliers': n_inliers, 'n_sources': n_sources,
            'observed_rms': observed, 'observed_rms_err': observed * rel,
            'observed_axes': observed_axes, 'observed_axes_err': observed_axes * rel_axes,
            'whitened_std_axes': whitened_axes, 'whitened_std_axes_err': whitened_axes * rel_whitened}
    few = n_sources < FEW_SOURCES
    msg(f"  Binned scatter errors from {BOOTSTRAP_DRAWS} resamples of each bin's sources; {n_bins} bins of "
        f"{n_sources.min()} to {n_sources.max()} sources"
        + (f"; fewer than {FEW_SOURCES} sources at SNR " + ", ".join(f"{s:.3g}" for s in bins['snr_median'][few])
           if few.any() else ""))
    for q in (16, 50, 84):
        bins[f'predicted_rms_p{q}'] = np.percentile(combined, q, axis=0)
        bins[f'predicted_axes_p{q}'] = np.percentile(predicted, q, axis=0)
    # Realised scatter against the model, draw by draw, then the median over draws; one degree of freedom per bin.
    error = bins['observed_rms_err']
    bins['chi'] = np.median((observed_combined - combined) / error, axis=0)
    ratio = combined / observed_combined
    bins['fraction'] = 100.0 * (np.median(ratio, axis=0) - 1.0)
    bins['scatter_ratio'] = np.median(observed_combined / combined, axis=0)
    bins['fraction_err'] = 100.0 * np.median(ratio, axis=0) * error / observed
    bins['chi2'], bins['dof'] = float(np.sum(bins['chi'] ** 2)), n_bins
    msg(f"  Binned scatter about the posterior draws against the model: chi2/dof {bins['chi2']:.1f}/{n_bins}")

    # Minor-to-major scatter ratio in arcsec: observed, as the model predicts it, and as the beam alone would.
    v_min, v_maj = noise_variances(data, point, opts)
    observed_ratio = np.sqrt(np.mean(residual[inlier, 0] ** 2) / np.mean(residual[inlier, 1] ** 2))
    axes = {'observed_ratio': float(observed_ratio),
            'observed_ratio_err': float(observed_ratio / np.sqrt(inlier.sum())),
            'model_ratio': float(np.sqrt(np.mean(v_min[inlier]) / np.mean(v_maj[inlier]))),
            'beam_ratio': float(np.sqrt(np.mean(data.var_min[data.epoch][inlier])
                                        / np.mean(data.var_maj[data.epoch][inlier])))}
    msg(f"  Minor/major scatter ratio: observed {axes['observed_ratio']:.3f} "
        f"+/- {axes['observed_ratio_err']:.3f}, model {axes['model_ratio']:.3f}, "
        f"beam-shaped noise {axes['beam_ratio']:.3f}")

    return {'residual': residual, 'scaled': scaled, 'radial': np.hypot(scaled[:, 0], scaled[:, 1]),
            'whitened': whitened, 'draw': white['draw'], 'inlier': inlier,
            'whitened_std_minor': float(std_minor), 'whitened_std_major': float(std_major),
            'wasserstein_to_normal': w1, 'snr_bins': bins, 'axes': axes,
            'model_curve': model_curve(data, opts, draws, n_draws)}


def snr_histograms(data, parameters, diag, width_dex=HIST_BIN_DEX, min_count=MIN_PER_HIST_BIN, n_grid=200):
    """
    Residuals about one posterior draw in broad SNR bins, with the distribution the model expects.

    Residuals x - mu - tau along each epoch's minor and major beam axes about
    one posterior draw (see realised_scatter), so they carry the full model
    noise, all measurements including outliers, both in beam units (over
    sqrt(v')) and in mas. The model density in each bin is the fitted
    mixture, inlier normal with that draw's sigma plus outlier normal lambda
    times wider, summed over the bin's measurements at their own SNR and
    beam, with f and lambda at their posterior medians.

    Returns
    -------
    list of dict, one per bin
        lo, hi, n; for 'beam' and 'mas': grid (n_grid,), data (n, 2) and
        expected (2, n_grid) measurements per unit, with robust (1.4826 MAD)
        and predicted (rms of the inlier model) standard deviations per axis.
    """
    index, edges = snr_bins(data.snr, width_dex, min_count)
    draw = diag['draw']
    sigma = {'beam': np.sqrt(draw['variance'] / draw['beam']), 'mas': np.sqrt(draw['variance']) * 1e3}
    values = {'beam': draw['residual'] / np.sqrt(draw['beam']), 'mas': draw['residual'] * 1e3}
    f = parameters['outlier_fraction']['median'] if 'outlier_fraction' in parameters else 0.0
    lam = parameters['outlier_scale']['median'] if 'outlier_scale' in parameters else 1.0

    bins = []
    for b in range(len(edges) - 1):
        m = index == b
        out = {'lo': float(edges[b]), 'hi': float(edges[b + 1]), 'n': int(m.sum())}
        for unit in ('beam', 'mas'):
            x, s = values[unit][m], sigma[unit][m]
            half = 1.2 * np.percentile(np.abs(x), 99)
            grid = np.linspace(-half, half, n_grid)
            expected = np.empty((2, n_grid))
            for k in range(2):
                z = grid[None, :] / s[:, k, None]
                inlier = np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * s[:, k, None])
                outlier = np.exp(-0.5 * (z / lam) ** 2) / (np.sqrt(2 * np.pi) * lam * s[:, k, None])
                expected[k] = np.sum((1 - f) * inlier + f * outlier, axis=0)
            out[unit] = {'grid': grid, 'data': x, 'expected': expected,
                         'robust_sd': 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0),
                         'predicted_sd': np.sqrt(np.mean(s ** 2, axis=0))}
        bins.append(out)
    msg(f"  SNR histograms: {len(bins)} bins of at least {width_dex:g} dex and {min_count} measurements")
    return bins


def peak_chi(data, parameters, flux, scale, threshold):
    """
    Measured peaks against the fitted true flux densities, to show the shape of the peak errors.

    For every detection whose peak passes the censoring threshold, chi =
    (p - S (1 + g_e)) / sigma_p, sigma_p = sqrt((k rms)^2 + (f_frac S (1 + g_e))^2),
    with k the fitted peak noise, g_e the epoch's flux scale and f_frac the
    per-detection fractional error (each 0 when not fitted), which should
    follow N(0, 1): a Gaussian core with a few extreme peaks points to
    outliers (blends, mismatches), a wider spread at high SNR to a
    proportional error the model does not take up.

    Returns
    -------
    dict or None
        snr (n,), the fitted SNR; chi (n,); source (n,); s_code (n,); and the
        fraction beyond 3 and 5. None without a flux density fit.
    """
    if flux is None:
        return None
    k = parameters['peak_noise']['median']
    measured = data.peak >= threshold * data.rms
    ffrac = parameters['flux_fractional']['median'] if 'flux_fractional' in parameters else 0.0
    gain = 1.0 + (scale['median'][data.epoch[measured]] if scale is not None else 0.0)
    true = flux['median'][data.source[measured]] * gain
    chi = (data.peak[measured] - true) / np.hypot(k * data.rms[measured], ffrac * true)
    out = {'snr': true / data.rms[measured], 'chi': chi, 'source': data.source[measured],
           's_code': data.s_code[measured],
           'beyond_3': float(np.mean(np.abs(chi) > 3)), 'beyond_5': float(np.mean(np.abs(chi) > 5))}
    worst = np.bincount(out['source'][np.abs(chi) > 5], minlength=data.n_sources)
    msg(f"  Peak chi: {out['beyond_3'] * 100:.2f}% beyond 3, {out['beyond_5'] * 100:.2f}% beyond 5 "
        f"(Gaussian: 0.27%, 0.00006%); beyond 5 from {int((worst > 0).sum())} sources, at most {worst.max()} "
        f"from one")
    return out


def morphology_split(data, diag, p_outlier, peaks, threshold, width_dex=HIST_BIN_DEX, min_count=MIN_PER_HIST_BIN):
    """
    Position residuals and peak chi of single-Gaussian detections against the rest, in broad SNR bins.

    The groups are PyBDSF S_Code S (one Gaussian, alone in its island) and
    C or M (sharing an island, or several Gaussians). A whitened scatter near
    1 means the noise model describes that group. Whitened residuals are about
    one posterior draw (see realised_scatter) over the inlier noise, all
    measurements included, so std reacts to tails and the
    robust sd (1.4826 MAD) to the core alone. Computed only with
    MCMC.single_gaussian_only false, when C and M detections are fitted.

    Returns
    -------
    dict
        groups (MORPHOLOGY_GROUPS); edges, snr_median and n (n_bins,); per
        group (2, n_bins): count, fraction, std, robust_sd, outlier (fraction
        with p_outlier > 0.5), peak_extreme (fraction of censoring-passing
        peaks with |chi| > 5) and peak_median (median chi); share of all
        detections, position outliers and extreme peaks per group (2,).
    """
    index, edges = snr_bins(data.snr, width_dex, min_count)
    n_bins = len(edges) - 1
    group = (data.s_code != S_CODES.index('S')).astype(int)
    w = diag['whitened']
    measured = data.peak >= threshold * data.rms
    chi = np.full(data.n_obs, np.nan)
    if peaks is not None:
        chi[measured] = peaks['chi']

    out = {'groups': MORPHOLOGY_GROUPS, 'edges': edges,
           'snr_median': np.array([np.median(data.snr[index == b]) for b in range(n_bins)]),
           'n': np.bincount(index, minlength=n_bins)}
    for key in ('count', 'fraction', 'std', 'robust_sd', 'outlier', 'peak_extreme', 'peak_median'):
        out[key] = np.full((2, n_bins), np.nan)
    for g in range(2):
        for b in range(n_bins):
            m = (group == g) & (index == b)
            out['count'][g, b] = m.sum()
            out['fraction'][g, b] = m.sum() / out['n'][b]
            if not m.any():
                continue
            out['std'][g, b] = np.sqrt(np.mean(w[m] ** 2))
            out['robust_sd'][g, b] = 1.4826 * np.median(np.abs(w[m]))
            out['outlier'][g, b] = np.mean(p_outlier[m] > 0.5)
            chi_m = chi[m & measured]
            if chi_m.size:
                out['peak_extreme'][g, b] = np.mean(np.abs(chi_m) > 5)
                out['peak_median'][g, b] = np.median(chi_m)

    def share(mask):
        return np.bincount(group[mask], minlength=2) / max(mask.sum(), 1)

    out['share_all'] = share(np.ones(data.n_obs, dtype=bool))
    out['share_outlier'] = share(p_outlier > 0.5)
    out['share_peak_extreme'] = share(np.abs(np.nan_to_num(chi)) > 5)
    for name, key in (('detections', 'share_all'), ('position outliers', 'share_outlier'),
                      ('peaks beyond 5', 'share_peak_extreme')):
        msg(f"  Share of {name}: " + ", ".join(f"{s} {f * 100:.1f}%" for s, f in zip(MORPHOLOGY_GROUPS, out[key])))
    msg("  Whitened position std (robust sd) by SNR bin and S_Code:")
    for b in range(n_bins):
        cells = [f"{s} {out['std'][g, b]:.3f} ({out['robust_sd'][g, b]:.3f}) N={out['count'][g, b]:.0f}"
                 for g, s in enumerate(MORPHOLOGY_GROUPS) if out['count'][g, b]]
        msg(f"    SNR {edges[b]:.0f}-{edges[b + 1]:.0f}: " + "; ".join(cells))
    return out


def epoch_affine_fits(data, opts, point, p_outlier, weighting='model'):
    """
    Full affine fit of each epoch's residual field distortion.

    The inlier residuals x - mu - tau of an epoch are fitted against the true
    positions z (RA, Dec, relative to the phase centre) as r = t + L z,
    a residual translation t and a 2x2 map L = [[a, b], [c, d]], reported as

        scale    s = (a + d) / 2      rotation  omega = (c - b) / 2
        stretch  g1 = (a - d) / 2     shear     g2 = (b + c) / 2

    all dimensionless. A field with no residual distortion has all four at zero.

    Parameters
    ----------
    weighting : {'model', 'equal'}
        'model' weights each measurement by its inverse beam-shaped noise
        covariance at the posterior median, and inflates the errors by
        sqrt(reduced chi2) where that exceeds 1, so they follow the actual
        scatter. 'equal' weights all measurements alike and takes the errors
        from the scatter of the fit.

    Returns
    -------
    L : ndarray, shape (n_epochs, 2, 2)
    terms, errors : ndarray, shape (n_epochs, 4)
        scale, rotation, stretch and shear with their standard errors; NaN for
        epochs with fewer than MIN_SOURCES_AFFINE inlier sources.
    """
    r = data.x - point['mu'][data.source] - point['tau'].T[data.epoch]
    z = point['mu'][data.source]
    if weighting == 'model':
        v_min, v_maj = noise_variances(data, point, opts)
        u_min, u_maj = data.u_min[data.epoch], data.u_maj[data.epoch]
        # Inverse noise covariance of each measurement in (RA, Dec), shape (n, 2, 2).
        weight = (u_min[:, :, None] * u_min[:, None, :] / v_min[:, None, None]
                  + u_maj[:, :, None] * u_maj[:, None, :] / v_maj[:, None, None])
    else:
        weight = np.broadcast_to(np.eye(2), (data.n_obs, 2, 2))

    # Design rows: r_E = t_E + a z_E + b z_N and r_N = t_N + c z_E + d z_N.
    design = np.zeros((data.n_obs, 2, 6))
    design[:, 0, 0] = design[:, 1, 1] = 1.0
    design[:, 0, 2:4] = z
    design[:, 1, 4:6] = z
    to_terms = 0.5 * np.array([[0, 0, 1, 0, 0, 1], [0, 0, 0, -1, 1, 0],
                               [0, 0, 1, 0, 0, -1], [0, 0, 0, 1, 1, 0]])

    L = np.full((data.n_epochs, 2, 2), np.nan)
    terms = np.full((data.n_epochs, 4), np.nan)
    errors = np.full((data.n_epochs, 4), np.nan)
    inlier = p_outlier < 0.5
    for e in range(data.n_epochs):
        m = (data.epoch == e) & inlier
        n = np.count_nonzero(m)
        if n < MIN_SOURCES_AFFINE:
            continue
        cov = np.linalg.inv(np.einsum('nki,nkl,nlj->ij', design[m], weight[m], design[m]))
        theta = cov @ np.einsum('nki,nkl,nl->i', design[m], weight[m], r[m])
        res = r[m] - np.einsum('nki,i->nk', design[m], theta)
        reduced_chi2 = np.einsum('nk,nkl,nl->', res, weight[m], res) / (2 * n - 6)
        cov = cov * (max(reduced_chi2, 1.0) if weighting == 'model' else reduced_chi2)
        L[e] = theta[2:].reshape(2, 2)
        terms[e] = to_terms @ theta
        errors[e] = np.sqrt(np.diag(to_terms @ cov @ to_terms.T))
    return L, terms, errors


def flatness(terms, errors):
    """
    Test each affine term for consistency with zero over all epochs.

    Returns
    -------
    dict
        Per term: weighted mean and its error, chi2 against zero, degrees of
        freedom, p-value, and the number of epochs more than 3 sigma from zero.
    """
    out = {}
    for k, name in enumerate(AFFINE_TERMS):
        ok = np.isfinite(terms[:, k]) & np.isfinite(errors[:, k]) & (errors[:, k] > 0)
        t, err = terms[ok, k], errors[ok, k]
        w = 1.0 / err ** 2
        chi2_stat = float(np.sum((t / err) ** 2))
        out[name] = {'mean': float(np.sum(w * t) / np.sum(w)), 'mean_err': float(1.0 / np.sqrt(np.sum(w))),
                     'chi2': chi2_stat, 'dof': int(ok.sum()),
                     'p_value': float(chi2_distribution.sf(chi2_stat, ok.sum())),
                     'n_beyond_3sigma': int(np.sum(np.abs(t / err) > 3))}
    return out


def aberration_amplitude(config, data, aberration_info, scale, scale_err):
    """
    Amplitude k of the aberration scale, from the epochs' residual scale.

    Fits s_e = k_fit * s(t) + c over epochs, where s(t) is the predicted
    aberration scale; c absorbs any constant scale, which the true positions
    take up. After the correction the residual should not follow s(t), so
    k = 1 + k_fit; without the correction, k = k_fit directly. Either way
    k = 1 means the predicted aberration is exactly what the data contain.

    Returns
    -------
    dict or None
        k, its error, k_fit, whether the correction was applied, and the
        predicted scale per epoch. None for simulated data or without epoch times.
    """
    if is_simulated(config):
        msg("  Aberration amplitude: not applicable to simulated data")
        return None
    try:
        times = load_epoch_times(data.n_epochs, retained_indices=data.catalog_epochs)
    except (FileNotFoundError, ValueError) as e:
        msg(f"  Aberration amplitude: no epoch times ({e})")
        return None

    predicted = aberration_scale(parse_phase_center(config['BOOKKEEPING']['phase_ctr']), times,
                                 observatory_location(config))
    ok = np.isfinite(scale) & np.isfinite(scale_err) & (scale_err > 0)
    X = np.column_stack([predicted[ok] * 1e5, np.ones(ok.sum())])          # 1e-5 units for conditioning
    w = 1.0 / (scale_err[ok] * 1e5) ** 2
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    k_fit, _ = cov @ X.T @ (w * scale[ok] * 1e5)
    corrected = bool(aberration_info.get('corrected', False))
    k = 1.0 + k_fit if corrected else k_fit
    msg(f"  Aberration amplitude k = {k:.3f} +/- {np.sqrt(cov[0, 0]):.3f} "
        f"({'residual slope' if corrected else 'direct slope'} {k_fit:+.3f})")
    return {'k': float(k), 'k_err': float(np.sqrt(cov[0, 0])), 'k_fit': float(k_fit),
            'corrected': corrected, 'predicted_scale': predicted}


def affine_diagnostics(data, opts, point, p_outlier, config, aberration_info):
    """
    Residual field distortion per epoch, its flatness and the aberration amplitude.

    Computed twice for comparison: with the noise-model weights and with equal
    weights (see epoch_affine_fits).

    Returns
    -------
    dict
        'model' and 'equal', each with L, terms and errors per epoch, flatness
        per term and the aberration amplitude (or None); and ellipse_gain, the
        exaggeration that makes the largest noise-model-weighted distortion
        ELLIPSE_DISTORTION in the deformation-ellipse figure.
    """
    out = {}
    for weighting in ('model', 'equal'):
        L, terms, errors = epoch_affine_fits(data, opts, point, p_outlier, weighting)
        flat = flatness(terms, errors)
        for name, f in flat.items():
            msg(f"  Affine {name:<9} ({weighting} weights) mean {f['mean'] * 1e5:+.3f} +/- "
                f"{f['mean_err'] * 1e5:.3f} x 1e-5, chi2/dof {f['chi2']:.1f}/{f['dof']}, p {f['p_value']:.3f}, "
                f"{f['n_beyond_3sigma']} epochs beyond 3 sigma")
        out[weighting] = {'L': L, 'terms': terms, 'errors': errors, 'flatness': flat,
                          'aberration': aberration_amplitude(config, data, aberration_info,
                                                             terms[:, 0], errors[:, 0])}

    L = out['model']['L']
    largest = np.nanmax([np.linalg.norm(m, 2) for m in L if np.all(np.isfinite(m))] or [0.0])
    out['ellipse_gain'] = float(ELLIPSE_DISTORTION / largest if largest > 0 else 1.0)
    out['ellipse_distortion'] = ELLIPSE_DISTORTION
    msg(f"  Deformation ellipses exaggerated {out['ellipse_gain']:.3g} times")
    return out


def simulation_truth(config, data, opts):
    """
    Injected values of the fit parameters, for a simulated project.

    Read from the simulator's truth files beside the images: A and the SNR
    shape (flat and catalogue modes, where they are injected), B (for a fit of
    the injected kind, fractional or absolute), alpha, the peak noise k and the outlier fraction and
    width (flat and catalogue modes), sigma_tau as the scatter of the injected epoch
    translations over the fitted epochs, f_sys as the scatter of the injected
    epoch flux scales (see injected_flux_scales), and f_frac.

    The simulator injects A SNR^-beta [(1 + SNR_b / SNR) / (1 + SNR_b)]^gamma,
    A at SNR 1, so the fit's A is that term at SNR_0. A shape parameter has a
    truth only where the injected form is one the fit can take: beta for a
    power fit when gamma = 0, gamma (and SNR_b when gamma is not 0) for a
    broken fit when beta = 1.

    Returns
    -------
    dict or None
        Parameter name -> injected value, in the fit's units; None for real data.
    """
    if not is_simulated(config):
        return None
    image_dir = Path(config['BOOKKEEPING']['image_dir'])
    with open(image_dir / SIMULATION_TRUTH_FILE, 'r') as f:
        sim = json.load(f)['simulation_parameters']

    truth = {}
    if sim['mode'] in ('flat', 'catalogue'):
        beta, snr_break, gamma = (float(sim.get(k) or default)
                                  for k, default in (('beta', 1.0), ('snr_break', 1.0), ('snr_gamma', 0.0)))
        pivot = opts['snr_pivot']
        truth['A'] = (float(sim['a_parameter']) * pivot ** -beta
                      * ((1.0 + snr_break / pivot) / (1.0 + snr_break)) ** gamma)
        if opts['snr_term'] == 'power' and gamma == 0.0:
            truth['beta'] = beta
        if opts['snr_term'] == 'broken' and beta == 1.0:
            truth['snr_gamma'] = gamma
            if gamma != 0.0:
                truth['snr_break'] = snr_break
    # The injected floor is a fraction of the beam or, with simulate.py --b-absolute, arcsec; it has a truth only
    # for a fit of the same kind.
    injected_fractional = sim['b_units'] == 'beam'
    if opts['fit_b'] and opts['b_fractional'] == injected_fractional:
        truth['B'] = float(sim['b_parameter'])
    elif opts['fit_b']:
        msg(f"  Injected B is {'a fraction of the beam' if injected_fractional else 'in arcsec'}, the fit's "
            f"is {'fractional' if opts['b_fractional'] else 'absolute'}; no truth shown for B")
    if opts['fit_alpha']:
        truth['alpha'] = float(sim['alpha'])
    if sim['mode'] in ('flat', 'catalogue'):
        if opts['fit_flux']:
            truth['peak_noise'] = float(sim['peak_noise'])
        if opts['outliers'] and sim['outlier_fraction'] > 0:
            truth['outlier_fraction'] = float(sim['outlier_fraction'])
            truth['outlier_scale'] = float(sim['outlier_scale'])

    offsets_file = image_dir / 'per_epoch_offsets.csv'
    if opts['fit_tau'] and offsets_file.exists():
        table = np.genfromtxt(offsets_file, delimiter=',', names=True)
        injected = np.column_stack([table['global_ra_offset_arcsec'], table['global_dec_offset_arcsec']])
        if len(injected) > data.catalog_epochs.max():
            truth['sigma_tau'] = float(np.sqrt(np.mean(np.var(injected[data.catalog_epochs], axis=0))))
    injected = injected_flux_scales(config, data)
    if opts['fit_fsys'] and injected is not None:
        truth['flux_systematic'] = float(np.std((1.0 + injected) / np.mean(1.0 + injected)))
    if opts['fit_ffrac'] and 'flux_fractional' in sim:
        truth['flux_fractional'] = float(sim['flux_fractional'])
    msg(f"  Simulation truth: {', '.join(f'{k} = {v:.4g}' for k, v in truth.items())}")
    return truth


def injected_flux_scales(config, data):
    """
    A simulation's injected flux scale g_e of each fitted epoch, from per_epoch_offsets.csv.

    Returns
    -------
    ndarray, shape (n_epochs,), or None
        None for real data or when the file does not cover the catalogue's epochs.
    """
    if not is_simulated(config):
        return None
    path = Path(config['BOOKKEEPING']['image_dir']) / 'per_epoch_offsets.csv'
    if not path.exists():
        return None
    table = np.genfromtxt(path, delimiter=',', names=True)
    if len(table) <= data.catalog_epochs.max():
        return None
    return np.asarray(table['flux_scale'], dtype=float)[data.catalog_epochs]


def epoch_metadata(data, results_dir=RESULTS_DIR):
    """
    Image name, observation time and decimal year of each fitted epoch.

    Read from observation_properties.json through the catalogue epoch
    indices. Entries are None when the file or its times are unavailable.

    Returns
    -------
    dict
        image_identifier, obs_isot and year, each a list or array of length
        n_epochs, or None.
    """
    meta = {'image_identifier': None, 'obs_isot': None, 'year': None}
    props_file = Path(results_dir) / 'observation_properties.json'
    if props_file.exists():
        with open(props_file, 'r') as f:
            props = json.load(f)
        for key in ('image_identifier', 'obs_isot'):
            if key in props:
                meta[key] = [props[key][e] for e in data.catalog_epochs]
    try:
        times = load_epoch_times(data.n_epochs, results_dir=results_dir, retained_indices=data.catalog_epochs)
        meta['year'] = times.decimalyear
    except (FileNotFoundError, ValueError) as e:
        msg(f"  Epoch times unavailable ({e}); translations are indexed by epoch only")
    return meta


def _vector_products(draws):
    """
    Summaries of posterior draws of 2-vectors.

    Parameters
    ----------
    draws : ndarray, shape (n_draws, n_items, 2)
        RA and Dec components, arcsec.

    Returns
    -------
    dict
        median, mean, sd (n_items, 2); corr_ra_dec (n_items,), the
        correlation between the RA and Dec components of each item;
        covariance (n_items, 2, n_items, 2), with covariance[a, i, b, j] the
        covariance of component i of item a with component j of item b
        (0 = RA, 1 = Dec); draws as given.
    """
    n_draws, n_items, _ = draws.shape
    flat = draws.reshape(n_draws, 2 * n_items)
    covariance = np.cov(flat, rowvar=False).reshape(n_items, 2, n_items, 2)
    sd = draws.std(axis=0, ddof=1)
    diag = np.arange(n_items)
    return {
        'median': np.median(draws, axis=0),
        'mean': draws.mean(axis=0),
        'sd': sd,
        'corr_ra_dec': covariance[diag, 0, diag, 1] / (sd[:, 0] * sd[:, 1]),
        'covariance': covariance,
        'draws': draws,
    }


def translation_products(idata, data, meta):
    """
    Everything known about the epoch translations, for correcting any position in an epoch.

    Built from all posterior draws, so the covariance includes the
    uncertainty of the noise model and the correlations between epochs
    (from the zero-sum constraint) and between RA and Dec (from the
    beam shape). To correct a target, subtract a draw of an epoch's
    translation from each draw of the target position, or use the
    covariance for a Gaussian approximation.

    Returns
    -------
    dict
        The _vector_products summaries over epochs (draws shape
        (n_draws, n_epochs, 2)), plus catalog_epochs, image_identifier,
        obs_isot and year to identify each epoch.
    """
    tau = idata.posterior['tau'].values                     # (chain, draw, 2, n_epochs)
    draws = np.moveaxis(tau.reshape(-1, *tau.shape[2:]), 1, 2)
    products = _vector_products(draws)
    products.update({'catalog_epochs': data.catalog_epochs, **meta})
    return products


def position_products(idata, data, phase_centre, flux):
    """
    Everything known about the true source positions and peak flux densities.

    Flux densities are the posterior median and standard deviation of the
    fitted values, or, with MCMC.snr_from_median_flux, the median detected
    value with its robust error (1.2533 x 1.4826 MAD / sqrt(n)). snr is that
    flux density over the median rms of the source's detections.

    Returns
    -------
    dict
        The _vector_products summaries over sources (RA and Dec offsets
        from the phase centre, arcsec; draws shape (n_draws, n_sources, 2)),
        plus ra_deg and dec_deg of the posterior median, catalog_sources, the
        number of epochs each source was measured in, flux_Jy, flux_err_Jy,
        snr and flux_method ('fitted' or 'median').
    """
    mu = idata.posterior['mu'].values                       # (chain, draw, n_sources, 2)
    products = _vector_products(mu.reshape(-1, *mu.shape[2:]))
    ra, dec = tangent_plane_to_radec(products['median'], phase_centre)
    groups = [data.source == j for j in range(data.n_sources)]
    if flux is not None:
        value, error, method = flux['median'], flux['sd'], 'fitted'
    else:
        value = data.source_flux_median
        error = np.array([1.2533 * 1.4826 * np.median(np.abs(data.peak[m] - value[j])) / np.sqrt(m.sum())
                          for j, m in enumerate(groups)])
        method = 'median'
    products.update({'ra_deg': ra, 'dec_deg': dec, 'catalog_sources': data.catalog_sources,
                     'n_epochs': np.bincount(data.source, minlength=data.n_sources),
                     'flux_Jy': value, 'flux_err_Jy': error, 'flux_method': method,
                     'snr': value / np.array([np.median(data.rms[m]) for m in groups])})
    return products


def posterior_draws(idata):
    """
    Every posterior draw of each scalar parameter, kept per chain.

    Returns
    -------
    dict
        One array of shape (n_chains, n_draws) per scalar parameter in the
        model, and 'diverging', the divergence flag of every draw.
    """
    out = {name: idata.posterior[name].values for name in SCALAR_PARAMETERS if name in idata.posterior}
    out['diverging'] = idata.sample_stats['diverging'].values.astype(bool)
    return out


# ============================================================================
# OUTPUTS
# ============================================================================

def tangent_plane_to_radec(xy_arcsec, phase_centre):
    """RA and Dec in degrees from orthographic (RA, Dec) offsets in arcsec, shape (n, 2)."""
    a0, d0 = phase_centre.ra.rad, phase_centre.dec.rad
    e_ra = np.array([-np.sin(a0), np.cos(a0), 0.0])
    e_dec = np.array([-np.sin(d0) * np.cos(a0), -np.sin(d0) * np.sin(a0), np.cos(d0)])
    centre = phase_centre.cartesian.xyz.value
    xi, eta = xy_arcsec[:, 0] / ARCSEC_PER_RAD, xy_arcsec[:, 1] / ARCSEC_PER_RAD
    n = np.outer(xi, e_ra) + np.outer(eta, e_dec) + np.outer(np.sqrt(1.0 - xi ** 2 - eta ** 2), centre)
    return np.degrees(np.arctan2(n[:, 1], n[:, 0])) % 360.0, np.degrees(np.arcsin(n[:, 2]))


def write_epoch_offsets(trans, results_dir=RESULTS_DIR):
    """
    Epoch translations to results/fit_epoch_offsets.txt (table) and fit_translations.npz (complete).

    The table has one row per fitted epoch: posterior median and standard
    deviation of the RA and Dec translation (arcsec), the correlation
    between them, the epoch's column in the matched catalogue, and its
    observation time. Subtract the translation from an epoch's positions to
    correct it.

    The npz file holds every posterior draw, the median, mean and standard
    deviation, the full covariance between all epochs and both components,
    and the epoch identifiers, as documented in translation_products.
    """
    path = Path(results_dir) / 'fit_epoch_offsets.txt'
    obs_isot = trans['obs_isot'] or ['unknown'] * len(trans['median'])
    with open(path, 'w') as f:
        f.write("# Epoch translations from the joint fit, arcsec, summing to zero over epochs\n")
        f.write("# final_ra_offset, final_dec_offset: posterior median, RA and Dec\n")
        f.write("# final_ra_err, final_dec_err: posterior standard deviation\n")
        f.write("# corr_ra_dec: correlation between the RA and Dec translation\n")
        f.write("# epoch_idx: fitted epoch; catalog_epoch: column in the matched catalogue\n")
        f.write("# Full covariance between epochs and all posterior draws: fit_translations.npz\n")
        f.write("epoch_idx catalog_epoch final_ra_offset final_dec_offset final_ra_err final_dec_err "
                "corr_ra_dec obs_isot\n")
        for e, (m, sd, r) in enumerate(zip(trans['median'], trans['sd'], trans['corr_ra_dec'])):
            f.write(f"{e:3d} {trans['catalog_epochs'][e]:4d} {m[0]:10.6f} {m[1]:10.6f} {sd[0]:10.6f} "
                    f"{sd[1]:10.6f} {r:7.4f} {obs_isot[e]}\n")
    msg(f"Epoch translations saved to {path}")

    np.savez(TRANSLATIONS_FILE,
             **{k: np.asarray(v) for k, v in trans.items() if v is not None},
             description="Epoch translations (RA, Dec; arcsec) from the AstKAT joint fit. "
                         "draws (n_draws, n_epochs, 2); median, mean, sd (n_epochs, 2); "
                         "covariance (n_epochs, 2, n_epochs, 2) with [a, i, b, j] the covariance of "
                         "component i of epoch a and component j of epoch b (0 RA, 1 Dec).")
    msg(f"Translation draws and covariance saved to {TRANSLATIONS_FILE}")


def write_flux_scales(scale, results_dir=RESULTS_DIR):
    """
    Epoch flux scales to results/fit_flux_scales.txt.

    One row per fitted epoch: posterior median and standard deviation of
    g_e, the 16th and 84th percentiles, the epoch's column in the matched
    catalogue and its observation time. Divide an epoch's flux densities by
    1 + g_e to put it on the common scale.
    """
    path = Path(results_dir) / 'fit_flux_scales.txt'
    obs_isot = scale['obs_isot'] or ['unknown'] * len(scale['median'])
    with open(path, 'w') as f:
        f.write("# Epoch flux scales g_e from the joint fit: each epoch's flux densities are 1 + g_e times the\n")
        f.write("# common scale; summing to zero over epochs\n")
        f.write("# flux_scale, flux_scale_err: posterior median and standard deviation; p16, p84: percentiles\n")
        f.write("# epoch_idx: fitted epoch; catalog_epoch: column in the matched catalogue\n")
        f.write("epoch_idx catalog_epoch flux_scale flux_scale_err p16 p84 obs_isot\n")
        for e in range(len(scale['median'])):
            f.write(f"{e:3d} {scale['catalog_epochs'][e]:4d} {scale['median'][e]:9.5f} {scale['sd'][e]:9.5f} "
                    f"{scale['p16'][e]:9.5f} {scale['p84'][e]:9.5f} {obs_isot[e]}\n")
    msg(f"Epoch flux scales saved to {path}")


def write_source_positions(pos, results_dir=RESULTS_DIR):
    """
    True source positions to results/fit_source_positions.txt (table) and .npz (complete).

    The table has, per source, its row in the matched catalogue, posterior
    median RA and Dec, the RA and Dec offsets from the phase centre with
    their standard deviations (arcsec), the correlation between them, the
    number of epochs it was measured in, and its peak flux density with error
    (Jy) and SNR (see position_products). The npz file holds every draw and
    the full covariance, as documented in position_products.
    """
    path = Path(results_dir) / 'fit_source_positions.txt'
    method = ('fitted, non-detections censored (posterior median and standard deviation)'
              if pos['flux_method'] == 'fitted' else 'median detected value and its robust error, not fitted')
    with open(path, 'w') as f:
        f.write("# True source positions from the joint fit (posterior median and standard deviation)\n")
        f.write("# RA and Dec offsets from the phase centre in arcsec; full covariance in the .npz file\n")
        f.write(f"# Peak flux densities in Jy: {method}; snr over the median rms of the detections\n")
        f.write("catalog_index ra_deg dec_deg ra_offset_arcsec dec_offset_arcsec ra_err_arcsec dec_err_arcsec "
                "corr_ra_dec n_epochs flux_Jy flux_err_Jy snr\n")
        for j in range(len(pos['median'])):
            m, sd = pos['median'][j], pos['sd'][j]
            f.write(f"{pos['catalog_sources'][j]:5d} {pos['ra_deg'][j]:12.8f} {pos['dec_deg'][j]:12.8f} "
                    f"{m[0]:12.5f} {m[1]:12.5f} {sd[0]:9.5f} {sd[1]:9.5f} {pos['corr_ra_dec'][j]:7.4f} "
                    f"{pos['n_epochs'][j]:4d} {pos['flux_Jy'][j]:12.5e} {pos['flux_err_Jy'][j]:11.4e} "
                    f"{pos['snr'][j]:9.2f}\n")
    msg(f"True source positions saved to {path}")

    np.savez(POSITIONS_FILE, **{k: np.asarray(v) for k, v in pos.items()},
             description="True source positions (RA, Dec offsets from the phase centre; arcsec) "
                         "from the AstKAT joint fit. draws (n_draws, n_sources, 2); median, mean, sd "
                         "(n_sources, 2); covariance (n_sources, 2, n_sources, 2); flux_Jy, flux_err_Jy "
                         "and snr (n_sources,) by flux_method.")
    msg(f"Position draws and covariance saved to {POSITIONS_FILE}")


def write_outlier_probabilities(data, p_outlier, results_dir=RESULTS_DIR):
    """
    Posterior outlier probability of every measurement to results/fit_outlier_probabilities.txt.

    Rows are indexed by the matched catalogue's source row and epoch column.
    """
    path = Path(results_dir) / 'fit_outlier_probabilities.txt'
    with open(path, 'w') as f:
        f.write("# Posterior probability that each measurement belongs to the outlier component\n")
        f.write("catalog_source catalog_epoch p_outlier\n")
        for j, e, p in zip(data.catalog_sources[data.source], data.catalog_epochs[data.epoch], p_outlier):
            f.write(f"{j:5d} {e:4d} {p:.5f}\n")
    n_likely = int(np.sum(p_outlier > 0.5))
    msg(f"Outlier probabilities saved to {path} ({n_likely} measurements more likely outliers than not)")


def affine_summary_lines(affine):
    """Text-summary lines for the per-epoch affine check (see affine_diagnostics)."""
    lines = [
        "",
        "Residual field distortion (per-epoch affine fit, units of 1e-5)",
        "--------------------------------------------------------------",
        f"{'term':<12}{'weights':<9}{'mean':>10}{'error':>10}{'chi2':>10}{'dof':>6}{'p':>9}{'> 3 sigma':>11}",
    ]
    for name in AFFINE_TERMS:
        for weighting in ('model', 'equal'):
            f = affine[weighting]['flatness'][name]
            lines.append(f"{name:<12}{weighting:<9}{f['mean'] * 1e5:>10.3f}{f['mean_err'] * 1e5:>10.3f}"
                         f"{f['chi2']:>10.1f}{f['dof']:>6d}{f['p_value']:>9.3f}{f['n_beyond_3sigma']:>11d}")
    for weighting in ('model', 'equal'):
        amp = affine[weighting]['aberration']
        lines.append(f"{f'Aberration amplitude k ({weighting} weights)':<52}"
                     + (f"{amp['k']:.3f} +/- {amp['k_err']:.3f}" if amp else "not applicable"))
    lines.append(f"{'Deformation ellipses exaggerated by':<52}{affine['ellipse_gain']:.3g}")
    return lines


def write_summary_text(results, results_dir=RESULTS_DIR):
    """
    Readable summary of the fit to results/fit_summary.txt.

    Header with the data and settings, convergence of every scalar parameter
    (flagged where R-hat > 1.01 or bulk ESS < 400), the marginal posterior of
    each in its natural units, and goodness-of-fit measures.
    """
    opts, data, conv = results['model'], results['data'], results['convergence']
    truth = results.get('truth_parameters')
    mcmc = results['config'].get('MCMC', {}) or {}
    n_chains, n_draws = results['posterior_draws']['diverging'].shape
    units = {'B': (1.0, 'beam') if opts['b_fractional'] else (1e3, 'mas'), 'sigma_tau': (1e3, 'mas')}
    b_mode = 'off' if not opts['fit_b'] else ('fractional' if opts['b_fractional'] else 'absolute')
    lines = [
        "AstKAT joint fit summary",
        "========================",
        f"{'Project':<20}{results['config']['BOOKKEEPING']['project_name']}",
        f"{'Fitted':<20}{results['created']}",
        f"{'Data':<20}{data['snr'].size} measurements, {len(data['source_median'])} sources, "
        f"{len(data['var_min'])} epochs",
        f"{'Model':<20}translations {'fitted' if opts.get('fit_tau', True) else 'off'}, "
        f"SNR term {opts['snr_term']}, A at SNR {opts['snr_pivot']:g}, "
        f"B {b_mode}, "
        f"alpha {'fitted' if opts['fit_alpha'] else 'fixed at 1'}, "
        f"outlier mixture {'on' if opts['outliers'] else 'off'}"
        + (f", errors inflated x{opts['error_inflation']:.3f}" if opts['error_inflation'] > 1 else ""),
        f"{'Sampling':<20}{results['sampler']}, {n_chains} chains x {n_draws} draws "
        f"(+{mcmc.get('mcmc_warmup_samples', 1000)} tuning), {results['sampling_minutes']:.1f} min",
        "",
        "Convergence",
        "-----------",
        f"{'parameter':<20}{'R-hat':>8}{'ESS':>10}",
    ]
    for name, d in conv['parameters'].items():
        flag = "   WARNING" if d['r_hat'] > 1.01 or d['ess_bulk'] < 400 else ""
        lines.append(f"{name:<20}{d['r_hat']:>8.3f}{d['ess_bulk']:>10.0f}{flag}")
    lines += [
        f"{'Worst R-hat over true positions and translations':<52}"
        f"{conv['max_r_hat_positions_and_translations']:.3f}",
        f"{'Worst R-hat over source flux densities':<52}"
        + (f"{conv['max_r_hat_flux']:.3f}" if conv.get('max_r_hat_flux') is not None else "not fitted"),
        f"{'Divergent transitions':<52}{conv['divergences']}",
        f"{'Draws within 2% of a prior bound':<52}"
        + (', '.join(f"{n} {v * 100:.1f}%" for n, v in conv['near_bound'].items()) or 'none'),
        f"{f'Fitted fluxes beyond x{FLUX_OFFSET_FACTOR:g} of their median peak':<52}"
        + (f"{conv['flux_offsets']['low']} below, {conv['flux_offsets']['high']} above"
           if conv.get('flux_offsets') else "not fitted"),
        "",
        "Marginal posteriors",
        "-------------------",
        f"{'parameter':<20}{'unit':<6}{'median':>11}{'-68%':>11}{'+68%':>11}{'mean':>11}{'sd':>11}"
        f"{'2.5%':>11}{'97.5%':>11}" + (f"{'truth':>11}" if truth else ""),
    ]
    for name, p in results['parameters'].items():
        scale, unit = units.get(name, (1.0, ''))
        v = {k: p[k] * scale for k in ('median', 'p16', 'p84', 'mean', 'sd', 'p2.5', 'p97.5')}
        true_value = f"{truth[name] * scale:>11.4g}" if truth and name in truth else ""
        lines.append(f"{name:<20}{unit:<6}{v['median']:>11.4g}{v['median'] - v['p16']:>11.3g}"
                     f"{v['p84'] - v['median']:>11.3g}{v['mean']:>11.4g}{v['sd']:>11.3g}"
                     f"{v['p2.5']:>11.4g}{v['p97.5']:>11.4g}{true_value}")
    diag, loo = results['diagnostics'], results['loo']
    lines += [
        "",
        "Goodness of fit",
        "---------------",
        f"{'Whitened inlier residual std, minor axis':<52}{diag['whitened_std_minor']:.3f}",
        f"{'Whitened inlier residual std, major axis':<52}{diag['whitened_std_major']:.3f}",
        f"{'Wasserstein distance to N(0, 1)':<52}{diag['wasserstein_to_normal']:.4f}",
        f"{'Minor/major scatter ratio, observed':<52}{diag['axes']['observed_ratio']:.3f} +/- "
        f"{diag['axes']['observed_ratio_err']:.3f}",
        f"{'Minor/major scatter ratio, model':<52}{diag['axes']['model_ratio']:.3f}",
        f"{'Minor/major scatter ratio, beam-shaped noise':<52}{diag['axes']['beam_ratio']:.3f}",
        f"{'PSIS-LOO elpd':<52}{loo['elpd_loo']:.1f} +/- {loo['se']:.1f} (p_loo {loo['p_loo']:.1f})",
        f"{'Measurements with Pareto k > 0.7':<52}{loo['n_pareto_k_above_0.7']}",
        f"{'Likely outliers (probability > 0.5)':<52}{int(np.sum(results['p_outlier'] > 0.5))} of "
        f"{data['snr'].size}",
    ]
    lines += affine_summary_lines(diag['affine'])
    path = Path(results_dir) / SUMMARY_TEXT_FILE
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    msg(f"Readable fit summary saved to {path}")


def write_summary(summary, results_dir=RESULTS_DIR):
    """Fit settings, data counts, parameter estimates and diagnostics to results/fit_summary.json."""
    path = Path(results_dir) / 'fit_summary.json'
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    msg(f"Fit summary saved to {path}")


# ============================================================================
# SAVED RESULTS
# ============================================================================

def save_results(results, path=RESULTS_FILE):
    """
    Save the complete fit results to results/fit_results.pkl.

    Plain dicts and numpy arrays only, so the file loads from any script.
    The full posterior is in fit_trace.nc (see load_results).
    """
    with open(path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    msg(f"Fit results saved to {path}")


def load_results(path=RESULTS_FILE, with_trace=True):
    """
    Load saved fit results.

    Parameters
    ----------
    path : str or Path
        fit_results.pkl.
    with_trace : bool
        Also load the full posterior from fit_trace.nc into results['idata'].

    Returns
    -------
    dict
        Keys: created, config, model, data, aberration, parameters,
        convergence, loo, p_outlier, diagnostics, epoch_offsets (see
        translation_products; None without fitted translations), epochs
        (image names, times and catalogue indices of the fitted epochs),
        source_positions (see position_products), source_flux (see
        flux_products; None without a flux density fit),
        posterior_draws, sampling_minutes, trace_file (and idata if requested).
    """
    with open(path, 'rb') as f:
        results = pickle.load(f)
    if with_trace:
        results['idata'] = az.from_netcdf(results['trace_file'])
    return results


def parse_args():
    """Command-line options for fit.py."""
    parser = argparse.ArgumentParser(description="AstKAT joint astrometric fit")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('-y', '--yes', action='store_true', help='re-fit, replacing any existing fit')
    choice.add_argument('-n', '--no', action='store_true', help='keep an existing fit and only redo the figures')
    return parser.parse_args()


def should_refit(args, path=RESULTS_FILE):
    """
    Whether to run a new fit.

    Always when no saved fit exists. Otherwise -y or -n decide; without
    either, ask on the terminal (default no). Without a terminal, as in a
    batch job, the saved fit is kept.
    """
    if not Path(path).exists():
        return True
    if args.yes or args.no:
        return args.yes

    with open(path, 'rb') as f:
        saved = pickle.load(f)
    m = saved['model']
    msg(f"Existing fit from {saved['created']}: {saved['data']['snr'].size} measurements, "
        f"SNR term {m.get('snr_term', 'old')}, B {'fractional' if m['b_fractional'] else 'absolute'}, "
        f"alpha {'fitted' if m['fit_alpha'] else 'fixed'}, "
        f"translations {'fitted' if m.get('fit_tau', True) else 'off'}")
    if not sys.stdin.isatty():
        msg("No terminal to ask on; keeping the existing fit (use -y to re-fit)")
        return False
    return input("Re-fit and replace it? [y/N] ").strip().lower() in ('y', 'yes')


def make_plots(results):
    """Draw the fit figures with plot.py, if it is present."""
    try:
        from . import plot as plotting
    except ImportError:
        try:
            import plot as plotting
        except ModuleNotFoundError as e:
            if e.name != 'plot':
                raise
            msg("plot.py not found; figures skipped")
            return
    plotting.fit_diagnostics(results)


def run_fit(config):
    """
    Fit the project's data and save every output.

    Returns
    -------
    dict
        The results also written to fit_results.pkl (see load_results).
    """
    mcmc = config.get('MCMC', {}) or {}

    # 1. Data, corrected, cut and flattened to one row per detection.
    data, aberration_info = prepare_data(config)

    # 2. Model, true positions and epoch translations fitted jointly with the noise terms.
    model, opts = build_model(data, config)

    # 3. Sample, then check convergence before trusting any number below.
    idata, seconds, sampler = sample_posterior(model, config)
    idata.to_netcdf(str(TRACE_FILE))
    msg(f"Posterior saved to {TRACE_FILE}")
    msg("Convergence:")
    convergence = check_convergence(idata)
    convergence['near_bound'] = check_bounds(idata, opts['bounds'])
    convergence['flux_offsets'] = check_flux_offsets(idata, data)

    # 4. Posterior summaries. Pointwise terms use a thinned set of draws to bound memory.
    msg("Parameters (median, 68% interval):")
    parameters = parameter_summary(idata)
    # From here on every SNR is the fitted one, so diagnostics and figures see debiased values.
    flux = flux_products(idata)
    meta = epoch_metadata(data)
    scale = flux_scale_products(idata, data, meta, injected_flux_scales(config, data))
    if flux is not None:
        catalogued = data.snr
        data.snr = fitted_snr(data, flux, scale)
        msg(f"  Fitted SNR over median-flux SNR: median {np.median(data.snr / catalogued):.3f}, "
            f"5th percentile {np.percentile(data.snr / catalogued, 5):.3f}")
    n_draws = int(mcmc.get('posterior_predictive_draws', 1000))
    snr_bin, edges = snr_bins(data.snr)
    log_like, p_outlier, thinned = pointwise_terms(data, opts, idata, n_draws)
    loo = loo_summary(log_like, thinned)

    # 5. Diagnostics, judged on the inliers at the posterior median.
    msg("Diagnostics:")
    point = posterior_point(idata, data.n_epochs)
    draws = posterior_draws(idata)
    realised = realised_scatter(data, opts, idata, n_draws, snr_bin, p_outlier < 0.5)
    diagnostics = compute_diagnostics(data, opts, point, p_outlier, snr_bin, edges, realised, draws, n_draws)
    diagnostics['affine'] = affine_diagnostics(data, opts, point, p_outlier, config, aberration_info)
    diagnostics['snr_histograms'] = snr_histograms(data, parameters, diagnostics)
    diagnostics['peak_chi'] = peak_chi(data, parameters, flux, scale, opts['detection'][0])
    diagnostics['morphology'] = (None if mcmc['single_gaussian_only'] else
                                 morphology_split(data, diagnostics, p_outlier, diagnostics['peak_chi'],
                                                  opts['detection'][0]))
    # Diagnostics above are against the fitted model; from here on every error carries the inflation.
    diagnostics['inflation'] = set_error_inflation(opts, diagnostics['snr_bins'])

    # 6. Outputs from all posterior draws, so uncertainties and correlations carry the noise-model
    #    uncertainty. Epoch translations are what later steps subtract from each epoch's positions.
    phase_centre = parse_phase_center(config['BOOKKEEPING']['phase_ctr'])
    translations = translation_products(idata, data, meta) if opts['fit_tau'] else None
    positions = position_products(idata, data, phase_centre, flux)
    if translations is not None:
        write_epoch_offsets(translations)
    else:
        msg("No translations fitted (MCMC.fit_translations false); offsets.py estimates them for every epoch")
    write_source_positions(positions)
    write_outlier_probabilities(data, p_outlier)
    if scale is not None:
        write_flux_scales(scale)

    results = {
        'created': Time.now().isot[:19],
        'config': config,
        'model': opts,
        'data': asdict(data),
        'aberration': aberration_info,
        'parameters': parameters,
        'convergence': convergence,
        'loo': loo,
        'p_outlier': p_outlier,
        'diagnostics': diagnostics,
        'epoch_offsets': translations,
        'epochs': dict(meta, catalog_epochs=data.catalog_epochs),
        'source_positions': positions,
        'source_flux': flux,
        'flux_scale': scale,
        'posterior_draws': draws,
        'sampling_minutes': seconds / 60.0,
        'sampler': sampler,
        'truth_parameters': simulation_truth(config, data, opts),
        'trace_file': str(TRACE_FILE),
    }
    save_results(results)
    write_summary({k: results[k] for k in ('created', 'model', 'aberration', 'parameters', 'convergence',
                                            'loo', 'sampling_minutes')}
                  | {'data': {'n_obs': data.n_obs, 'n_sources': data.n_sources, 'n_epochs': data.n_epochs,
                              'pixel_arcsec': data.pixel_arcsec},
                     'n_likely_outliers': int(np.sum(p_outlier > 0.5)),
                     'whitened_std_minor': diagnostics['whitened_std_minor'],
                     'whitened_std_major': diagnostics['whitened_std_major'],
                     'wasserstein_to_normal': diagnostics['wasserstein_to_normal'],
                     'axis_ratio': diagnostics['axes'],
                     'affine': {w: {'flatness': diagnostics['affine'][w]['flatness'],
                                    'aberration_amplitude': (
                                        {k: v for k, v in diagnostics['affine'][w]['aberration'].items()
                                         if k != 'predicted_scale'} if diagnostics['affine'][w]['aberration']
                                        else None)}
                                for w in ('model', 'equal')}})
    write_summary_text(results)
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the joint fit for the project in the current directory, or reuse a saved one."""
    args = parse_args()
    title("AstKAT joint fit")
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)

    # A saved fit is only replaced on request; reusing it just redraws the figures.
    if should_refit(args):
        results = run_fit(config)
    else:
        results = load_results(with_trace=False)
        msg(f"Using the saved fit from {results['created']}")

    make_plots(results)


if __name__ == "__main__":
    main()
