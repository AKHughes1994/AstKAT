"""
AstKAT epoch offsets.

Uses the joint fit's frame, the fitted true source positions and noise
model, to give every epoch of the matched catalogue a translation, including
epochs the fit itself left out (another frequency band, anomalous noise).
Run after fit.py.

Usage
    python offsets.py                 # compares with simulation truth automatically when present
    python offsets.py --truth FILE    # compare with a given per_epoch_offsets.csv
"""

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from astropy.time import Time
from scipy.spatial import cKDTree
from scipy.special import ndtr

try:
    from .utils import msg, title, parse_phase_center, column_index, S_CODES, AMBIGUOUS, DEFAULT_SEED, beam_frame
    from .fit import (FitData, load_catalog, load_results, tangent_plane_arcsec, epoch_beams,
                      noise_variances, epoch_metadata, _vector_products, affine_diagnostics,
                      affine_summary_lines, FWHM_TO_SIGMA)
except ImportError:
    from utils import msg, title, parse_phase_center, column_index, S_CODES, AMBIGUOUS, DEFAULT_SEED, beam_frame
    from fit import (FitData, load_catalog, load_results, tangent_plane_arcsec, epoch_beams,
                     noise_variances, epoch_metadata, _vector_products, affine_diagnostics,
                     affine_summary_lines, FWHM_TO_SIGMA)


CONFIG_FILE = Path('astkat_files') / 'astkat.yaml'
RESULTS_DIR = Path('results')
OFFSETS_RESULTS_FILE = RESULTS_DIR / 'offsets_results.pkl'
TRANSLATIONS_FILE = RESULTS_DIR / 'offsets_translations_allepochs.npz'
TRANSLATIONS_TABLE = RESULTS_DIR / 'offsets_translations_allepochs.csv'
SUMMARY_FILE = RESULTS_DIR / 'offsets_summary.txt'

MIN_SOURCES = 5                  # epochs with fewer fitted sources get no translation
N_ITERATIONS = 10                # expectation-maximisation steps per posterior draw
N_GRID = 300                     # points on which each epoch's expected residual distribution is evaluated


# ============================================================================
# DATA
# ============================================================================

def prepare_all_epochs(config, results):
    """
    Measurements of the fitted sources in every epoch of the matched catalogue.

    The catalogue is loaded and corrected for aberration exactly as fit.py
    does, with no epoch cuts, and projected about the same phase centre. Only
    the sources the fit used are kept, since only they have fitted true
    positions, and with MCMC.single_gaussian_only only their S_Code S
    detections, as in the fit.

    Parameters
    ----------
    config : dict
        Parsed astkat.yaml.
    results : dict
        Saved fit results (see fit.load_results).

    Returns
    -------
    data : FitData
        One row per measurement; epoch indexes all catalogue epochs and
        source indexes the fitted sources.
    in_fit : ndarray of bool, shape (n_catalogue_epochs,)
        Epochs the joint fit used.
    """
    fitted = results['data']
    catalog, _ = load_catalog(config)
    catalog = catalog[fitted['catalog_sources']]

    ra, dec = catalog[:, :, column_index('ra_deg')], catalog[:, :, column_index('dec_deg')]
    peak, rms = catalog[:, :, column_index('peak_flux_Jy')], catalog[:, :, column_index('rms_Jy')]
    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(peak) & np.isfinite(rms)
    # The same detections as the fit: C and M detections are left out when it left them out.
    if results['config']['MCMC']['single_gaussian_only']:
        valid &= catalog[:, :, column_index('s_code')] == S_CODES.index('S')
    source, epoch = np.nonzero(valid)
    # Every measurement's SNR is the source's flux density from the fit, times the epoch's fitted flux scale
    # (1 for epochs not in the fit), over its own rms: the fitted value, or with MCMC.snr_from_median_flux
    # the median detected peak.
    flux = (results['source_flux']['median'] if results['source_flux'] is not None
            else fitted['source_flux_median'])
    gain = np.ones(catalog.shape[1])
    if results['flux_scale'] is not None:
        gain[fitted['catalog_epochs']] += results['flux_scale']['median']
    snr = flux[:, None] * gain[None, :] / rms

    var_min, var_maj, u_min, u_maj = epoch_beams(catalog)
    phase_centre = parse_phase_center(config['BOOKKEEPING']['phase_ctr'])
    data = FitData(x=tangent_plane_arcsec(ra[valid], dec[valid], phase_centre), snr=snr[valid],
                   source=source, epoch=epoch, var_min=var_min, var_maj=var_maj, u_min=u_min, u_maj=u_maj,
                   source_median=fitted['source_median'], pixel_arcsec=fitted['pixel_arcsec'],
                   catalog_sources=fitted['catalog_sources'], catalog_epochs=np.arange(catalog.shape[1]),
                   rms=rms[valid])

    in_fit = np.zeros(catalog.shape[1], dtype=bool)
    in_fit[fitted['catalog_epochs']] = True
    per_epoch = np.bincount(epoch, minlength=catalog.shape[1])
    msg(f"Epochs: {catalog.shape[1]} in the catalogue, {int(in_fit.sum())} fitted, "
        f"{int((~in_fit).sum())} to estimate in the fitted frame")
    msg(f"Measurements: {data.n_obs} of {data.n_sources} fitted sources; "
        f"{per_epoch.min()} to {per_epoch.max()} per epoch")
    return data, in_fit


# ============================================================================
# TRANSLATIONS IN THE FITTED FRAME
# ============================================================================

def thinned_draws(idata, n_draws):
    """
    About n_draws posterior draws of mu, tau and the scalar parameters, spread evenly over all chains.

    Returns
    -------
    dict
        Each entry with draws along the first axis: mu (n, n_sources, 2),
        tau (n, 2, n_fitted_epochs), source_flux (n, n_sources) and
        flux_scale (n, n_fitted_epochs) when fitted, and one array (n,) per
        scalar parameter.
    """
    post = idata.posterior
    n_total = post.sizes['chain'] * post.sizes['draw']
    step = max(1, n_total // max(n_draws, 1))
    names = ['mu'] + [n for n in ('tau', 'A', 'beta', 'snr_break', 'snr_gamma', 'B', 'alpha', 'outlier_fraction',
                                  'outlier_scale', 'sigma_tau', 'source_flux', 'flux_scale', 'peak_noise',
                                  'flux_fractional') if n in post]
    return {n: post[n].values.reshape(n_total, *post[n].shape[2:])[::step] for n in names}


def _weighted_translation(data, r, weight_min, weight_maj, prior_precision):
    """
    Weighted-mean translation of every epoch, and its covariance.

    Each measurement's residual r (RA, Dec) is weighted by its precision,
    weight_min and weight_maj along the epoch's beam axes, and a Gaussian
    prior of the given precision centred on zero is added.

    Returns
    -------
    tau : ndarray, shape (n_epochs, 2)
    cov : ndarray, shape (n_epochs, 2, 2)
    """
    u_min, u_maj = data.u_min[data.epoch], data.u_maj[data.epoch]
    p_xx = weight_min * u_min[:, 0] ** 2 + weight_maj * u_maj[:, 0] ** 2
    p_xy = weight_min * u_min[:, 0] * u_min[:, 1] + weight_maj * u_maj[:, 0] * u_maj[:, 1]
    p_yy = weight_min * u_min[:, 1] ** 2 + weight_maj * u_maj[:, 1] ** 2
    def total(weights):
        return np.bincount(data.epoch, weights=weights, minlength=data.n_epochs)

    s_xx, s_xy, s_yy = total(p_xx) + prior_precision, total(p_xy), total(p_yy) + prior_precision
    b_x, b_y = total(p_xx * r[:, 0] + p_xy * r[:, 1]), total(p_xy * r[:, 0] + p_yy * r[:, 1])

    det = s_xx * s_yy - s_xy ** 2
    cov = np.stack([np.stack([s_yy, -s_xy], -1), np.stack([-s_xy, s_xx], -1)], -2) / det[:, None, None]
    tau = np.einsum('eij,ej->ei', cov, np.column_stack([b_x, b_y]))
    return tau, cov


def _sample_2d(cov, rng):
    """One draw from N(0, cov) for each 2x2 covariance, shape (n, 2)."""
    l11 = np.sqrt(cov[:, 0, 0])
    l21 = cov[:, 0, 1] / l11
    l22 = np.sqrt(np.maximum(cov[:, 1, 1] - l21 ** 2, 0.0))
    z = rng.standard_normal((len(cov), 2))
    return np.column_stack([l11 * z[:, 0], l21 * z[:, 0] + l22 * z[:, 1]])


def frozen_frame_translations(data, opts, draws, rng):
    """
    Every epoch's translation with the fit's true positions and noise model held fixed.

    For each posterior draw the residuals r = x - mu are modelled as
    tau + noise, with the fit's beam-shaped noise and outlier mixture. tau is
    found by expectation-maximisation: each measurement's inlier probability
    w sets its weight, rho = w + (1 - w) / lambda^2, in a weighted mean that
    includes the fit's prior on translations (width sigma_tau; none if the fit
    had no translations). One tau is
    then drawn from its Gaussian uncertainty, so across draws the result
    carries both the epoch's own noise and the uncertainty of the frame.

    Returns
    -------
    samples : ndarray, shape (n_draws, n_epochs, 2)
    p_outlier : ndarray, shape (n_obs,)
        Posterior outlier probability of each measurement.
    """
    n_draws = len(draws['mu'])
    u_min, u_maj = data.u_min[data.epoch], data.u_maj[data.epoch]
    samples = np.empty((n_draws, data.n_epochs, 2))
    p_outlier = np.zeros(data.n_obs)

    for k in range(n_draws):
        p = {name: float(draws[name][k]) if name in draws else default
             for name, default in (('A', 0.0), ('beta', 1.0), ('snr_break', 1.0), ('snr_gamma', 0.0), ('B', 0.0),
                                   ('alpha', 1.0))}
        # The same draw's true flux densities and epoch flux scales set each measurement's SNR.
        p['snr'] = (draws['source_flux'][k][data.source] * draws['gain'][k][data.epoch] / data.rms
                    if 'source_flux' in draws else None)
        v_min, v_maj = noise_variances(data, p, opts)
        r = data.x - draws['mu'][k][data.source]
        prior = 1.0 / float(draws['sigma_tau'][k]) ** 2 if 'sigma_tau' in draws else 0.0
        f = float(draws['outlier_fraction'][k]) if opts['outliers'] else 0.0
        lam2 = float(draws['outlier_scale'][k]) ** 2 if opts['outliers'] else 1.0

        rho, w = np.ones(data.n_obs), np.ones(data.n_obs)
        for _ in range(N_ITERATIONS if opts['outliers'] else 1):
            tau, cov = _weighted_translation(data, r, rho / v_min, rho / v_maj, prior)
            if opts['outliers']:
                d = r - tau[data.epoch]
                chi2 = np.sum(d * u_min, axis=1) ** 2 / v_min + np.sum(d * u_maj, axis=1) ** 2 / v_maj
                log_ratio = np.log(f) - np.log(lam2) - 0.5 * chi2 / lam2 - (np.log1p(-f) - 0.5 * chi2)
                w = 1.0 / (1.0 + np.exp(np.clip(log_ratio, -700, 700)))
                rho = w + (1.0 - w) / lam2

        p_outlier += 1.0 - w
        samples[k] = tau + _sample_2d(cov, rng)
    return samples, p_outlier / n_draws


def all_epoch_translations(data, in_fit, opts, draws, rng):
    """
    Translations of every catalogue epoch in the fitted frame.

    Fitted epochs take the joint fit's own posterior draws; the other epochs
    take the frozen-frame estimate (see frozen_frame_translations), paired
    draw by draw with the same true positions, so covariances between fitted
    and added epochs are consistent. The frozen-frame estimate is also made
    for the fitted epochs and compared with the fit as a check of the method.

    Returns
    -------
    dict
        The _vector_products summaries over all epochs (draws shape
        (n_draws, n_epochs, 2)), plus in_fit, n_sources per epoch,
        frame_check (n_epochs,), the largest of the RA and Dec
        differences between the fit and the frozen-frame estimate in units of
        their combined standard deviation (NaN for added epochs), and
        p_outlier per measurement.
    """
    frozen, p_outlier = frozen_frame_translations(data, opts, draws, rng)
    n_sources = np.bincount(data.epoch, minlength=data.n_epochs)
    combined = frozen.copy()
    frame_check = np.full(data.n_epochs, np.nan)

    # Without fitted translations every epoch takes the frozen-frame estimate.
    if 'tau' in draws:
        fitted = np.nonzero(in_fit)[0]
        fit_draws = np.moveaxis(draws['tau'], 1, 2)                      # (n, n_fitted, 2)
        combined[:, fitted] = fit_draws
        # The frozen-frame estimate should reproduce the fit on the epochs where both exist.
        diff = fit_draws.mean(axis=0) - frozen[:, fitted].mean(axis=0)
        sigma = np.sqrt(fit_draws.var(axis=0) + frozen[:, fitted].var(axis=0))
        frame_check[fitted] = np.max(np.abs(diff) / sigma, axis=1)
        msg(f"Frame check on {len(fitted)} fitted epochs: frozen-frame estimate differs from the fit by "
            f"{np.sqrt(np.mean(diff ** 2)) * 1e3:.2f} mas rms, at most {np.nanmax(frame_check):.2f} sigma")
    else:
        msg("The fit has no translations; every epoch is estimated in the fitted frame")
    estimated = ~in_fit if 'tau' in draws else np.ones(data.n_epochs, dtype=bool)
    combined[:, estimated & (n_sources < MIN_SOURCES)] = np.nan

    products = _vector_products(combined)
    products.update({'in_fit': in_fit, 'n_sources': n_sources, 'frame_check': frame_check, 'p_outlier': p_outlier})
    added = (~in_fit) & np.isfinite(products['median'][:, 0])
    msg(f"Translations for {int(in_fit.sum())} fitted and {int(added.sum())} added epochs"
        + (f"; {int(((~in_fit) & ~added).sum())} epochs with fewer than {MIN_SOURCES} sources left out"
           if ((~in_fit) & ~added).any() else ""))
    return products


def offsets_affine(config, results, data, trans):
    """
    The per-epoch affine check of fit.py, repeated over every epoch with a translation.

    Uses the fit's true positions and noise terms at their posterior medians,
    the all-epoch translations and each measurement's outlier probability, so
    added epochs are judged in exactly the same way as fitted ones.

    Returns
    -------
    dict
        As fit.affine_diagnostics, for all catalogue epochs.
    """
    point = {'mu': results['source_positions']['median'], 'tau': trans['median'].T}
    for name, default in (('A', 0.0), ('beta', 1.0), ('snr_break', 1.0), ('snr_gamma', 0.0), ('B', 0.0),
                          ('alpha', 1.0)):
        point[name] = results['parameters'][name]['median'] if name in results['parameters'] else default
    msg("Affine check over every epoch:")
    return affine_diagnostics(data, results['model'], point, trans['p_outlier'], config, results['aberration'])


def epoch_residuals(results, data, trans):
    """
    Each measurement's offset from its fitted true position, and the distribution the model expects per epoch.

    For an epoch with translation tau, a measurement with RA-Dec noise
    variances (var_E, var_N) is expected at tau with the fit's mixture of an
    inlier normal and an outlier normal lambda times wider. Summed over the
    epoch's measurements this gives the expected number density of residuals
    on each axis, which a histogram of the residuals should follow in both
    position and width. Noise terms are at their posterior medians.

    Returns
    -------
    dict
        residual (n_obs, 2) in arcsec; grid (N_GRID,) of residual values in
        arcsec, symmetric about zero and shared by all epochs; expected
        (n_epochs, 2, N_GRID), measurements per arcsec (NaN for epochs
        without a translation).
    """
    params = results['parameters']
    point = {name: params[name]['median'] if name in params else default
             for name, default in (('A', 0.0), ('beta', 1.0), ('snr_break', 1.0), ('snr_gamma', 0.0), ('B', 0.0),
                                   ('alpha', 1.0))}
    residual = data.x - results['source_positions']['median'][data.source]
    # Against the fitted model, as the fit's own diagnostics are, so without the conservative inflation.
    v_min, v_maj = noise_variances(data, point, dict(results['model'], error_inflation=1.0))
    u_min, u_maj = data.u_min[data.epoch], data.u_maj[data.epoch]
    var = v_min[:, None] * u_min ** 2 + v_maj[:, None] * u_maj ** 2          # (n_obs, 2): RA, Dec

    tau = trans['median']
    inlier = trans['p_outlier'] < 0.5
    half = 1.15 * max(np.percentile(np.abs(residual[inlier]), 99.5), np.nanmax(np.abs(tau) + trans['sd']))
    grid = np.linspace(-half, half, N_GRID)
    f = params['outlier_fraction']['median'] if 'outlier_fraction' in params else 0.0
    lam2 = params['outlier_scale']['median'] ** 2 if 'outlier_scale' in params else 1.0

    expected = np.full((data.n_epochs, 2, N_GRID), np.nan)
    for e in np.nonzero(np.isfinite(tau[:, 0]))[0]:
        m = data.epoch == e
        for k in range(2):
            d2 = (grid[None, :] - tau[e, k]) ** 2
            s2 = var[m, k][:, None]
            inl = np.exp(-0.5 * d2 / s2) / np.sqrt(2 * np.pi * s2)
            out = np.exp(-0.5 * d2 / (lam2 * s2)) / np.sqrt(2 * np.pi * lam2 * s2)
            expected[e, k] = np.sum((1 - f) * inl + f * out, axis=0)
    return {'residual': residual, 'grid': grid, 'expected': expected}


# ============================================================================
# SIMULATION TRUTH
# ============================================================================

def compare_with_truth(trans, path):
    """
    Compare every epoch's translation with the offsets injected by a simulation.

    Rows of per_epoch_offsets.csv follow the catalogue's epochs. The fitted
    translations sum to zero over the fitted epochs, which fixes the frame,
    while the injected offsets need not; the mean injected offset over the
    fitted epochs is therefore removed before comparing.

    Returns
    -------
    dict or None
        truth (n_epochs, 2) after that shift, in arcsec; diff = recovered -
        truth; chi = diff / sd; chi2 and dof from the full translation
        covariance (pseudo-inverse, since the zero sum removes two degrees of
        freedom); rms in mas and chi mean and standard deviation per axis,
        for fitted and for added epochs. None if the file does not match the
        catalogue's epochs.
    """
    table = np.genfromtxt(path, delimiter=',', names=True)
    truth = np.column_stack([table['global_ra_offset_arcsec'], table['global_dec_offset_arcsec']])
    if len(truth) != len(trans['median']):
        msg(f"WARNING: {path} has {len(truth)} epochs but the catalogue has {len(trans['median'])}; "
            f"truth comparison skipped")
        return None

    in_fit = np.asarray(trans['in_fit'])
    truth = truth - truth[in_fit].mean(axis=0)
    diff = trans['median'] - truth
    chi = diff / trans['sd']
    idx = np.nonzero(np.isfinite(diff[:, 0]))[0]
    cov = trans['covariance'][idx][:, :, idx, :].reshape(2 * len(idx), 2 * len(idx))
    d = diff[idx].ravel()
    chi2 = float(d @ np.linalg.pinv(cov, rcond=1e-10) @ d)
    dof = int(np.linalg.matrix_rank(cov, tol=1e-10 * np.max(np.abs(cov))))

    out = {'truth': truth, 'diff': diff, 'chi': chi, 'chi2': chi2, 'dof': dof, 'file': str(path)}
    for name, sel in (('fitted', in_fit), ('added', ~in_fit)):
        sel = sel & np.isfinite(diff[:, 0])
        if sel.any():
            out[name] = {'n': int(sel.sum()), 'rms_mas': np.sqrt(np.mean(diff[sel] ** 2, axis=0)) * 1e3,
                         'chi_mean': chi[sel].mean(axis=0), 'chi_std': chi[sel].std(axis=0)}
            o = out[name]
            msg(f"  Truth, {o['n']} {name} epochs: rms {o['rms_mas'][0]:.2f} (RA), "
                f"{o['rms_mas'][1]:.2f} (Dec) mas; chi {o['chi_mean'][0]:+.2f} +/- {o['chi_std'][0]:.2f}, "
                f"{o['chi_mean'][1]:+.2f} +/- {o['chi_std'][1]:.2f}")
    msg(f"  Truth chi2/dof with the full covariance {chi2:.1f}/{dof}")
    return out


def compare_sources_with_truth(config, results, in_fit, offsets_path):
    """
    Compare the fitted true positions and peak flux densities with the sources a simulation injected.

    Reads true_source_positions.csv beside per_epoch_offsets.csv. Each fitted
    source is matched to the nearest injected source within half the largest
    beam FWHM. The fitted positions are in the frame of the fitted epochs, so
    the injected positions are shifted by the mean injected epoch offset over
    those epochs before comparing. Flux densities are those of
    fit_source_positions: fitted, or the median detected values in the old
    mode (MCMC.snr_from_median_flux).

    Returns
    -------
    dict or None
        For the matched sources: fit_index, true_index (row in
        true_source_positions.csv) and catalog_sources, injected_snr (the
        injected flux density over the median local rms), diff
        (n, 2), recovered minus injected RA and Dec in arcsec, sd (n, 2),
        chi = diff / sd, true_xy (n, 2) injected positions relative to the
        phase centre in arcsec, injected_flux, flux_ratio (recovered over
        injected), flux_ratio_err and flux_chi; plus n_fitted, rms_mas,
        chi_mean, chi_std, chi2 and dof from the full position covariance,
        median_flux_ratio, flux_chi_mean, flux_chi_std, flux_scale_mean (the
        mean injected 1 + g_e of the fitted epochs, where the ratio should
        sit), flux_method and
        file. None if the file is missing or nothing matches.
    """
    path = Path(offsets_path).parent / 'true_source_positions.csv'
    if not path.exists():
        msg(f"  {path} not found; source comparison skipped")
        return None
    table = np.genfromtxt(path, delimiter=',', names=True)
    offsets_table = np.genfromtxt(offsets_path, delimiter=',', names=True)
    injected = np.column_stack([offsets_table['global_ra_offset_arcsec'], offsets_table['global_dec_offset_arcsec']])

    # Injected positions in the fitted frame: true position plus the mean injected offset over fitted epochs.
    phase_centre = parse_phase_center(config['BOOKKEEPING']['phase_ctr'])
    true_xy = tangent_plane_arcsec(table['ra_deg'], table['dec_deg'], phase_centre) + injected[in_fit].mean(axis=0)

    pos = results['source_positions']
    tolerance = 0.5 * np.sqrt(np.max(results['data']['var_maj'])) / FWHM_TO_SIGMA
    distance, nearest = cKDTree(true_xy).query(pos['median'])
    matched = np.nonzero(distance < tolerance)[0]
    if matched.size == 0:
        msg(f"  No fitted source lies within {tolerance:.2f} arcsec of an injected one; source comparison skipped")
        return None

    diff = pos['median'][matched] - true_xy[nearest[matched]]
    sd = pos['sd'][matched]
    cov = pos['covariance'][matched][:, :, matched, :].reshape(2 * matched.size, 2 * matched.size)
    d = diff.ravel()
    chi2 = float(d @ np.linalg.solve(cov, d))

    injected = table['flux_density'][nearest[matched]]
    flux, flux_err = pos['flux_Jy'][matched], pos['flux_err_Jy'][matched]
    flux_chi = (flux - injected) / flux_err
    # The fitted flux scales sum to zero, so the fitted flux densities sit on the mean scale of the fitted
    # epochs: mean(1 + g_e) of the injected scales times the injected value (1 without injected scales).
    scale = (float(np.mean(1.0 + offsets_table['flux_scale'][in_fit]))
             if 'flux_scale' in offsets_table.dtype.names else 1.0)

    # The injected SNR, so a source's place on the axis does not depend on its own fit: the injected value on
    # the fitted epochs' mean scale over the same median local rms the fitted SNR uses (fitted flux / fitted SNR).
    injected_snr = injected * scale * pos['snr'][matched] / flux

    chi = diff / sd
    out = {'fit_index': matched, 'true_index': nearest[matched], 'catalog_sources': pos['catalog_sources'][matched],
           'injected_snr': injected_snr,
           'diff': diff, 'sd': sd, 'chi': chi, 'true_xy': true_xy[nearest[matched]], 'injected_flux': injected,
           'flux_ratio': flux / injected, 'flux_ratio_err': flux_err / injected, 'flux_chi': flux_chi,
           'n_fitted': len(pos['median']), 'rms_mas': np.sqrt(np.mean(diff ** 2, axis=0)) * 1e3,
           'chi_mean': chi.mean(axis=0), 'chi_std': chi.std(axis=0), 'chi2': chi2, 'dof': 2 * matched.size,
           'median_flux_ratio': float(np.median(flux / injected)), 'flux_chi_mean': float(flux_chi.mean()),
           'flux_chi_std': float(flux_chi.std()), 'flux_method': str(pos['flux_method']),
           'flux_scale_mean': scale, 'file': str(path)}
    msg(f"  Sources: {matched.size} of {len(pos['median'])} fitted matched; rms {out['rms_mas'][0]:.2f} (RA), "
        f"{out['rms_mas'][1]:.2f} (Dec) mas; chi {out['chi_mean'][0]:+.2f} +/- {out['chi_std'][0]:.2f}, "
        f"{out['chi_mean'][1]:+.2f} +/- {out['chi_std'][1]:.2f}; chi2/dof {chi2:.1f}/{out['dof']}")
    msg(f"  Flux densities ({out['flux_method']}): median recovered/injected {out['median_flux_ratio']:.3f}, "
        f"against {scale:.4f} expected from the mean injected flux scale of the fitted epochs; "
        f"chi {out['flux_chi_mean']:+.2f} +/- {out['flux_chi_std']:.2f}")
    return out


CAUSES = ('pybdsf', 'matching')   # stages at which a source can be lost in an epoch


def completeness_with_truth(config, results, in_fit, sources):
    """
    Fraction of fitted epochs in which each matched source was detected, against its true SNR, and why it was not.

    The true SNR is the injected peak flux density over each epoch's median
    rms, the noise the censored likelihood uses. Two fractions are kept: pairs
    PyBDSF detected at all, whatever their S_Code, and pairs measured,
    detected with a peak at or above T rms (T = MCMC.flux_censor_snr). The
    measured fraction is taken over the pairs the fit sees: with
    MCMC.single_gaussian_only, C and M detections are not observed there, so
    they leave its numerator and denominator alike. It should then follow the
    step the fit assumes, (1 - e) Phi((SNR - T) / sqrt(k^2 + (f_frac SNR)^2)),
    with k the fitted peak noise, f_frac the per-detection fractional error
    (0 when not fitted) and e MCMC.detection_floor. The epoch flux scales
    shift the step epoch by epoch and are left out. Every setting is read from the fit's own
    configuration.

    Each miss is traced back through the pipeline at the source's injected
    position in that epoch (true position plus the epoch's injected offset):
    no PyBDSF source within the matching ellipse ('pybdsf'), or a PyBDSF
    source that matching did not assign ('matching'). Below about SNR 3.5 the fitted
    sources are those detected often enough to pass the epoch cuts, so
    completeness there is biased high.

    Returns
    -------
    dict
        snr_centre, completeness (detected) and measured with their binomial
        errors, n and n_measured (pairs the fit sees) per log-spaced bin; lost, the fraction of pairs lost at each
        stage of CAUSES per bin, empty when there are no per-epoch PyBDSF catalogues to trace misses through;
        threshold, width and floor; model_snr and model, the assumed step.
    """
    project = config['BOOKKEEPING']['project_name']
    with np.load(RESULTS_DIR / f"matched_catalog_{project}.npz") as npz:
        catalog = npz['data'][:, in_fit, :]
    epoch_rms = np.nanmedian(catalog[:, :, column_index('rms_Jy')], axis=0)
    rows = catalog[results['source_positions']['catalog_sources'][sources['fit_index']]]
    detected = np.isfinite(rows[:, :, column_index('ra_deg')]) & np.isfinite(rows[:, :, column_index('peak_flux_Jy')])
    true_snr = sources['injected_flux'][:, None] / epoch_rms[None, :]

    # The fit's settings, so the figure shows the step the fit used even if astkat.yaml has changed since.
    mcmc = results['config']['MCMC']
    censor = float(mcmc['flux_censor_snr'])
    # Not observed by the fit: C and M detections when it left them out, and ambiguous matches always.
    code = rows[:, :, column_index('s_code')]
    left_out = ((detected & (code != S_CODES.index('S')) if mcmc['single_gaussian_only']
                 else np.zeros(detected.shape, dtype=bool)) | (code == AMBIGUOUS))
    with np.errstate(invalid='ignore'):
        measured = (detected & ~left_out
                    & (rows[:, :, column_index('peak_flux_Jy')] >= censor * rows[:, :, column_index('rms_Jy')]))

    # Where each source sat in each epoch's image: its true position moved by that epoch's injected offset.
    table = np.genfromtxt(sources['file'], delimiter=',', names=True)
    offsets_table = np.genfromtxt(Path(sources['file']).parent / 'per_epoch_offsets.csv', delimiter=',', names=True)
    injected = np.column_stack([offsets_table['global_ra_offset_arcsec'],
                                offsets_table['global_dec_offset_arcsec']])[in_fit]
    phase_centre = parse_phase_center(config['BOOKKEEPING']['phase_ctr'])
    true_xy = tangent_plane_arcsec(table['ra_deg'][sources['true_index']], table['dec_deg'][sources['true_index']],
                                   phase_centre)
    with open(RESULTS_DIR / 'observation_properties.json', 'r') as f:
        props = json.load(f)
    obs_isot = np.asarray(props['obs_isot'])[in_fit]
    beams = np.column_stack([np.asarray(props[k], dtype=float)[in_fit] for k in ('bmaj_deg', 'bmin_deg', 'bpa_deg')])
    match_threshold = float(config['THRESHOLDS']['match_threshold'])

    # Misses are traced through the per-epoch PyBDSF catalogues; a catalogue simulation has none.
    cause = np.full(detected.shape, -1)
    traced = all((Path('pybdsf_files') / f'catalog_{t}_{project}.fits').exists() for t in obs_isot)
    if not traced:
        msg("  No per-epoch PyBDSF catalogues: misses are not traced to a stage")
    for e in np.nonzero((~detected).any(axis=0))[0] if traced else []:
        missed = np.nonzero(~detected[:, e])[0]
        position = true_xy[missed] + injected[e]
        with fits.open(Path('pybdsf_files') / f'catalog_{obs_isot[e]}_{project}.fits') as hdul:
            raw = tangent_plane_arcsec(hdul[1].data['RA'], hdul[1].data['DEC'], phase_centre)
        # The matching region of matching.match_epochs: the epoch's beam ellipse scaled by 2 match_threshold.
        in_raw = (cKDTree(beam_frame(raw, *beams[e])).query(beam_frame(position, *beams[e]))[0] < match_threshold
                  if len(raw) else np.zeros(len(missed), bool))
        cause[missed, e] = np.where(in_raw, 1, 0)

    edges = 10 ** np.arange(0.0, 3.0 + 1e-9, 0.1)
    index = np.digitize(true_snr.ravel(), edges) - 1
    keep = (index >= 0) & (index < len(edges) - 1)
    seen = keep & ~left_out.ravel()

    def counts(use):
        return np.bincount(index[use], minlength=len(edges) - 1)

    def fraction(weights, use):
        total = counts(use)
        return np.divide(np.bincount(index[use], weights=weights[use], minlength=len(edges) - 1), total,
                         out=np.full(total.shape, np.nan), where=total > 0)

    def binomial_error(f, total):
        return np.sqrt(np.divide(f * (1 - f), total, out=np.full(total.shape, np.nan), where=total > 0))

    n, n_measured = counts(keep), counts(seen)
    frac = fraction(detected.ravel().astype(float), keep)
    frac_measured = fraction(measured.ravel().astype(float), seen)
    lost = ({name: fraction((cause.ravel() == k).astype(float), keep) for k, name in enumerate(CAUSES)}
            if traced else {})
    out = {'snr_centre': np.sqrt(edges[:-1] * edges[1:]), 'completeness': frac,
           'completeness_err': binomial_error(frac, n),
           'measured': frac_measured, 'measured_err': binomial_error(frac_measured, n_measured),
           'n': n, 'n_measured': n_measured, 'lost': lost, 'threshold': censor,
           'pybdsf_threshold': float(results['config']['PYBDSF']['snr_threshold']),
           # Without a flux density fit there is no fitted peak noise; the step is then drawn for the rms alone.
           'width': float(results['parameters']['peak_noise']['median']) if 'peak_noise' in results['parameters']
           else 1.0,
           'flux_fractional': (float(results['parameters']['flux_fractional']['median'])
                               if 'flux_fractional' in results['parameters'] else 0.0),
           'floor': float(mcmc['detection_floor']), 'detect_threshold': float(mcmc['detection_censor_snr'])}
    out['model_snr'] = np.logspace(0.0, 3.0, 300)
    sigma = np.hypot(out['width'], out['flux_fractional'] * out['model_snr'])
    out['model'] = (1.0 - out['floor']) * ndtr((out['model_snr'] - out['threshold']) / sigma)
    out['model_detected'] = (1.0 - out['floor']) * ndtr((out['model_snr'] - out['detect_threshold']) / sigma)

    n_missed = int((~detected).sum())
    shares = ', '.join(f"{name} {np.sum(cause == k) / max(n_missed, 1) * 100:.0f}%" for k, name in enumerate(CAUSES))
    msg(f"  Completeness from {int(n.sum())} source-epoch pairs at true SNR 1 to 1000; {n_missed} misses"
        + (f": {shares}" if traced else ""))
    if left_out.any():
        msg(f"    {int(left_out.sum())} C or M detections ({left_out.sum() / detected.size * 100:.2f}% of pairs) "
            f"count as detected but are left out of the measured fraction, as in the fit")
    for low, high in ((5, 10), (10, 30), (30, 1000)):
        sel = (true_snr >= low) & (true_snr < high)
        if sel.any():
            msg(f"    true SNR {low}-{high}: detected {detected[sel].mean() * 100:.1f}%"
                + ("; lost to " + ', '.join(f"{name} {np.mean(cause[sel] == k) * 100:.1f}%"
                                            for k, name in enumerate(CAUSES)) if traced else ""))
    return out


def truth_file(args, config):
    """The simulation truth file to compare with: --truth, or per_epoch_offsets.csv beside the images."""
    if args.truth is not None:
        return args.truth
    path = Path(config['BOOKKEEPING'].get('image_dir', '.')) / 'per_epoch_offsets.csv'
    return path if path.exists() else None


# ============================================================================
# OUTPUTS
# ============================================================================

def write_translations(trans, meta):
    """
    Every epoch's translation to offsets_translations_allepochs.csv (table) and .npz (complete).

    The table has one row per catalogue epoch: whether the fit used it, its
    observation time, the posterior median and standard deviation of the RA
    and Dec translation (arcsec), their correlation, the number of fitted
    sources measured, and the frame check for fitted epochs. The npz file holds
    every draw and the full covariance between all epochs and both components.
    """
    obs_isot = meta['obs_isot'] or ['unknown'] * len(trans['median'])
    with open(TRANSLATIONS_TABLE, 'w') as f:
        f.write("catalog_epoch,in_fit,obs_isot,ra_offset_arcsec,dec_offset_arcsec,ra_err_arcsec,dec_err_arcsec,"
                "corr_ra_dec,n_sources,frame_check_sigma\n")
        for e, (m, sd) in enumerate(zip(trans['median'], trans['sd'])):
            f.write(f"{e},{int(trans['in_fit'][e])},{obs_isot[e]},{m[0]:.6f},{m[1]:.6f},{sd[0]:.6f},{sd[1]:.6f},"
                    f"{trans['corr_ra_dec'][e]:.4f},{trans['n_sources'][e]},{trans['frame_check'][e]:.3f}\n")
    msg(f"Epoch translations saved to {TRANSLATIONS_TABLE}")

    np.savez(TRANSLATIONS_FILE, **{k: np.asarray(v) for k, v in trans.items() if k != 'p_outlier'},
             **{k: np.asarray(v) for k, v in meta.items() if v is not None},
             description="Translations (RA, Dec; arcsec) of every catalogue epoch in the AstKAT joint fit's "
                         "frame. draws (n_draws, n_epochs, 2); median, mean, sd (n_epochs, 2); covariance "
                         "(n_epochs, 2, n_epochs, 2) with [a, i, b, j] the covariance of component i of epoch a "
                         "and component j of epoch b (0 RA, 1 Dec); in_fit marks epochs the fit used.")
    msg(f"Translation draws and covariance saved to {TRANSLATIONS_FILE}")


def write_offsets_summary(offsets):
    """Readable summary of the all-epoch translations and their affine check to results/offsets_summary.txt."""
    trans = offsets['translations']
    in_fit = trans['in_fit']
    has = np.isfinite(trans['median'][:, 0])
    lines = [
        "AstKAT epoch offsets summary",
        "============================",
        f"{'Fit used':<52}{offsets['fit_created']}",
        f"{'Catalogue epochs':<52}{len(in_fit)}",
        f"{'Fitted epochs':<52}{int(in_fit.sum())}",
        f"{'Added epochs with a translation':<52}{int((~in_fit & has).sum())}",
        f"{f'Added epochs left out (fewer than {MIN_SOURCES} sources)':<52}{int((~in_fit & ~has).sum())}",
        f"{'Largest frame-check difference (fitted epochs)':<52}"
        + (f"{np.nanmax(trans['frame_check']):.2f} sigma" if np.isfinite(trans['frame_check']).any()
           else "not applicable (no fitted translations)"),
        f"{'Translation rms, RA and Dec':<52}"
        f"{np.sqrt(np.nanmean(trans['median'][:, 0] ** 2)) * 1e3:.1f}, "
        f"{np.sqrt(np.nanmean(trans['median'][:, 1] ** 2)) * 1e3:.1f} mas",
    ]
    lines += affine_summary_lines(offsets['affine'])
    truth = offsets.get('truth')
    if truth:
        lines += ["", "Comparison with simulation truth", "--------------------------------",
                  f"{'Truth file':<52}{truth['file']}",
                  f"{'chi2 / dof with the full covariance':<52}{truth['chi2']:.1f} / {truth['dof']}"]
        for name in ('fitted', 'added'):
            if name not in truth:
                continue
            o, label = truth[name], f"{name.capitalize()} epochs"
            rms, mean, std = o['rms_mas'], o['chi_mean'], o['chi_std']
            lines += [f"{label + ' (' + str(o['n']) + '), rms RA, Dec':<52}{rms[0]:.2f}, {rms[1]:.2f} mas",
                      f"{label + ', chi mean +/- std, RA':<52}{mean[0]:+.2f} +/- {std[0]:.2f}",
                      f"{label + ', chi mean +/- std, Dec':<52}{mean[1]:+.2f} +/- {std[1]:.2f}"]
    src = offsets.get('truth_sources')
    if src:
        rms, mean, std = src['rms_mas'], src['chi_mean'], src['chi_std']
        lines += ["", "Sources against simulation truth", "--------------------------------",
                  f"{'Truth file':<52}{src['file']}",
                  f"{'Fitted sources matched to injected ones':<52}{len(src['fit_index'])} of {src['n_fitted']}",
                  f"{'Position rms, RA and Dec':<52}{rms[0]:.2f}, {rms[1]:.2f} mas",
                  f"{'Position chi mean +/- std, RA':<52}{mean[0]:+.2f} +/- {std[0]:.2f}",
                  f"{'Position chi mean +/- std, Dec':<52}{mean[1]:+.2f} +/- {std[1]:.2f}",
                  f"{'Position chi2 / dof with the full covariance':<52}{src['chi2']:.1f} / {src['dof']}",
                  f"{'Flux densities':<52}{src['flux_method']}",
                  f"{'Median flux density ratio, recovered / injected':<52}{src['median_flux_ratio']:.3f}",
                  f"{'Flux density chi mean +/- std':<52}{src['flux_chi_mean']:+.2f} +/- "
                  f"{src['flux_chi_std']:.2f}"]
    with open(SUMMARY_FILE, 'w') as f:
        f.write("\n".join(lines) + "\n")
    msg(f"Readable offsets summary saved to {SUMMARY_FILE}")


def save_offsets_results(results, data, trans, meta, affine, residuals, truth, truth_sources=None):
    """
    Save everything later steps and figures need to results/offsets_results.pkl (plain types only).

    Returns
    -------
    dict
        The saved results.
    """
    out = {'created': Time.now().isot[:19], 'fit_created': results['created'], 'model': results['model'],
           'data': asdict(data), 'translations': trans, 'epochs': meta, 'affine': affine,
           'residuals': residuals, 'truth': truth, 'truth_sources': truth_sources}
    with open(OFFSETS_RESULTS_FILE, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    msg(f"Offsets results saved to {OFFSETS_RESULTS_FILE}")
    return out


def make_plots(offsets):
    """Draw the offsets figures with plot.py, if it is present."""
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
    plotting.offsets_figures(offsets)


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Command-line options for offsets.py."""
    parser = argparse.ArgumentParser(description="AstKAT epoch translations for every epoch")
    parser.add_argument('--truth', type=Path, default=None,
                        help='per_epoch_offsets.csv of a simulation to compare with '
                             '(default: the one beside the images, if present)')
    return parser.parse_args()


def main():
    """Give every epoch of the project a translation in the joint fit's frame."""
    args = parse_args()
    title("AstKAT epoch offsets")
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)

    # 1. The fitted frame and every epoch's measurements of the fitted sources.
    results = load_results(with_trace=True)
    msg(f"Using the fit from {results['created']}")
    if results['model']['error_inflation'] > 1:
        msg(f"Measurement errors inflated x{results['model']['error_inflation']:.3f}, as in the fit "
            f"(MCMC.error_inflation)")
    data, in_fit = prepare_all_epochs(config, results)

    # 2. Translations for every epoch, drawn alongside the fit's own posterior.
    mcmc = config.get('MCMC', {}) or {}
    draws = thinned_draws(results['idata'], int(mcmc.get('posterior_predictive_draws', 1000)))
    # Each draw's 1 + g_e over every catalogue epoch; epochs not in the fit have no fitted flux scale.
    draws['gain'] = np.ones((len(draws['mu']), data.n_epochs))
    if 'flux_scale' in draws:
        draws['gain'][:, results['data']['catalog_epochs']] += draws['flux_scale']
    rng = np.random.default_rng(int(config['BOOKKEEPING'].get('seed', DEFAULT_SEED)))
    trans = all_epoch_translations(data, in_fit, results['model'], draws, rng)

    # 3. The fit's affine check repeated over every epoch, added epochs included, and each
    #    epoch's residuals with the distribution the model expects for them.
    affine = offsets_affine(config, results, data, trans)
    residuals = epoch_residuals(results, data, trans)

    # 4. For simulations, the translations against the injected offsets.
    path = truth_file(args, config)
    truth = truth_sources = None
    if path is not None:
        msg(f"Comparison with simulation truth ({path}):")
        truth = compare_with_truth(trans, path)
        truth_sources = compare_sources_with_truth(config, results, in_fit, path)
        if truth_sources is not None:
            truth_sources['completeness'] = completeness_with_truth(config, results, in_fit, truth_sources)

    # 5. Outputs; the npz file is what corrects a target in any epoch.
    meta = epoch_metadata(data)
    write_translations(trans, meta)
    offsets = save_offsets_results(results, data, trans, meta, affine, residuals, truth, truth_sources)
    write_offsets_summary(offsets)

    # 6. Figures; epochs added after the fit are drawn as stars.
    make_plots(offsets)
    return offsets


if __name__ == "__main__":
    main()
