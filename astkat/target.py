"""
AstKAT target astrometry.

Corrects a target's measured positions with the epoch translations from
offsets.py and gives the corrected positions their full covariance: each
measurement's own noise from the fitted noise model, plus the translation
uncertainty, shared by measurements in the same epoch and correlated between
epochs. Run after fit.py and offsets.py.

A target's SNR comes from one measured peak, so the noise model is averaged
over its uncertainty (peak noise k and fractional error f_frac from the
fit) rather than evaluated at the measured value, which near the threshold
is boosted by the detection itself. Flux densities are also put on the
common flux scale of the fit, divided by 1 + g_e of their epoch.

Input is a text table, one row per measurement, with columns flux, rms, ra,
dec, then optionally bmaj, bmin, bpa, then time. Columns are found by header
name (see COLUMN_ALIASES) or, without a header, taken in that order. RA and
Dec are decimal degrees or sexagesimal, times ISO-8601 or MJD. Rows without
a beam take the beam of their epoch.

Usage
    python target.py target.txt
    python target.py target.txt --beam-unit deg
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii as ascii_io
from astropy.time import Time
from scipy.stats import truncnorm

try:
    from .utils import (msg, title, parse_phase_center, observatory_location, correct_positions_for_aberration,
                        is_simulated, aberration_correction_enabled, DEFAULT_SEED)
    from .fit import (FitData, load_results, tangent_plane_arcsec, tangent_plane_to_radec,
                      noise_variances, FWHM_TO_SIGMA)
    from .offsets import TRANSLATIONS_FILE, CONFIG_FILE, thinned_draws
except ImportError:
    from utils import (msg, title, parse_phase_center, observatory_location, correct_positions_for_aberration,
                       is_simulated, aberration_correction_enabled, DEFAULT_SEED)
    from fit import (FitData, load_results, tangent_plane_arcsec, tangent_plane_to_radec,
                     noise_variances, FWHM_TO_SIGMA)
    from offsets import TRANSLATIONS_FILE, CONFIG_FILE, thinned_draws


RESULTS_DIR = Path('results')
MATCH_TOLERANCE_HOURS = 12.0     # largest gap between a row's time and its epoch's DATE-OBS

CORE_FIELDS = ('flux', 'rms', 'ra', 'dec')
BEAM_FIELDS = ('bmaj', 'bmin', 'bpa')
COLUMN_ALIASES = {
    'flux': ('flux', 'flux_density', 'total_flux', 'peak_flux', 's', 's_peak', 's_total'),
    'rms': ('rms', 'rms_error', 'noise', 'local_rms', 'err_rms', 'e_rms', 'sigma'),
    'ra': ('ra', 'right_ascension', 'alpha'),
    'dec': ('dec', 'declination', 'delta'),
    'bmaj': ('bmaj', 'maj', 'major', 'dc_maj'),
    'bmin': ('bmin', 'min', 'minor', 'dc_min'),
    'bpa': ('bpa', 'pa', 'position_angle', 'dc_pa'),
    'time': ('time', 'obs_time', 'date_obs', 'datetime', 'isot', 'mjd', 'obs_isot'),
}
BEAM_UNIT_TO_ARCSEC = {'arcsec': 1.0, 'arcmin': 60.0, 'deg': 3600.0}


# ============================================================================
# INPUT
# ============================================================================

def parse_positions(ra_col, dec_col):
    """RA and Dec in degrees from decimal-degree or sexagesimal (hourangle RA) columns."""
    ra, dec = [str(v) for v in ra_col], [str(v) for v in dec_col]
    try:
        return np.array(ra, dtype=float), np.array(dec, dtype=float)
    except ValueError:
        coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
        return coords.ra.deg, coords.dec.deg


def parse_times(time_col):
    """Observation times from an ISO-8601 or MJD column."""
    values = [str(v) for v in time_col]
    try:
        return Time(values, format='isot', scale='utc')
    except ValueError:
        return Time(np.array(values, dtype=float), format='mjd', scale='utc')


def load_catalog(path, beam_unit='arcsec'):
    """
    Read the target catalogue.

    Returns
    -------
    dict
        flux, rms, ra_deg, dec_deg (n,); time (Time); bmaj_arcsec,
        bmin_arcsec, bpa_deg (n,), or None without beam columns.
    """
    table = ascii_io.read(str(path))
    names = {c.lower(): c for c in table.colnames}
    def find(field):
        return next((names[a] for a in COLUMN_ALIASES[field] if a in names), None)

    # astropy names header-less columns col1, col2, ...
    headerless = all(re.match(r'col\d+$', c) for c in table.colnames)
    if not headerless and all(find(f) for f in CORE_FIELDS + ('time',)):
        mapping = {f: find(f) for f in CORE_FIELDS + ('time',)}
        if all(find(f) for f in BEAM_FIELDS):
            mapping.update({f: find(f) for f in BEAM_FIELDS})
    elif len(table.colnames) == 8:
        mapping = dict(zip(CORE_FIELDS + BEAM_FIELDS + ('time',), table.colnames))
    elif len(table.colnames) == 5:
        mapping = dict(zip(CORE_FIELDS + ('time',), table.colnames))
    else:
        raise ValueError(f"{path}: need columns flux, rms, ra, dec (, bmaj, bmin, bpa), time; "
                         f"found {table.colnames}")

    ra, dec = parse_positions(table[mapping['ra']], table[mapping['dec']])
    cat = {'flux': np.asarray(table[mapping['flux']], dtype=float),
           'rms': np.asarray(table[mapping['rms']], dtype=float),
           'ra_deg': ra, 'dec_deg': dec, 'time': parse_times(table[mapping['time']]),
           'bmaj_arcsec': None, 'bmin_arcsec': None, 'bpa_deg': None}
    if 'bmaj' in mapping:
        scale = BEAM_UNIT_TO_ARCSEC[beam_unit]
        cat['bmaj_arcsec'] = np.asarray(table[mapping['bmaj']], dtype=float) * scale
        cat['bmin_arcsec'] = np.asarray(table[mapping['bmin']], dtype=float) * scale
        cat['bpa_deg'] = np.asarray(table[mapping['bpa']], dtype=float)
    msg(f"Loaded {path}: {len(ra)} measurements")
    return cat


def load_translations(path=TRANSLATIONS_FILE):
    """Translations of every catalogue epoch from offsets.py (see offsets.write_translations)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; run offsets.py first")
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def match_epochs(times, trans, tolerance_hours):
    """
    Catalogue epoch of each row, the one with the nearest DATE-OBS.

    Every row must be within tolerance_hours of an epoch with a translation.
    """
    epoch_times = Time([str(t) for t in trans['obs_isot']], format='isot', scale='utc')
    gap = np.abs(times.mjd[:, None] - epoch_times.mjd[None, :]) * 24.0
    epoch = np.argmin(gap, axis=1)
    nearest = gap[np.arange(len(epoch)), epoch]
    bad = np.nonzero((nearest > tolerance_hours) | ~np.isfinite(trans['median'][epoch, 0]))[0]
    if bad.size:
        rows = "\n".join(f"  row {i}: {times[i].isot}, nearest epoch {epoch[i]} {nearest[i]:.1f} h away"
                         + ("" if nearest[i] > tolerance_hours else ", no translation") for i in bad)
        raise ValueError(f"{bad.size} rows have no usable epoch:\n{rows}")
    msg(f"Rows matched to epochs, largest time gap {nearest.max():.2f} h")
    return epoch


def epoch_beam(cat, epoch, results_dir=RESULTS_DIR):
    """Beam of each row (bmaj, bmin in arcsec, bpa in deg), from the catalogue or its epoch."""
    if cat['bmaj_arcsec'] is not None:
        return cat['bmaj_arcsec'], cat['bmin_arcsec'], cat['bpa_deg']
    with open(Path(results_dir) / 'observation_properties.json', 'r') as f:
        props = json.load(f)
    msg("Beams taken from each row's epoch")
    return (np.asarray(props['bmaj_deg'])[epoch] * 3600.0, np.asarray(props['bmin_deg'])[epoch] * 3600.0,
            np.asarray(props['bpa_deg'])[epoch])


# ============================================================================
# CORRECTION
# ============================================================================

def correct_aberration(cat, epoch, trans, config):
    """
    Remove differential aberration at each epoch's DATE-OBS, under the same rule as offsets.py.

    Skipped for simulated projects and when MCMC.correct_differential_aberration
    is false in astkat.yaml, exactly as for the reference sources.
    """
    if is_simulated(config) or not aberration_correction_enabled(config):
        msg("Aberration correction not applied, as for the reference sources")
        return cat['ra_deg'], cat['dec_deg']
    times = Time([str(trans['obs_isot'][e]) for e in epoch], format='isot', scale='utc')
    return correct_positions_for_aberration(cat['ra_deg'], cat['dec_deg'],
                                            parse_phase_center(config['BOOKKEEPING']['phase_ctr']),
                                            times, observatory_location(config))


def noise_covariance(snr, bmaj, bmin, bpa, opts, draws, rng=None):
    """
    Measurement noise covariance of each row in (RA, Dec), averaged over posterior draws, arcsec^2.

    Uses the fit's inlier noise model with each row's own beam. The SNR is
    one measured peak over its rms, so in each draw every row's true SNR is
    drawn from N(SNR, sqrt(k^2 + (f_frac SNR)^2)) truncated at zero, with k
    and f_frac that draw's peak noise and fractional error; the average over
    draws then carries the SNR uncertainty, which matters near the threshold.
    Without rng (the SNR is the true one, as when simulating) or without a
    flux density fit, the SNR is used as it is.

    Returns
    -------
    ndarray, shape (n, 2, 2)
    """
    n = len(snr)
    pa = np.radians(bpa)
    u_maj = np.column_stack([np.sin(pa), np.cos(pa)])
    u_min = np.column_stack([-np.cos(pa), np.sin(pa)])
    # Each row is its own epoch, so fit.noise_variances applies its beam directly.
    rows = FitData(x=np.zeros((n, 2)), snr=snr, source=np.zeros(n, dtype=int), epoch=np.arange(n),
                   var_min=(bmin * FWHM_TO_SIGMA) ** 2, var_maj=(bmaj * FWHM_TO_SIGMA) ** 2,
                   u_min=u_min, u_maj=u_maj, source_median=np.zeros((1, 2)), pixel_arcsec=0.0,
                   catalog_sources=np.zeros(1, dtype=int), catalog_epochs=np.arange(n))

    cov = np.zeros((n, 2, 2))
    n_draws = len(draws['mu'])
    for k in range(n_draws):
        p = {name: float(draws[name][k]) if name in draws else default
             for name, default in (('A', 0.0), ('beta', 1.0), ('snr_break', 1.0), ('snr_gamma', 0.0), ('B', 0.0),
                                   ('alpha', 1.0))}
        if rng is not None and 'peak_noise' in draws:
            ffrac = float(draws['flux_fractional'][k]) if 'flux_fractional' in draws else 0.0
            spread = np.hypot(float(draws['peak_noise'][k]), ffrac * snr)
            p['snr'] = truncnorm.rvs(-snr / spread, np.inf, loc=snr, scale=spread, random_state=rng)
        v_min, v_maj = noise_variances(rows, p, opts)
        cov += (v_min[:, None, None] * u_min[:, :, None] * u_min[:, None, :]
                + v_maj[:, None, None] * u_maj[:, :, None] * u_maj[:, None, :])
    return cov / n_draws


def radial_offsets(p, cov, reference=0):
    """
    Distance of every row from the first row's corrected position, and its error from the full covariance.

    d_i = p_i - p_ref, so Cov(d_i) = C_ii + C_rr - C_ir - C_ri; the radial error
    is that projected on the direction of d_i. Correlations through shared
    or correlated epochs are therefore included.

    Returns
    -------
    r, r_err : ndarray, shape (n,), arcsec
        Zero for the reference row itself.
    """
    d = p - p[reference]
    r = np.hypot(d[:, 0], d[:, 1])
    g = np.divide(d, r[:, None], out=np.zeros_like(d), where=r[:, None] > 0)
    n = len(p)
    rows = np.arange(n)
    cov_d = (cov[rows, :, rows, :] + cov[reference, :, reference, :][None]
             - cov[rows, :, reference, :] - cov[reference, :, rows, :])
    return r, np.sqrt(np.einsum('ia,iab,ib->i', g, cov_d, g))


def correct_target(x, epoch, noise_cov, trans):
    """
    Corrected positions and their full covariance.

    p_i = x_i - tau_e(i), with
    Cov(p_i, p_j) = delta_ij Sigma_i + C_tau[e(i), e(j)].

    Returns
    -------
    p : ndarray, shape (n, 2)
        RA and Dec offsets from the phase centre, arcsec.
    cov : ndarray, shape (n, 2, n, 2)
        cov[i, a, j, b] is the covariance of component a of row i and component b of row j.
    """
    n = len(epoch)
    p = x - trans['median'][epoch]
    cov = trans['covariance'][epoch][:, :, epoch, :].copy()
    cov[np.arange(n), :, np.arange(n), :] += noise_cov
    return p, cov


# ============================================================================
# SIMULATION TRUTH
# ============================================================================

def truth_file(args):
    """The true track to compare with: --truth, or {stem}_truth.csv beside the catalogue (from simulate.py)."""
    if args.truth is not None:
        return args.truth
    path = args.catalog.with_name(f'{args.catalog.stem}_truth.csv')
    return path if path.exists() else None


def compare_with_truth(p, cov, path, phase_centre):
    """
    Corrected positions against the true track a simulation injected.

    The simulated measurements already sit in the fitted frame, so no shift
    is needed: corrected minus true is the translation error plus the noise.

    Returns
    -------
    dict or None
        true_xy (n, 2) arcsec; true_radial (n,), distance from the first
        true position; diff (n, 2) corrected minus true, arcsec;
        chi = diff / sd per component; chi2 and dof from the full
        covariance; rms_mas, chi_mean, chi_std per axis; file. None if
        the rows do not match the catalogue.
    """
    table = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')
    if len(table) != len(p):
        msg(f"WARNING: {path} has {len(table)} rows but the catalogue has {len(p)}; truth comparison skipped")
        return None
    true_xy = tangent_plane_arcsec(np.asarray(table['true_ra_deg'], float), np.asarray(table['true_dec_deg'], float),
                                   phase_centre)
    n = len(p)
    diff = p - true_xy
    sd = np.sqrt(cov[np.arange(n), :, np.arange(n), :][:, [0, 1], [0, 1]])
    chi = diff / sd
    d = diff.ravel()
    chi2 = float(d @ np.linalg.solve(cov.reshape(2 * n, 2 * n), d))
    out = {'true_xy': true_xy, 'true_radial': np.hypot(*(true_xy - true_xy[0]).T),
           'diff': diff, 'chi': chi, 'chi2': chi2, 'dof': 2 * n,
           'rms_mas': np.sqrt(np.mean(diff ** 2, axis=0)) * 1e3, 'chi_mean': chi.mean(axis=0),
           'chi_std': chi.std(axis=0), 'file': str(path)}
    msg(f"Truth ({path}): rms {out['rms_mas'][0]:.2f} (RA), {out['rms_mas'][1]:.2f} (Dec) mas; "
        f"chi {out['chi_mean'][0]:+.2f} +/- {out['chi_std'][0]:.2f}, "
        f"{out['chi_mean'][1]:+.2f} +/- {out['chi_std'][1]:.2f}; chi2/dof {chi2:.1f}/{2 * n}")
    return out


# ============================================================================
# OUTPUT
# ============================================================================

def corrected_fluxes(cat, epoch, results):
    """
    Flux densities on the common flux scale of the fit, and their errors.

    Each row's flux is divided by 1 + g_e of its epoch (posterior median of
    the fit's flux scale). The error combines the peak noise k rms and the
    fractional error f_frac of the fit, as for the field sources, with the
    error of g_e:
    sigma = sqrt((k rms)^2 + (f_frac S)^2 + (S sigma_g / (1 + g))^2) / (1 + g).
    Epochs without a fitted scale (other bands, where it would mix the gain
    with the spectral index) are NaN.

    Returns
    -------
    dict or None
        scale, scale_err, flux, flux_err, each (n,); None without fitted flux scales.
    """
    scale = results['flux_scale']
    if scale is None:
        return None
    n_epochs = max(int(scale['catalog_epochs'].max()), int(epoch.max())) + 1
    g, g_err = np.full(n_epochs, np.nan), np.full(n_epochs, np.nan)
    g[scale['catalog_epochs']], g_err[scale['catalog_epochs']] = scale['median'], scale['sd']
    g, g_err = g[epoch], g_err[epoch]
    parameters = results['parameters']
    k = parameters['peak_noise']['median']
    ffrac = parameters['flux_fractional']['median'] if 'flux_fractional' in parameters else 0.0
    gain, flux = 1.0 + g, cat['flux']
    error = np.sqrt((k * cat['rms']) ** 2 + (ffrac * flux) ** 2 + (flux * g_err / gain) ** 2) / gain
    msg(f"Flux densities on the common flux scale: {int(np.isfinite(g).sum())} of {len(g)} rows "
        f"(epochs without a fitted scale left NaN); scales {np.nanmin(g) * 100:+.1f}% to {np.nanmax(g) * 100:+.1f}%")
    return {'scale': g, 'scale_err': g_err, 'flux': flux / gain, 'flux_err': error}


def write_results(stem, cat, ra, dec, epoch, trans, snr, bmaj, bmin, bpa, noise_cov, cov, r, r_err, fluxes,
                  results_dir=RESULTS_DIR):
    """
    Corrected positions to results/target_{stem}.csv and the full covariance to target_{stem}_covariance.npz.

    Errors are along RA (RA cos Dec) and Dec, arcsec. *_meas is the
    measurement noise alone, *_corr the translation alone, and RA_err,
    Dec_err the total; RA_Dec_corr is the RA-Dec correlation of the total.
    Radial_offset is the distance from the first row's corrected position,
    with its error from the full covariance (see radial_offsets). With
    fitted flux scales, Flux_scale and Flux_scale_err are the epoch's g_e and
    Flux_corrected, Flux_corrected_err the flux density on the common scale
    (see corrected_fluxes).
    """
    n = len(epoch)
    diag = np.arange(n)
    total = cov[diag, :, diag, :]
    tau_cov = trans['covariance'][epoch, :, epoch, :]
    sd = np.sqrt(total[:, [0, 1], [0, 1]])
    columns = {
        'Source_Index': diag, 'Flux': cat['flux'], 'RMS_Error': cat['rms'], 'SNR': snr,
        'RA_deg_raw': cat['ra_deg'], 'Dec_deg_raw': cat['dec_deg'], 'RA_deg': ra, 'Dec_deg': dec,
        'Time_MJD': cat['time'].mjd, 'Epoch': epoch, 'In_fit': trans['in_fit'][epoch].astype(int),
        'RA_corr_applied_arcsec': trans['median'][epoch, 0], 'Dec_corr_applied_arcsec': trans['median'][epoch, 1],
        'RA_err_meas_arcsec': np.sqrt(noise_cov[:, 0, 0]), 'Dec_err_meas_arcsec': np.sqrt(noise_cov[:, 1, 1]),
        'RA_err_corr_arcsec': np.sqrt(tau_cov[:, 0, 0]), 'Dec_err_corr_arcsec': np.sqrt(tau_cov[:, 1, 1]),
        'RA_err_arcsec': sd[:, 0], 'Dec_err_arcsec': sd[:, 1], 'RA_Dec_corr': total[:, 0, 1] / (sd[:, 0] * sd[:, 1]),
        'Radial_offset_arcsec': r, 'Radial_offset_err_arcsec': r_err,
        'BMAJ_arcsec': bmaj, 'BMIN_arcsec': bmin, 'BPA_deg': bpa,
    }
    if fluxes is not None:
        columns.update({'Flux_scale': fluxes['scale'], 'Flux_scale_err': fluxes['scale_err'],
                        'Flux_corrected': fluxes['flux'], 'Flux_corrected_err': fluxes['flux_err']})
    path = Path(results_dir) / f'target_{stem}.csv'
    np.savetxt(path, np.column_stack(list(columns.values())), delimiter=',', comments='', fmt='%.10g',
               header=','.join(columns))
    msg(f"Corrected positions saved to {path}")

    cov_path = Path(results_dir) / f'target_{stem}_covariance.npz'
    np.savez(cov_path, ra_deg=ra, dec_deg=dec, covariance=cov, epoch=epoch,
             description="Corrected target positions. covariance (n, 2, n, 2), arcsec^2, with [i, a, j, b] the "
                         "covariance of component a of row i and component b of row j (0 RA, 1 Dec).")
    msg(f"Full covariance saved to {cov_path}")


def write_summary(stem, target, aberrated, results_dir=RESULTS_DIR):
    """Readable summary of the corrected target, and the truth comparison if any, to results/target_{stem}_summary.txt."""
    n = len(target['epoch'])
    diag = np.arange(n)
    total = np.sqrt(target['cov'][diag, :, diag, :][:, [0, 1], [0, 1]]) * 1e3
    meas = np.sqrt(target['noise_cov'][:, [0, 1], [0, 1]]) * 1e3
    corr = np.sqrt(np.maximum(total ** 2 - meas ** 2, 0.0))
    lines = [
        "AstKAT target summary",
        "=====================",
        f"{'Catalogue':<52}{target['catalog']}",
        f"{'Rows':<52}{n} in {len(np.unique(target['epoch']))} epochs "
        f"({int((~target['in_fit']).sum())} in epochs added after the fit)",
        f"{'Aberration correction':<52}{'applied' if aberrated else 'not applied'}",
        f"{'SNR':<52}{target['snr'].min():.1f} to {target['snr'].max():.1f}",
        f"{'Median error, measurement, RA and Dec':<52}{np.median(meas[:, 0]):.2f}, {np.median(meas[:, 1]):.2f} mas",
        f"{'Median error, translation, RA and Dec':<52}{np.median(corr[:, 0]):.2f}, {np.median(corr[:, 1]):.2f} mas",
        f"{'Median error, total, RA and Dec':<52}{np.median(total[:, 0]):.2f}, {np.median(total[:, 1]):.2f} mas",
    ]
    truth = target.get('truth')
    if truth:
        rms, mean, std = truth['rms_mas'], truth['chi_mean'], truth['chi_std']
        lines += ["", "Comparison with simulation truth", "--------------------------------",
                  f"{'Truth file':<52}{truth['file']}",
                  f"{'rms, RA and Dec':<52}{rms[0]:.2f}, {rms[1]:.2f} mas",
                  f"{'Chi mean +/- std, RA':<52}{mean[0]:+.2f} +/- {std[0]:.2f}",
                  f"{'Chi mean +/- std, Dec':<52}{mean[1]:+.2f} +/- {std[1]:.2f}",
                  f"{'chi2 / dof with the full covariance':<52}{truth['chi2']:.1f} / {truth['dof']}"]
    path = Path(results_dir) / f'target_{stem}_summary.txt'
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    msg(f"Readable target summary saved to {path}")


def make_plots(target):
    """Draw the target figure with plot.py, if it is present."""
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
    plotting.target_figure(target)


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Command-line options for target.py."""
    parser = argparse.ArgumentParser(description="AstKAT target positions corrected with the epoch translations")
    parser.add_argument('catalog', type=Path, help='target catalogue, one row per measurement')
    parser.add_argument('--beam-unit', choices=sorted(BEAM_UNIT_TO_ARCSEC), default='arcsec',
                        help='unit of bmaj and bmin in the catalogue (default: arcsec)')
    parser.add_argument('--match-tolerance-hours', type=float, default=MATCH_TOLERANCE_HOURS,
                        help=f'largest gap between a row and its epoch (default: {MATCH_TOLERANCE_HOURS:g})')
    parser.add_argument('--truth', type=Path, default=None,
                        help='true track of a simulated target (default: {stem}_truth.csv beside the catalogue, '
                             'if present)')
    return parser.parse_args()


def main():
    """Correct a target's positions with the epoch translations and give them a full covariance."""
    args = parse_args()
    title("AstKAT target astrometry")
    results = load_results(with_trace=True)
    config = results['config']
    if results['model']['error_inflation'] > 1:
        msg(f"Measurement errors inflated x{results['model']['error_inflation']:.3f}, as in the fit "
            f"(MCMC.error_inflation)")
    trans = load_translations()
    cat = load_catalog(args.catalog, args.beam_unit)

    # Each row takes its epoch's translation, beam and aberration correction.
    epoch = match_epochs(cat['time'], trans, args.match_tolerance_hours)
    bmaj, bmin, bpa = epoch_beam(cat, epoch)
    with open(CONFIG_FILE, 'r') as f:
        project_config = yaml.safe_load(f)
    aberrated = not is_simulated(project_config) and aberration_correction_enabled(project_config)
    ra, dec = correct_aberration(cat, epoch, trans, project_config)

    # Same projection about the same phase centre as the fit.
    phase_centre = parse_phase_center(config['BOOKKEEPING']['phase_ctr'])
    x = tangent_plane_arcsec(ra, dec, phase_centre)

    snr = cat['flux'] / cat['rms']
    draws = thinned_draws(results['idata'], int(config['MCMC'].get('posterior_predictive_draws', 1000)))
    rng = np.random.default_rng(int(project_config['BOOKKEEPING'].get('seed', DEFAULT_SEED)))
    noise_cov = noise_covariance(snr, bmaj, bmin, bpa, results['model'], draws, rng)
    if 'peak_noise' in draws:
        msg(f"Measurement errors averaged over each row's SNR uncertainty (peak noise k "
            f"{np.median(draws['peak_noise']):.3f}"
            + (f", f_frac {np.median(draws['flux_fractional']):.4f}" if 'flux_fractional' in draws else "") + ")")
    fluxes = corrected_fluxes(cat, epoch, results)
    p, cov = correct_target(x, epoch, noise_cov, trans)

    ra_corr, dec_corr = tangent_plane_to_radec(p, phase_centre)
    stem = args.catalog.stem
    r, r_err = radial_offsets(p, cov)
    write_results(stem, cat, ra_corr, dec_corr, epoch, trans, snr, bmaj, bmin, bpa, noise_cov, cov, r, r_err,
                  fluxes)

    # For a simulated target, the corrected track against the injected one.
    path = truth_file(args)
    truth = compare_with_truth(p, cov, path, phase_centre) if path is not None else None

    target = {'stem': stem, 'catalog': str(args.catalog), 'mjd': cat['time'].mjd, 'epoch': epoch,
              'in_fit': trans['in_fit'][epoch].astype(bool), 'snr': snr, 'measured': x, 'p': p, 'cov': cov,
              'noise_cov': noise_cov, 'radial': r, 'radial_err': r_err, 'truth': truth}
    write_summary(stem, target, aberrated)
    make_plots(target)
    return target


if __name__ == "__main__":
    main()
