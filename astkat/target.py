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

Optional columns sys_ra, sys_dec and sys_cov (arcsec, arcsec, arcsec^2) give a
row its own systematic error, whatever its origin. It is added to that row's
noise covariance, on top of the fitted noise model; NaN adds nothing, and
sys_cov defaults to zero. --no-systematic ignores these columns, and with
--floor-only rows with a systematic take only the floor B from the noise
model, not its SNR term, for a systematic that already holds the thermal noise.

With sys_ra_lo, sys_ra_hi, sys_dec_lo and sys_dec_hi as well (arcsec, towards
lower and higher RA or Dec) the systematic is asymmetric. The covariance stays
symmetric, using sys_ra and sys_dec or, without them, the mean of each pair,
and the results gain lower and upper errors on RA, Dec and the radial offset,
each side added in quadrature to the symmetric remainder.

Radial offsets are measured from a reference: by default the first row's
corrected position; with --reference-rows START END the mean of those rows'
corrected positions (inclusive, 0-based), leaving out any given with
--reference-exclude; or with --reference a fixed
position in decimal degrees, hmsdms or colon-separated hms dms, taken as
exact. A reference built from rows carries its own
uncertainty and its correlation with every row, through the full
covariance, into each offset's error.

Usage
    python target.py target.txt
    python target.py target.txt --beam-unit deg
    python target.py target.txt --reference-rows 0 9
    python target.py target.txt --reference-rows 0 9 --reference-exclude 3 7
    python target.py target.txt --reference-rows 0 40 --reference-exclude 3 12-15
    python target.py target.txt --reference 261.93046 -16.20533
    python target.py target.txt --reference 17h27m43.31s -16d12m19.2s
    python target.py target.txt --reference 17:27:43.31 -16:12:19.2
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
    'sys_ra': ('sys_ra', 'sys_ra_arcsec'),
    'sys_dec': ('sys_dec', 'sys_dec_arcsec'),
    'sys_cov': ('sys_cov', 'sys_cov_arcsec2'),
    'sys_ra_lo': ('sys_ra_lo', 'sys_ra_lo_arcsec'),
    'sys_ra_hi': ('sys_ra_hi', 'sys_ra_hi_arcsec'),
    'sys_dec_lo': ('sys_dec_lo', 'sys_dec_lo_arcsec'),
    'sys_dec_hi': ('sys_dec_hi', 'sys_dec_hi_arcsec'),
}
SYSTEMATIC_SIDE_FIELDS = ('sys_ra_lo', 'sys_ra_hi', 'sys_dec_lo', 'sys_dec_hi')
CATALOGUE_HELP = """catalogue columns, found by header name in any order:
  required      flux, rms, ra, dec, time
  beam          bmaj, bmin, bpa (rows without a beam take their epoch's)
  symmetric     sys_ra, sys_dec (arcsec), with sys_cov (arcsec^2) optional
  asymmetric    sys_ra_lo, sys_ra_hi, sys_dec_lo, sys_dec_hi (arcsec), all four

Without a header, columns are read by position as flux, rms, ra, dec, (bmaj, bmin, bpa,) time,
and systematic columns are not available.

A systematic adds to that row's noise model. Asymmetric sides add lower and upper errors on
RA, Dec and the radial offset to the results; the covariance stays symmetric.
"""
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
        bmin_arcsec, bpa_deg (n,), or None without beam columns;
        systematic_cov (n, 2, 2), arcsec^2, or None without a systematic;
        systematic_sides (n, 2, 2), arcsec, [row, RA or Dec, lower or upper],
        or None without the _lo and _hi columns.
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
                         f"found {table.colnames}. Systematic columns need a header line")

    ra, dec = parse_positions(table[mapping['ra']], table[mapping['dec']])
    cat = {'flux': np.asarray(table[mapping['flux']], dtype=float),
           'rms': np.asarray(table[mapping['rms']], dtype=float),
           'ra_deg': ra, 'dec_deg': dec, 'time': parse_times(table[mapping['time']]),
           'bmaj_arcsec': None, 'bmin_arcsec': None, 'bpa_deg': None, 'systematic_cov': None,
           'systematic_sides': None}
    if 'bmaj' in mapping:
        scale = BEAM_UNIT_TO_ARCSEC[beam_unit]
        cat['bmaj_arcsec'] = np.asarray(table[mapping['bmaj']], dtype=float) * scale
        cat['bmin_arcsec'] = np.asarray(table[mapping['bmin']], dtype=float) * scale
        cat['bpa_deg'] = np.asarray(table[mapping['bpa']], dtype=float)
    sides = [find(f) for f in SYSTEMATIC_SIDE_FIELDS]
    if any(sides) and not all(sides):
        raise ValueError(f"{path}: an asymmetric systematic needs all of {', '.join(SYSTEMATIC_SIDE_FIELDS)}")
    if not headerless and all(sides):
        cat['systematic_sides'] = np.nan_to_num(np.column_stack([np.asarray(table[c], dtype=float) for c in sides])).reshape(-1, 2, 2)
    if not headerless and (cat['systematic_sides'] is not None or (find('sys_ra') and find('sys_dec'))):
        if find('sys_ra') and find('sys_dec'):
            sys_ra = np.nan_to_num(np.asarray(table[find('sys_ra')], dtype=float))
            sys_dec = np.nan_to_num(np.asarray(table[find('sys_dec')], dtype=float))
        else:
            sys_ra, sys_dec = cat['systematic_sides'].mean(axis=2).T
        sys_cov = np.nan_to_num(np.asarray(table[find('sys_cov')], dtype=float)) if find('sys_cov') else np.zeros(len(ra))
        cat['systematic_cov'] = np.array([[[a ** 2, c], [c, b ** 2]] for a, b, c in zip(sys_ra, sys_dec, sys_cov)])
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


def noise_covariance(snr, bmaj, bmin, bpa, opts, draws, rng=None, floor_only=False):
    """
    Measurement noise covariance of each row in (RA, Dec), averaged over posterior draws, arcsec^2.

    Uses the fit's inlier noise model with each row's own beam. The SNR is
    one measured peak over its rms, so in each draw every row's true SNR is
    drawn from N(SNR, sqrt(k^2 + (f_frac SNR)^2)) truncated at zero, with k
    and f_frac that draw's peak noise and fractional error; the average over
    draws then carries the SNR uncertainty, which matters near the threshold.
    Without rng (the SNR is the true one, as when simulating) or without a
    flux density fit, the SNR is used as it is. With floor_only the SNR term
    is dropped and only the floor B remains.

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
        if floor_only:
            p['A'] = 0.0
        elif rng is not None and 'peak_noise' in draws:
            ffrac = float(draws['flux_fractional'][k]) if 'flux_fractional' in draws else 0.0
            spread = np.hypot(float(draws['peak_noise'][k]), ffrac * snr)
            p['snr'] = truncnorm.rvs(-snr / spread, np.inf, loc=snr, scale=spread, random_state=rng)
        v_min, v_maj = noise_variances(rows, p, opts)
        cov += (v_min[:, None, None] * u_min[:, :, None] * u_min[:, None, :]
                + v_maj[:, None, None] * u_maj[:, :, None] * u_maj[:, None, :])
    return cov / n_draws


def reference_position(p, cov, weights=None, fixed=None):
    """
    The reference of the radial offsets: a weighted mean of rows, or a fixed position.

    For rows, ref = sum_j w_j p_j, so Cov(p_i, ref) = sum_j w_j C_ij and
    Cov(ref) = sum_i w_i Cov(p_i, ref). A fixed position is exact.

    Parameters
    ----------
    p : ndarray, shape (n, 2)
    cov : ndarray, shape (n, 2, n, 2)
    weights : ndarray, shape (n,), summing to 1, or None with fixed
    fixed : ndarray, shape (2,), or None
        Tangent-plane offsets of a fixed reference, arcsec.

    Returns
    -------
    ref : ndarray, shape (2,)
    ref_cov : ndarray, shape (2, 2)
    cross : ndarray, shape (n, 2, 2), Cov(p_i, ref)
    """
    if fixed is not None:
        return np.asarray(fixed, float), np.zeros((2, 2)), np.zeros((len(p), 2, 2))
    cross = np.einsum('iajb,j->iab', cov, weights)
    return weights @ p, np.einsum('i,iab->ab', weights, cross), cross


def radial_offsets(p, cov, ref, ref_cov, cross):
    """
    Distance of every row from the reference, and its error from the full covariance.

    d_i = p_i - ref, so Cov(d_i) = C_ii + Cov(ref) - Cov(p_i, ref) - Cov(ref, p_i);
    the radial error is that projected on the direction of d_i. Correlations
    through shared or correlated epochs, and the reference's own error, are
    therefore included.

    Returns
    -------
    r, r_err : ndarray, shape (n,), arcsec
        r_err is 0 for a row at the reference itself.
    """
    d = p - ref
    r = np.hypot(d[:, 0], d[:, 1])
    g = np.divide(d, r[:, None], out=np.zeros_like(d), where=r[:, None] > 0)
    rows = np.arange(len(p))
    cov_d = cov[rows, :, rows, :] + ref_cov[None] - cross - cross.transpose(0, 2, 1)
    return r, np.sqrt(np.einsum('ia,iab,ib->i', g, cov_d, g))


def asymmetric_errors(p, cov, ref, r_err, systematic_cov, sides):
    """
    Lower and upper errors with an asymmetric systematic, each side in quadrature with the symmetric remainder.

    Each row's symmetric systematic in the total covariance is swapped for its
    lower or upper side, keeping its RA-Dec correlation. For the radial offset
    the upper error takes the side pointing away from the reference on each
    axis, and the lower error the side pointing towards it. The reference
    itself stays symmetric. An axis with no sides keeps its symmetric error.

    Parameters
    ----------
    p : ndarray, shape (n, 2)
    cov : ndarray, shape (n, 2, n, 2)
    ref : ndarray, shape (2,)
    r_err : ndarray, shape (n,)
    systematic_cov : ndarray, shape (n, 2, 2)
    sides : ndarray, shape (n, 2, 2)
        [row, RA or Dec, lower or upper], arcsec.

    Returns
    -------
    dict
        position_lo, position_hi (n, 2) and radial_lo, radial_hi (n,), arcsec.
    """
    rows = np.arange(len(p))
    total = cov[rows, :, rows, :][:, [0, 1], [0, 1]]
    symmetric = np.sqrt(systematic_cov[:, [0, 1], [0, 1]])
    sides = np.where(sides.sum(axis=2, keepdims=True) > 0, sides, symmetric[:, :, None])
    correlation = np.divide(systematic_cov[:, 0, 1], symmetric[:, 0] * symmetric[:, 1],
                            out=np.zeros(len(p)), where=symmetric[:, 0] * symmetric[:, 1] > 0)

    def position_error(side):
        return np.sqrt(np.maximum(total + sides[:, :, side] ** 2 - symmetric ** 2, 0.0))

    d = p - ref
    r = np.hypot(d[:, 0], d[:, 1])
    g = np.divide(d, r[:, None], out=np.zeros_like(d), where=r[:, None] > 0)
    away = (g >= 0).astype(int)

    def radial_error(side_index):
        side = np.take_along_axis(sides, side_index[:, :, None], axis=2)[:, :, 0]
        change = (np.sum(g ** 2 * (side ** 2 - symmetric ** 2), axis=1)
                  + 2.0 * g[:, 0] * g[:, 1] * (correlation * side[:, 0] * side[:, 1] - systematic_cov[:, 0, 1]))
        return np.sqrt(np.maximum(r_err ** 2 + change, 0.0))

    return {'position_lo': position_error(0), 'position_hi': position_error(1),
            'radial_lo': radial_error(1 - away), 'radial_hi': radial_error(away)}


def parse_reference(tokens):
    """
    RA and Dec in degrees from --reference: decimal degrees, hmsdms or colon-separated hms dms.

    Two decimal numbers are degrees; anything else is read as RA in hours and
    Dec in degrees, e.g. 17h27m43.31s -16d12m19.2s or 17:27:43.31 -16:12:19.2,
    as one quoted argument or two.
    """
    text = " ".join(tokens).replace(",", " ")
    parts = text.split()
    try:
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except ValueError:
        pass
    try:
        coord = SkyCoord(text, unit=(u.hourangle, u.deg), frame='icrs')
    except ValueError as e:
        raise ValueError(f"--reference '{text}': not decimal degrees, hmsdms or colon-separated hms dms") from e
    return float(coord.ra.deg), float(coord.dec.deg)


def parse_excluded_rows(tokens):
    """
    Row indices from --reference-exclude.

    Parameters
    ----------
    tokens : list of str
        Single rows, or inclusive ranges such as 12-15.

    Returns
    -------
    list of int
        Sorted and unique.
    """
    rows = set()
    for token in tokens:
        match = re.fullmatch(r'(\d+)(?:-(\d+))?', token)
        if match is None:
            raise ValueError(f"--reference-exclude '{token}': give a row or a range such as 12-15")
        first, last = int(match[1]), int(match[2] or match[1])
        if last < first:
            raise ValueError(f"--reference-exclude '{token}': the range ends before it starts")
        rows.update(range(first, last + 1))
    return sorted(rows)


def reference_spec(args, n, phase_centre):
    """
    The reference from the command line: weights over rows, or a fixed position, and a description for the log.

    Returns
    -------
    weights : ndarray, shape (n,), or None
    fixed : ndarray, shape (2,), arcsec offsets from the phase centre, or None
    label : str
    """
    if args.reference_exclude is not None and args.reference_rows is None:
        raise ValueError("--reference-exclude needs --reference-rows")
    if args.reference is not None:
        ra, dec = parse_reference(args.reference)
        fixed = tangent_plane_arcsec(np.array([ra]), np.array([dec]), phase_centre)[0]
        return None, fixed, f"fixed position ({ra:.7f}, {dec:+.7f}) deg, taken as exact"
    start, end = args.reference_rows if args.reference_rows is not None else (0, 0)
    if not 0 <= start <= end < n:
        raise ValueError(f"--reference-rows {start} {end} outside rows 0 to {n - 1}, or START after END")
    excluded = parse_excluded_rows(args.reference_exclude) if args.reference_exclude is not None else []
    outside = [row for row in excluded if not start <= row <= end]
    if outside:
        raise ValueError(f"--reference-exclude rows {outside} are outside rows {start} to {end}")
    rows = [row for row in range(start, end + 1) if row not in excluded]
    if not rows:
        raise ValueError(f"--reference-exclude leaves none of rows {start} to {end}")
    weights = np.zeros(n)
    weights[rows] = 1.0 / len(rows)
    label = f"row {start}" if start == end else f"mean of rows {start} to {end}"
    if excluded:
        label += f", excluding {', '.join(str(row) for row in excluded)}"
    if len(rows) > 1:
        label += f" ({len(rows)} rows)"
    return weights, None, label


def format_position(ra_deg, dec_deg, sep='hms'):
    """A position in sexagesimal (hmsdms, or colons with sep=':') and in decimal degrees."""
    c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame='icrs')
    dms = 'dms' if sep == 'hms' else sep
    return (f"{c.ra.to_string(unit=u.hourangle, sep=sep, precision=5, pad=True)} "
            f"{c.dec.to_string(unit=u.deg, sep=dms, precision=4, alwayssign=True, pad=True)}  "
            f"({ra_deg:.7f}, {dec_deg:+.7f} deg)")


def reference_jitter(ra_deg, dec_deg, weights):
    """
    Scatter of the reference rows about their mean, on the sphere, and the error it gives the reference.

    The centre is the rows' weighted mean direction; each row's offset from
    it along RA (RA cos Dec) and Dec comes from spherical_offsets_to. The
    jitter is the weighted standard deviation of those offsets (the sample
    one for equal weights) and the error on the reference
    jitter * sqrt(sum w^2), jitter / sqrt(n) for equal weights. It holds
    everything that varies between rows, real motion included, so it checks
    the covariance rather than replacing it.

    Returns
    -------
    dict or None
        ra_deg, dec_deg of the centre; jitter, error (2,), RA and Dec,
        arcsec; n. None for fewer than two rows.
    """
    rows = np.nonzero(weights)[0]
    if len(rows) < 2:
        return None
    w = weights[rows] / weights[rows].sum()
    coords = SkyCoord(ra_deg[rows] * u.deg, dec_deg[rows] * u.deg, frame='icrs')
    xyz = coords.cartesian.xyz.value @ w
    centre = SkyCoord(np.degrees(np.arctan2(xyz[1], xyz[0])) % 360.0 * u.deg,
                      np.degrees(np.arcsin(xyz[2] / np.linalg.norm(xyz))) * u.deg, frame='icrs')
    d_ra, d_dec = centre.spherical_offsets_to(coords)
    d = np.column_stack([d_ra.to_value(u.arcsec), d_dec.to_value(u.arcsec)])
    jitter = np.sqrt(w @ (d - w @ d) ** 2 / (1.0 - np.sum(w ** 2)))
    return {'ra_deg': float(centre.ra.deg), 'dec_deg': float(centre.dec.deg), 'jitter': jitter,
            'error': jitter * np.sqrt(np.sum(w ** 2)), 'n': len(rows)}


def predicted_jitter(cov, weights):
    """
    Jitter of the reference rows expected from the full covariance, RA and Dec, arcsec.

    The rows' residuals from their weighted mean are r = M p with
    M = I - 1 w^T, so the expectation of the estimator in reference_jitter is
    (sum_i w_i C_ii - w^T C w) / (1 - sum w^2) for each component, with C
    that component's covariance between the rows.
    """
    rows = np.nonzero(weights)[0]
    w = weights[rows] / weights[rows].sum()
    c = cov[rows][:, :, rows, :]
    var = [w @ np.diag(c[:, a, :, a]) - w @ c[:, a, :, a] @ w for a in (0, 1)]
    return np.sqrt(np.maximum(var, 0.0) / (1.0 - np.sum(w ** 2)))


def composite_reference_cov(ref_cov, weights, measured, predicted):
    """
    Reference covariance with the jitter the covariance does not explain added in quadrature.

    The excess jitter sqrt(max(measured^2 - predicted^2, 0)) per component is
    scatter beyond the noise model and translations (underestimated errors
    or real motion over the rows); it adds (excess^2 sum w^2) to the
    reference variance. What the covariance already holds is not counted
    twice.

    Returns
    -------
    cov : ndarray, shape (2, 2)
    excess : ndarray, shape (2,), arcsec
    """
    excess = np.sqrt(np.maximum(measured ** 2 - predicted ** 2, 0.0))
    return ref_cov + np.diag(excess ** 2 * np.sum(weights ** 2)), excess


def reference_lines(ref):
    """Readable lines on the reference: its position, error, and the jitter of its rows with and without translations."""
    lines = [f"{'Reference':<40}{ref['label']}"]
    for name, row in (('  start row', 'start'), ('  end row', 'end')):
        if row in ref:
            lines.append(f"{name:<40}{ref[row]['index']}: MJD {ref[row]['mjd']:.5f}, "
                         f"DATE-OBS {ref[row]['date_obs']} (epoch {ref[row]['epoch']})")
    if 'raw_ra_deg' in ref:
        lines += [f"{'  without translations':<40}{format_position(ref['raw_ra_deg'], ref['raw_dec_deg'])}",
                  f"{'  with translations':<40}{format_position(ref['ra_deg'], ref['dec_deg'])}",
                  f"{'  shift from the translations':<40}{ref['shift'][0]:+.4f} (RA), {ref['shift'][1]:+.4f} (Dec) arcsec"]
    else:
        lines.append(f"{'  position':<40}{format_position(ref['ra_deg'], ref['dec_deg'])}")
    if ref['fixed']:
        lines.append(f"{'  error':<40}none, taken as exact")
    else:
        lines.append(f"{'  error from the full covariance':<40}{ref['error'][0]:.4f} (RA), "
                     f"{ref['error'][1]:.4f} (Dec) arcsec")
    before, after = ref.get('before'), ref.get('after')
    if before and after:
        ratio = before['jitter'] / after['jitter']
        predicted, excess, composite = ref['predicted'], ref['excess'], ref['composite']
        lines += [f"  Jitter of the {after['n']} reference rows, arcsec   jitter RA  jitter Dec   error RA  error Dec",
                  f"    without translations {before['jitter'][0]:20.4f} {before['jitter'][1]:11.4f} "
                  f"{before['error'][0]:10.4f} {before['error'][1]:10.4f}",
                  f"    with translations {after['jitter'][0]:23.4f} {after['jitter'][1]:11.4f} "
                  f"{after['error'][0]:10.4f} {after['error'][1]:10.4f}",
                  f"    expected from the covariance {predicted[0]:12.4f} {predicted[1]:11.4f}",
                  f"    excess over expected {excess[0]:20.4f} {excess[1]:11.4f}",
                  f"    improvement, without / with {ratio[0]:12.2f}x {ratio[1]:10.2f}x",
                  f"{'  composite error, covariance + excess':<40}{composite[0]:.4f} (RA), {composite[1]:.4f} (Dec) "
                  f"arcsec, used for the radial offsets",
                  "    excess jitter is scatter the covariance does not explain, real motion included"]
    elif not ref['fixed']:
        lines.append(f"{'  jitter':<40}not available from a single row")
    return lines


def error_decimals(err):
    """Decimal places to quote an error to: one significant figure, two when it starts with 1."""
    if not np.isfinite(err) or err <= 0:
        return 4
    exponent = int(np.floor(np.log10(err)))
    leading = int(np.round(err / 10.0 ** exponent, 6))
    return max(-exponent + (1 if leading == 1 else 0), 0)


def publication_lines(ref):
    """
    The reference position to quote: RA hms and Dec dms, each +/- a conservative 1-sigma error.

    The error is the larger of the composite error (covariance plus excess
    jitter) and the error from the measured jitter alone, per component. The
    RA error is given in seconds of time and in arcsec (RA cos Dec), the Dec
    error in arcsec, and each coordinate to the decimal places of its error.
    """
    head = ["Reference position to quote (conservative 1 sigma)",
            "--------------------------------------------------"]
    c = SkyCoord(ref['ra_deg'] * u.deg, ref['dec_deg'] * u.deg, frame='icrs')
    if ref['fixed']:
        return head + [f"  {format_position(ref['ra_deg'], ref['dec_deg'])}, a fixed position, no error"]
    err = ref.get('composite', ref['error'])
    if ref.get('after') is not None:
        err = np.maximum(err, ref['after']['error'])
    ra_s = err[0] / (15.0 * np.cos(np.radians(ref['dec_deg'])))
    n_ra, n_dec, n_ra_arcsec = error_decimals(ra_s), error_decimals(err[1]), error_decimals(err[0])
    ra = c.ra.to_string(unit=u.hourangle, sep='hms', precision=n_ra, pad=True)
    dec = c.dec.to_string(unit=u.deg, sep='dms', precision=n_dec, alwayssign=True, pad=True)
    return head + [f"  RA   {ra} +/- {ra_s:.{n_ra}f} s ({err[0]:.{n_ra_arcsec}f} arcsec)",
                   f"  Dec  {dec} +/- {err[1]:.{n_dec}f} arcsec"]


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


def compare_with_truth(p, cov, path, phase_centre, weights, fixed):
    """
    Corrected positions against the true track a simulation injected.

    The simulated measurements already sit in the fitted frame, so no shift
    is needed: corrected minus true is the translation error plus the noise.

    Returns
    -------
    dict or None
        true_xy (n, 2) arcsec; true_radial (n,), distance from the same
        reference built from the true positions (or the fixed position); diff (n, 2) corrected minus true, arcsec;
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
    true_ref = fixed if fixed is not None else weights @ true_xy
    out = {'true_xy': true_xy, 'true_radial': np.hypot(*(true_xy - true_ref).T),
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
                  asymmetric=None, results_dir=RESULTS_DIR):
    """
    Corrected positions to results/target_{stem}.csv and the full covariance to target_{stem}_covariance.npz.

    Errors are along RA (RA cos Dec) and Dec, arcsec. *_meas is the
    measurement noise alone, *_corr the translation alone, and RA_err,
    Dec_err the total; RA_Dec_corr is the RA-Dec correlation of the total.
    Radial_offset is the distance from the reference (see reference_spec),
    with its error from the full covariance (see radial_offsets). With
    fitted flux scales, Flux_scale and Flux_scale_err are the epoch's g_e and
    Flux_corrected, Flux_corrected_err the flux density on the common scale
    (see corrected_fluxes). With an asymmetric systematic, RA_err_lo, RA_err_hi,
    Dec_err_lo and Dec_err_hi are towards lower and higher RA or Dec and
    Radial_offset_err_lo, Radial_offset_err_hi towards and away from the
    reference (see asymmetric_errors).
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
    if asymmetric is not None:
        columns.update({'RA_err_lo_arcsec': asymmetric['position_lo'][:, 0], 'RA_err_hi_arcsec': asymmetric['position_hi'][:, 0],
                        'Dec_err_lo_arcsec': asymmetric['position_lo'][:, 1], 'Dec_err_hi_arcsec': asymmetric['position_hi'][:, 1],
                        'Radial_offset_err_lo_arcsec': asymmetric['radial_lo'],
                        'Radial_offset_err_hi_arcsec': asymmetric['radial_hi']})
    path = Path(results_dir) / f'target_{stem}.csv'
    np.savetxt(path, np.column_stack(list(columns.values())), delimiter=',', comments='', fmt='%.10g',
               header=','.join(columns))
    msg(f"Corrected positions saved to {path}")

    cov_path = Path(results_dir) / f'target_{stem}_covariance.npz'
    np.savez(cov_path, ra_deg=ra, dec_deg=dec, covariance=cov, epoch=epoch,
             description="Corrected target positions. covariance (n, 2, n, 2), arcsec^2, with [i, a, j, b] the "
                         "covariance of component a of row i and component b of row j (0 RA, 1 Dec).")
    msg(f"Full covariance saved to {cov_path}")


def row_lines(target):
    """
    One line per row, arcsec: corrected position, its total and measurement errors, the translation applied
    and its error, and the radial offset from the reference with its error and significance.
    """
    n = len(target['epoch'])
    diag = np.arange(n)
    total = np.sqrt(target['cov'][diag, :, diag, :][:, [0, 1], [0, 1]])
    meas = np.sqrt(target['noise_cov'][:, [0, 1], [0, 1]])
    tau, tau_err = target['translation'], target['translation_err']
    r, r_err = target['radial'], target['radial_err']
    sig = np.divide(r, r_err, out=np.zeros_like(r), where=r_err > 0)
    asymmetric = target['asymmetric']
    lines = [f"{'Row':>4} {'Ep':>3} {'MJD':>10} {'DATE-OBS':>23} {'SNR':>6}  {'RA':>14} {'Dec':>14}  {'err RA':>7} {'err Dec':>7}  "
             f"{'meas RA':>7} {'meas Dec':>8}  {'tau RA':>8} {'tau Dec':>8} {'err RA':>7} {'err Dec':>7}  "
             f"{'radial':>8} {'err':>7} {'sigma':>6}"
             + (f"  {'RA lo':>7} {'RA hi':>7} {'Dec lo':>7} {'Dec hi':>7} {'rad lo':>7} {'rad hi':>7}" if asymmetric is not None else "")]
    for i in range(n):
        c = SkyCoord(target['ra_deg'][i] * u.deg, target['dec_deg'][i] * u.deg, frame='icrs')
        lines.append(f"{i:>4} {target['epoch'][i]:>3} {target['mjd'][i]:>10.4f} {target['date_obs'][i]:>23} "
                     f"{target['snr'][i]:>6.1f}  "
                     f"{c.ra.to_string(unit=u.hourangle, sep=':', precision=4, pad=True):>14} "
                     f"{c.dec.to_string(unit=u.deg, sep=':', precision=3, alwayssign=True, pad=True):>14}  "
                     f"{total[i, 0]:7.4f} {total[i, 1]:7.4f}  {meas[i, 0]:7.4f} {meas[i, 1]:8.4f}  "
                     f"{tau[i, 0]:+8.4f} {tau[i, 1]:+8.4f} {tau_err[i, 0]:7.4f} {tau_err[i, 1]:7.4f}  "
                     f"{r[i]:8.4f} {r_err[i]:7.4f} {sig[i]:6.1f}"
                     + (f"  {asymmetric['position_lo'][i, 0]:7.4f} {asymmetric['position_hi'][i, 0]:7.4f} "
                        f"{asymmetric['position_lo'][i, 1]:7.4f} {asymmetric['position_hi'][i, 1]:7.4f} "
                        f"{asymmetric['radial_lo'][i]:7.4f} {asymmetric['radial_hi'][i]:7.4f}" if asymmetric is not None else ""))
    return lines


def write_summary(stem, target, aberrated, results_dir=RESULTS_DIR):
    """
    Readable summary of the corrected target, and the truth comparison if any, to
    results/target_{stem}_summary.txt, and to the log. Errors and offsets in arcsec.
    """
    n = len(target['epoch'])
    diag = np.arange(n)
    total = np.sqrt(target['cov'][diag, :, diag, :][:, [0, 1], [0, 1]])
    meas = np.sqrt(target['noise_cov'][:, [0, 1], [0, 1]])
    tau, tau_err = target['translation'], target['translation_err']
    r, r_err = target['radial'], target['radial_err']
    sig = np.divide(r, r_err, out=np.zeros_like(r), where=r_err > 0)
    mjd = target['mjd']

    def stats(name, values):
        return (f"  {name:<30}{np.median(values[:, 0]):8.4f} {np.min(values[:, 0]):8.4f} {np.max(values[:, 0]):8.4f}"
                f"   {np.median(values[:, 1]):8.4f} {np.min(values[:, 1]):8.4f} {np.max(values[:, 1]):8.4f}")

    lines = [
        "AstKAT target summary",
        "=====================",
        f"{'Catalogue':<40}{target['catalog']}",
        f"{'Rows':<40}{n} in {len(np.unique(target['epoch']))} epochs "
        f"({int((~target['in_fit']).sum())} in epochs added after the fit)",
        f"{'Time span':<40}MJD {mjd.min():.4f} to {mjd.max():.4f} ({mjd.max() - mjd.min():.2f} d)",
        f"{'Aberration correction':<40}{'applied' if aberrated else 'not applied'}",
        f"{'SNR':<40}{target['snr'].min():.1f} to {target['snr'].max():.1f} (median {np.median(target['snr']):.1f})",
        "",
        "Reference of the radial offsets",
        "-------------------------------",
        *reference_lines(target['reference']),
        "",
        "Errors and translations, arcsec",
        "-------------------------------",
        f"  {'':<30}{'RA: median':>8}{'min':>9}{'max':>9}   {'Dec: median':>8}{'min':>8}{'max':>9}",
        stats('error, measurement', meas),
        stats('error, translation', tau_err),
        stats('error, total', total),
        stats('translation applied', tau),
        "",
        "Radial offsets from the reference, arcsec",
        "-----------------------------------------",
        f"{'  Median, largest':<40}{np.median(r):.4f}, {r.max():.4f} (row {int(np.argmax(r))})",
        f"{'  Median error':<40}{np.median(r_err[r_err > 0]) if np.any(r_err > 0) else 0.0:.4f}",
        f"{'  Rows more than 3 sigma from it':<40}{int((sig > 3).sum())} of {n}",
        f"{'  Largest significance':<40}{sig.max():.1f} sigma (row {int(np.argmax(sig))})",
        "",
        "Per row, arcsec (tau: translation subtracted; errors 1 sigma)",
        "-------------------------------------------------------------",
        *row_lines(target),
    ]
    truth = target.get('truth')
    if truth:
        rms, mean, std = truth['rms_mas'] / 1e3, truth['chi_mean'], truth['chi_std']
        lines += ["", "Comparison with simulation truth", "--------------------------------",
                  f"{'Truth file':<40}{truth['file']}",
                  f"{'rms, RA and Dec':<40}{rms[0]:.4f}, {rms[1]:.4f} arcsec",
                  f"{'Chi mean +/- std, RA':<40}{mean[0]:+.2f} +/- {std[0]:.2f}",
                  f"{'Chi mean +/- std, Dec':<40}{mean[1]:+.2f} +/- {std[1]:.2f}",
                  f"{'chi2 / dof with the full covariance':<40}{truth['chi2']:.1f} / {truth['dof']}"]
    title("Target summary")
    for line in lines[2:]:
        msg(line)
    # The position to quote is printed last, by main.
    lines += ["", *publication_lines(target['reference'])]
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
    parser = argparse.ArgumentParser(description="AstKAT target positions corrected with the epoch translations",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=CATALOGUE_HELP)
    parser.add_argument('catalog', type=Path, help='target catalogue, one row per measurement')
    parser.add_argument('--beam-unit', choices=sorted(BEAM_UNIT_TO_ARCSEC), default='arcsec',
                        help='unit of bmaj and bmin in the catalogue (default: arcsec)')
    parser.add_argument('--match-tolerance-hours', type=float, default=MATCH_TOLERANCE_HOURS,
                        help=f'largest gap between a row and its epoch (default: {MATCH_TOLERANCE_HOURS:g})')
    parser.add_argument('--truth', type=Path, default=None,
                        help='true track of a simulated target (default: {stem}_truth.csv beside the catalogue, '
                             'if present)')
    parser.add_argument('--reference-exclude', nargs='+', metavar='ROW', default=None,
                        help='rows left out of --reference-rows: single rows or inclusive ranges such as 12-15')
    parser.add_argument('--no-systematic', action='store_true',
                        help='ignore the catalogue\'s systematic columns')
    parser.add_argument('--floor-only', action='store_true',
                        help='rows with a systematic take only the floor B from the noise model')
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument('--reference', nargs='+', metavar='POSITION', default=None,
                           help='fixed reference of the radial offsets, taken as exact: decimal degrees '
                                '(261.93046 -16.20533), hmsdms (17h27m43.31s -16d12m19.2s) or colon-separated '
                                'hms dms (17:27:43.31 -16:12:19.2), quoted or not')
    reference.add_argument('--reference-rows', nargs=2, type=int, metavar=('START', 'END'), default=None,
                           help='reference of the radial offsets the mean of these rows\' corrected positions, '
                                'inclusive, 0-based (default: row 0)')
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
    if args.floor_only and (cat['systematic_cov'] is None or args.no_systematic):
        raise ValueError("--floor-only needs systematic columns (sys_ra and sys_dec, or all four "
                         "sys_*_lo/hi), and not --no-systematic")
    if cat['systematic_cov'] is not None and not args.no_systematic:
        if args.floor_only:
            has_systematic = (cat['systematic_cov'][:, 0, 0] + cat['systematic_cov'][:, 1, 1]) > 0
            floor_cov = noise_covariance(snr, bmaj, bmin, bpa, results['model'], draws, floor_only=True)
            noise_cov = np.where(has_systematic[:, None, None], floor_cov, noise_cov)
            msg(f"Floor only, without the SNR term, for the {has_systematic.sum()} rows with a systematic")
        noise_cov = noise_cov + cat['systematic_cov']
        msg(f"Systematic errors added to {np.count_nonzero(cat['systematic_cov'][:, 0, 0] + cat['systematic_cov'][:, 1, 1])} of "
            f"{len(snr)} rows, on top of the noise model")
    fluxes = corrected_fluxes(cat, epoch, results)
    p, cov = correct_target(x, epoch, noise_cov, trans)

    ra_corr, dec_corr = tangent_plane_to_radec(p, phase_centre)
    stem = args.catalog.stem
    weights, fixed, label = reference_spec(args, len(p), phase_centre)
    ref, ref_cov, cross = reference_position(p, cov, weights, fixed)
    ref_ra, ref_dec = tangent_plane_to_radec(ref[None], phase_centre)
    reference = {'label': label, 'ra_deg': float(ref_ra[0]), 'dec_deg': float(ref_dec[0]),
                 'error': np.sqrt(np.diag(ref_cov)), 'fixed': fixed is not None}
    if weights is not None:
        # The same rows without the translations, to show what the translations do to the reference.
        rows = np.nonzero(weights)[0]
        for key, i in (('start', rows[0]), ('end', rows[-1])):
            reference[key] = {'index': int(i), 'mjd': float(cat['time'][i].mjd), 'epoch': int(epoch[i]),
                              'date_obs': str(trans['obs_isot'][epoch[i]])}
        raw = weights @ x
        raw_ra, raw_dec = tangent_plane_to_radec(raw[None], phase_centre)
        reference.update(raw_ra_deg=float(raw_ra[0]), raw_dec_deg=float(raw_dec[0]), shift=ref - raw,
                         before=reference_jitter(ra, dec, weights),
                         after=reference_jitter(ra_corr, dec_corr, weights))
        if reference['after'] is not None:
            # Composite reference error: the covariance plus the jitter it does not explain.
            predicted = predicted_jitter(cov, weights)
            ref_cov, excess = composite_reference_cov(ref_cov, weights, reference['after']['jitter'], predicted)
            reference.update(predicted=predicted, excess=excess, composite=np.sqrt(np.diag(ref_cov)))
    for line in reference_lines(reference):
        msg(line)
    r, r_err = radial_offsets(p, cov, ref, ref_cov, cross)
    asymmetric = None
    if cat['systematic_sides'] is not None and not args.no_systematic:
        asymmetric = asymmetric_errors(p, cov, ref, r_err, cat['systematic_cov'], cat['systematic_sides'])
        msg("Asymmetric errors added on RA, Dec and the radial offset")
    write_results(stem, cat, ra_corr, dec_corr, epoch, trans, snr, bmaj, bmin, bpa, noise_cov, cov, r, r_err,
                  fluxes, asymmetric)

    # For a simulated target, the corrected track against the injected one.
    path = truth_file(args)
    truth = compare_with_truth(p, cov, path, phase_centre, weights, fixed) if path is not None else None

    target = {'stem': stem, 'catalog': str(args.catalog), 'mjd': cat['time'].mjd, 'epoch': epoch,
              'in_fit': trans['in_fit'][epoch].astype(bool), 'snr': snr, 'measured': x, 'p': p, 'cov': cov,
              'noise_cov': noise_cov, 'radial': r, 'radial_err': r_err, 'asymmetric': asymmetric, 'reference': reference, 'truth': truth,
              'ra_deg': ra_corr, 'dec_deg': dec_corr, 'date_obs': [str(trans['obs_isot'][e]) for e in epoch],
              'translation': trans['median'][epoch],
              'translation_err': np.sqrt(trans['covariance'][epoch, :, epoch, :][:, [0, 1], [0, 1]])}
    write_summary(stem, target, aberrated)
    make_plots(target)
    msg("")
    for line in publication_lines(reference):
        msg(line)
    return target


if __name__ == "__main__":
    main()
