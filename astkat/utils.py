"""
AstKAT shared utilities.

Logging, bookkeeping of observation metadata, system resources, catalogue
conventions, spherical position helpers, and the differential aberration
correction applied to every catalogue before fitting.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import bottleneck as bn
import psutil
import astropy.constants as const
import astropy.units as u
from astropy.coordinates import (SkyCoord, EarthLocation, GCRS,
                                 get_body_barycentric, get_body_barycentric_posvel)
from astropy.time import Time


# Random seed shared by simulation, MCMC fitting and posterior draws. A project
# overrides it with BOOKKEEPING.seed in astkat.yaml (initialise.py --seed).
DEFAULT_SEED = 19950325

# File the simulators write next to the images; its presence marks a simulated project.
SIMULATION_TRUTH_FILE = 'expected_values.json'

# Columns of the matched catalogue array, shape (n_sources, n_epochs, n_columns).
# flux_ratio is the fitted Gaussian's total flux over its peak, maj_ratio and min_ratio its axes over
# the beam's, s_code the index in S_CODES of PyBDSF's S_Code.
CATALOG_COLUMNS = ('ra_deg', 'dec_deg', 'peak_flux_Jy', 'rms_Jy', 'snr',
                   'phase_offset_deg', 'bmaj_deg', 'bmin_deg', 'bpa_deg', 'freq_Hz',
                   'flux_ratio', 'maj_ratio', 'min_ratio', 's_code')

# PyBDSF S_Code: S one Gaussian alone in its island, C one Gaussian in an island shared with
# other sources, M several Gaussians. A is set by matching.py: an epoch where several sources
# claimed the reference source, so none was kept, or where it was blended with a neighbour that took
# the source inside its ellipse; every other column is NaN, and it is not observed.
S_CODES = ('S', 'C', 'M', 'A')
AMBIGUOUS = S_CODES.index('A')

# PyBDSF thresh_pix for the epoch images, in sigma: the default PYBDSF.snr_threshold, and the
# detection threshold of simulate.py's catalogue mode.
PYBDSF_SNR_THRESHOLD = 4.0


def column_index(name):
    """Index of a named column in the matched catalogue array (see CATALOG_COLUMNS)."""
    return CATALOG_COLUMNS.index(name)


def epoch_frequencies(catalog):
    """Each epoch's frequency in Hz, the median over its detections, from a matched catalogue (sources, epochs, columns)."""
    with np.errstate(all='ignore'):
        return bn.nanmedian(catalog[:, :, column_index('freq_Hz')], axis=0)


def frequency_bands(freq, tolerance):
    """
    Band of each epoch: epochs within tolerance (fractional) of a band's frequency share the band.

    The most common frequency takes every epoch within tolerance of it as
    band 0, the dominant band; the most common of the rest takes band 1, and
    so on. Epochs without a frequency are -1. matching.py and fit.py both
    group epochs this way, with MCMC.frequency_match_tolerance.
    """
    freq = np.asarray(freq, dtype=float)
    band = np.full(freq.shape, -1)
    remaining = np.isfinite(freq)
    label = 0
    while remaining.any():
        values, counts = np.unique(freq[remaining], return_counts=True)
        centre = values[np.argmax(counts)]
        members = remaining & (np.abs(freq - centre) <= centre * tolerance)
        band[members] = label
        remaining &= ~members
        label += 1
    return band


# ============================================================================
# LOGGING
# ============================================================================

_log_file_handle = None   # opened on the first msg() call; False if opening failed


def _get_log_file():
    """
    Per-run log file under logs/, named after the script and its start time.

    For example logs/matching_20260907_153312.log. Returns None when no log
    file can be opened (e.g. a read-only filesystem), in which case messages
    go to stdout only.
    """
    global _log_file_handle
    if _log_file_handle is None:
        try:
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            script_name = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else 'astkat'
            log_path = log_dir / f"{script_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            _log_file_handle = open(log_path, 'a')
        except Exception:
            _log_file_handle = False
    return _log_file_handle or None


def msg(txt):
    """
    Print a timestamped message and append it to the run's log file.

    Parameters
    ----------
    txt : str
        Message text.
    """
    line = time.strftime(' %Y-%m-%d %H:%M:%S | ') + txt
    print(line, flush=True)

    # A failed log write never interrupts the run.
    log_file = _get_log_file()
    if log_file:
        try:
            log_file.write(line + '\n')
            log_file.flush()
        except Exception:
            pass


def title(txt):
    """Log a stage title, === txt ===, between blank lines."""
    msg('')
    msg(f"=== {txt} ===")
    msg('')


# ============================================================================
# OBSERVATION BOOKKEEPING
# ============================================================================

def remove_truth_csv_rows_by_position(csv_path, positions_to_remove, reason=""):
    """
    Drop rows from a simulation truth file (per_epoch_offsets.csv) by row position.

    Used when an epoch is dropped and only its position in the catalogue's
    epoch order is known. Row order in the truth file follows the catalogue's
    epoch order. Does nothing if the file is absent, as for real data.

    Parameters
    ----------
    csv_path : str or Path
        Path to per_epoch_offsets.csv.
    positions_to_remove : iterable of int
        0-based row positions to drop.
    reason : str, optional
        Short reason for the log message.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return

    with open(csv_path, 'r', newline='') as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else None

    positions = set(positions_to_remove)
    kept_rows = [r for i, r in enumerate(rows) if i not in positions]
    if len(kept_rows) == len(rows):
        return

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    removed_epochs = [rows[i]['epoch_index'] for i in sorted(positions) if i < len(rows)]
    suffix = f" ({reason})" if reason else ""
    msg(f"Removed {len(removed_epochs)} epochs from {csv_path.name}{suffix}, {len(kept_rows)} left: "
        f"epoch_index {removed_epochs}")


def remove_observation_properties_entries(obs_props_path, positions_to_remove=None,
                                          image_identifiers_to_remove=None, reason=""):
    """
    Drop epochs from observation_properties.json, keeping every list field aligned.

    Epochs are selected either by 0-based position or by image_identifier;
    pass exactly one of the two. Does nothing if the file is absent.

    Parameters
    ----------
    obs_props_path : str or Path
        Path to observation_properties.json.
    positions_to_remove : iterable of int, optional
        0-based positions to drop.
    image_identifiers_to_remove : iterable of str, optional
        image_identifier values to drop.
    reason : str, optional
        Short reason for the log message.
    """
    obs_props_path = Path(obs_props_path)
    if not obs_props_path.exists():
        return

    with open(obs_props_path, 'r') as f:
        data = json.load(f)

    identifiers = data.get('image_identifier', [])
    n = len(identifiers)
    if n == 0:
        return

    # Every per-epoch list is filtered with the same mask so all fields stay aligned.
    if image_identifiers_to_remove is not None:
        drop_ids = set(image_identifiers_to_remove)
        keep_mask = [img_id not in drop_ids for img_id in identifiers]
    elif positions_to_remove is not None:
        drop_positions = set(positions_to_remove)
        keep_mask = [i not in drop_positions for i in range(n)]
    else:
        return

    if all(keep_mask):
        return

    removed_ids = [identifiers[i] for i, keep in enumerate(keep_mask) if not keep]
    for key, values in data.items():
        if isinstance(values, list) and len(values) == n:
            data[key] = [v for v, keep in zip(values, keep_mask) if keep]

    with open(obs_props_path, 'w') as f:
        json.dump(data, f, indent=4)

    suffix = f" ({reason})" if reason else ""
    msg(f"Removed {len(removed_ids)} of {n} epochs from {obs_props_path.name}{suffix}: {removed_ids}")


def is_simulated(config):
    """
    True for a simulated project.

    Read from BOOKKEEPING.simulated in astkat.yaml, or inferred from the
    truth file the simulators write next to the images (SIMULATION_TRUTH_FILE).
    Simulated catalogues contain no aberration, so the correction is skipped.

    Parameters
    ----------
    config : dict
        Parsed astkat.yaml.
    """
    bookkeeping = config['BOOKKEEPING']
    return bool(bookkeeping.get('simulated')) or (Path(bookkeeping['image_dir']) / SIMULATION_TRUTH_FILE).exists()


def parse_phase_center(phase_center):
    """
    Phase centre as an ICRS SkyCoord.

    Parameters
    ----------
    phase_center : str or SkyCoord
        For example '17:27:43.31 -16:12:04.2' (RA in hours) or
        '17h27m43.31s -16d12m04.2s'.

    Returns
    -------
    SkyCoord
    """
    if isinstance(phase_center, SkyCoord):
        return phase_center.icrs
    try:
        return SkyCoord(phase_center).icrs
    except ValueError:
        ra_str, dec_str = phase_center.split(' ', 1)
        return SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg)).icrs


# ============================================================================
# SYSTEM RESOURCES
# ============================================================================

def get_system_resources(memory_threshold=0.5):
    """
    Detect CPU, memory and disk resources and log a summary.

    Parameters
    ----------
    memory_threshold : float
        Fraction of available memory treated as usable, normally
        BOOKKEEPING.memory_safety_fraction in astkat.yaml.

    Returns
    -------
    dict
        Core counts, memory and disk figures, and suggested thread and
        process counts.
    """
    cpu_physical = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)

    memory = psutil.virtual_memory()
    total_memory_gb = memory.total / (1024 ** 3)
    available_memory_gb = memory.available / (1024 ** 3)
    usable_memory_gb = available_memory_gb * memory_threshold
    usable_memory_pixels = int(usable_memory_gb * 1024 ** 3 / 4)   # float32 pixels

    free_disk_gb = psutil.disk_usage('.').free / (1024 ** 3)

    optimal_threads_io = min(12, cpu_logical)
    optimal_processes_cpu = max(1, cpu_physical)

    title("System resources")
    msg(f"CPU {cpu_physical} physical, {cpu_logical} logical cores; memory {total_memory_gb:.1f} GB total, "
        f"{available_memory_gb:.1f} GB available, {usable_memory_gb:.1f} GB usable ({memory_threshold * 100:.0f}%, "
        f"{usable_memory_pixels:,} float32 pixels); disk {free_disk_gb:.1f} GB free")
    msg(f"Workers: {optimal_threads_io} I/O threads, {optimal_processes_cpu} processes")

    return {
        'cpu_physical': cpu_physical,
        'cpu_logical': cpu_logical,
        'memory_total_gb': total_memory_gb,
        'memory_available_gb': available_memory_gb,
        'usable_memory_gb': usable_memory_gb,
        'usable_memory_pixels': usable_memory_pixels,
        'free_disk_gb': free_disk_gb,
        'optimal_threads_io': optimal_threads_io,
        'optimal_processes_cpu': optimal_processes_cpu,
    }


def load_system_resources(files_dir='astkat_files'):
    """
    System resources saved by initialise.py, or a fresh detection if unavailable.

    Parameters
    ----------
    files_dir : str or Path
        Directory holding system_resources.json.

    Returns
    -------
    dict
    """
    system_file = Path(files_dir) / "system_resources.json"
    try:
        with open(system_file, 'r') as f:
            system_resources = json.load(f)
        msg(f"System resources from {system_file}")
        return system_resources
    except FileNotFoundError:
        msg(f"{system_file} not found; detecting system resources")
    except json.JSONDecodeError as e:
        msg(f"{system_file} unreadable ({e}); detecting system resources")
    return get_system_resources()


# ============================================================================
# SPHERICAL POSITION HELPERS
# ============================================================================

ARCSEC_PER_RAD = np.degrees(1.0) * 3600.0


def tangent_plane_arcsec(ra_deg, dec_deg, phase_centre):
    """Orthographic (RA, Dec) offsets from the phase centre in arcsec, shape (n, 2)."""
    a0, d0 = phase_centre.ra.rad, phase_centre.dec.rad
    e_ra = np.array([-np.sin(a0), np.cos(a0), 0.0])
    e_dec = np.array([-np.sin(d0) * np.cos(a0), -np.sin(d0) * np.sin(a0), np.cos(d0)])
    n = _radec_to_unit(ra_deg, dec_deg)
    return np.column_stack([e_ra @ n, e_dec @ n]) * ARCSEC_PER_RAD


def beam_frame(xy, bmaj_deg, bmin_deg, bpa_deg):
    """
    Tangent-plane positions in beam units: along the major axis over BMAJ and along the minor axis over BMIN.

    BPA is East of North, so the major axis points along (sin BPA, cos BPA)
    and the minor axis along (-cos BPA, sin BPA), as in fit.epoch_beams.
    Distances in this frame are separations in units of the FWHM axes; a
    distance of at most d is the beam's FWHM ellipse scaled by 2 d (d = 0.75,
    1.5 times the beam).

    Parameters
    ----------
    xy : ndarray, shape (n, 2)
        (RA, Dec) offsets, arcsec (see tangent_plane_arcsec).
    bmaj_deg, bmin_deg, bpa_deg : float or ndarray of shape (n,)
        The beam, one for all rows or one per row.

    Returns
    -------
    ndarray, shape (n, 2)
    """
    pa = np.radians(bpa_deg)
    major = (xy[:, 0] * np.sin(pa) + xy[:, 1] * np.cos(pa)) / (np.asarray(bmaj_deg) * 3600.0)
    minor = (-xy[:, 0] * np.cos(pa) + xy[:, 1] * np.sin(pa)) / (np.asarray(bmin_deg) * 3600.0)
    return np.column_stack([major, minor])


# ============================================================================
# DIFFERENTIAL ABERRATION
# ============================================================================
#
# What it is
#   The observer moves at ~30 km/s, so every apparent position is displaced
#   toward the direction of motion by up to v/c ~ 20 arcsec. The correlator
#   tracks the apparent phase centre and the imager labels it with its
#   catalogue position, so this bulk shift never appears in the images. What
#   remains is the difference between each source's displacement and the
#   phase centre's.
#
# Model
#   With a(x) the apparent direction of a true direction x (unit 3-vectors in
#   ICRS axes) and p the phase centre, a measured position is
#
#       x_meas = norm( a(x) - a(p) + p )
#
#   where a(x) is solar light deflection followed by exact special-relativistic
#   aberration. To first order this is an isotropic scale about the phase
#   centre,
#
#       s(t) = -v_obs(t) . p / c ,
#
#   of up to ~1e-4, i.e. ~160 mas at 0.45 deg from the phase centre, varying
#   over the year with the Earth's orbit. The shift is purely radial, s sin(rho)
#   at angular distance rho, so the residual beyond the linear scale is cubic,
#   0.002 mas at 0.45 deg. Solar deflection adds a few mas across a 0.45 deg
#   field within ~10 deg of the Sun, as a shear rather than a scale.
#
# Correction
#   Since a(x) - a(p) + p is parallel to x_meas and |a(x)| = 1,
#
#       a(x) = k x_meas + Dp,   k = -(x_meas . Dp) + sqrt((x_meas . Dp)^2 + 1 - |Dp|^2),
#
#   with Dp = a(p) - p. Aberration is then undone exactly (the same formula
#   with -v) and deflection by a short fixed-point iteration. The observer's
#   velocity and the Sun's position are evaluated once per epoch from
#   astropy's ephemeris, plus the observatory's diurnal velocity when a
#   location is given. Shifts are applied to 3-vectors in ICRS axes; applying
#   them in the apparent frame instead would add a spurious rotation of order
#   beta * tan(dec).
#
# Zero point
#   A misplaced zero point (e.g. visibilities rephased away from the tracking
#   centre) leaves s(t)(p' - p), the same for every source in an epoch, which
#   the per-epoch translations absorb. Only the scale has to be right for
#   relative astrometry.

# MeerKAT array reference position. The location only sets the diurnal term
# (<= 1.6e-6 in scale), so its precision is irrelevant. Override with
# MCMC.observatory_location in astkat.yaml.
MEERKAT_LAT_DEG = -30.711056
MEERKAT_LON_DEG = 21.443889
MEERKAT_HEIGHT_M = 1035.0

_C_KM_S = const.c.to(u.km / u.s).value
_SUN_SCHWARZSCHILD_AU = 1.97412574336e-8   # 2GM/c^2 of the Sun in au (SOFA SRS)

_diurnal_fallback_warned = False


def observatory_location(config=None):
    """
    Observatory location from astkat.yaml, defaulting to MeerKAT.

    Parameters
    ----------
    config : dict, optional
        Parsed astkat.yaml. Reads MCMC.observatory_location (lat_deg,
        lon_deg, height_m); missing keys fall back to MeerKAT.

    Returns
    -------
    EarthLocation
    """
    loc = ((config or {}).get('MCMC', {}) or {}).get('observatory_location') or {}
    return EarthLocation.from_geodetic(lon=float(loc.get('lon_deg', MEERKAT_LON_DEG)) * u.deg,
                                       lat=float(loc.get('lat_deg', MEERKAT_LAT_DEG)) * u.deg,
                                       height=float(loc.get('height_m', MEERKAT_HEIGHT_M)) * u.m)


def load_epoch_times(n_epochs, results_dir='results', retained_indices=None):
    """
    Observation time (DATE-OBS) of each catalogue epoch, from observation_properties.json.

    observation_properties.json is sorted by time in the same epoch order as
    the matched catalogue. For a catalogue that has had epochs removed, pass
    the image indices of the epochs it kept.

    Parameters
    ----------
    n_epochs : int
        Number of catalogue epochs the times must align with.
    results_dir : str or Path
        Directory holding observation_properties.json.
    retained_indices : sequence of int, optional
        Image indices of the catalogue's epochs.

    Returns
    -------
    Time, shape (n_epochs,)

    Raises
    ------
    FileNotFoundError, ValueError
        Missing file or obs_isot, a length mismatch, or unusable timestamps.
        The correction must not run on misaligned or guessed times.
    """
    props_file = Path(results_dir) / 'observation_properties.json'
    if not props_file.exists():
        raise FileNotFoundError(f"{props_file} not found: epoch times are needed for the aberration correction")
    with open(props_file, 'r') as f:
        iso = json.load(f).get('obs_isot')
    if iso is None:
        raise ValueError(f"{props_file} has no obs_isot entries")

    if retained_indices is not None:
        retained_indices = list(retained_indices)
        if len(retained_indices) and max(retained_indices) >= len(iso):
            raise ValueError(f"retained_indices reach image {max(retained_indices)} but "
                             f"{props_file} has only {len(iso)} images")
        iso = [iso[i] for i in retained_indices]

    if len(iso) != n_epochs:
        raise ValueError(f"{props_file} gives {len(iso)} observation times for {n_epochs} catalog epochs")
    bad = [i for i, t in enumerate(iso) if t is None or str(t).strip().lower() in ('', 'unknown', 'none', 'nan')]
    if bad:
        raise ValueError(f"Unusable obs_isot for epoch(s) {bad[:10]}{'...' if len(bad) > 10 else ''}")
    return Time(list(iso), format='isot', scale='utc')


def _radec_to_unit(ra_deg, dec_deg):
    """Unit 3-vectors, shape (3, n), from RA and Dec in degrees."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _unit_to_radec(xyz):
    """RA and Dec in degrees from 3-vectors of shape (3, n), normalising first."""
    xyz = xyz / np.linalg.norm(xyz, axis=0)
    ra = np.degrees(np.arctan2(xyz[1], xyz[0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(xyz[2], -1.0, 1.0)))
    return ra, dec


def _angle_arcsec(a, b):
    """Angle between unit 3-vectors (columns) in arcsec, stable at small separations."""
    return np.degrees(2.0 * np.arcsin(np.clip(0.5 * np.linalg.norm(a - b, axis=0), 0.0, 1.0))) * 3600.0


def _observer_posvel(location, times):
    """
    Observatory GCRS position and velocity, or None for a geocentric observer.

    The diurnal term needs Earth-rotation (IERS) data. If astropy cannot get
    them, for example offline, this warns once and falls back to a geocentric
    observer, which drops at most 1.6e-6 in scale.
    """
    global _diurnal_fallback_warned
    if location is None:
        return None
    try:
        return location.get_gcrs_posvel(times)
    except Exception as e:
        if not _diurnal_fallback_warned:
            msg(f"WARNING: observer Earth-rotation data unavailable ({type(e).__name__}: {e}); "
                f"using a geocentric observer (drops the diurnal term, <= 1.6e-6 in scale)")
            _diurnal_fallback_warned = True
        return None


def _as_source_epoch(ra_deg, dec_deg, times):
    """(S, E) position arrays and a flag for 1-D input, which is one epoch per element."""
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    if ra.shape != dec.shape:
        raise ValueError(f"ra and dec shapes differ: {ra.shape} vs {dec.shape}")
    squeeze = ra.ndim == 1
    if squeeze:
        ra, dec = ra[None, :], dec[None, :]
    if ra.ndim != 2 or len(times) != ra.shape[1]:
        raise ValueError(f"expected (n_sources, n_epochs) positions with n_epochs times; "
                         f"got positions {np.shape(ra_deg)} and {len(times)} times")
    return ra, dec, squeeze


def observer_state(times, location=None):
    """
    Observer velocity and position relative to the Sun at each time, in ICRS axes.

    Parameters
    ----------
    times : Time, shape (n,)
    location : EarthLocation, optional
        Observatory. None gives a geocentric observer (no diurnal term).

    Returns
    -------
    v : ndarray, shape (3, n)
        Observer barycentric velocity in units of c.
    e_sun : ndarray, shape (3, n)
        Unit vector from the Sun to the observer.
    d_sun : ndarray, shape (n,)
        Sun-observer distance in au.
    """
    # Earth's orbital motion from astropy's ephemeris, plus the observatory's rotation with the Earth.
    earth_pos, earth_vel = get_body_barycentric_posvel('earth', times)
    helio = (earth_pos - get_body_barycentric('sun', times)).xyz.to(u.au).value
    vel = earth_vel.xyz.to(u.km / u.s).value

    posvel = _observer_posvel(location, times)
    if posvel is not None:
        helio = helio + posvel[0].xyz.to(u.au).value
        vel = vel + posvel[1].xyz.to(u.km / u.s).value

    d_sun = np.linalg.norm(helio, axis=0)
    return vel / _C_KM_S, helio / d_sun, d_sun


def _solar_deflection(p, e_sun, d_sun):
    """
    Displacement of unit vectors p (3, n) by the Sun's gravity, as in SOFA iauLdsun.

    The displacement is perpendicular to p. d_sun has shape (n,).
    """
    pe = np.sum(p * e_sun, axis=0)
    floor = 1e-6 / np.maximum(d_sun ** 2, 1.0)
    w = _SUN_SCHWARZSCHILD_AU / d_sun / np.maximum(1.0 + pe, floor)
    return w * (e_sun - pe * p)


def aberrate(p, v):
    """
    Exact special-relativistic aberration of unit vectors p for velocity v (units of c).

    Both arrays have shape (3, n). The inverse is aberrate(p_apparent, -v).
    Matches SOFA iauAb apart from its ~1e-12 rad gravitational term, which
    cancels when positions are referred to the phase centre.
    """
    inv_gamma = np.sqrt(1.0 - np.sum(v * v, axis=0))
    pv = np.sum(p * v, axis=0)
    w = p * inv_gamma + (1.0 + pv / (1.0 + inv_gamma)) * v
    return w / np.linalg.norm(w, axis=0)


def apparent_direction(x, v, e_sun, d_sun):
    """Apparent direction a(x) of true directions x (3, n): solar deflection, then aberration."""
    return aberrate(x + _solar_deflection(x, e_sun, d_sun), v)


def true_direction(a, v, e_sun, d_sun, max_iter=8, tol_rad=1e-15):
    """
    Inverse of apparent_direction.

    Aberration is undone exactly. Deflection has no closed-form inverse, so
    it is removed by fixed-point iteration, which converges in two to four
    steps (each step gains a factor of the deflection gradient, <= 1e-3 even
    a degree from the Sun).

    Returns
    -------
    x : ndarray, shape (3, n)
        True directions.
    last_step : float
        Largest change in the final iteration, in radians.
    """
    # Aberration is undone exactly by aberrating with the opposite velocity.
    p = aberrate(a, -v)

    # Deflection is removed iteratively; it is at most milliarcseconds, so this converges fast.
    x, last_step = p, 0.0
    for _ in range(max_iter):
        x_new = p - _solar_deflection(x, e_sun, d_sun)
        x_new = x_new / np.linalg.norm(x_new, axis=0)
        last_step = float(np.max(np.linalg.norm(x_new - x, axis=0))) if x.shape[1] else 0.0
        x = x_new
        if last_step < tol_rad:
            break
    return x, last_step


def _anchor_context(phase_center, times, location):
    """Observer state per epoch and the phase-centre displacement Dp = a(p) - p, shape (3, E)."""
    v, e_sun, d_sun = observer_state(times, location)

    # The phase centre's own shift, which the imaging already removed from every position.
    p = np.repeat(phase_center.cartesian.xyz.value[:, None], len(times), axis=1)
    return v, e_sun, d_sun, apparent_direction(p, v, e_sun, d_sun) - p


def apply_aberration_to_positions(ra_deg, dec_deg, phase_center, times, location=None):
    """
    Forward model: true positions to positions as measured in the image.

    Used to inject aberration into simulations and to test the correction.
    NaN positions are passed through unchanged.

    Parameters
    ----------
    ra_deg, dec_deg : ndarray, shape (n_sources, n_epochs) or (n,)
        True positions in degrees. 1-D input is one epoch per element.
    phase_center : str or SkyCoord
    times : Time, shape (n_epochs,) or (n,)
    location : EarthLocation, optional
        Observatory. None gives a geocentric observer (no diurnal term).

    Returns
    -------
    (ra_deg, dec_deg) with the input shape
    """
    pc = parse_phase_center(phase_center)
    ra, dec, squeeze = _as_source_epoch(ra_deg, dec_deg, times)
    out_ra, out_dec = ra.copy(), dec.copy()
    ok = np.isfinite(ra) & np.isfinite(dec)

    if np.any(ok):
        e = np.nonzero(ok)[1]
        v, e_sun, d_sun, dp = _anchor_context(pc, times, location)
        x = _radec_to_unit(ra[ok], dec[ok])

        # Apparent direction, measured relative to the phase centre's apparent shift.
        a = apparent_direction(x, v[:, e], e_sun[:, e], d_sun[e])
        out_ra[ok], out_dec[ok] = _unit_to_radec(a - dp[:, e])

    if squeeze:
        return out_ra[0], out_dec[0]
    return out_ra, out_dec


def correct_positions_for_aberration(ra_deg, dec_deg, phase_center, times, location=None,
                                     return_convergence=False):
    """
    Inverse model: measured positions to true positions, in closed form.

    NaN positions are passed through unchanged.

    Parameters
    ----------
    ra_deg, dec_deg : ndarray, shape (n_sources, n_epochs) or (n,)
        Measured positions in degrees. 1-D input is one epoch per element.
    phase_center : str or SkyCoord
    times : Time, shape (n_epochs,) or (n,)
    location : EarthLocation, optional
        Observatory. None gives a geocentric observer (no diurnal term).
    return_convergence : bool
        Also return the largest change in the final deflection iteration, in arcsec.

    Returns
    -------
    (ra_deg, dec_deg) with the input shape, plus the convergence figure if requested
    """
    pc = parse_phase_center(phase_center)
    ra, dec, squeeze = _as_source_epoch(ra_deg, dec_deg, times)
    out_ra, out_dec = ra.copy(), dec.copy()
    ok = np.isfinite(ra) & np.isfinite(dec)
    last_change = 0.0

    if np.any(ok):
        e = np.nonzero(ok)[1]
        v, e_sun, d_sun, dp = _anchor_context(pc, times, location)
        m = _radec_to_unit(ra[ok], dec[ok])
        dp_e = dp[:, e]

        # Restore the phase centre's shift: a(x) = k m + Dp, with k fixed by |a(x)| = 1.
        m_dp = np.sum(m * dp_e, axis=0)
        k = -m_dp + np.sqrt(m_dp ** 2 + 1.0 - np.sum(dp_e ** 2, axis=0))

        # Then undo aberration and deflection to reach the true direction.
        x, last_step = true_direction(k * m + dp_e, v[:, e], e_sun[:, e], d_sun[e])
        out_ra[ok], out_dec[ok] = _unit_to_radec(x)
        last_change = float(np.degrees(last_step) * 3600.0)

    if squeeze:
        out_ra, out_dec = out_ra[0], out_dec[0]
    if return_convergence:
        return out_ra, out_dec, last_change
    return out_ra, out_dec


def aberration_scale(phase_center, times, location=None):
    """
    First-order differential aberration, the isotropic scale about the phase centre.

        s(t) = -v_obs(t) . p / c

    Positions are displaced by s times their offset from the phase centre.
    This is the prediction used by template tests such as diagnose_affine.py;
    the correction itself uses the exact model.

    Parameters
    ----------
    phase_center : str or SkyCoord
    times : Time
    location : EarthLocation, optional

    Returns
    -------
    ndarray, dimensionless, one value per time
    """
    pc = parse_phase_center(phase_center)
    v, _, _ = observer_state(times, location)
    return -(pc.cartesian.xyz.value @ v)


# ============================================================================
# DIFFERENTIAL ABERRATION, ASTROPY GCRS REFERENCE
# ============================================================================
#
# The same model computed through astropy's ICRS -> GCRS transform, which
# applies the same physics (annual and diurnal aberration, solar deflection)
# as a black box, for comparison with the closed form above; nothing in the
# pipeline calls it.
#
# It is slower because astropy evaluates the ephemeris and Earth rotation for
# every position rather than every epoch, and its inverse is a fixed-point
# iteration that calls the transform each step. That iteration omits the
# length |a(x) - Dp| of the forward model, which leaves a second-order bias of
# up to (beta sin rho)^2: about 1e-4 mas at 0.45 deg and 2e-3 mas at 1.6 deg.

def _apparent_minus_true_gcrs(ra_deg, dec_deg, times, location=None):
    """
    a(x) - x through astropy's GCRS frame, as 3-vectors in ICRS axes, shape (3, n).

    ra_deg, dec_deg and times are flat and of equal length. GCRS axes are
    aligned with ICRS, so the difference of unit vectors is the displacement.
    """
    c = SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg, frame='icrs')
    posvel = _observer_posvel(location, times)
    if posvel is None:
        frame = GCRS(obstime=times)
    else:
        frame = GCRS(obstime=times, obsgeoloc=posvel[0], obsgeovel=posvel[1])
    return c.transform_to(frame).cartesian.xyz.value - c.cartesian.xyz.value


def _phase_center_displacement_gcrs(phase_center, times, location):
    """a(p) - p for the phase centre at each epoch through astropy's GCRS frame, shape (3, E)."""
    n = len(times)
    return _apparent_minus_true_gcrs(np.full(n, phase_center.ra.deg), np.full(n, phase_center.dec.deg),
                                     times, location)


def apply_aberration_to_positions_gcrs(ra_deg, dec_deg, phase_center, times, location=None):
    """
    Reference forward model through astropy's GCRS frame. Same interface as apply_aberration_to_positions.
    """
    pc = parse_phase_center(phase_center)
    ra, dec, squeeze = _as_source_epoch(ra_deg, dec_deg, times)
    out_ra, out_dec = ra.copy(), dec.copy()
    ok = np.isfinite(ra) & np.isfinite(dec)

    if np.any(ok):
        e = np.nonzero(ok)[1]
        dp = _phase_center_displacement_gcrs(pc, times, location)
        x = _radec_to_unit(ra[ok], dec[ok])
        d = _apparent_minus_true_gcrs(ra[ok], dec[ok], times[e], location)
        out_ra[ok], out_dec[ok] = _unit_to_radec(x + d - dp[:, e])

    if squeeze:
        return out_ra[0], out_dec[0]
    return out_ra, out_dec


def correct_positions_for_aberration_gcrs(ra_deg, dec_deg, phase_center, times, location=None,
                                          n_iter=3, return_convergence=False):
    """
    Reference inverse through astropy's GCRS frame, by fixed-point iteration.

    Solves x = norm(x_meas - [a(x) - x] + [a(p) - p]) from x = x_meas. Each
    iteration shrinks the error by ~1e-4, so three iterations converge. Same
    interface as correct_positions_for_aberration, plus n_iter.
    """
    pc = parse_phase_center(phase_center)
    ra, dec, squeeze = _as_source_epoch(ra_deg, dec_deg, times)
    out_ra, out_dec = ra.copy(), dec.copy()
    ok = np.isfinite(ra) & np.isfinite(dec)
    last_change = 0.0

    if np.any(ok):
        e = np.nonzero(ok)[1]
        t_obs = times[e]
        dp = _phase_center_displacement_gcrs(pc, times, location)[:, e]
        y = _radec_to_unit(ra[ok], dec[ok])
        x = y.copy()
        for _ in range(max(int(n_iter), 1)):
            xr, xd = _unit_to_radec(x)
            x_new = y - _apparent_minus_true_gcrs(xr, xd, t_obs, location) + dp
            x_new /= np.linalg.norm(x_new, axis=0)
            last_change = float(np.max(_angle_arcsec(x_new, x)))
            x = x_new
        out_ra[ok], out_dec[ok] = _unit_to_radec(x)

    if squeeze:
        out_ra, out_dec = out_ra[0], out_dec[0]
    if return_convergence:
        return out_ra, out_dec, last_change
    return out_ra, out_dec


# ============================================================================
# ABERRATION CORRECTION IN THE PIPELINE
# ============================================================================

def correct_catalog_for_aberration(catalog, phase_center, times, location=None, ra_idx=0, dec_idx=1):
    """
    Remove differential aberration from a matched catalogue's positions.

    Run straight after loading, before any filtering, so every later step sees
    corrected positions. Only the RA and Dec columns change; phase_offset_deg
    is left as is, since the change (< 0.1 arcsec) is far below any filter
    threshold.

    Parameters
    ----------
    catalog : ndarray, shape (n_sources, n_epochs, n_columns)
        Matched catalogue (see CATALOG_COLUMNS).
    phase_center : str or SkyCoord
        BOOKKEEPING.phase_ctr.
    times : Time, shape (n_epochs,)
        From load_epoch_times, aligned with the catalogue's epochs.
    location : EarthLocation, optional
        From observatory_location(config).
    ra_idx, dec_idx : int
        Column indices of RA and Dec in degrees.

    Returns
    -------
    ndarray
        A corrected copy of the catalogue.
    """
    if catalog.shape[1] != len(times):
        raise ValueError(f"catalog has {catalog.shape[1]} epochs but {len(times)} times were given")

    pc = parse_phase_center(phase_center)
    ra, dec = catalog[:, :, ra_idx], catalog[:, :, dec_idx]
    new_ra, new_dec, conv = correct_positions_for_aberration(ra, dec, pc, times, location,
                                                            return_convergence=True)
    out = catalog.copy()
    out[:, :, ra_idx] = new_ra
    out[:, :, dec_idx] = new_dec

    # The shifts grow with distance from the phase centre, up to ~0.2 arcsec.
    ok = np.isfinite(ra) & np.isfinite(dec)
    if np.any(ok):
        shift = _angle_arcsec(_radec_to_unit(ra[ok], dec[ok]), _radec_to_unit(new_ra[ok], new_dec[ok]))
        s = aberration_scale(pc, times, location)
        msg(f"Differential aberration: {ok.sum()} positions over {catalog.shape[1]} epochs "
            f"({times.min().isot[:10]} to {times.max().isot[:10]}), scale {s.min() * 1e5:+.2f} to "
            f"{s.max() * 1e5:+.2f} x 1e-5, shift median {np.median(shift) * 1e3:.1f} mas, max "
            f"{np.max(shift) * 1e3:.1f} mas, inverse converged to {conv * 1e3:.1e} mas")
    return out


def aberration_correction_enabled(config):
    """True if MCMC.correct_differential_aberration is set in astkat.yaml (False if absent)."""
    return bool(((config or {}).get('MCMC', {}) or {}).get('correct_differential_aberration', False))


def apply_aberration_correction_from_config(catalog, config, results_dir='results'):
    """
    Correct a freshly loaded matched catalogue for differential aberration, as configured.

    Call on the full catalogue straight after loading, before any filtering,
    because only then do its epochs map one-to-one onto
    observation_properties.json. Skipped for simulated projects (see
    is_simulated) and when MCMC.correct_differential_aberration is false.

    Parameters
    ----------
    catalog : ndarray, shape (n_sources, n_epochs, n_columns)
        The unfiltered matched catalogue.
    config : dict
        Parsed astkat.yaml (MCMC.correct_differential_aberration,
        MCMC.observatory_location, BOOKKEEPING.phase_ctr).
    results_dir : str or Path
        Directory holding observation_properties.json.

    Returns
    -------
    (catalog, info)
        The corrected catalogue, or the input unchanged when skipped, and a
        JSON-serialisable record with 'corrected' plus either the reason it
        was skipped or the time range, first-order scale range and
        observatory location.
    """
    if is_simulated(config):
        msg("Differential aberration not corrected: simulated data contain none")
        return catalog, {'corrected': False, 'reason': 'simulated data'}
    if not aberration_correction_enabled(config):
        msg("Differential aberration not corrected (MCMC.correct_differential_aberration false)")
        return catalog, {'corrected': False, 'reason': 'disabled in astkat.yaml'}

    # Each epoch's observation time sets the observer's velocity, and so the size of the shift.
    times = load_epoch_times(catalog.shape[1], results_dir=results_dir)
    location = observatory_location(config)
    phase_center = parse_phase_center(config['BOOKKEEPING']['phase_ctr'])
    corrected = correct_catalog_for_aberration(catalog, phase_center, times, location)
    s = aberration_scale(phase_center, times, location)
    info = {
        'corrected': True,
        'model': 'closed form (annual + diurnal aberration, solar deflection), zero point at phase centre',
        'n_epochs': int(catalog.shape[1]),
        'time_start': times.min().isot,
        'time_end': times.max().isot,
        'first_order_scale_min': float(s.min()),
        'first_order_scale_max': float(s.max()),
        'observatory_location': {'lat_deg': float(location.lat.deg),
                                 'lon_deg': float(location.lon.deg),
                                 'height_m': float(location.height.to(u.m).value)},
    }
    return corrected, info
