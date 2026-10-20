"""
AstKAT source matching.

Cross-matches the epoch catalogues against a reference catalogue (the deep
image if there is one, otherwise THRESHOLDS.ref_index), keeps sources that
are detected often enough, point-like over their epochs and not variable,
drops epochs with too few sources, and saves the matched catalogue.

Usage
    python matching.py

Writes
    results/matched_catalog_{project}.npz
        data, shape (n_sources, n_epochs, n_columns), NaN where a source was
        not detected; columns as in utils.CATALOG_COLUMNS; peak_noise, the
        measured scatter k of a fitted peak over the rms.
"""

import json
from pathlib import Path

import numpy as np
import bottleneck as bn
import yaml
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from scipy.spatial import cKDTree
from scipy.stats import chi2, norm

try:
    from .utils import (msg, title, parse_phase_center, column_index, CATALOG_COLUMNS, S_CODES, AMBIGUOUS,
                        epoch_frequencies, frequency_bands, tangent_plane_arcsec, beam_frame,
                        remove_truth_csv_rows_by_position, remove_observation_properties_entries)
except ImportError:
    from utils import (msg, title, parse_phase_center, column_index, CATALOG_COLUMNS, S_CODES, AMBIGUOUS,
                       epoch_frequencies, frequency_bands, tangent_plane_arcsec, beam_frame,
                       remove_truth_csv_rows_by_position, remove_observation_properties_entries)


CONFIG_FILE = Path('astkat_files') / 'astkat.yaml'
FILTERED_DIR = Path('astkat_files')
RESULTS_DIR = Path('results')

RA, DEC = column_index('ra_deg'), column_index('dec_deg')
FLUX, RMS, SNR = column_index('peak_flux_Jy'), column_index('rms_Jy'), column_index('snr')
NEIGHBOUR_BEAMS = 2.0      # a matched source closer than this many BMAJ counts as a neighbour
SELECTION_SNR_EDGES = (0.0, 5.0, 10.0, 30.0, 100.0, 1000.0, np.inf)   # median detected SNR bins of the selection table
VARIABILITY_CLIP = 5.0          # a flux this many robust deviations from its band's median is a failed fit, not variability
PEAK_NOISE_SNR = (10.0, 30.0)   # sources whose peak scatter measures k: untruncated, flux scale negligible
PEAK_NOISE_MIN_DETECTIONS = 10  # ... each detected at least this often


def load_json(path):
    """Contents of a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


# ============================================================================
# REFERENCE AND CROSS-MATCHING
# ============================================================================

def project_phase_centre(config):
    """Phase centre from astkat.yaml; raises if extraction.py has not set it yet."""
    centre = parse_phase_center(config['BOOKKEEPING']['phase_ctr'])
    if centre.separation(SkyCoord(0 * u.deg, 0 * u.deg)).arcsec < 1e-6:
        raise ValueError("BOOKKEEPING.phase_ctr is still the placeholder; run extraction.py first")
    msg(f"Phase centre: {config['BOOKKEEPING']['phase_ctr']}")
    return centre


def reference_catalogue(config):
    """
    The reference catalogue and the epoch catalogues, in time order.

    The deep catalogue is the reference when it exists. Otherwise the epoch
    catalogue at THRESHOLDS.ref_index is, and it also stays in the epoch list.

    Returns
    -------
    reference : dict
        Filtered catalogue contents.
    epoch_files : list of Path
    ref_epoch : int or None
        Epoch index of the reference, or None for the deep catalogue.
    """
    project_name = config['BOOKKEEPING']['project_name']
    deep_file = FILTERED_DIR / f'filtered_catalog_deep_{project_name}.json'
    epoch_files = sorted(f for f in FILTERED_DIR.glob('filtered_catalog_*.json') if f != deep_file)
    if not epoch_files:
        raise FileNotFoundError(f"No filtered epoch catalogues in {FILTERED_DIR}; run filtering.py first")
    msg(f"Found {len(epoch_files)} epoch catalogues")

    if deep_file.exists():
        msg(f"Reference: deep catalogue {deep_file.name}")
        return load_json(deep_file), epoch_files, None

    ref_epoch = int(config['THRESHOLDS']['ref_index'])
    if ref_epoch >= len(epoch_files):
        raise IndexError(f"THRESHOLDS.ref_index {ref_epoch} exceeds the {len(epoch_files)} epoch catalogues")
    msg(f"Reference: epoch {ref_epoch}, {epoch_files[ref_epoch].name} (no deep catalogue)")
    return load_json(epoch_files[ref_epoch]), epoch_files, ref_epoch


def epoch_table(cat, phase_centre):
    """Rows of one filtered catalogue in CATALOG_COLUMNS order, shape (n, n_columns)."""
    coords = SkyCoord(np.array(cat['ra_deg']) * u.deg, np.array(cat['dec_deg']) * u.deg, frame='icrs')
    peak, rms = np.array(cat['peak_Jy']), np.array(cat['rms_Jy'])
    values = {
        'ra_deg': cat['ra_deg'], 'dec_deg': cat['dec_deg'],
        'peak_flux_Jy': peak, 'rms_Jy': rms, 'snr': peak / rms,
        'phase_offset_deg': coords.separation(phase_centre).deg,
        'bmaj_deg': cat['beam_bmaj_deg'], 'bmin_deg': cat['beam_bmin_deg'], 'bpa_deg': cat['beam_bpa_deg'],
        'freq_Hz': cat['freq_Hz'],
        'flux_ratio': cat['flux_ratio'], 'maj_ratio': cat['maj_ratio'], 'min_ratio': cat['min_ratio'],
        's_code': cat['s_code'],
    }
    return np.column_stack([values[name] for name in CATALOG_COLUMNS]), coords


def match_epochs(reference, epoch_files, ref_epoch, phase_centre, match_threshold):
    """
    Place each epoch's sources against the reference sources.

    Separations are measured in the epoch's beam, along each axis in units
    of that axis's FWHM (utils.beam_frame), so the matching region follows
    the beam's shape. A source matches the reference source nearest in that
    measure within match_threshold, an ellipse with semi-axes match_threshold
    times BMAJ and BMIN (0.75: 1.5 times the beam). Reference sources matched
    by more than one source in an epoch keep none of them and are marked
    ambiguous (S_Code A) in that epoch. So is a reference source left without
    a match while a source assigned to another lies inside its ellipse: the
    two were blended, so it was not seen on its own rather than missed.

    Returns
    -------
    ndarray, shape (n_reference, n_epochs, n_columns), NaN where unmatched
    """
    ref_xy = tangent_plane_arcsec(np.array(reference['ra_deg']), np.array(reference['dec_deg']), phase_centre)
    catalog = np.full((len(ref_xy), len(epoch_files), len(CATALOG_COLUMNS)), np.nan)

    for e, epoch_file in enumerate(epoch_files):
        cat = reference if e == ref_epoch else load_json(epoch_file)
        if not cat['ra_deg']:
            msg(f"{epoch_file.name}: no sources")
            continue
        table, _ = epoch_table(cat, phase_centre)

        if e == ref_epoch:
            catalog[:, e, :] = table
            msg(f"{epoch_file.name}: reference epoch, {len(table)} sources")
            continue

        beam = (cat['beam_bmaj_deg'][0], cat['beam_bmin_deg'][0], cat['beam_bpa_deg'][0])
        xy = tangent_plane_arcsec(table[:, RA], table[:, DEC], phase_centre)
        ref_beam, epoch_beam = beam_frame(ref_xy, *beam), beam_frame(xy, *beam)
        distance, nearest = cKDTree(ref_beam).query(epoch_beam)
        close = np.nonzero(distance < match_threshold)[0]
        targets, counts = np.unique(nearest[close], return_counts=True)
        unique_targets = set(targets[counts == 1])
        keep = np.array([i for i in close if nearest[i] in unique_targets], dtype=int)
        if keep.size:
            catalog[nearest[keep], e, :] = table[keep]
        catalog[targets[counts > 1], e, column_index('s_code')] = AMBIGUOUS
        # Unmatched, yet a source assigned to a neighbour lies inside the ellipse: blended, not missed.
        unmatched = np.setdiff1d(np.arange(len(ref_xy)), targets)
        inside = cKDTree(epoch_beam).query_ball_point(ref_beam[unmatched], match_threshold)
        blended = unmatched[np.array([len(i) > 0 for i in inside], dtype=bool)]
        catalog[blended, e, column_index('s_code')] = AMBIGUOUS

        n_ambiguous = int(np.sum(counts > 1))
        msg(f"{epoch_file.name}: matched {keep.size} sources within {match_threshold * beam[0] * 3600:.2f} x "
            f"{match_threshold * beam[1] * 3600:.2f} arcsec at {beam[2]:.0f} deg"
            + (f"; {n_ambiguous} reference sources matched ambiguously" if n_ambiguous else "")
            + (f"; {blended.size} blended with a neighbour" if blended.size else "")
            + ("; both marked A" if n_ambiguous and blended.size else
               "; marked A" if n_ambiguous or blended.size else ""))
    return catalog


# ============================================================================
# SOURCE AND EPOCH SELECTION
# ============================================================================

def variability_significance(flux, rms, band, k, frac_flux_sys, min_epochs_per_band):
    """
    Variability of one source in sigma, from the chi-squared of its fluxes, robust to failed fits.

    Epochs are grouped by frequency band (utils.frequency_bands, one label per
    epoch) and each band is compared with its own median, so spectral-index
    offsets between bands do not count as variability. Each measurement's
    error is the larger of k times its rms (k from peak_noise_factor) and
    frac_flux_sys times the band's median flux (flux-scale calibration).

    A failed PyBDSF fit can report a peak far from the image's (a near-zero
    peak beside a bright source, or a several-Gaussian island's inflated one),
    and one such epoch would dominate the chi-squared. So within each band the
    measurements' deviations from the median, in units of their errors, are
    compared with their own robust spread (1.4826 MAD, at least 1); those more
    than VARIABILITY_CLIP spreads out are set aside before the chi-squared. A
    truly variable source has a wide spread, so its epochs are kept and it is
    still caught. Bands with fewer than min_epochs_per_band epochs left are
    skipped. Per-band chi-squared values are summed and converted to a
    one-sided significance.

    Returns
    -------
    significance : float
        In sigma, or NaN if no band had enough epochs.
    n_clipped : int
        Measurements set aside as failed fits.
    """
    ok = np.isfinite(flux) & np.isfinite(rms)
    flux, rms, band = flux[ok], rms[ok], band[ok]
    chi2_stat, dof, n_clipped = 0.0, 0, 0
    for label in np.unique(band):
        in_band = band == label
        f = flux[in_band]
        err = np.maximum(k * rms[in_band], frac_flux_sys * abs(np.median(f)))
        z = (f - np.median(f)) / err
        spread = max(1.4826 * np.median(np.abs(z - np.median(z))), 1.0)
        good = np.abs(z - np.median(z)) <= VARIABILITY_CLIP * spread
        n_clipped += int((~good).sum())
        if good.sum() < min_epochs_per_band:
            continue
        f, err = f[good], err[good]
        chi2_stat += np.sum((f - np.median(f)) ** 2 / err ** 2)
        dof += int(good.sum()) - 1
    if dof <= 0:
        return np.nan, n_clipped
    p_value = np.clip(chi2.sf(chi2_stat, dof), 1e-10, 1 - 1e-10)
    return norm.isf(p_value), n_clipped


def peak_noise_factor(catalog, candidates, snr_range=PEAK_NOISE_SNR, min_detections=PEAK_NOISE_MIN_DETECTIONS):
    """
    Scatter of a fitted peak over the image rms, k, measured from the data.

    The residuals (peak - median peak) / rms of every candidate source with
    median detected SNR inside snr_range and at least min_detections
    detections are pooled, and k is their robust (1.4826 MAD) scatter. The
    range keeps sources bright enough that the detection threshold does not
    truncate their peaks and faint enough that flux-scale errors do not
    dominate; the MAD keeps the few variable ones out.
    """
    peak, rms = catalog[:, :, FLUX], catalog[:, :, RMS]
    with np.errstate(all='ignore'):
        median_snr = bn.nanmedian(catalog[:, :, SNR], axis=1)
        z = (peak - bn.nanmedian(peak, axis=1)[:, None]) / rms
    use = (candidates & (median_snr >= snr_range[0]) & (median_snr < snr_range[1])
           & (np.sum(np.isfinite(peak), axis=1) >= min_detections))
    if not use.any():
        raise ValueError(f"No source at median SNR {snr_range[0]:g}-{snr_range[1]:g} with at least "
                         f"{min_detections} detections to measure the peak noise from")
    pooled = z[use]
    k = float(1.4826 * np.median(np.abs(pooled[np.isfinite(pooled)])))
    msg(f"Peak noise: k = {k:.3f} rms, from {int(use.sum())} sources at median SNR "
        f"{snr_range[0]:g}-{snr_range[1]:g}")
    return k


def nearest_neighbour_beams(catalog):
    """Distance from each source's median position to the nearest other matched source, in median BMAJ."""
    with np.errstate(all='ignore'):
        ra, dec = bn.nanmedian(catalog[:, :, RA], axis=1), bn.nanmedian(catalog[:, :, DEC], axis=1)
    seen = np.isfinite(ra) & np.isfinite(dec)
    distance = np.full(len(ra), np.inf)
    coords = SkyCoord(ra[seen] * u.deg, dec[seen] * u.deg, frame='icrs')
    _, separation, _ = match_coordinates_sky(coords, coords, nthneighbor=2)
    distance[seen] = separation.deg / bn.nanmedian(catalog[:, :, column_index('bmaj_deg')])
    return distance


def snr_counts(snr, edges=SELECTION_SNR_EDGES):
    """Number of sources in each SNR bin, then the total."""
    return list(np.histogram(snr, bins=edges)[0]) + [snr.size]


def select_sources(catalog, thresholds, tolerance, radius):
    """
    Keep sources detected in enough epochs that are point-like, mostly single Gaussians and not variable.

    Sources further than radius (MCMC.max_phase_offset_deg, by their median
    phase offset) from the phase centre are removed first, as the fit never
    uses them.
    A source needs max(epoch_fraction * n_epochs, epoch_min) detections (all
    epochs if there are fewer than epoch_min). It is point-like when the
    medians over its detections satisfy |Gaussian total flux / peak - 1| <
    flux_threshold and |axis / beam axis - 1| < size_threshold on both axes;
    testing medians, not single epochs, keeps every detection of a point
    source however noisy. At least single_fraction of its detections must
    have PyBDSF S_Code S (one Gaussian alone in its island); a source mostly
    C or M is blended or structured, which can displace even its S epochs.
    Its variability significance must lie below variability_threshold;
    sources that cannot be tested are kept. Epochs are grouped into
    frequency bands with tolerance (MCMC.frequency_match_tolerance) as the fit
    groups them: the variability test compares each band with its own
    median, and the peak noise k is measured in the dominant band only, so a
    spectral index between bands does not inflate it.

    Logs a table of the sources each step removes in bins of median detected
    SNR (SELECTION_SNR_EDGES), with the point-like tests failed (a source can
    fail several) and how many removed sources have another matched source
    within NEIGHBOUR_BEAMS times BMAJ.

    Returns
    -------
    catalog : ndarray
        The kept sources.
    k : float
        The measured peak noise (see peak_noise_factor).
    """
    n_sources, n_epochs, _ = catalog.shape
    with np.errstate(all='ignore'):
        median_snr = bn.nanmedian(catalog[:, :, SNR], axis=1)
    neighbour = nearest_neighbour_beams(catalog) < NEIGHBOUR_BEAMS
    keep = np.isfinite(median_snr)
    rows = [('Matched', snr_counts(median_snr[keep]))]
    notes = []

    def remove(label, passed):
        """Record the sources a step removes, narrow keep to those that pass, and return the removed."""
        removed = keep & ~passed
        rows.append((label, [-n for n in snr_counts(median_snr[removed])]))
        keep[:] = keep & passed
        return removed

    with np.errstate(all='ignore'):
        offset = bn.nanmedian(catalog[:, :, column_index('phase_offset_deg')], axis=1)
    remove(f"Phase centre (<= {radius:g} deg)", offset <= radius)

    epoch_min = float(thresholds['epoch_min'])
    if n_epochs < epoch_min:
        need = n_epochs
        msg(f"Warning: only {n_epochs} epochs, fewer than epoch_min ({epoch_min:.0f}); requiring all of them")
    else:
        need = max(float(thresholds['epoch_fraction']) * n_epochs, epoch_min)
    remove(f"Detection (>= {need:.0f} epochs)", np.sum(np.isfinite(catalog[:, :, FLUX]), axis=1) >= need)

    # A merged reference source (two sources the deep image could not split) is matched ambiguously in most epochs.
    code = catalog[:, :, column_index('s_code')]
    ambiguous_fraction = float(thresholds['ambiguous_fraction'])
    with np.errstate(all='ignore'):
        ambiguous = np.sum(code == AMBIGUOUS, axis=1) / np.sum(np.isfinite(code), axis=1)
    remove(f"Unblended (ambiguous <= {ambiguous_fraction * 100:.0f}%)", ambiguous <= ambiguous_fraction)

    flux_threshold, size_threshold = float(thresholds['flux_threshold']), float(thresholds['size_threshold'])
    with np.errstate(all='ignore'):
        median = {name: bn.nanmedian(catalog[:, :, column_index(name)], axis=1)
                  for name in ('flux_ratio', 'maj_ratio', 'min_ratio')}
    fails = {'flux ratio': ~(np.abs(median['flux_ratio'] - 1.0) < flux_threshold),
             'major': ~(np.abs(median['maj_ratio'] - 1.0) < size_threshold),
             'minor': ~(np.abs(median['min_ratio'] - 1.0) < size_threshold)}
    removed = remove("Point-like (medians)", ~(fails['flux ratio'] | fails['major'] | fails['minor']))
    notes.append("Point-like failures: " + ", ".join(f"{name} {int((removed & fail).sum())}"
                                                     for name, fail in fails.items())
                 + f"; neighbour < {NEIGHBOUR_BEAMS:g} beams: {int((removed & neighbour).sum())}")

    single_fraction = float(thresholds['single_fraction'])
    with np.errstate(all='ignore'):
        single = (np.sum(code == S_CODES.index('S'), axis=1) / np.sum(np.isfinite(code) & (code != AMBIGUOUS), axis=1)
                  >= single_fraction)
    removed = remove(f"Single Gaussian (S >= {single_fraction * 100:.0f}%)", single)
    notes.append(f"Single Gaussian removals with a neighbour < {NEIGHBOUR_BEAMS:g} beams: "
                 f"{int((removed & neighbour).sum())}")

    # Peaks scatter by more than the rms (position and size fitted, correlated noise), so the
    # variability test uses the measured factor, from the dominant band alone.
    band = frequency_bands(epoch_frequencies(catalog), tolerance)
    if band.max() > 0:
        msg(f"Frequency bands (within {tolerance * 100:g}%): {band.max() + 1}, the dominant one with "
            f"{int((band == 0).sum())} of {n_epochs} epochs")
    k = peak_noise_factor(catalog[:, band == 0], keep)
    significance = np.full(n_sources, np.nan)
    clipped = 0
    for i in np.nonzero(keep)[0]:
        significance[i], n = variability_significance(catalog[i, :, FLUX], catalog[i, :, RMS], band, k,
                                                   float(thresholds['frac_flux_sys']),
                                                   float(thresholds['min_epochs_per_freq_group']))
        clipped += n
    threshold = float(thresholds['variability_threshold'])
    untested = int((keep & np.isnan(significance)).sum())
    remove(f"Variability (< {threshold:g} sigma)", ~(significance >= threshold))
    if untested:
        notes.append(f"Variability: {untested} sources kept untested")
    if clipped:
        notes.append(f"Variability: {clipped} detections more than {VARIABILITY_CLIP:g} robust deviations from their "
                     f"source's median flux set aside as failed fits")
    rows.append(('Kept', snr_counts(median_snr[keep])))

    labels = [f"<{SELECTION_SNR_EDGES[1]:g}"] + [f"{lo:g}-{hi:g}" for lo, hi in
                                                zip(SELECTION_SNR_EDGES[1:-2], SELECTION_SNR_EDGES[2:-1])]
    labels += [f">{SELECTION_SNR_EDGES[-2]:g}", 'all']
    msg("Source selection by median detected SNR:")
    msg(f"  {'':<34}" + "".join(f"{label:>10}" for label in labels))
    for label, counts in rows:
        msg(f"  {label:<34}" + "".join(f"{c:>10}" if c else f"{'.':>10}" for c in counts))
    for note in notes:
        msg(f"  {note}")
    return catalog[keep], k


def drop_sparse_epochs(catalog, config):
    """
    Remove epochs with fewer than THRESHOLDS.min_sources_per_epoch sources.

    The same epochs are removed from observation_properties.json and, for
    simulations, from per_epoch_offsets.csv, so all three stay aligned.
    """
    minimum = config['THRESHOLDS'].get('min_sources_per_epoch', 10)
    sparse = np.sum(np.isfinite(catalog[:, :, RA]), axis=0) < minimum
    if not sparse.any():
        msg(f"Epoch filter (< {minimum} sources): no epochs removed")
        return catalog

    positions = np.nonzero(sparse)[0].tolist()
    reason = f"< {minimum} sources detected"
    image_dir = config['BOOKKEEPING'].get('image_dir')
    if image_dir:
        remove_truth_csv_rows_by_position(Path(image_dir) / 'per_epoch_offsets.csv', positions, reason=reason)
    remove_observation_properties_entries(RESULTS_DIR / 'observation_properties.json',
                                          positions_to_remove=positions, reason=reason)
    msg(f"Epoch filter (< {minimum} sources): removed epochs {positions}, {int((~sparse).sum())} kept")
    return catalog[:, ~sparse, :]


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Match, select and save the project's sources."""
    title("AstKAT source matching")
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)

    phase_centre = project_phase_centre(config)

    # Sources are identified against one reference list: the deep image if available.
    reference, epoch_files, ref_epoch = reference_catalogue(config)
    msg(f"Reference catalogue: {len(reference['ra_deg'])} sources")

    # One row per reference source and one column per epoch; NaN where a source is not detected.
    catalog = match_epochs(reference, epoch_files, ref_epoch, phase_centre,
                           float(config['THRESHOLDS']['match_threshold']))
    # Keep sources detected often enough and with steady flux, as variable sources make poor references.
    catalog, k = select_sources(catalog, config['THRESHOLDS'], float(config['MCMC']['frequency_match_tolerance']),
                                float(config['MCMC']['max_phase_offset_deg']))

    # Epochs with too few sources cannot constrain their own translation in the fit.
    catalog = drop_sparse_epochs(catalog, config)

    # The measured peak noise k is where fit.py starts its own.
    out_file = RESULTS_DIR / f"matched_catalog_{config['BOOKKEEPING']['project_name']}.npz"
    np.savez(out_file, data=catalog, columns=list(CATALOG_COLUMNS), shape=catalog.shape, peak_noise=k,
             description="AstKAT matched source catalogue (n_sources, n_epochs, n_columns); "
                         "snr is each detection's peak flux over its rms; peak_noise the measured k")
    msg(f"Saved matched catalogue: {out_file} ({catalog.shape[0]} sources x {catalog.shape[1]} epochs)")


if __name__ == "__main__":
    main()
