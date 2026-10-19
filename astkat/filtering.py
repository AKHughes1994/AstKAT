"""
AstKAT catalogue conversion.

Converts each PyBDSF source list to JSON with its epoch's restoring beam and,
per source, the quantities matching.py's per-source tests use: the Gaussian
total flux over the peak, the fitted axes over the beam's and the S_Code (an
index into utils.S_CODES). Epoch catalogues keep every detection, since
matching.py tests each source on its medians over epochs and noise in one
epoch should never remove a detection. The deep catalogue, which supplies the
reference source list, keeps only sources at or above THRESHOLDS.snr_threshold.

Usage
    python filtering.py

Writes
    astkat_files/filtered_catalog_{DATE-OBS}_{project}.json   one per epoch catalogue
    astkat_files/filtered_catalog_deep_{project}.json
"""

import json
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

try:
    from .utils import msg, title, S_CODES
except ImportError:
    from utils import msg, title, S_CODES


CONFIG_FILE = Path('astkat_files') / 'astkat.yaml'
CATALOG_DIR = Path('pybdsf_files')
OUTPUT_DIR = Path('astkat_files')
RESULTS_DIR = Path('results')


def epoch_beams(results_dir=RESULTS_DIR):
    """
    Restoring beam of each epoch, keyed by DATE-OBS.

    Read from observation_properties.json, which extraction.py fills from the
    image headers. BPA is East of North, as for PyBDSF's sky-plane sizes.

    Returns
    -------
    dict
        DATE-OBS string -> (bmaj_deg, bmin_deg, bpa_deg)
    """
    with open(Path(results_dir) / 'observation_properties.json', 'r') as f:
        props = json.load(f)
    return {t: (bmaj, bmin, bpa) for t, bmaj, bmin, bpa in
            zip(props['obs_isot'], props['bmaj_deg'], props['bmin_deg'], props['bpa_deg'])}


def image_beam(image_path):
    """Restoring beam (bmaj_deg, bmin_deg, bpa_deg) from an image header, BPA wrapped to [-90, 90)."""
    header = fits.getheader(image_path)
    return header['BMAJ'], header['BMIN'], ((header['BPA'] + 90) % 180) - 90


def convert_catalog(cat_file, beam, snr_threshold=None):
    """
    Write one PyBDSF source list as JSON, with the beam and the per-source test quantities.

    Parameters
    ----------
    cat_file : Path
        PyBDSF source list (FITS).
    beam : tuple
        (bmaj_deg, bmin_deg, bpa_deg) of the image.
    snr_threshold : float or None
        Keep only sources with Peak_flux / isl_rms at or above this; None keeps every source.
    """
    with fits.open(cat_file) as hdul:
        data = hdul[1].data
        freq = hdul[1].header['FREQ0']

    mask = (np.ones(len(data), dtype=bool) if snr_threshold is None
            else data['Peak_flux'] / data['isl_rms'] >= snr_threshold)
    n_kept = int(np.count_nonzero(mask))
    output = {
        'ra_deg': data['RA'][mask].tolist(),
        'dec_deg': data['DEC'][mask].tolist(),
        'peak_Jy': data['Peak_flux'][mask].tolist(),
        'rms_Jy': data['isl_rms'][mask].tolist(),
        'beam_bmaj_deg': [float(beam[0])] * n_kept,
        'beam_bmin_deg': [float(beam[1])] * n_kept,
        'beam_bpa_deg': [float(beam[2])] * n_kept,
        'freq_Hz': [float(freq)] * n_kept,
        'flux_ratio': (data['Total_flux'][mask] / data['Peak_flux'][mask]).tolist(),
        'maj_ratio': (data['Maj'][mask] / beam[0]).tolist(),
        'min_ratio': (data['Min'][mask] / beam[1]).tolist(),
        's_code': [S_CODES.index(code.strip()) for code in data['S_Code'][mask]],
    }
    out_file = OUTPUT_DIR / f"filtered_{cat_file.stem}.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=4)
    msg(f"  {len(mask)} sources: {out_file}" if snr_threshold is None
        else f"  {n_kept} of {len(mask)} sources with SNR >= {snr_threshold:g}: {out_file}")


def main():
    """Convert every PyBDSF catalogue of the project."""
    title("AstKAT catalogue conversion")
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    project_name = config['BOOKKEEPING']['project_name']
    deep_image = config['BOOKKEEPING'].get('deep_image', False)

    # Each catalogue is stored with its own image's restoring beam, against which its sizes are tested.
    beams = epoch_beams()
    deep_name = f"catalog_deep_{project_name}"
    prefix, suffix = "catalog_", f"_{project_name}"

    for cat_file in sorted(CATALOG_DIR.glob('catalog_*.fits')):
        msg(f"Converting {cat_file.stem}")
        # Every deep source becomes a row of the matched catalogue, so a spurious one costs a row matched
        # only by chance; the SNR cut keeps the reference list to real sources.
        if cat_file.stem == deep_name:
            if not isinstance(deep_image, str):
                raise ValueError(f"{cat_file.name} exists but BOOKKEEPING.deep_image is not set")
            convert_catalog(cat_file, image_beam(deep_image), float(config['THRESHOLDS']['snr_threshold']))
            continue

        # Epoch catalogues are named catalog_{DATE-OBS}_{project}.
        obs_isot = cat_file.stem[len(prefix):-len(suffix)]
        if obs_isot not in beams:
            raise KeyError(f"{cat_file.name}: DATE-OBS {obs_isot} not in observation_properties.json; "
                           f"re-run extraction.py")
        convert_catalog(cat_file, beams[obs_isot])


if __name__ == "__main__":
    main()
