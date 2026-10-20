"""
AstKAT source extraction.

Runs PyBDSF on every epoch image and on the deep image, records the
observation properties of each epoch, and sets the project phase centre in
astkat.yaml from the image headers.

Usage
    python extraction.py

Writes
    pybdsf_files/catalog_{DATE-OBS}_{project}.fits   one source list per epoch
    pybdsf_files/catalog_deep_{project}.fits         source list of the deep image
    results/observation_properties.json              per-epoch metadata, in time order
"""

import csv
import json
import sys
import re
import warnings
from pathlib import Path

import yaml
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits

try:
    from .utils import msg, title, remove_observation_properties_entries, is_simulated
except ImportError:
    from utils import msg, title, remove_observation_properties_entries, is_simulated

# PyBDSF's Gaussian fitter reports hitting its evaluation limit on some islands; harmless here.
warnings.filterwarnings("ignore", message=r"Number of calls to function has reached maxfev",
                        category=RuntimeWarning)


CONFIG_FILE = Path('astkat_files') / 'astkat.yaml'
CATALOG_DIR = Path('pybdsf_files')
RESULTS_DIR = Path('results')

DEEP_SNR_THRESHOLD = 5.0        # PyBDSF thresh_pix for the deep image
DEEP_ISLAND_THRESHOLD = 3.0     # PyBDSF thresh_isl for the deep image
DEEP_RMS_FACTOR = 0.25          # a fixed BOOKKEEPING.rms is scaled by this for the deep image
PHASE_CENTRE_TOLERANCE_ARCSEC = 1e-3


# ============================================================================
# IMAGES AND PHASE CENTRE
# ============================================================================

def find_images(config):
    """
    Epoch images and the deep image.

    Epoch images are every FITS file in BOOKKEEPING.image_dir apart from the
    deep image, sorted by name. The deep image (BOOKKEEPING.deep_image) may
    sit anywhere.

    Returns
    -------
    (epoch_images, deep_image)
        A list of paths, and a path or None.
    """
    bookkeeping = config['BOOKKEEPING']
    image_dir = Path(bookkeeping.get('image_dir', '.'))
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    deep_image = bookkeeping.get('deep_image', False)
    deep_path = Path(deep_image) if isinstance(deep_image, str) else None
    if deep_path is not None and not deep_path.exists():
        raise FileNotFoundError(f"Deep image not found: {deep_path}")

    epoch_images = [str(p) for p in sorted(image_dir.glob('*.fits'))
                    if deep_path is None or p.name != deep_path.name]
    if not epoch_images:
        raise FileNotFoundError(f"No epoch images found in {image_dir}")

    msg(f"Found {len(epoch_images)} epoch images in {image_dir}")
    if deep_path is not None:
        msg(f"Deep image: {deep_path}")
    return epoch_images, (str(deep_path) if deep_path is not None else None)


def _header_phase_centre(header):
    """Phase centre (CRVAL1, CRVAL2 in their CUNIT units) as a SkyCoord, or None if absent."""
    ra, dec = header.get('CRVAL1'), header.get('CRVAL2')
    if ra is None or dec is None:
        return None
    return SkyCoord(Angle(float(ra), unit=header.get('CUNIT1', 'deg')),
                    Angle(float(dec), unit=header.get('CUNIT2', 'deg')), frame='icrs')


def common_phase_centre(images):
    """
    Phase centre shared by all epoch images.

    Raises
    ------
    ValueError
        If any image lacks CRVAL1/CRVAL2 or its phase centre differs from the
        first image's by more than PHASE_CENTRE_TOLERANCE_ARCSEC.
    """
    centre = _header_phase_centre(fits.getheader(images[0]))
    if centre is None:
        raise ValueError(f"No phase centre (CRVAL1/CRVAL2) in {Path(images[0]).name}")

    mismatches = []
    for image in images[1:]:
        other = _header_phase_centre(fits.getheader(image))
        if other is None:
            mismatches.append(f"{Path(image).name}: no CRVAL1/CRVAL2")
        elif centre.separation(other).arcsec > PHASE_CENTRE_TOLERANCE_ARCSEC:
            mismatches.append(f"{Path(image).name}: RA={other.ra.deg:.9f}, Dec={other.dec.deg:.9f} deg, "
                              f"{centre.separation(other).arcsec:.4f} arcsec from the first image")
    if mismatches:
        for line in mismatches[:10]:
            msg(f"  {line}")
        raise ValueError(f"{len(mismatches)} epoch image(s) do not share the phase centre of "
                         f"{Path(images[0]).name} (tolerance {PHASE_CENTRE_TOLERANCE_ARCSEC} arcsec)")
    return centre


def phase_centre_string(centre):
    """A phase centre as BOOKKEEPING.phase_ctr stores it, 'HH:MM:SS.sssss +DD:MM:SS.ssss'."""
    return (f"{centre.ra.wrap_at('24h').to_string(unit='hourangle', sep=':', precision=5)} "
            f"{centre.dec.to_string(unit='deg', sep=':', precision=4, alwayssign=True)}")


def write_phase_centre(centre, config_file=CONFIG_FILE):
    """Set BOOKKEEPING.phase_ctr in astkat.yaml (see phase_centre_string)."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config['BOOKKEEPING']['phase_ctr'] = phase_centre_string(centre)
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    msg(f"Phase centre {config['BOOKKEEPING']['phase_ctr']} written to {config_file}")


# ============================================================================
# OBSERVATION PROPERTIES
# ============================================================================

def image_properties(image_path):
    """
    Header properties of one epoch image.

    Returns
    -------
    dict
        image_identifier, obs_isot (DATE-OBS), bmaj_deg, bmin_deg, bpa_deg
        (wrapped to [-90, 90)), freq_ghz (CRVAL3), cdelt1_deg and cdelt2_deg
        (absolute pixel sizes).
    """
    header = fits.getheader(image_path)
    freq_hz = header.get('CRVAL3', 0.0)
    return {
        'image_identifier': Path(image_path).name,
        'obs_isot': header.get('DATE-OBS', 'unknown'),
        'bmaj_deg': header.get('BMAJ', 0.0),
        'bmin_deg': header.get('BMIN', 0.0),
        'bpa_deg': ((header.get('BPA', 0.0) + 90) % 180) - 90,
        'freq_ghz': freq_hz / 1e9 if freq_hz > 0 else 0.0,
        'cdelt1_deg': abs(header.get('CDELT1', 0.0)),
        'cdelt2_deg': abs(header.get('CDELT2', 0.0)),
    }


def write_observation_properties(images, results_dir=RESULTS_DIR):
    """
    Save the properties of all epoch images to observation_properties.json, in time order.

    Each property is stored as a list with one entry per epoch, so the file
    lines up with the matched catalogue's epoch axis.
    """
    properties = sorted((image_properties(im) for im in images), key=lambda p: p['obs_isot'])
    table = {key: [p[key] for p in properties] for key in properties[0]}
    out_file = Path(results_dir) / 'observation_properties.json'
    with open(out_file, 'w') as f:
        json.dump(table, f, indent=4)
    msg(f"Saved observation properties for {len(properties)} epochs: {out_file}")
    return out_file


def remove_failed_epoch(image_path, results_dir=RESULTS_DIR):
    """
    Drop an epoch whose extraction failed from the bookkeeping files.

    Removes its entry from observation_properties.json and, for simulations
    whose per_epoch_offsets.csv sits next to the images, the truth row whose
    epoch_index matches the number at the end of the image file name.
    """
    remove_observation_properties_entries(Path(results_dir) / 'observation_properties.json',
                                          image_identifiers_to_remove=[Path(image_path).name],
                                          reason="PyBDSF extraction failed")

    match = re.search(r'(\d+)(?=\.fits$)', Path(image_path).name)
    csv_path = Path(image_path).parent / 'per_epoch_offsets.csv'
    if not match or not csv_path.exists():
        return

    epoch_index = int(match.group(1))
    with open(csv_path, 'r', newline='') as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if int(r['epoch_index']) != epoch_index]
    if len(kept) == len(rows):
        return

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(kept)
    msg(f"Removed epoch {epoch_index} from {csv_path.name} (PyBDSF extraction failed)")


# ============================================================================
# SOURCE EXTRACTION
# ============================================================================

def run_pybdsf(image_path, config, output_dir=CATALOG_DIR, overwrite=False, deep=False):
    """
    Run PyBDSF on one image and save its source list as a FITS catalogue.

    Uses the PYBDSF section of astkat.yaml. The deep image uses
    DEEP_SNR_THRESHOLD and DEEP_ISLAND_THRESHOLD instead. A fixed
    BOOKKEEPING.rms replaces PyBDSF's noise map (scaled by DEEP_RMS_FACTOR
    for the deep image); in a simulated project an epoch image's own NOISE
    header, the unit rms times its flux scale, takes its place.

    Parameters
    ----------
    image_path : str
    config : dict
        Parsed astkat.yaml.
    output_dir : str or Path
    overwrite : bool
        Re-run even if the catalogue exists.
    deep : bool
        The image is the deep image.

    Returns
    -------
    bool
        True on success or when an existing catalogue is kept.
    """
    pybdsf = config.get('PYBDSF', {})
    project_name = config['BOOKKEEPING'].get('project_name', 'astkat')
    image_name = Path(image_path).name

    if deep:
        output_file = Path(output_dir) / f'catalog_deep_{project_name}.fits'
    else:
        obs_isot = fits.getheader(image_path).get('DATE-OBS', 'unknown')
        output_file = Path(output_dir) / f'catalog_{obs_isot}_{project_name}.fits'
    if output_file.exists() and not overwrite:
        msg(f"Catalogue exists for {image_name}; skipped (PYBDSF.overwrite is false)")
        return True

    thresh_pix = DEEP_SNR_THRESHOLD if deep else pybdsf['snr_threshold']
    thresh_isl = DEEP_ISLAND_THRESHOLD if deep else pybdsf['thresh_isl']
    extra = dict(pybdsf.get('zargs') or {})
    rms = config['BOOKKEEPING'].get('rms', False)
    if rms is not False:
        noise = None if deep or not is_simulated(config) else fits.getheader(image_path).get('NOISE')
        extra['rms_value'] = rms * DEEP_RMS_FACTOR if deep else (float(noise) if noise is not None else rms)
        extra['rms_map'] = False

    msg(f"PyBDSF on {image_name}: thresh_pix={thresh_pix}, thresh_isl={thresh_isl}"
        + (f", extra arguments {extra}" if extra else ""))
    try:
        import bdsf
        img = bdsf.process_image(image_path,
                                 thresh_pix=thresh_pix,
                                 thresh_isl=thresh_isl,
                                 fix_to_beam=pybdsf.get('fix_to_beam', True),
                                 group_by_isl=True,
                                 flag_maxsize_fwhm=pybdsf.get('flag_maxsize_fwhm', 0.1),
                                 **extra)
        img.write_catalog(format='fits', catalog_type='srl', outfile=str(output_file), clobber=True)
    except Exception as e:
        msg(f"PyBDSF failed on {image_name}: {e}")
        return False

    msg(f"Saved catalogue: {output_file}")
    return True


def extract_sources_from_images():
    """Extract sources from every epoch image and the deep image of the project."""
    title("AstKAT source extraction")
    try:
        import bdsf  # noqa: F401  loaded here so the other steps run without PyBDSF installed
    except ImportError:
        sys.exit("PyBDSF is not installed; install it with: pip install 'astkat[extraction]' (see README)")
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    for d in (CATALOG_DIR, RESULTS_DIR):
        if not d.exists():
            raise FileNotFoundError(f"Directory {d} not found; run initialise.py first")

    epoch_images, deep_image = find_images(config)

    # All epochs must share one phase centre; later steps measure every position relative to it.
    write_phase_centre(common_phase_centre(epoch_images))

    # Per-epoch times, beams and pixel sizes, in time order to line up with the catalogue epochs.
    obs_file = write_observation_properties(epoch_images)

    overwrite = config.get('PYBDSF', {}).get('overwrite', False)
    n_ok = 0
    for i, image in enumerate(epoch_images, 1):
        msg("")
        msg(f"({i}/{len(epoch_images)}) {image}")
        if run_pybdsf(image, config, overwrite=overwrite):
            n_ok += 1
        else:
            # A failed epoch is dropped from the bookkeeping so the epoch lists stay aligned.
            remove_failed_epoch(image)

    # The deep image is extracted with its own thresholds; it becomes the matching reference.
    if deep_image:
        msg("")
        msg(f"Deep image {deep_image}")
        if run_pybdsf(deep_image, config, overwrite=overwrite, deep=True):
            n_ok += 1

    title("AstKAT extraction complete")
    msg(f"  Successful extractions: {n_ok}/{len(epoch_images) + (1 if deep_image else 0)}")
    msg(f"  Observation properties: {obs_file}")
    msg(f"  Catalogues: {CATALOG_DIR}")


if __name__ == "__main__":
    extract_sources_from_images()
