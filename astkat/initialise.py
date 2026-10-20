"""
AstKAT project initialisation.

Creates the project directories, writes astkat_files/astkat.yaml and
system_resources.json, and builds the stacked deep image, the reference
source list for cross-matching. Every image is stacked by default; -s
START,END stacks a range, -d uses an existing deep image instead and
--no-stack uses an epoch catalogue (THRESHOLDS.ref_index) as the reference.

Usage
    python initialise.py -n PROJECT -i IMAGE_DIR [-s START,END | -d DEEP_IMAGE | --no-stack] [-r RMS]
                         [--seed SEED] [--simulated]
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml

try:
    from .utils import msg, title, get_system_resources, DEFAULT_SEED, SIMULATION_TRUTH_FILE, \
        PYBDSF_SNR_THRESHOLD
    from .stacking import homogenise_and_stack_images
    from .extraction import common_phase_centre, phase_centre_string
except ImportError:
    from utils import msg, title, get_system_resources, DEFAULT_SEED, SIMULATION_TRUTH_FILE, \
        PYBDSF_SNR_THRESHOLD
    from stacking import homogenise_and_stack_images
    from extraction import common_phase_centre, phase_centre_string


PROJECT_DIRECTORIES = ('pybdsf_files', 'astkat_files', 'plots', 'results', 'logs')


# ============================================================================
# COMMAND LINE
# ============================================================================

def parse_stack_image_arg(value):
    """
    Parse --stack-image.

    'true'/'false' (or 1/0, yes/no) give a boolean; anything else, such as
    '0,10' for a range of image indices, is passed to the stacker as a string.
    """
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    return value


def parse_rms_arg(value):
    """Parse --rms: a number in Jy/beam, or 'false' to let PyBDSF measure the noise."""
    if value.lower() in ('false', '0', 'no'):
        return False
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"RMS value must be a number or 'false', got: {value}")


def setup_argument_parser():
    """Command-line options for initialise.py."""
    parser = argparse.ArgumentParser(
        description="Initialise an AstKAT project: directories, astkat.yaml and the stacked deep image"
    )
    parser.add_argument('-n', '--name', default='astkat',
                        help='Project name (default: astkat)')
    parser.add_argument('-i', '--image-dir', default='.',
                        help='Directory holding the epoch images (default: current directory)')
    parser.add_argument('-s', '--stack-image', type=parse_stack_image_arg, nargs='?', const='true', default=True,
                        help='Images to stack into the deep image: all (default, or "true") or an index range '
                             'such as "0,10"')
    parser.add_argument('-d', '--deep-image', default=False,
                        help='Path to an existing deep image to use instead of stacking')
    parser.add_argument('--no-stack', action='store_true',
                        help='Build no deep image; an epoch catalogue (THRESHOLDS.ref_index) is the reference')
    parser.add_argument('-r', '--rms', type=parse_rms_arg, default=False,
                        help='Fixed image RMS in Jy/beam, or "false" to let PyBDSF measure it (default: false)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Random seed used throughout the pipeline (default: {DEFAULT_SEED})')
    parser.add_argument('--simulated', action='store_true',
                        help=f'Mark the project as simulated, which turns off the aberration correction. '
                             f'Detected automatically when {SIMULATION_TRUTH_FILE} is in the image directory')
    return parser


# ============================================================================
# CONFIGURATION
# ============================================================================

def create_project_config(name, deep_image, stack_image, image_dir, rms=False,
                          seed=DEFAULT_SEED, simulated=False):
    """
    Default astkat.yaml contents for a new project.

    Parameters
    ----------
    name : str
        Project name.
    deep_image : str or False
        Path to an existing deep image, used when not stacking.
    stack_image : bool or str
        Stacking selection (see parse_stack_image_arg).
    image_dir : str
        Directory holding the epoch images.
    rms : float or False
        Fixed image RMS in Jy/beam, or False for PyBDSF's own estimate.
    seed : int
        Random seed for simulation, MCMC fitting and posterior draws.
    simulated : bool
        Simulated project; turns off the aberration correction.

    Returns
    -------
    dict
    """
    image_dir = str(Path(image_dir).absolute())

    return {
        'BOOKKEEPING': {
            'image_dir': image_dir,                 # directory holding the epoch images
            'stacked_image': stack_image,           # images to stack: True (all), "start,end", or False
            'deep_image': f"{image_dir}/{name}_stacked.fits" if stack_image is not False else deep_image,
            'project_name': name,
            'phase_ctr': "0h00m00s 0d00m00s",       # set from the image headers by extraction.py
            'rms': rms,                             # fixed image noise in Jy/beam, or False for PyBDSF's map
            'seed': seed,                           # random seed for stacking, sampling and posterior draws
            'simulated': bool(simulated),           # simulated data contain no aberration
            'memory_safety_fraction': 0.5,          # share of free memory the stacker may use
        },
        'PYBDSF': {
            'snr_threshold': PYBDSF_SNR_THRESHOLD,  # detection threshold (thresh_pix), in sigma; below MCMC.flux_censor_snr
            'thresh_isl': 3.0,                      # island boundary threshold, in sigma
            'fix_to_beam': True,                    # fit Gaussians with the beam's shape
            'flag_maxsize_fwhm': 0.1,
            'overwrite': False,                     # re-run PyBDSF when a catalogue already exists
            'zargs': {'quiet': True},               # extra keyword arguments passed to bdsf.process_image
        },
        'THRESHOLDS': {
            'flux_threshold': 0.15,                 # point-like if the median |total flux / peak flux - 1| is below this
            'size_threshold': 0.15,                 # ... and the median |size / beam - 1| on both axes is below this
            'single_fraction': 0.5,                 # least fraction of a source's detections with PyBDSF S_Code S
            'ambiguous_fraction': 0.1,              # most epochs (as a share of those with a match) a source may be
                                                    # ... ambiguous or blended (S_Code A); more is a blended pair
            'snr_threshold': 7.0,                   # least SNR of a deep-catalogue source; the deep sources are the
                                                    # ... reference list, so this keeps spurious ones out
            'match_threshold': 0.75,                # cross-matching ellipse, semi-axes this times each epoch's BMAJ
                                                    # ... and BMIN (0.75: 1.5 times the beam)
            'ref_index': 0,                         # reference epoch when there is no deep image
            'epoch_min': 5,                         # a source must be detected in at least this many epochs
            'epoch_fraction': 0.0,                  # ... and in at least this share of all epochs
            'variability_threshold': 3.0,           # sources more variable than this, in sigma, are dropped
            'frac_flux_sys': 0.15,                  # flux-scale uncertainty used in the variability test
            'min_epochs_per_freq_group': 5,         # same-band epochs needed to test a source's variability
            'min_sources_per_epoch': 10,            # epochs with fewer matched sources are dropped
        },
        'MCMC': {
            # Sampling
            'mcmc_samples': 1000,                   # posterior draws per chain
            'mcmc_warmup_samples': 1000,            # tuning steps per chain
            'mcmc_acceptance_rate': 0.95,           # NUTS target acceptance (0.8 to 0.99)
            'mcmc_chains': 4,
            'mcmc_cores': None,                     # None uses all but one CPU
            'nuts_sampler': 'nutpie',               # 'nutpie' (fastest on CPU), 'pymc', or 'numpyro'
            'posterior_predictive_draws': 1000,     # draws used for the diagnostic plots

            # Data selection
            'max_phase_offset_deg': 0.45,           # sources further from the phase centre are not fitted
            'enable_frequency_filtering': True,     # fit only the dominant frequency band
            'frequency_match_tolerance': 0.1,       # fractional width of that band
            'enable_rms_filtering': False,          # drop epochs with anomalous noise
            'rms_outlier_sigma_threshold': 3.0,     # ... beyond this many robust sigma from the median

            # Differential aberration: removes the annual and diurnal scale about the phase centre
            # (up to ~1e-4, ~160 mas at 0.45 deg) before fitting. Always off for simulated data.
            'correct_differential_aberration': not simulated,
            'observatory_location': {               # MeerKAT; only sets the diurnal term (<= 1.6e-6 in scale)
                'lat_deg': -30.711056,
                'lon_deg': 21.443889,
                'height_m': 1035.0,
            },

            # Noise model: sigma^2 = (A / SNR)^2 * beam variance + B^2, plus an outlier component
            'fit_translations': True,               # False absorbs epoch shifts into the noise (very conservative)
            'enable_systematic_floor': True,        # fit the floor B
            'systematic_floor_as_fraction': True,   # True: B as a fraction of the beam; False: B in arcsec
            'snr_term': 'fixed',                    # SNR term of the position error, A the error at SNR_0:
                                                    # ... fixed   A SNR_0 / SNR, SNR_0 = 1 (textbook)
                                                    # ... power   A (SNR / SNR_0)^-beta
                                                    # ... broken  A (SNR_0 / SNR) [(1 + SNR_b / SNR) / (1 + SNR_b / SNR_0)]^gamma,
                                                    #             slope -1 above SNR_b, -1 - gamma below
            'snr_pivot': 10.0,                      # SNR_0 for power and broken
            'fit_beam_shape_exponent': True,        # fit alpha (1 = noise shaped like the beam, 0 = isotropic)
            'enable_outlier_mixture': True,         # model a fraction of measurements as outliers
            'error_inflation': True,                # False: none. True: after the fit, scale every error by the
                                                    # ... smallest factor that puts all binned residuals at or
                                                    # ... below the model. A number >= 1: that factor

            # Source peak flux densities: fitted, with faint detections and non-detections censored, so SNRs
            # near the threshold are not biased high
            'snr_from_median_flux': False,          # True: SNR from the median detected flux density, not fitted
            'flux_censor_snr': 5.0,                 # peaks below this many rms count as censored; at least 0.5
                                                    # ... above PYBDSF.snr_threshold, so detection there is complete
            'detection_floor': 0.01,                # chance a source is missed for reasons other than its flux density
            'detection_censor_snr': round(PYBDSF_SNR_THRESHOLD - 0.2, 2),
                                                    # detection as a cut on the fitted peak, in rms: PyBDSF's
                                                    # ... threshold less the brightest pixel's ~0.2 rms excess over
                                                    # ... the fitted peak; detections below flux_censor_snr lie between
                                                    # ... it and flux_censor_snr, non-detections below it
            'fit_flux_systematic': True,            # fit each epoch's flux scale g_e (a gain error shared by its
                                                    # ... sources), spread f_sys; False: every g_e = 0
            'fit_flux_fractional': True,            # fit a fractional peak error f_frac per detection, not shared
                                                    # ... (primary beam, direction-dependent errors), added in
                                                    # ... quadrature to k rms; False: f_frac = 0
            'single_gaussian_only': True,           # fit only PyBDSF S_Code S detections; C and M (blended,
                                                    # ... several Gaussians) are left out, not counted as missed
        },
        'PRIORS': {
            # Each bounded parameter is [min, max]; the comment gives its distribution.
            'A': [0.0, 2.0],                        # SNR term in beam units, uniform
            'beta': [0.5, 1.5],                     # power: SNR exponent, uniform, starts at 1
            'snr_break': [5.0, 100.0],              # broken: break SNR_b, log-uniform
            'snr_gamma': [-1.0, 0.5],               # broken: gamma, uniform, starts at 0 (the textbook 1 / SNR);
                                                    # ... -1 is flat below SNR_b, so errors never rise with SNR
            'B': [0.0, 0.1],                        # floor as a fraction of beam sigma, uniform
            'B_absolute': [0.0, None],              # floor in arcsec when not fractional, uniform; None = 0.1 FWHM
            'alpha': [-1.0, 2.0],                   # beam-shape exponent, uniform, starts at 1
            'outlier_fraction': [0.0, 0.05],        # Beta(a, b) stretched over [min, max]; max < 0.5 stops swaps
            'outlier_fraction_beta': [2.0, 8.0],    # ... [a, b]: zero density at 0, mode 0.006, mean 0.01
            'outlier_scale': [4.0, 20.0],           # outlier width over inlier width, log-uniform
            'peak_noise': 0.5,                      # noise k on a fitted peak over the rms: log-normal sigma about
                                                    # ... 1 (rms alone); starts at the k matching.py measures
            'flux_systematic': 0.05,                # spread f_sys of the epoch flux scales: half-normal scale,
                                                    # ... starts at 0.02
            'flux_fractional': 0.03,                # per-detection fractional peak error f_frac: half-normal scale,
                                                    # ... starts at 0.01
            'source_flux': 1.0,                     # true peak flux density: log-normal sigma in ln S about its
                                                    # ... median detected value
            'sigma_tau': 0.5,                       # epoch translations: half-normal scale of their spread, arcsec
            'position_scale': 10.0,                 # true positions: normal about the median, this many errors on
                                                    # ... the median wide, at least one pixel
        },
    }


# ============================================================================
# PROJECT SETUP
# ============================================================================

def create_project_structure(base_dir='.'):
    """Create the project directories and return them by name."""
    directories = {name: Path(base_dir) / name for name in PROJECT_DIRECTORIES}
    for path in directories.values():
        path.mkdir(exist_ok=True)
        msg(f"Created directory: {path}")
    return directories


def build_stacked_image(config, system_resources, stacked_image_path):
    """Build the stacked deep image unless it already exists."""
    if stacked_image_path.exists():
        msg(f"Stacked image already exists, skipping stacking: {stacked_image_path}")
        return

    msg("Creating stacked image...")
    homogenise_and_stack_images(config, system_resources)
    if not stacked_image_path.exists():
        raise FileNotFoundError(f"Stacking finished but no image was written to {stacked_image_path}")
    msg("Stacked image created")


def initialise_project():
    """Set up a new AstKAT project from the command line."""
    args = setup_argument_parser().parse_args()
    title(f"AstKAT project initialisation: {args.name}")

    # Check the inputs before anything is written.
    image_path = Path(args.image_dir)
    if not image_path.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    if args.deep_image is not False:
        if not Path(args.deep_image).exists():
            raise FileNotFoundError(f"Deep image file not found: {args.deep_image}")
        msg(f"Deep image verified: {args.deep_image}")

    # Simulators leave a truth file next to the images; simulated data skip the aberration correction.
    truth_file = image_path / SIMULATION_TRUTH_FILE
    simulated = args.simulated or truth_file.exists()
    if simulated:
        msg("Simulated project: differential aberration correction turned off")
    mode = None
    if truth_file.exists():
        with open(truth_file, 'r') as f:
            mode = json.load(f)['simulation_parameters']['mode']

    # An existing deep image, or --no-stack, replaces the stack. A catalogue simulation is never matched,
    # so it needs no reference image.
    stack = (False if args.no_stack or args.deep_image is not False or mode == 'catalogue'
             else args.stack_image)

    # Project directories and astkat.yaml, the configuration every later step reads.
    directories = create_project_structure()
    config = create_project_config(args.name, args.deep_image, stack, str(image_path),
                                   args.rms, args.seed, simulated)

    # Flat and catalogue simulations define their SNRs for a unit rms.
    if mode in ('flat', 'catalogue'):
        config['BOOKKEEPING']['rms'] = 1.0
        msg(f"{mode.capitalize()} simulation: BOOKKEEPING.rms set to 1")
    if mode == 'catalogue':
        # The images share one phase centre, which extraction.py would otherwise record.
        images = sorted(p for p in image_path.glob('*.fits') if not p.name.endswith('_stacked.fits'))
        config['BOOKKEEPING']['phase_ctr'] = phase_centre_string(common_phase_centre(images))

    config_file = directories['astkat_files'] / "astkat.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=None, sort_keys=False, indent=2)
    msg(f"Configuration saved to: {config_file}")

    # CPU and memory limits, saved once and reused by the stacker.
    system_resources = get_system_resources(config['BOOKKEEPING']['memory_safety_fraction'])
    system_file = directories['astkat_files'] / "system_resources.json"
    with open(system_file, 'w') as f:
        json.dump(system_resources, f, indent=2)
    msg(f"System resources saved to: {system_file}")

    # A catalogue simulation brings its own matched catalogue and observation properties.
    if mode == 'catalogue':
        shutil.copy(image_path / 'matched_catalog.npz', directories['results'] / f"matched_catalog_{args.name}.npz")
        shutil.copy(image_path / 'observation_properties.json', directories['results'] / 'observation_properties.json')
        msg(f"Catalogue simulation: matched catalogue and observation properties copied to {directories['results']}; "
            f"run fit.py next")

    # The stacked deep image becomes the reference catalogue for cross-matching.
    if stack is not False:
        build_stacked_image(config, system_resources, image_path / f"{args.name}_stacked.fits")

    title("AstKAT project initialisation complete")
    msg(f"  Project name: {args.name}")
    msg(f"  Image directory: {image_path.absolute()}")
    msg(f"  Stacked image mode: {stack}")
    msg(f"  Deep image: {config['BOOKKEEPING']['deep_image']}")
    msg(f"  Simulated: {simulated}")
    msg(f"  Configuration file: {config_file}")
    msg(f"  System resources file: {system_file}")


if __name__ == "__main__":
    initialise_project()
