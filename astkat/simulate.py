#!/usr/bin/env python3
"""
AstKAT image simulation.

Makes a set of epoch images of one field with known source positions, known
astrometric errors and known per-epoch translations, for testing the
pipeline end to end.

Modes
    noise   realistic images: noise whose autocorrelation is the beam, the
            systematic floor B injected, and the SNR-dependent error left to
            PyBDSF's own fitting
    flat    a flat background with every error injected directly:
            sigma = sqrt(s^2 + B^2) in beam units, with
            s = A SNR^-beta [(1 + SNR_b / SNR) / (1 + SNR_b)]^gamma (A at SNR 1),
            an optional outlier fraction, and peaks p = S + k N(0, 1) at
            unit rms, so PyBDSF detects exactly the peaks at or above its
            threshold
    catalogue
            the joint model's own data, written straight to the matched
            catalogue and observation_properties.json so fit.py can run
            without extraction or matching: positions, outliers and peaks
            as in flat mode, and a detection only where p >=
            CATALOGUE_DETECTION_SNR; the flat images are written too
    target  a moving target in an existing project's epochs, written as a
            catalogue for target.py; needs fit.py and offsets.py outputs and
            runs from the project directory

    In noise, flat and catalogue modes each epoch's image (sources and noise)
    is multiplied by 1 + g_e, a flux-scale error shared by every source, with
    g_e normal of sigma f_sys (--flux-sys), and each source's peak by
    1 + f_frac N(0, 1), an error of its own (--flux-frac).

Usage
    python simulate.py --mode noise -b 0.01
    python simulate.py --mode flat -a 1.0 -b 0.01
    python simulate.py --mode catalogue -a 1.0 -b 0.01 -k 1.3
    python simulate.py --mode target -i 20 --v0 3 --decel 0.04

Writes, in the output directory
    {prefix}NNNN.fits                 one image per epoch
    true_source_positions.csv/.reg    true source positions (and *_pixels.reg)
    expected_values.json              injected parameters and per-epoch truth
    per_epoch_offsets.csv             injected epoch translations, RA and Dec in arcsec, and flux scales g_e
    catalogue mode also
    matched_catalog.npz               the matched catalogue, columns as in utils.CATALOG_COLUMNS
    observation_properties.json       per-epoch image properties, as extraction.py writes them
    target mode instead
    target_trajectory.txt             the target catalogue, input to target.py
    target_trajectory_truth.csv       true track and injected translation per epoch

Writes, in plots/ of the working directory
    noise mode with correlated noise
    simulate_noise_correlation.pdf/.png   the simulated noise autocorrelation against the beam
"""

import argparse
import csv
import json
import secrets
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.table import QTable
from astropy.time import Time
from astropy.wcs import WCS
from photutils.datasets import make_model_image
from scipy.ndimage import label, map_coordinates
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

try:
    from .utils import msg, title, DEFAULT_SEED, SIMULATION_TRUTH_FILE, CATALOG_COLUMNS, S_CODES, \
        PYBDSF_SNR_THRESHOLD
    from .extraction import image_properties
    from .plot import simulation_noise_figure
except ImportError:
    from utils import msg, title, DEFAULT_SEED, SIMULATION_TRUTH_FILE, CATALOG_COLUMNS, S_CODES, \
        PYBDSF_SNR_THRESHOLD
    from extraction import image_properties
    from plot import simulation_noise_figure


# Per-mode settings: beam size limits, file prefix and default output directory.
MODES = {
    'noise': {'min_beam_pixels': 4.0, 'max_beam_ratio': 3.0, 'prefix': 'sim_img_', 'output_dir': 'sim_images'},
    'flat': {'min_beam_pixels': 5.0, 'max_beam_ratio': 5.0, 'prefix': 'sim_flat_img_',
             'output_dir': 'sim_images_flat'},
    'catalogue': {'min_beam_pixels': 5.0, 'max_beam_ratio': 5.0, 'prefix': 'sim_cat_img_',
                  'output_dir': 'sim_catalogue'},
    'target': {'output_dir': 'sim_target'},
}

PIXEL_SCALE = 1.0                # arcsec per pixel
IMAGE_SIZE = 2560                # pixels per side
N_IMAGES = 100
N_SOURCES = 500
EDGE_BUFFER = 50                 # pixels kept free of sources at the image edge
MIN_BEAM_RATIO = 1.15            # least BMAJ / BMIN; PyBDSF's fix_to_beam fits peaks ~5% low for near-circular beams

SNR_POWERLAW = -1.5              # source SNR distribution, P(SNR) ~ SNR^-1.5
MIN_SNR, MAX_SNR = 4.0, 1000.0
MIN_RMS, MAX_RMS = 1.0, 100.0    # noise mode: range of the field's RMS when not given
RMS_VARIATION = 0.05             # noise mode: fractional epoch-to-epoch RMS scatter
FLAT_BACKGROUND = 1e-5           # flat mode: background level ...
FLAT_BACKGROUND_SIGMA = 1e-3     # ... and a tiny pixel scatter, far below the unit RMS of the SNR scale
CATALOGUE_DETECTION_SNR = PYBDSF_SNR_THRESHOLD   # catalogue mode: a peak at or above this many rms is detected
CATALOGUE_FILE = 'matched_catalog.npz'
OBSERVATION_FILE = 'observation_properties.json'

EPOCH_OFFSET_SIGMA = 0.2         # injected epoch translation per axis, arcsec
START_DATE = '1994-09-20T03:00:00'
CADENCE_DAYS = 1.0               # days between epochs
FREQ_HZ = 1.284e9                # MeerKAT L band
BANDWIDTH_HZ = 856e6
BAD_FREQ_FACTOR = 10.0           # frequency of the deliberately wrong epochs (--n-bad-freq-epochs)
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

NOISE_CHECK_SIZE = 512           # noise mode: central pixels per side whose autocorrelation is checked
NOISE_CHECK_MAX_LAG = 3.0        # largest lag checked, in units of BMAJ
NOISE_CHECK_LOBE = 0.2           # the connected region above this rho around zero lag is fitted


# ============================================================================
# FIELD AND BEAMS
# ============================================================================

def random_pointing(rng):
    """Random field centre (RA, Dec) in degrees, uniform on the sky away from the poles."""
    return rng.uniform(0.0, 360.0), np.degrees(np.arcsin(rng.uniform(-0.9, 0.9)))


def make_beam(bmaj, bmin, bpa):
    """Beam dict from FWHM axes in pixels and position angle in degrees (East of North)."""
    return {'bmaj_pixels': bmaj, 'bmin_pixels': bmin, 'bpa_deg': bpa,
            'bmaj_sigma': bmaj * FWHM_TO_SIGMA, 'bmin_sigma': bmin * FWHM_TO_SIGMA}


def reference_beam(rng, min_pixels, max_ratio):
    """Random beam with an axis ratio log-uniform in [MIN_BEAM_RATIO, max_ratio], so elongated beams are rare."""
    bmin = min_pixels + 2.0 * rng.random()
    ratio = np.exp(rng.uniform(np.log(MIN_BEAM_RATIO), np.log(max_ratio)))
    return make_beam(bmin * ratio, bmin, rng.uniform(-90.0, 90.0))


def perturbed_beam(rng, reference, min_pixels):
    """
    Beam with each axis scaled by 0.66-1.33 of the reference and a new random position angle.

    The major axis is stretched where needed so BMAJ / BMIN is at least MIN_BEAM_RATIO.
    """
    axes = sorted([max(reference['bmaj_pixels'] * rng.uniform(0.66, 1.33), min_pixels),
                   max(reference['bmin_pixels'] * rng.uniform(0.66, 1.33), min_pixels)])
    return make_beam(max(axes[1], MIN_BEAM_RATIO * axes[0]), axes[0], rng.uniform(-90.0, 90.0))


def epoch_beam_list(rng, n_epochs, mode, same_beam=False, circular=False):
    """One beam per epoch: the reference beam first, then perturbations of it unless same_beam."""
    settings = MODES[mode]
    ref = reference_beam(rng, settings['min_beam_pixels'], settings['max_beam_ratio'])
    beams = [ref] + [ref if same_beam else perturbed_beam(rng, ref, settings['min_beam_pixels'])
                     for _ in range(n_epochs - 1)]
    if circular:
        beams = [make_beam(b['bmaj_pixels'], b['bmaj_pixels'], b['bpa_deg']) for b in beams]
    return beams


def truncated_powerlaw(rng, alpha, xmin, xmax, size):
    """Samples from P(x) ~ x^alpha on [xmin, xmax] by inverse-transform sampling."""
    u_ = rng.uniform(0.0, 1.0, size)
    if abs(alpha + 1.0) < 1e-8:
        return xmin * (xmax / xmin) ** u_
    a1 = alpha + 1.0
    return (u_ * (xmax ** a1 - xmin ** a1) + xmin ** a1) ** (1.0 / a1)


# ============================================================================
# POSITIONS AND ERRORS
# ============================================================================

def pixels_to_radec(x, y, centre, image_size):
    """RA and Dec in degrees of 0-indexed pixel positions, with the images' SIN projection."""
    header = fits.Header()
    header.update({'NAXIS': 2, 'CTYPE1': 'RA---SIN', 'CTYPE2': 'DEC--SIN',
                   'CRPIX1': image_size / 2 + 1, 'CRPIX2': image_size / 2 + 1,
                   'CRVAL1': float(centre[0]), 'CRVAL2': float(centre[1]),
                   'CDELT1': -PIXEL_SCALE / 3600.0, 'CDELT2': PIXEL_SCALE / 3600.0,
                   'CUNIT1': 'deg', 'CUNIT2': 'deg'})
    return WCS(header).wcs_pix2world(np.asarray(x, float) + 1.0, np.asarray(y, float) + 1.0, 1)


def error_sigma(snr, A, beta, snr_break, gamma, beam, B, b_absolute):
    """
    Astrometric error along the beam's major and minor axes, each in units of that axis's beam sigma.

    The SNR term is A SNR^-beta [(1 + snr_break / SNR) / (1 + snr_break)]^gamma,
    A at SNR 1; beta = 1 and gamma = 0 give the textbook A / SNR. The floor B
    adds in quadrature: a fraction of each axis's beam sigma, or with
    b_absolute B arcsec on both axes, B over that axis's beam sigma here.

    Returns
    -------
    sigma_major, sigma_minor : ndarray
    """
    s = A * snr ** -beta * ((1.0 + snr_break / snr) / (1.0 + snr_break)) ** gamma
    if b_absolute:
        floor_major = B / (beam['bmaj_sigma'] * PIXEL_SCALE)
        floor_minor = B / (beam['bmin_sigma'] * PIXEL_SCALE)
    else:
        floor_major = floor_minor = B
    return np.sqrt(s ** 2 + floor_major ** 2), np.sqrt(s ** 2 + floor_minor ** 2)


def beam_offsets(rng, beam, sigma_major, sigma_minor):
    """
    Random position offsets in pixels for errors given in beam units.

    Each source gets a normal offset of width sigma_major along the beam's
    major axis and sigma_minor along its minor axis, each scaled by that axis's
    beam sigma and rotated by the position angle (major axis along
    (sin BPA, cos BPA) in RA, Dec). Pixel x increases to the West.
    """
    major = rng.normal(0.0, sigma_major) * beam['bmaj_sigma'] * PIXEL_SCALE
    minor = rng.normal(0.0, sigma_minor) * beam['bmin_sigma'] * PIXEL_SCALE
    pa = np.radians(beam['bpa_deg'])
    east = minor * np.cos(pa) + major * np.sin(pa)
    north = major * np.cos(pa) - minor * np.sin(pa)
    return -east / PIXEL_SCALE, north / PIXEL_SCALE


def write_true_positions(output_dir, positions, fluxes, centre, image_size):
    """True source positions to true_source_positions.csv and DS9 region files (sky and pixel)."""
    x, y = positions
    ra, dec = pixels_to_radec(x, y, centre, image_size)
    with open(output_dir / 'true_source_positions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_id', 'ra_deg', 'dec_deg', 'x_pixels', 'y_pixels', 'flux_density'])
        for i in range(len(x)):
            writer.writerow([i] + [f"{v:.17g}" for v in (ra[i], dec[i], x[i], y[i], fluxes[i])])

    style = 'dashlist=8 3 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1'
    coords = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
    with open(output_dir / 'true_source_positions.reg', 'w') as f:
        f.write(f"# Region file format: DS9 version 4.1\nglobal color=green width=1 {style}\nfk5\n")
        for i, c in enumerate(coords):
            f.write(f"circle({c.ra.to_string(unit=u.hourangle, sep=':', precision=6, pad=True)},"
                    f"{c.dec.to_string(sep=':', precision=5, pad=True, alwayssign=True)},5\") # text={{S{i}}}\n")
    with open(output_dir / 'true_source_positions_pixels.reg', 'w') as f:
        f.write(f"# Region file format: DS9 version 4.1\nglobal color=red width=2 {style}\nimage\n")
        for i in range(len(x)):
            f.write(f"circle({x[i] + 1:.9f},{y[i] + 1:.9f},5) # text={{P{i}}}\n")
    msg(f"True source positions saved to {output_dir / 'true_source_positions.csv'} (with DS9 regions)")


# ============================================================================
# IMAGES
# ============================================================================

def fits_header(beam, centre, image_size, epoch, frequency_hz, mode, cadence_days, rms=None):
    """4-D radio FITS header (RA, Dec, frequency, Stokes) for one simulated epoch."""
    scale = PIXEL_SCALE / 3600.0
    header = fits.Header()
    header.update({
        'SIMPLE': True, 'BITPIX': -32, 'NAXIS': 4, 'NAXIS1': image_size, 'NAXIS2': image_size,
        'NAXIS3': 1, 'NAXIS4': 1, 'BSCALE': 1.0, 'BZERO': 0.0, 'BUNIT': 'Jy/beam',
        'BMAJ': beam['bmaj_pixels'] * scale, 'BMIN': beam['bmin_pixels'] * scale, 'BPA': beam['bpa_deg'],
        'CTYPE1': 'RA---SIN', 'CRPIX1': image_size / 2 + 1, 'CRVAL1': centre[0], 'CDELT1': -scale, 'CUNIT1': 'deg',
        'CTYPE2': 'DEC--SIN', 'CRPIX2': image_size / 2 + 1, 'CRVAL2': centre[1], 'CDELT2': scale, 'CUNIT2': 'deg',
        'CTYPE3': 'FREQ', 'CRPIX3': 1, 'CRVAL3': frequency_hz, 'CDELT3': BANDWIDTH_HZ, 'CUNIT3': 'Hz',
        'CTYPE4': 'STOKES', 'CRPIX4': 1, 'CRVAL4': 1.0, 'CDELT4': 1.0,
        'EQUINOX': 2000.0, 'SPECSYS': 'TOPOCENT',
        'DATE-OBS': Time(Time(START_DATE, format='isot').mjd + epoch * cadence_days, format='mjd').isot,
        'OBSERVER': 'AstKAT simulation', 'TELESCOP': 'MeerKAT', 'INSTRUME': f'Simulated ({mode})',
    })
    if rms is not None:
        header['NOISE'] = (rms, 'RMS noise level')
    return header


def correlated_noise(rng, image_size, beam, rms, correlated=True):
    """
    Gaussian noise of the given RMS whose autocorrelation is the beam, or white if correlated is False.

    White noise convolved with a Gaussian kernel of covariance K has
    autocorrelation of covariance 2 K, so the kernel has the beam's
    orientation and each axis's sigma divided by sqrt(2). An interferometer's
    thermal noise has the PSF as its autocorrelation, here the Gaussian beam.
    """
    noise = rng.standard_normal((image_size, image_size))
    if correlated:
        size = max(51, int(6 * max(beam['bmaj_sigma'], beam['bmin_sigma']))) | 1
        kernel = Gaussian2DKernel(x_stddev=beam['bmaj_sigma'] / np.sqrt(2.0),
                                  y_stddev=beam['bmin_sigma'] / np.sqrt(2.0),
                                  theta=np.radians(beam['bpa_deg'] + 90), x_size=size, y_size=size,
                                  mode='center').array
        noise = fftconvolve(noise, kernel, mode='same')
    return noise / np.sqrt(np.mean(noise ** 2)) * rms


def noise_autocorrelation(field, max_lag_pix):
    """
    Unbiased autocorrelation coefficient of a mean-subtracted square field.

    Zero-padding to twice the field size removes circular wrap-around, and
    dividing by the number of pixel pairs at each lag removes the taper of
    the raw sum toward large lags.

    Parameters
    ----------
    field : ndarray, shape (n, n)
        Pixel values.
    max_lag_pix : int
        Largest lag L kept along each axis, in pixels.

    Returns
    -------
    ndarray, shape (2 L + 1, 2 L + 1)
        Correlation coefficient indexed [dy, dx], 1 at zero lag in the centre.
    """
    n = field.shape[0]
    residual = field - field.mean()
    shape = (2 * n, 2 * n)
    acf = np.fft.fftshift(np.fft.irfft2(np.abs(np.fft.rfft2(residual, s=shape)) ** 2, s=shape))
    pairs = np.fft.fftshift(np.fft.irfft2(np.abs(np.fft.rfft2(np.ones_like(residual), s=shape)) ** 2, s=shape))
    # Zero lag sits at index n after the shift
    window = slice(n - max_lag_pix, n + max_lag_pix + 1)
    acf = acf[window, window] / np.round(pairs[window, window])
    return acf / acf[max_lag_pix, max_lag_pix]


def pixel_covariance(fwhm_major, fwhm_minor, pa_deg):
    """
    Covariance of an elliptical Gaussian in pixel coordinates (x West, y North).

    Parameters
    ----------
    fwhm_major, fwhm_minor : float
        FWHM along each axis, pixels.
    pa_deg : float
        Position angle of the major axis, degrees East of North.

    Returns
    -------
    ndarray, shape (2, 2)
        Covariance in pixels^2.
    """
    t = np.radians(pa_deg + 90.0)
    u_major, u_minor = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    return ((fwhm_major * FWHM_TO_SIGMA) ** 2 * np.outer(u_major, u_major)
            + (fwhm_minor * FWHM_TO_SIGMA) ** 2 * np.outer(u_minor, u_minor))


def gaussian_map(dx, dy, cov):
    """Unit-peak Gaussian exp(-1/2 d^T cov^-1 d) at pixel offsets (dx, dy), any matching shape."""
    inv = np.linalg.inv(cov)
    return np.exp(-0.5 * (inv[0, 0] * dx ** 2 + 2.0 * inv[0, 1] * dx * dy + inv[1, 1] * dy ** 2))


def noise_correlation_check(noise, beam, keep_maps=False):
    """
    Autocorrelation of the centre of an epoch's noise field, fitted and compared with its beam.

    The autocorrelation of the central NOISE_CHECK_SIZE pixels is computed out
    to NOISE_CHECK_MAX_LAG BMAJ, and an elliptical Gaussian is fitted to its
    central lobe, the connected region above NOISE_CHECK_LOBE containing zero
    lag, starting from the beam.

    Parameters
    ----------
    noise : ndarray, shape (n, n)
        The epoch's noise field, before sources are added.
    beam : dict
        The epoch's beam, from make_beam.
    keep_maps : bool
        Also return the maps and beam-axis cuts that plot.simulation_noise_figure draws.

    Returns
    -------
    dict
        fwhm_pixels : ndarray, shape (2,)
            Fitted noise correlation FWHM, major and minor.
        pa_deg : float
            Its position angle, degrees East of North, in [-90, 90).
        ratio : ndarray, shape (2,)
            fwhm_pixels over the beam FWHM, major and minor; 1 when the
            autocorrelation is the beam.
        With keep_maps, also rho, beam_rho and self_rho (2 L + 1, 2 L + 1),
        the measured map and the beam and beam self-convolution expectations;
        d_ra and d_dec, each lag's offset in arcsec; cuts, for 'major' and
        'minor' the offsets in arcsec and rho measured, for the beam and for
        the self-convolution; pixel_scale.
    """
    size = min(NOISE_CHECK_SIZE, noise.shape[0])
    start = (noise.shape[0] - size) // 2
    field = noise[start:start + size, start:start + size]
    max_lag = min(int(np.ceil(NOISE_CHECK_MAX_LAG * beam['bmaj_pixels'])), size // 4)
    rho = noise_autocorrelation(field, max_lag)
    lags = np.arange(-max_lag, max_lag + 1)
    dx, dy = np.meshgrid(lags, lags)

    labels, _ = label(rho > NOISE_CHECK_LOBE)
    lobe = labels == labels[max_lag, max_lag]

    def model(coords, amplitude, fwhm_a, fwhm_b, pa):
        return amplitude * gaussian_map(coords[0], coords[1], pixel_covariance(abs(fwhm_a), abs(fwhm_b), pa))

    p0 = [1.0, beam['bmaj_pixels'], beam['bmin_pixels'], beam['bpa_deg']]
    (_, fwhm_a, fwhm_b, pa), _ = curve_fit(model, (dx[lobe], dy[lobe]), rho[lobe], p0=p0)
    # Major axis first, its angle in [-90, 90) as the beams are drawn
    fwhm_a, fwhm_b = abs(fwhm_a), abs(fwhm_b)
    if fwhm_b > fwhm_a:
        fwhm_a, fwhm_b, pa = fwhm_b, fwhm_a, pa + 90.0
    fwhm = np.array([fwhm_a, fwhm_b])
    result = {'fwhm_pixels': fwhm, 'pa_deg': ((pa + 90.0) % 180.0) - 90.0,
              'ratio': fwhm / np.array([beam['bmaj_pixels'], beam['bmin_pixels']])}

    if keep_maps:
        beam_cov = pixel_covariance(beam['bmaj_pixels'], beam['bmin_pixels'], beam['bpa_deg'])
        t = np.radians(beam['bpa_deg'] + 90.0)
        steps = np.arange(-max_lag, max_lag + 1, dtype=float)
        cuts = {}
        for name, angle, sigma in (('major', t, beam['bmaj_sigma']), ('minor', t + 0.5 * np.pi, beam['bmin_sigma'])):
            # Bilinear samples along the axis through zero lag, rho indexed [dy, dx]
            measured = map_coordinates(rho, [max_lag + steps * np.sin(angle), max_lag + steps * np.cos(angle)],
                                       order=1)
            cuts[name] = {'offset': steps * PIXEL_SCALE, 'measured': measured,
                          'beam': np.exp(-0.5 * (steps / sigma) ** 2), 'self': np.exp(-0.25 * (steps / sigma) ** 2)}
        result.update(rho=rho, beam_rho=gaussian_map(dx, dy, beam_cov), self_rho=gaussian_map(dx, dy, 2.0 * beam_cov),
                      d_ra=-dx * PIXEL_SCALE, d_dec=dy * PIXEL_SCALE, cuts=cuts, pixel_scale=PIXEL_SCALE)
    return result


def render_sources(x, y, amplitudes, beam, image_size):
    """Image of beam-shaped point sources at pixel positions (x, y)."""
    table = QTable({'amplitude': amplitudes, 'x_mean': x, 'y_mean': y,
                    'x_stddev': np.full(len(x), beam['bmaj_sigma']),
                    'y_stddev': np.full(len(x), beam['bmin_sigma']),
                    'theta': np.full(len(x), np.radians(beam['bpa_deg'] + 90))})
    return make_model_image((image_size, image_size), Gaussian2D(), table, x_name='x_mean', y_name='y_mean')


# ============================================================================
# TRUTH FILES
# ============================================================================

def save_truth(output_dir, params, epochs):
    """
    Injected parameters and per-epoch truth to expected_values.json and per_epoch_offsets.csv.

    The epoch translations are RA and Dec in arcsec, in epoch order;
    later pipeline steps drop rows for any epoch they drop.
    """
    truth = {'simulation_parameters': params, 'pointing_center': params['pointing_center'], 'epoch_data': epochs}
    with open(output_dir / SIMULATION_TRUTH_FILE, 'w') as f:
        json.dump(truth, f, indent=2, default=float)
    with open(output_dir / 'per_epoch_offsets.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch_index', 'global_ra_offset_arcsec', 'global_dec_offset_arcsec', 'flux_scale'])
        for e in epochs:
            writer.writerow([e['epoch_index'], e['global_x_offset_arcsec'], e['global_y_offset_arcsec'],
                             e['flux_scale']])
    msg(f"Truth saved to {output_dir / SIMULATION_TRUTH_FILE} and {output_dir / 'per_epoch_offsets.csv'}")


# ============================================================================
# SIMULATION
# ============================================================================

def run_simulation(args):
    """Simulate every epoch of one field and write the images and truth files."""
    rng = np.random.default_rng(args.seed)
    settings = MODES[args.mode]
    output_dir = Path(args.output_dir or settings['output_dir'])
    output_dir.mkdir(exist_ok=True)
    # Catalogue mode injects every error as flat mode does, on the same unit-rms SNR scale.
    catalogue = args.mode == 'catalogue'
    flat = args.mode in ('flat', 'catalogue')
    A = args.a_parameter if flat else 0.0
    n_total = args.n_images + args.n_bad_freq_epochs
    title(f"AstKAT simulation, {args.mode} mode")
    msg(f"{args.n_images} epochs (+{args.n_bad_freq_epochs} at a wrong frequency), {args.n_sources} sources, "
        f"{args.image_size} x {args.image_size} pixels of {PIXEL_SCALE}\", "
        f"one epoch every {args.cadence_days} days")
    msg(f"Injected errors: A = {A} beam sigma at SNR 1"
        + (f", beta = {args.beta}, SNR_b = {args.snr_break}, gamma = {args.snr_gamma}" if flat else "")
        + f", B = {args.b_parameter}" + ("\" per axis" if args.b_absolute else " beam sigma")
        + ("" if flat else "; the SNR-dependent error comes from PyBDSF's fitting"))
    if flat:
        msg(f"Peak noise k = {args.peak_noise} rms; "
            f"outlier fraction {args.outlier_fraction} at {args.outlier_scale}x the width"
            + (f"; detection at peak >= {CATALOGUE_DETECTION_SNR:g} rms" if catalogue else ""))

    msg(f"Flux scale: each epoch's image times 1 + g_e, g_e normal with sigma f_sys = {args.flux_sys}; "
        f"each source's peak times 1 + f_frac N(0, 1), f_frac = {args.flux_frac}")

    # The intrinsic field, drawn first so every mode shares it for a given seed: centre, true positions,
    # source SNRs, the epoch translations and the epoch flux scales.
    centre = random_pointing(rng)
    x_true, y_true = (rng.uniform(EDGE_BUFFER, args.image_size - EDGE_BUFFER, args.n_sources) for _ in range(2))
    snr = truncated_powerlaw(rng, SNR_POWERLAW, MIN_SNR, MAX_SNR, args.n_sources)
    translations = rng.normal(0.0, EPOCH_OFFSET_SIGMA, (n_total, 2))
    if args.disable_epoch_offsets:
        translations[:] = 0.0
    # A flux-scale (gain) error multiplies the whole image, noise included, so SNRs are unchanged.
    flux_scale = args.flux_sys * rng.normal(size=n_total)
    field_rms = 1.0 if flat else (args.rms if args.rms is not None else rng.uniform(MIN_RMS, MAX_RMS))
    fluxes = snr * field_rms
    beams = epoch_beam_list(rng, n_total, args.mode, args.same_beam, args.circularise_beams)
    write_true_positions(output_dir, (x_true, y_true), fluxes, centre, args.image_size)
    msg(f"Field centre RA {centre[0]:.4f}, Dec {centre[1]:.4f} deg; SNR {snr.min():.1f} to {snr.max():.1f}"
        + ("" if flat else f"; field RMS {field_rms:.3f}"))

    epochs, rows, paths, noise_checks = [], [], [], []
    for e in range(n_total):
        beam = beams[e]
        noise_check = None
        frequency = FREQ_HZ * (BAD_FREQ_FACTOR if e >= args.n_images else 1.0)

        # Noise mode: the epoch's RMS varies, so each source's SNR does too.
        rms = (None if flat
               else max(abs(rng.normal(field_rms, args.rms_variation * field_rms)), 1e-6 * field_rms))
        epoch_snr = snr if flat else fluxes / rms

        # Source scatter in beam units, then the epoch translation shared by every source. Flat and catalogue
        # modes also draw outliers, the same scatter outlier_scale times wider, and each epoch's peak.
        shape = (args.beta, args.snr_break, args.snr_gamma) if flat else (1.0, 1.0, 0.0)
        sigma_major, sigma_minor = error_sigma(epoch_snr, A, *shape, beam, args.b_parameter, args.b_absolute)
        # Every mode: each source's own fractional error, not shared with the epoch's other sources.
        peak = fluxes * (1.0 + args.flux_frac * rng.normal(size=args.n_sources))
        if flat:
            wide = np.where(rng.random(args.n_sources) < args.outlier_fraction, args.outlier_scale, 1.0)
            sigma_major, sigma_minor = sigma_major * wide, sigma_minor * wide
            peak = peak + args.peak_noise * field_rms * rng.normal(size=args.n_sources)
        dx, dy = beam_offsets(rng, beam, sigma_major, sigma_minor)
        east, north = translations[e]
        x = x_true + dx - east / PIXEL_SCALE
        y = y_true + dy + north / PIXEL_SCALE

        image = render_sources(x, y, peak, beam, args.image_size)
        if flat:
            image += rng.normal(FLAT_BACKGROUND, FLAT_BACKGROUND_SIGMA, image.shape)
        else:
            noise = correlated_noise(rng, args.image_size, beam, rms, not args.no_spatial_correlation)
            image += noise
            # The noise autocorrelation against the beam, checked before sources are added
            if not args.no_spatial_correlation:
                noise_check = noise_correlation_check(noise, beam, keep_maps=not noise_checks)
                noise_checks.append(noise_check)
        gain = 1.0 + flux_scale[e]
        image *= gain
        # The epoch's rms scales with the gain; flat images record it in NOISE, which extraction.py reads.
        rms = (field_rms if flat else rms) * gain
        header = fits_header(beam, centre, args.image_size, e, frequency, args.mode, args.cadence_days, rms)
        path = output_dir / f"{settings['prefix']}{e:04d}.fits"
        fits.PrimaryHDU(data=image[np.newaxis, np.newaxis].astype(np.float32), header=header).writeto(
            path, overwrite=True)
        paths.append(path)
        if catalogue:
            rows.append(catalogue_epoch(x, y, peak * gain, field_rms * gain, beam, centre, args.image_size,
                                        frequency))

        epochs.append({'epoch_index': e, 'frequency_hz': frequency, 'rms_noise': rms,
                       'global_x_offset_arcsec': east, 'global_y_offset_arcsec': north,
                       'flux_scale': float(flux_scale[e]),
                       'median_snr': float(np.median(epoch_snr)), 'beam_parameters': beam,
                       **({'noise_correlation_fwhm_pixels': [float(v) for v in noise_check['fwhm_pixels']],
                           'noise_correlation_pa_deg': float(noise_check['pa_deg'])} if noise_check else {})})
        msg(f"Epoch {e:04d}: beam {beam['bmaj_pixels']:.1f} x {beam['bmin_pixels']:.1f} px at "
            f"{beam['bpa_deg']:.0f} deg, translation ({east:+.3f}, {north:+.3f})\", flux scale {gain:.3f}"
            + (f", RMS {rms:.2f}" if rms is not None else "")
            + (f", noise correlation / beam FWHM {noise_check['ratio'][0]:.2f} major, "
               f"{noise_check['ratio'][1]:.2f} minor" if noise_check else "")
            # Flat and catalogue: the realised peak scatter in rms before the flux scale; k when f_frac = 0.
            + (f", peak scatter {np.std(peak - fluxes) / field_rms:.3f} rms" if flat else "")
            + (f", {np.mean(peak >= CATALOGUE_DETECTION_SNR * field_rms) * 100:.1f}% detected" if catalogue else "")
            + (" (wrong frequency)" if e >= args.n_images else ""))

    save_truth(output_dir, {
        'mode': args.mode, 'n_images': args.n_images, 'n_sources': args.n_sources, 'image_size': args.image_size,
        'pixel_scale_arcsec': PIXEL_SCALE, 'cadence_days': args.cadence_days, 'seed': args.seed,
        'a_parameter': A, 'beta': args.beta if flat else None, 'snr_break': args.snr_break if flat else None,
        'snr_gamma': args.snr_gamma if flat else None,
        'b_parameter': args.b_parameter, 'b_units': 'arcsec' if args.b_absolute else 'beam', 'alpha': 1.0,
        'epoch_offset_sigma_arcsec': 0.0 if args.disable_epoch_offsets else EPOCH_OFFSET_SIGMA,
        'field_rms': field_rms, 'rms_variation_frac': 0.0 if flat else args.rms_variation,
        'powerlaw_alpha': SNR_POWERLAW, 'min_snr': MIN_SNR, 'max_snr': MAX_SNR,
        'n_bad_freq_epochs': args.n_bad_freq_epochs,
        'bad_freq_epoch_indices': list(range(args.n_images, n_total)), 'pointing_center': centre,
        'flux_systematic': args.flux_sys, 'flux_fractional': args.flux_frac,
        **({'peak_noise': args.peak_noise, 'outlier_fraction': args.outlier_fraction,
            'outlier_scale': args.outlier_scale} if flat else {}),
        **({'detection_snr': CATALOGUE_DETECTION_SNR} if catalogue else {}),
    }, epochs)
    if catalogue:
        write_catalogue(output_dir, np.stack(rows, axis=1), paths)
    if noise_checks:
        ratio = np.array([c['ratio'] for c in noise_checks])
        msg(f"Noise correlation / beam FWHM over {len(noise_checks)} epochs: major {np.median(ratio[:, 0]):.3f}, "
            f"minor {np.median(ratio[:, 1]):.3f} (median; 1 for the beam, {np.sqrt(2.0):.3f} for the beam "
            f"self-convolution)")
        simulation_noise_figure(dict(noise_checks[0], epochs=np.arange(len(noise_checks)), ratio=ratio))
    title(f"AstKAT simulation complete: {n_total} images in {output_dir}")


# ============================================================================
# CATALOGUE MODE
# ============================================================================

def catalogue_epoch(x, y, peak, rms, beam, centre, image_size, frequency):
    """
    One epoch of the matched catalogue, shape (n_sources, n_columns), NaN where a source is not detected.

    A source is detected where its drawn peak is at least CATALOGUE_DETECTION_SNR
    times the rms. Every detection is a single point-like Gaussian: the size
    and flux ratios are 1 and the S_Code is S.
    """
    ra, dec = pixels_to_radec(x, y, centre, image_size)
    offset = SkyCoord(ra * u.deg, dec * u.deg).separation(SkyCoord(centre[0] * u.deg, centre[1] * u.deg)).deg
    scale = PIXEL_SCALE / 3600.0
    values = {'ra_deg': ra, 'dec_deg': dec, 'peak_flux_Jy': peak, 'rms_Jy': rms, 'snr': peak / rms,
              'phase_offset_deg': offset, 'bmaj_deg': beam['bmaj_pixels'] * scale,
              'bmin_deg': beam['bmin_pixels'] * scale, 'bpa_deg': ((beam['bpa_deg'] + 90) % 180) - 90,
              'freq_Hz': frequency, 'flux_ratio': 1.0, 'maj_ratio': 1.0, 'min_ratio': 1.0,
              's_code': S_CODES.index('S')}
    table = np.column_stack([np.broadcast_to(np.asarray(values[name], float), peak.shape)
                             for name in CATALOG_COLUMNS])
    table[peak < CATALOGUE_DETECTION_SNR * rms] = np.nan
    return table


def write_catalogue(output_dir, catalog, images):
    """
    Write the matched catalogue and observation_properties.json for fit.py.

    The catalogue has matching.py's format, epochs in time order; the
    properties come from the images through extraction.image_properties, so
    both match what the pipeline would write. initialise.py copies them into
    the project.
    """
    np.savez(output_dir / CATALOGUE_FILE, data=catalog, columns=list(CATALOG_COLUMNS), shape=catalog.shape,
             description="AstKAT simulated matched catalogue (n_sources, n_epochs, n_columns), drawn from the "
                         "joint model; snr is each detection's peak over its rms")
    properties = [image_properties(path) for path in images]
    with open(output_dir / OBSERVATION_FILE, 'w') as f:
        json.dump({key: [p[key] for p in properties] for key in properties[0]}, f, indent=4)
    detected = np.isfinite(catalog[:, :, CATALOG_COLUMNS.index('ra_deg')])
    msg(f"Catalogue: {int(detected.sum())} detections of {catalog.shape[0]} sources over {catalog.shape[1]} "
        f"epochs ({detected.mean() * 100:.1f}% of pairs); saved to {output_dir / CATALOGUE_FILE} and "
        f"{output_dir / OBSERVATION_FILE}")


# ============================================================================
# TARGET TRAJECTORY
# ============================================================================

def target_trajectory(n_steps, v0, decel, rng):
    """
    A straight track in a random direction with a linearly falling speed.

    The speed at step t is v0 max(1 - decel t, 0) arcsec per step, so the
    target stops near step 1 / decel; it starts at the origin.

    Returns
    -------
    offset : ndarray, shape (n_steps, 2)
        RA and Dec distance travelled, arcsec.
    speed : ndarray, shape (n_steps,)
    direction_deg : float
        Position angle of the track, East of North.
    """
    theta = rng.uniform(0.0, 2.0 * np.pi)
    speed = np.clip(v0 * (1.0 - decel * np.arange(n_steps)), 0.0, None)
    distance = np.concatenate([[0.0], np.cumsum(speed[:-1])])
    return np.outer(distance, [np.sin(theta), np.cos(theta)]), speed, float(np.degrees(theta))


def run_target_simulation(args):
    """
    Simulate a moving target in an existing project's epochs, as a catalogue for target.py.

    Uses the project's fit and offsets.py outputs, so run from the project
    directory after both. One posterior draw sets everything injected: each
    epoch's translation and the noise model, with each epoch's own beam.
    Each measurement is the true position plus its epoch's translation plus
    noise, and carries differential aberration when the project corrects for
    it, so target.py should recover the true track within its errors.
    """
    import yaml
    try:
        from .utils import (parse_phase_center, observatory_location, apply_aberration_to_positions,
                            is_simulated, aberration_correction_enabled)
        from .fit import load_results, tangent_plane_arcsec, tangent_plane_to_radec
        from .offsets import CONFIG_FILE, thinned_draws, _sample_2d
        from .target import load_translations, noise_covariance
    except ImportError:
        from utils import (parse_phase_center, observatory_location, apply_aberration_to_positions,
                           is_simulated, aberration_correction_enabled)
        from fit import load_results, tangent_plane_arcsec, tangent_plane_to_radec
        from offsets import CONFIG_FILE, thinned_draws, _sample_2d
        from target import load_translations, noise_covariance

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir or MODES['target']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    title("AstKAT simulation, target mode")
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    results = load_results(with_trace=True)
    trans = load_translations()

    # The first n epochs with a translation, each with its own time and beam.
    epochs = np.nonzero(np.isfinite(trans['median'][:, 0]))[0][:args.n_images]
    times = Time([str(trans['obs_isot'][e]) for e in epochs], format='isot', scale='utc')
    with open(Path('results') / 'observation_properties.json', 'r') as f:
        props = json.load(f)
    bmaj, bmin, bpa = (np.asarray(props[k], dtype=float)[epochs] for k in ('bmaj_deg', 'bmin_deg', 'bpa_deg'))
    bmaj, bmin = bmaj * 3600.0, bmin * 3600.0

    # offsets.py thins the posterior the same way, so draw k of its translations
    # belongs to draw k of the noise parameters.
    draws = thinned_draws(results['idata'], int(config['MCMC'].get('posterior_predictive_draws', 1000)))
    if len(draws['mu']) != len(trans['draws']):
        raise ValueError("MCMC.posterior_predictive_draws has changed since offsets.py ran; re-run offsets.py")
    k = int(rng.integers(len(trans['draws'])))
    one_draw = {name: values[k:k + 1] for name, values in draws.items()}
    tau = trans['draws'][k][epochs]

    phase_centre = parse_phase_center(config['BOOKKEEPING']['phase_ctr'])
    start = parse_phase_center(args.start) if args.start else phase_centre
    offset, speed, direction = target_trajectory(len(epochs), args.v0, args.decel, rng)
    true_xy = tangent_plane_arcsec(np.array([start.ra.deg]), np.array([start.dec.deg]), phase_centre) + offset

    flux = np.clip(args.target_flux * (1.0 + args.target_flux_scatter * rng.standard_normal(len(epochs))),
                   1e-6 * args.target_flux, None)
    rms = np.full(len(epochs), args.target_rms)
    # The fitted noise itself, without any conservative inflation, so target.py's inflated errors are tested.
    noise_cov = noise_covariance(flux / rms, bmaj, bmin, bpa, dict(results['model'], error_inflation=1.0), one_draw)
    measured_xy = true_xy + tau + _sample_2d(noise_cov, rng)

    ra, dec = tangent_plane_to_radec(measured_xy, phase_centre)
    aberrated = not is_simulated(config) and aberration_correction_enabled(config)
    if aberrated:
        ra, dec = apply_aberration_to_positions(ra, dec, phase_centre, times, observatory_location(config))
    true_ra, true_dec = tangent_plane_to_radec(true_xy, phase_centre)

    catalogue = output_dir / 'target_trajectory.txt'
    with open(catalogue, 'w') as f:
        f.write("flux rms ra dec bmaj bmin bpa time\n")
        for i in range(len(epochs)):
            f.write(f"{flux[i]:.10g} {rms[i]:.10g} {ra[i]:.12f} {dec[i]:.12f} {bmaj[i]:.6f} {bmin[i]:.6f} "
                    f"{bpa[i]:.4f} {times[i].isot}\n")
    truth = output_dir / 'target_trajectory_truth.csv'
    with open(truth, 'w') as f:
        f.write("step,catalog_epoch,time_isot,true_ra_deg,true_dec_deg,true_ra_offset_arcsec,true_dec_offset_arcsec,"
                "speed_arcsec_per_step,tau_ra_arcsec,tau_dec_arcsec\n")
        for i, e in enumerate(epochs):
            f.write(f"{i},{e},{times[i].isot},{true_ra[i]:.12f},{true_dec[i]:.12f},{true_xy[i, 0]:.6f},"
                    f"{true_xy[i, 1]:.6f},{speed[i]:.6f},{tau[i, 0]:.6f},{tau[i, 1]:.6f}\n")

    msg(f"{len(epochs)} epochs from {times[0].isot} to {times[-1].isot}; posterior draw {k}")
    msg(f"Track at PA {direction:.1f} deg, {np.hypot(*offset[-1]):.3f} arcsec in total; "
        f"SNR {np.median(flux / rms):.0f} (median); aberration {'injected' if aberrated else 'not injected'}")
    msg(f"Catalogue saved to {catalogue}, truth to {truth}")
    msg(f"Test recovery with: python target.py {catalogue}")


def parse_args():
    """Command-line options for simulate.py."""
    parser = argparse.ArgumentParser(description="Simulate AstKAT epoch images with known astrometric errors")
    parser.add_argument('--mode', choices=sorted(MODES), default='noise',
                        help='noise: realistic noisy images; flat: errors injected directly; catalogue: '
                             'the model\'s own data written as the matched catalogue; target: a moving target '
                             'for target.py (default: noise)')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='output directory (default: sim_images, sim_images_flat, sim_catalogue or '
                             'sim_target)')
    parser.add_argument('-s', '--image-size', type=int, default=IMAGE_SIZE,
                        help=f'pixels per side (default {IMAGE_SIZE})')
    parser.add_argument('-i', '--n-images', type=int, default=N_IMAGES,
                        help=f"number of epochs (target mode: at most the project's epochs) (default {N_IMAGES})")
    parser.add_argument('-n', '--n-sources', type=int, default=N_SOURCES,
                        help=f'number of sources (default {N_SOURCES})')
    parser.add_argument('-a', '--a-parameter', type=float, default=1.0,
                        help='flat and catalogue modes: injected A, the SNR term at SNR 1 (default 1.0)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='flat and catalogue modes: injected SNR exponent, A SNR^-beta (default 1)')
    parser.add_argument('--snr-break', type=float, default=10.0,
                        help='flat and catalogue modes: injected break SNR_b of the SNR term (default 10)')
    parser.add_argument('--snr-gamma', type=float, default=0.0,
                        help='flat and catalogue modes: injected gamma; slope -beta - gamma below SNR_b '
                             '(default 0, no break)')
    parser.add_argument('-b', '--b-parameter', type=float, default=0.01,
                        help='injected floor B as a fraction of the beam sigma per axis, or in arcsec with '
                             '--b-absolute (default 0.01)')
    parser.add_argument('--b-absolute', action='store_true',
                        help='B in arcsec, the same on both axes and in every epoch, as an absolute-B fit assumes')
    parser.add_argument('-r', '--rms', type=float, default=None,
                        help=f'noise mode: field RMS (default: random in {MIN_RMS:g}-{MAX_RMS:g})')
    parser.add_argument('--rms-variation', type=float, default=RMS_VARIATION,
                        help=f'noise mode: fractional epoch-to-epoch RMS scatter (default {RMS_VARIATION})')
    parser.add_argument('--no-spatial-correlation', action='store_true', help='noise mode: white noise')
    parser.add_argument('--circularise-beams', action='store_true',
                        help='circular beams (BMIN = BMAJ); PyBDSF then fits peaks ~5%% low (see MIN_BEAM_RATIO)')
    parser.add_argument('--same-beam', action='store_true', help='the same beam in every epoch')
    parser.add_argument('--disable-epoch-offsets', action='store_true', help='no epoch translations')
    parser.add_argument('-nf', '--n-bad-freq-epochs', type=int, default=0,
                        help=f'extra epochs at {BAD_FREQ_FACTOR:g}x the frequency, to test frequency cuts '
                             '(default 0)')
    parser.add_argument('--cadence-days', type=float, default=CADENCE_DAYS,
                        help=f'days between epochs (default {CADENCE_DAYS:g})')
    parser.add_argument('--flux-sys', type=float, default=0.05,
                        help='sigma of the per-epoch flux scale error g_e; each image is multiplied by 1 + g_e '
                             '(default 0.05)')
    parser.add_argument('--flux-frac', type=float, default=0.02,
                        help="fractional error on each source's peak in each epoch, not shared between sources "
                             '(default 0.02)')
    seed = parser.add_mutually_exclusive_group()
    seed.add_argument('--seed', type=int, default=DEFAULT_SEED, help=f'random seed (default {DEFAULT_SEED})')
    seed.add_argument('-R', '--random-seed', action='store_true',
                      help='draw a fresh random seed; it is printed and saved with the truth')
    cat = parser.add_argument_group('flat and catalogue modes')
    cat.add_argument('-k', '--peak-noise', type=float, default=1.3,
                     help='noise on a peak over the rms (default 1.3, as PyBDSF measures)')
    cat.add_argument('--outlier-fraction', type=float, default=0.002,
                     help='fraction of positions drawn outlier-scale times wider (default 0.002)')
    cat.add_argument('--outlier-scale', type=float, default=6.0,
                     help='width of the outliers over the inlier width (default 6)')
    target = parser.add_argument_group('target mode')
    target.add_argument('--start', default=None, help="starting position, e.g. '17:27:43.3 -16:12:04' "
                                                      "(default: the phase centre)")
    target.add_argument('--v0', type=float, default=3.0, help='starting speed, arcsec per epoch (default 3)')
    target.add_argument('--decel', type=float, default=0.04,
                        help='fractional fall in speed per epoch; the target stops near epoch 1 / decel '
                             '(default 0.04)')
    target.add_argument('--target-flux', type=float, default=0.35,
                        help='mean flux density; with the default rms this gives SNR 7 (default 0.35)')
    target.add_argument('--target-flux-scatter', type=float, default=0.2,
                        help='fractional flux scatter per epoch (default 0.2)')
    target.add_argument('--target-rms', type=float, default=0.05,
                        help='rms, same units as the flux (default 0.05)')
    args = parser.parse_args()
    if args.random_seed:
        # A new seed each run; recorded in place of --seed so the run can be repeated.
        args.seed = secrets.randbits(32)
    msg(f"Random seed {args.seed}" + (" (drawn with -R; repeat with --seed)" if args.random_seed else ""))
    if args.n_bad_freq_epochs < 0:
        parser.error('--n-bad-freq-epochs must be >= 0')
    return args


def main():
    arguments = parse_args()
    if arguments.mode == 'target':
        run_target_simulation(arguments)
    else:
        run_simulation(arguments)
        if arguments.mode == 'noise':
            # Noise mode injects no outliers, but astkat init turns the outlier mixture on.
            msg("Noise mode injects no outliers: after astkat init, set MCMC.enable_outlier_mixture: false "
                "in astkat_files/astkat.yaml")


if __name__ == '__main__':
    main()
