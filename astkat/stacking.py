"""
AstKAT image stacking.

Builds the deep image used as the source-finding reference. The epoch images
are restricted to one frequency band and to a set of similar beams (a core
of BOOKKEEPING.stack_min_epochs with the smallest common beam, plus any that
widen it by at most stack_beam_growth; all of them if there are fewer),
convolved to their common beam (the smallest ellipse enclosing every
selected beam), and median-combined pixel by pixel.

Run by initialise.py with --stack-image, or on its own for an existing project:
    python stacking.py
"""

import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import bottleneck as bn
import scipy.signal
import yaml
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from scipy.spatial import ConvexHull

try:
    from .utils import msg, title, load_system_resources, DEFAULT_SEED
    from .plot import common_beam_figure
except ImportError:
    from utils import msg, title, load_system_resources, DEFAULT_SEED
    from plot import common_beam_figure


FREQUENCY_TOLERANCE = 0.1       # images within 10% in frequency count as one band
ELLIPSE_SAMPLES = 50            # points per beam outline when finding the common beam
PSF_SIZE = 101                  # pixels per side of the PSF images given to pypher
FWHM_TO_SIGMA = (8 * np.log(2)) ** -0.5

KERNELS_DIR = Path('temp_kernels')
HOMOGENISED_DIR = Path('temp_homogenised')


# ============================================================================
# FITS INPUT AND OUTPUT
# ============================================================================

def get_image(fitsfile):
    """
    2-D image data from a FITS file as float32, dropping degenerate leading axes.

    Returns None if the file cannot be read.
    """
    try:
        with fits.open(fitsfile) as hdul:
            data = hdul[0].data
            while data.ndim > 2:
                data = data[0]
            return np.array(data, dtype=np.float32)
    except Exception as e:
        msg(f"Error loading {fitsfile}: {e}")
        return None


def create_fits(image, header, fitsfile):
    """Write a 2-D or 3-D image as a 4-D radio FITS image (Stokes, frequency, y, x)."""
    if image.ndim == 2:
        image = image.reshape(1, 1, *image.shape)
    elif image.ndim == 3:
        image = image.reshape(1, *image.shape)
    fits.PrimaryHDU(data=image, header=header).writeto(fitsfile, overwrite=True)
    msg(f'Created FITS file: {fitsfile}')


# ============================================================================
# MINIMUM ENCLOSING ELLIPSE (WELZL)
# ============================================================================

class WelzlEllipse:
    """
    Smallest ellipse enclosing a set of 2-D points, by Welzl's randomised algorithm.

    Points are processed in random order. A point outside the current ellipse
    must lie on the boundary of the final one, so the boundary set grows until
    it defines the ellipse uniquely (3, 4 or 5 points). Expected time is linear
    in the number of points. The result does not depend on the random order;
    the generator is seeded only to make runs repeatable.

    Ellipses are returned as (centre, semi_major, semi_minor, rotation_rad).

    Reference: Welzl (1991), Smallest enclosing disks (balls and ellipsoids),
    New Results and New Trends in Computer Science, 359-370.
    """

    def __init__(self, tolerance=1e-12, max_recursion=1000, seed=DEFAULT_SEED):
        """
        Parameters
        ----------
        tolerance : float
            Relative singular-value threshold below which a matrix counts as singular.
        max_recursion : int
            Recursion depth at which the search gives up.
        seed : int
            Seed for the point order.
        """
        self.tolerance = tolerance
        self.max_recursion = max_recursion
        self.recursion_depth = 0
        self.rng = np.random.default_rng(seed)

    def is_singular(self, A):
        """True if A is singular or nearly so, judged by its singular values."""
        try:
            _, s, _ = np.linalg.svd(A)
            return np.min(s) < self.tolerance * np.max(s)
        except np.linalg.LinAlgError:
            return True

    def center_form_to_geometric(self, F, c):
        """
        Geometric parameters of the ellipse (x - c)^T F (x - c) = 1.

        Returns None if F is not positive definite.
        """
        try:
            w, V = np.linalg.eigh(F)
            if np.any(w <= 0):
                return None
            if V[1, 0] < 0:
                V[:, 0] = -V[:, 0]
            rotation = np.arctan2(V[1, 0], V[0, 0])
            return c, 1.0 / np.sqrt(np.min(w)), 1.0 / np.sqrt(np.max(w)), rotation
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            return None

    def ellipse_from_boundary5(self, S):
        """The unique ellipse through 5 points, shape (5, 2)."""
        if S.shape != (5, 2):
            return None
        try:
            x, y = S[:, 0], S[:, 1]
            A = np.column_stack((x ** 2, y ** 2, 2 * x * y, x, y))
            if self.is_singular(A):
                return None
            sol = np.linalg.solve(A, -np.ones(S.shape[0]))

            center_matrix = -2 * np.array([[sol[0], sol[2]], [sol[2], sol[1]]])
            if self.is_singular(center_matrix):
                return None
            c = np.linalg.solve(center_matrix, sol[3:5])

            A_ext = np.vstack([
                np.hstack([np.eye(3), -np.array([[sol[0], sol[2], sol[1]]]).T]),
                np.array([c[0] ** 2, 2 * c[0] * c[1], c[1] ** 2, -1]),
            ])
            s = np.linalg.solve(A_ext, np.array([0, 0, 0, 1]))
            F = np.array([[s[0], s[1]], [s[1], s[2]]])
            return self.center_form_to_geometric(F, c)
        except (np.linalg.LinAlgError, ValueError):
            return None

    def ellipse_from_boundary4(self, S):
        """
        The smallest ellipse through 4 points, shape (4, 2).

        The quadrilateral is mapped by rotation, shear and stretch to a cyclic
        one, the optimal swing angle is found from a cubic, and the enclosing
        circle is mapped back.
        """
        if S.shape != (4, 2):
            return None
        try:
            Sc = S - np.mean(S, axis=0)
            S = S[np.argsort(-np.arctan2(Sc[:, 1], Sc[:, 0])), :]

            # Intersection of the diagonals becomes the origin
            A = np.column_stack([S[2, :] - S[0, :], S[1, :] - S[3, :]])
            if self.is_singular(A):
                return None
            s = np.linalg.solve(A, S[1, :] - S[0, :])
            diag_intersect = S[0, :] + s[0] * (S[2, :] - S[0, :])
            S = S - diag_intersect

            # Rotate the first diagonal onto the x axis
            AC = S[2, :] - S[0, :]
            theta = np.arctan2(AC[1], AC[0])
            rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
            S = (rot_mat @ S.T).T

            # Shear the second diagonal onto the y axis
            denominator = S[3, 1] - S[1, 1]
            if abs(denominator) < 1e-12:
                return None
            shear_mat = np.array([[1.0, (S[1, 0] - S[3, 0]) / denominator], [0.0, 1.0]])
            S = (shear_mat @ S.T).T

            # Stretch to make the quadrilateral cyclic
            b = np.linalg.norm(S, axis=1)
            if np.any(b == 0):
                return None
            d = (b[1] * b[3]) / (b[2] * b[0])
            if d <= 0:
                return None
            stretch_mat = np.diag([d ** 0.25, d ** -0.25])
            S = (stretch_mat @ S.T).T

            # Optimal swing angle from a cubic
            a = np.linalg.norm(S, axis=1)
            coeff = np.zeros(4)
            coeff[0] = -4 * a[1] ** 2 * a[2] * a[0]
            coeff[1] = -4 * a[1] * (a[2] - a[0]) * (a[1] ** 2 - a[2] * a[0])
            coeff[2] = (3 * a[1] ** 2 * (a[1] ** 2 + a[2] ** 2)
                        - 8 * a[1] ** 2 * a[2] * a[0]
                        + 3 * (a[1] ** 2 + a[2] ** 2) * a[0] ** 2)
            coeff[3] = coeff[1] / 2.0
            roots = np.roots(coeff)
            valid = roots[(-1 < np.real(roots)) & (np.real(roots) < 1) & (np.abs(np.imag(roots)) < 1e-10)]
            if len(valid) == 0:
                return None
            theta = np.arcsin(np.real(valid[0]))

            cos_theta_sqrt = np.cos(theta) ** 0.5
            if cos_theta_sqrt == 0:
                return None
            D_mat = np.array([[1.0 / cos_theta_sqrt, np.sin(theta) / cos_theta_sqrt],
                              [0.0, cos_theta_sqrt]])
            S = (D_mat @ S.T).T

            # Circle through three of the transformed points
            boundary = S[:-1, :]
            A = np.vstack([-2 * boundary.T, np.ones(boundary.shape[0])]).T
            if self.is_singular(A):
                return None
            s = np.linalg.solve(A, -np.sum(boundary ** 2, axis=1))
            circle_c = s[:2]
            circle_r_sq = np.sum(circle_c ** 2) - s[2]
            if circle_r_sq <= 0:
                return None

            # Map the circle back to an ellipse in the original frame
            T_mat = D_mat @ stretch_mat @ shear_mat @ rot_mat
            ellipse_c = np.linalg.solve(T_mat, circle_c) + diag_intersect
            ellipse_F = T_mat.T @ T_mat / circle_r_sq
            return self.center_form_to_geometric(ellipse_F, ellipse_c)
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            return None

    def ellipse_from_boundary3(self, S):
        """The smallest ellipse through 3 points, shape (3, 2)."""
        if S.shape != (3, 2):
            return None
        try:
            c = np.mean(S, axis=0)
            Sc = S - c
            if self.is_singular(Sc):
                return None
            StS = Sc.T @ Sc
            if self.is_singular(StS):
                return None
            return self.center_form_to_geometric(1.5 * np.linalg.inv(StS), c)
        except (np.linalg.LinAlgError, ValueError):
            return None

    def is_point_inside(self, point, ellipse):
        """True if point lies inside or on the ellipse, to a relative tolerance of 1e-10."""
        if ellipse is None or point is None:
            return False
        try:
            c, a, b, t = ellipse
            v = point - c
            rot_mat = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
            F = rot_mat.T @ np.diag([1.0 / a ** 2, 1.0 / b ** 2]) @ rot_mat
            return v.T @ F @ v <= 1.0 + 1e-10
        except (ValueError, TypeError, IndexError):
            return False

    def sample_ellipse(self, ellipse, num_pts=100, endpoint=True):
        """
        Points on an origin-centred ellipse outline.

        Parameters
        ----------
        ellipse : tuple
            (semi_major, semi_minor, rotation_rad).
        num_pts : int
        endpoint : bool
            Repeat the first point at the end, to draw a closed curve.

        Returns
        -------
        ndarray, shape (num_pts, 2)
        """
        a, b, p = ellipse
        rot_mat = np.array([[np.cos(p), -np.sin(p)], [np.sin(p), np.cos(p)]])
        theta = np.linspace(0, 2 * np.pi, num_pts, endpoint=endpoint)
        return (rot_mat @ np.column_stack((a * np.cos(theta), b * np.sin(theta))).T).T

    def compute_minimum_enclosing_ellipse(self, points):
        """
        Smallest ellipse enclosing points, shape (n, 2).

        Returns (centre, semi_major, semi_minor, rotation_rad), or None on failure.
        """
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates")
        self.recursion_depth = 0
        return self._welzl_recursive(points, np.zeros((0, 2)))

    def _welzl_recursive(self, interior, boundary):
        """One step of Welzl's recursion over the remaining interior points."""
        self.recursion_depth += 1
        if self.recursion_depth > self.max_recursion:
            msg("Warning: maximum recursion depth reached in the Welzl algorithm")
            return None

        if interior.shape[0] == 0 or boundary.shape[0] == 5:
            self.recursion_depth -= 1
            return self._ellipse_from_boundary(boundary)

        i = self.rng.integers(interior.shape[0])
        p = interior[i, :]
        interior_wo_p = np.delete(interior, i, 0)

        ellipse = self._welzl_recursive(interior_wo_p, boundary)
        if not self.is_point_inside(p, ellipse):
            new_boundary = np.vstack([boundary, p]) if boundary.size > 0 else p.reshape(1, -1)
            ellipse = self._welzl_recursive(interior_wo_p, new_boundary)

        self.recursion_depth -= 1
        return ellipse

    def _ellipse_from_boundary(self, boundary):
        """Ellipse defined by 3, 4 or 5 boundary points; None for fewer."""
        n = boundary.shape[0]
        if n <= 2:
            return None
        if n == 3:
            return self.ellipse_from_boundary3(boundary)
        if n == 4:
            return self.ellipse_from_boundary4(boundary)
        return self.ellipse_from_boundary5(boundary)


# ============================================================================
# IMAGE AND BEAM SELECTION
# ============================================================================

def find_epoch_images(image_dir, project_name):
    """Epoch images in image_dir, sorted by name (time order), excluding the stacked image."""
    stacked_name = f'{project_name}_stacked.fits'
    images = sorted(str(p) for p in Path(image_dir).glob('*.fits') if p.name != stacked_name)
    if not images:
        raise FileNotFoundError(f"No FITS images found in {image_dir}")
    msg(f"Found {len(images)} images in {image_dir}")
    return images


def select_single_band(images):
    """
    Keep the most common frequency band (CRVAL3, within FREQUENCY_TOLERANCE).

    Beam homogenisation assumes one band. Images without frequency
    information are kept unless they sit alongside more than one band.
    """
    bands = {}   # representative frequency in GHz -> image paths
    no_freq = []
    for image in images:
        try:
            freq_ghz = fits.getheader(image).get('CRVAL3', 0) / 1e9
        except Exception as e:
            msg(f"Warning: could not read the frequency of {Path(image).name}: {e}")
            freq_ghz = 0
        if freq_ghz <= 0:
            no_freq.append(image)
            continue
        band = next((f for f in bands if abs(freq_ghz - f) <= f * FREQUENCY_TOLERANCE), freq_ghz)
        bands.setdefault(band, []).append(image)

    if not bands:
        msg("No frequency information in any image; using all images")
        return images

    msg(f"Frequency bands ({FREQUENCY_TOLERANCE * 100:.0f}% tolerance):")
    for freq, members in sorted(bands.items()):
        msg(f"  {freq:.3f} GHz: {len(members)} images")
    if no_freq:
        msg(f"  no frequency information: {len(no_freq)} images")

    if len(bands) == 1:
        return images

    main_band = max(bands, key=lambda f: len(bands[f]))
    msg(f"Using the {main_band:.3f} GHz band ({len(bands[main_band])} images); "
        f"{len(images) - len(bands[main_band])} images from other bands or without frequency left out")
    return bands[main_band]


def parse_image_selection(stacked_image_config, total_images):
    """
    Image indices selected by BOOKKEEPING.stacked_image.

    Parameters
    ----------
    stacked_image_config : True, str or list
        True for all images, "start,end" for images start to end - 1, or a
        list of indices.
    total_images : int

    Returns
    -------
    (indices, description)
    """
    if stacked_image_config is True:
        return list(range(total_images)), "all images"

    if isinstance(stacked_image_config, str):
        try:
            start, end = map(int, stacked_image_config.split(','))
        except ValueError:
            raise ValueError(f"stacked_image must be 'start,end'; got '{stacked_image_config}'")
        start, end = max(0, start), min(total_images, end)
        if start >= end:
            raise ValueError(f"stacked_image range is empty: start {start} >= end {end}")
        return list(range(start, end)), f"images {start} to {end - 1}"

    if isinstance(stacked_image_config, list):
        indices = [i for i in stacked_image_config if 0 <= i < total_images]
        return indices, f"images {indices}"

    raise ValueError(f"Unsupported stacked_image setting: {stacked_image_config}")


def read_beams(images):
    """Beam (BMAJ, BMIN, BPA in degrees) of each image, skipping images without a valid beam."""
    beams = []
    for image in images:
        try:
            header = fits.getheader(image)
        except Exception as e:
            msg(f"  {Path(image).name}: could not read header ({e}); skipped")
            continue
        bmaj, bmin, bpa = header.get('BMAJ', 0), header.get('BMIN', 0), header.get('BPA', 0)
        if bmaj > 1e-14 and bmin > 1e-14:
            beams.append({'bmaj': bmaj, 'bmin': bmin, 'bpa': bpa, 'path': image})
            msg(f"  {Path(image).name}: BMAJ={bmaj:.6f}, BMIN={bmin:.6f}, BPA={bpa:.2f} deg")
        else:
            msg(f"  {Path(image).name}: no valid beam (BMAJ={bmaj}, BMIN={bmin}); skipped")
    if not beams:
        raise ValueError("No valid beams found in the selected images")
    return beams


def second_moments(beams):
    """
    Second-moment matrix of each beam, (n, 2, 2) in deg^2, in (East, North).

    Sigma = R diag(BMAJ^2, BMIN^2) R^T with the major axis along (sin BPA,
    cos BPA): orientation counts as far as the beam is elongated, and BPA
    and BPA + 180 deg are the same beam.
    """
    pa = np.radians([b['bpa'] for b in beams])
    major = np.stack([np.sin(pa), np.cos(pa)], axis=-1)
    minor = np.stack([-np.cos(pa), np.sin(pa)], axis=-1)
    a2, b2 = np.array([b['bmaj'] for b in beams]) ** 2, np.array([b['bmin'] for b in beams]) ** 2
    return a2[:, None, None] * major[:, :, None] * major[:, None, :] + b2[:, None, None] * minor[:, :, None] * minor[:, None, :]


def select_beams(beams, min_epochs, growth, seed=DEFAULT_SEED):
    """
    The epochs to stack: a core of similar beams, then any others that barely widen its common beam.

    With fewer than min_epochs beams every one is stacked. Otherwise the
    core is the set of min_epochs beams, a seed and its nearest neighbours in
    second-moment distance (the Frobenius norm of the difference over the
    median BMAJ^2), whose common beam has the smallest area. Further beams
    join while the common beam's area stays within (1 + growth) of the
    core's: a beam inside the common beam always, as it costs no resolution,
    then each time the one that widens it least.

    Returns
    -------
    kept : list of beams
    note : str
        How the selection went, for the log and the FITS history.
    """
    if len(beams) < min_epochs:
        return beams, f"all {len(beams)} beams, fewer than BOOKKEEPING.stack_min_epochs ({min_epochs})"

    sigma = second_moments(beams).reshape(len(beams), 4)
    scale = np.median([b['bmaj'] for b in beams]) ** 2
    distance = np.linalg.norm(sigma[:, None, :] - sigma[None, :, :], axis=-1) / scale

    def area(members):
        target, _ = common_beam([beams[i] for i in members], seed=seed, quiet=True)
        return target[0] * target[1]

    cores = [tuple(sorted(np.argsort(distance[i], kind='stable')[:min_epochs])) for i in range(len(beams))]
    cores = list(dict.fromkeys(cores))
    core_areas = [area(c) for c in cores]
    kept = list(cores[int(np.argmin(core_areas))])
    core_area = min(core_areas)
    limit = (1.0 + growth) * core_area

    # Each candidate is tried against the current common beam alone: a beam enclosing both encloses every kept
    # beam, so the trial area is an upper bound, and the exact common beam is found once the set is chosen.
    target, _ = common_beam([beams[i] for i in kept], seed=seed, quiet=True)
    current = core_area
    inside = widened = 0
    rest = [i for i in range(len(beams)) if i not in kept]
    while rest:
        held = {'bmaj': target[0], 'bmin': target[1], 'bpa': target[2]}
        trials = [common_beam([held, beams[i]], seed=seed, quiet=True)[0] for i in rest]
        areas = np.array([t[0] * t[1] for t in trials])
        best = int(np.argmin(areas))
        if areas[best] > limit:
            break
        if areas[best] <= current * (1.0 + 1e-3):
            inside += 1
        else:
            widened += 1
        target, current = trials[best], areas[best]
        kept.append(rest.pop(best))

    note = (f"core of {min_epochs} beams with the smallest common beam, then {inside} beams inside it and "
            f"{widened} that widened its area by {100 * (current / core_area - 1):.1f}% "
            f"(BOOKKEEPING.stack_beam_growth {100 * growth:g}%); {len(kept)} of {len(beams)} beams")
    msg(f"Beam selection: {note}")
    return [beams[i] for i in kept], note


# ============================================================================
# COMMON BEAM
# ============================================================================

def common_beam(beams, seed=DEFAULT_SEED, quiet=False):
    """
    Smallest beam enclosing every input beam.

    Each beam outline is sampled at its 1-sigma extent; the minimum enclosing
    ellipse of the convex hull of all samples is the target beam. Every input
    beam can then be convolved to it.

    Returns
    -------
    target : (bmaj, bmin, bpa) in degrees
    geometry : dict
        Sampled points, hull points and the ellipse, for plotting.
    """
    solver = WelzlEllipse(seed=seed)
    outlines = [solver.sample_ellipse((b['bmaj'] * FWHM_TO_SIGMA, b['bmin'] * FWHM_TO_SIGMA,
                                       np.radians(b['bpa']) + 0.5 * np.pi),
                                      num_pts=ELLIPSE_SAMPLES, endpoint=False)
                for b in beams]
    points = np.vstack(outlines)
    hull_points = points[ConvexHull(points).vertices]

    ellipse = solver.compute_minimum_enclosing_ellipse(hull_points)
    if ellipse is None:
        raise RuntimeError("Failed to compute the minimum enclosing ellipse of the beams")
    centre, semi_major, semi_minor, rotation = ellipse

    target = (semi_major / FWHM_TO_SIGMA, semi_minor / FWHM_TO_SIGMA, np.degrees(rotation - 0.5 * np.pi))
    if not quiet:
        n_inside = sum(solver.is_point_inside(pt, ellipse) for pt in points)
        msg(f"Common beam: BMAJ={target[0]:.6f}, BMIN={target[1]:.6f}, BPA={target[2]:.2f} deg "
            f"({n_inside}/{len(points)} beam outline points enclosed)")

    outline = solver.sample_ellipse((semi_major, semi_minor, rotation), num_pts=100) + centre
    return target, {'outlines': outlines, 'hull': hull_points, 'ellipse': outline, 'centre': centre}


# ============================================================================
# CONVOLUTION TO THE COMMON BEAM
# ============================================================================

def _n_workers(resources, ny, nx, n_tasks, safety_factor):
    """Processes for per-image work, bounded by CPU cores, usable memory and the number of tasks."""
    per_image_gb = ny * nx * 4 * safety_factor / 1024 ** 3
    memory_limited = int(resources['usable_memory_gb'] / per_image_gb) if per_image_gb > 0 else n_tasks
    return max(1, min(resources['optimal_processes_cpu'], memory_limited, n_tasks))


def _beam_psf(bmaj, bmin, bpa, deg_per_pix):
    """Peak-normalised Gaussian PSF image for a beam in degrees."""
    psf = Gaussian2DKernel(x_stddev=bmaj / deg_per_pix * FWHM_TO_SIGMA,
                           y_stddev=bmin / deg_per_pix * FWHM_TO_SIGMA,
                           theta=np.radians(bpa) + 0.5 * np.pi,
                           x_size=PSF_SIZE, y_size=PSF_SIZE, mode='center').array
    return psf / np.amax(psf)


def _pypher_command():
    """The pypher executable: beside the running Python (an unactivated venv), else on PATH."""
    beside = Path(sys.executable).parent / 'pypher'
    return str(beside) if beside.exists() else (shutil.which('pypher') or 'pypher')


def _kernel_task(beam, target):
    """
    Convolution kernel taking one image's beam to the target beam, made with pypher.

    Runs in a worker process. Returns (image_path, kernel_path or None, log message).
    """
    image = beam['path']
    name = Path(image).name.replace('-image.fits', '')
    kernel_file = KERNELS_DIR / f'{name}-kernel.fits'
    psf_in = KERNELS_DIR / f'{name}-psf-in.fits'
    psf_out = KERNELS_DIR / f'{name}-psf-out.fits'
    try:
        header = fits.getheader(image)
        deg_per_pix = abs(header['CDELT1'])
        psf_header = header.copy()
        psf_header['NAXIS1'] = psf_header['NAXIS2'] = PSF_SIZE
        fits.writeto(psf_in, _beam_psf(beam['bmaj'], beam['bmin'], beam['bpa'], deg_per_pix),
                     psf_header, overwrite=True)
        fits.writeto(psf_out, _beam_psf(*target, deg_per_pix), psf_header, overwrite=True)
        subprocess.run([_pypher_command(), str(psf_in), str(psf_out), str(kernel_file)],
                       check=True, capture_output=True, text=True)
        return image, str(kernel_file), f"Kernel made for {name}"
    except subprocess.CalledProcessError as e:
        return image, None, f"pypher failed for {name}: {e.stderr.strip() or e}"
    except Exception as e:
        return image, None, f"Kernel failed for {name}: {e}"
    finally:
        psf_in.unlink(missing_ok=True)
        psf_out.unlink(missing_ok=True)


def _convolve_task(image, kernel_file, target):
    """
    One image convolved with its kernel and written with the target beam in its header.

    Runs in a worker process. Returns (image_path, output_path or None, log message).
    """
    name = Path(image).name.replace('-image.fits', '')
    out_file = HOMOGENISED_DIR / f'{name}_homogenised.fits'
    data, kernel = get_image(image), get_image(kernel_file)
    if data is None or kernel is None:
        return image, None, f"Could not load the image or kernel for {name}"
    try:
        convolved = scipy.signal.fftconvolve(data, kernel, mode='same')
        header = fits.getheader(image)
        header['BMAJ'], header['BMIN'], header['BPA'] = target
        header['HISTORY'] = f'Convolved to the common beam with {Path(kernel_file).name}'
        create_fits(convolved.astype(np.float32), header, str(out_file))
        return image, str(out_file), f"Convolved {name}"
    except Exception as e:
        return image, None, f"Convolution failed for {name}: {e}"


def _run_parallel(task, jobs, n_workers, label):
    """Run task(*job) for each job across processes; return {image_path: output_path} for successes."""
    outputs = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(task, *job) for job in jobs]
        for n_done, future in enumerate(as_completed(futures), 1):
            image, output, message = future.result()
            msg(f"  {message} ({n_done}/{len(jobs)})")
            if output is not None:
                outputs[image] = output
    msg(f"{label}: {len(outputs)} of {len(jobs)} succeeded")
    return outputs


def homogenise_images(beams, target, resources):
    """
    Convolve each image to the target beam, in parallel.

    Returns the convolved image paths in time order.
    """
    for d in (KERNELS_DIR, HOMOGENISED_DIR):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()

    header = fits.getheader(beams[0]['path'])
    ny, nx = header['NAXIS2'], header['NAXIS1']

    n_workers = _n_workers(resources, ny, nx, len(beams), safety_factor=1.5)
    msg(f"Making convolution kernels with {n_workers} process(es)...")
    kernels = _run_parallel(_kernel_task, [(b, target) for b in beams], n_workers, "Kernels")
    if not kernels:
        raise RuntimeError("No convolution kernels could be made")

    images = sorted(kernels)
    n_workers = _n_workers(resources, ny, nx, len(images), safety_factor=4.0)   # FFT buffers need headroom
    msg(f"Convolving {len(images)} images with {n_workers} process(es)...")
    convolved = _run_parallel(_convolve_task, [(im, kernels[im], target) for im in images], n_workers,
                              "Convolution")
    if not convolved:
        raise RuntimeError("No images could be convolved to the common beam")
    return [convolved[im] for im in images if im in convolved]


# ============================================================================
# MEDIAN STACK
# ============================================================================

def _median_rows(images, y0, y1):
    """Pixel-wise median over images of rows y0:y1, read through memory maps."""
    block = None
    for i, image in enumerate(images):
        with fits.open(image, memmap=True) as hdul:
            data = hdul[0].data
            while data.ndim > 2:
                data = data[0]
            if block is None:
                block = np.empty((len(images), y1 - y0, data.shape[1]), dtype=np.float32)
            block[i] = data[y0:y1]
    return bn.median(block, axis=0)


def median_stack(images, resources):
    """
    Pixel-wise median of same-sized images, computed in row blocks.

    Blocks are sized so that all worker processes together stay within
    resources['usable_memory_gb'], and are processed in parallel. The result
    is identical to the median of the full stack held in memory.

    Parameters
    ----------
    images : list of str
    resources : dict
        From utils.get_system_resources.

    Returns
    -------
    ndarray, float32, shape (ny, nx)
    """
    with fits.open(images[0], memmap=True) as hdul:
        ny, nx = hdul[0].data.shape[-2:]

    n_workers = max(1, resources['optimal_processes_cpu'])
    bytes_per_row = len(images) * nx * 4 * 2          # the block plus bottleneck's working copy
    budget = resources['usable_memory_gb'] * 1024 ** 3 / n_workers
    rows = int(min(ny, max(1, budget // bytes_per_row)))
    blocks = [(y0, min(ny, y0 + rows)) for y0 in range(0, ny, rows)]
    n_workers = min(n_workers, len(blocks))
    msg(f"Median of {len(images)} images ({ny} x {nx}) in {len(blocks)} block(s) of up to {rows} rows "
        f"with {n_workers} process(es)")

    result = np.empty((ny, nx), dtype=np.float32)
    if n_workers == 1:
        for y0, y1 in blocks:
            result[y0:y1] = _median_rows(images, y0, y1)
        return result

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_median_rows, images, y0, y1): (y0, y1) for y0, y1 in blocks}
        for future in as_completed(futures):
            y0, y1 = futures[future]
            result[y0:y1] = future.result()
    return result


# ============================================================================
# WORKFLOW
# ============================================================================

def homogenise_and_stack_images(config, resources):
    """
    Build the stacked deep image for a project.

    Steps: find the epoch images, keep one frequency band, apply the
    BOOKKEEPING.stacked_image selection, choose a similar set of beams
    (select_beams, BOOKKEEPING.stack_min_epochs and stack_beam_growth),
    convolve to their common beam, and take the pixel-wise median.

    Parameters
    ----------
    config : dict
        Parsed astkat.yaml (BOOKKEEPING.image_dir, project_name, stacked_image, seed).
    resources : dict
        From utils.get_system_resources.

    Writes
    ------
    {image_dir}/{project_name}_stacked.fits
    plots/{project_name}_common_beam.pdf and .png
    """
    bookkeeping = config['BOOKKEEPING']
    image_dir = bookkeeping['image_dir']
    project_name = bookkeeping.get('project_name', 'astkat')
    title("AstKAT beam homogenisation and stacking")

    # One frequency band, then the user's selection of images.
    images = select_single_band(find_epoch_images(image_dir, project_name))
    indices, selection = parse_image_selection(bookkeeping.get('stacked_image', True), len(images))
    images = [images[i] for i in indices]
    if not images:
        raise ValueError("No images selected for stacking")
    msg(f"Selection: {selection} ({len(images)} images)")

    # Beams of different shape or orientation widen the common beam, so only a similar set is stacked.
    msg("Reading beams...")
    beams = read_beams(images)
    kept, filter_note = select_beams(beams, int(bookkeeping['stack_min_epochs']),
                                     float(bookkeeping['stack_beam_growth']), seed=bookkeeping.get('seed', DEFAULT_SEED))
    kept = sorted(kept, key=lambda b: b['path'])

    # The smallest beam that every kept beam can be convolved to.
    target, geometry = common_beam(kept, seed=bookkeeping.get('seed', DEFAULT_SEED))
    common_beam_figure(beams, kept, target, geometry, project_name)

    # Convolve each image to the common beam, then take the pixel-wise median, which rejects
    # transients and artefacts present in only a few epochs.
    convolved = homogenise_images(kept, target, resources)
    stacked = median_stack(convolved, resources)

    header = fits.getheader(convolved[0])
    header['HISTORY'] = f'Median of {len(convolved)} images convolved to a common beam'
    header['HISTORY'] = f'Beam filter: {filter_note}'
    header['NSTACKED'] = len(convolved)
    output_file = Path(image_dir) / f'{project_name}_stacked.fits'
    create_fits(stacked, header, str(output_file))

    # Kernels and convolved images are temporary.
    for d in (KERNELS_DIR, HOMOGENISED_DIR):
        shutil.rmtree(d, ignore_errors=True)

    title("AstKAT stacking complete")
    msg(f"  Selected: {len(images)} images ({selection}); beams kept: {len(kept)}; stacked: {len(convolved)}")
    msg(f"  Common beam: BMAJ={target[0]:.6f}, BMIN={target[1]:.6f}, BPA={target[2]:.2f} deg")
    msg(f"  Output: {output_file}")


def main():
    with open(Path('astkat_files') / 'astkat.yaml', 'r') as f:
        project_config = yaml.safe_load(f)
    homogenise_and_stack_images(project_config, load_system_resources())


if __name__ == "__main__":
    main()
