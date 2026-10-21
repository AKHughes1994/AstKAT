"""
AstKAT plotting library.

Every figure the pipeline makes, in one place with one publication style.
Pipeline scripts compute their results and call these functions; nothing
here computes a fit quantity. Figures are written to plots/ as PDF and PNG.

Used by
    stacking.py   common_beam_figure
    fit.py        fit_diagnostics
    offsets.py    offsets_figures
    target.py     target_figure
    simulate.py   simulation_noise_figure
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from astropy.time import Time

try:
    from .utils import msg, S_CODES
except ImportError:
    from utils import msg, S_CODES


PLOTS_DIR = Path('plots')
SAVE_FORMATS = ('pdf', 'png')
DPI = 300

EPOCH_FORMATS = ('png',)        # per-epoch figures, one per epoch; add 'pdf' for vector copies
DATA_COLOUR = 'black'           # measurements and binned data
MODEL_COLOUR = 'C0'             # fitted model, its bands and reference lines
MUTED_COLOUR = '0.45'           # background points and secondary data
HIGHLIGHT_COLOUR = 'C3'         # values flagged for attention
AXIS_COLOURS = ('C3', 'C0')     # right ascension (red) and declination (blue) components
AXIS_NAMES = ('RA', 'Dec')
COLOUR_MAP = 'Blues'            # sequential colour map; light points get a black edge to stay visible
GROUP_COLOURS = {'S': DATA_COLOUR, 'C or M': HIGHLIGHT_COLOUR}   # PyBDSF S_Code groups


# ============================================================================
# STYLE AND OUTPUT
# ============================================================================

def set_style():
    """Publication style: serif STIX fonts, inward ticks on all sides with minor ticks, no legend frame."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.linewidth': 1.0,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 6, 'ytick.major.size': 6,
        'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'xtick.minor.width': 0.7, 'ytick.minor.width': 0.7,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'legend.frameon': False,
        'legend.fontsize': 10,
        'savefig.bbox': 'tight',
    })


def save_figure(fig, stem, formats=SAVE_FORMATS, quiet=False):
    """Save a figure as plots/{stem} in each format and close it; stem may include a subdirectory."""
    for ext in formats:
        path = PLOTS_DIR / f'{stem}.{ext}'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=DPI)
        if not quiet:
            msg(f"Saved {path}")
    plt.close(fig)


# ============================================================================
# STACKING
# ============================================================================

def common_beam_figure(beams, kept, target, geometry, project_name):
    """
    The common beam of the stacked image and the beams it encloses.

    Left: outlines of the kept beams at their 1-sigma extent, their convex
    hull and the common beam, on the sky (right ascension increasing to the left). Right: BMAJ
    against BMIN of all beams, with the kept ones and the common beam marked.
    Legends sit above the panels.

    Parameters
    ----------
    beams, kept : list of dict
        All beams and the beams used, each with 'bmaj' and 'bmin' in degrees.
    target : tuple
        Common beam (bmaj, bmin, bpa) in degrees.
    geometry : dict
        From stacking.common_beam: 'outlines', 'hull' and 'ellipse' in degrees.
    project_name : str
        Used in the file name, plots/{project_name}_common_beam.
    """
    set_style()
    fig, (ax_sky, ax_size) = plt.subplots(1, 2, figsize=(10.0, 4.6), gridspec_kw={'wspace': 0.3})

    # Outlines are sampled with x pointing West; (-x, y) is (RA, Dec) in arcsec.
    to_sky = np.array([-3600.0, 3600.0])
    for i, pts in enumerate(geometry['outlines']):
        closed = np.vstack([pts, pts[:1]]) * to_sky
        ax_sky.plot(closed[:, 0], closed[:, 1], color=MUTED_COLOUR, lw=0.6, alpha=0.6,
                    label='Kept beams (1$\\sigma$)' if i == 0 else None)
    hull = np.vstack([geometry['hull'], geometry['hull'][:1]]) * to_sky
    ax_sky.plot(hull[:, 0], hull[:, 1], '--', color=DATA_COLOUR, lw=0.9, label='Convex hull')
    ellipse = geometry['ellipse'] * to_sky
    ax_sky.plot(ellipse[:, 0], ellipse[:, 1], '-', color=MODEL_COLOUR, lw=1.8, label='Common beam')
    ax_sky.set_aspect('equal')
    ax_sky.invert_xaxis()
    ax_sky.set_xlabel('Right ascension offset (arcsec)')
    ax_sky.set_ylabel('Declination offset (arcsec)')
    # Legends sit above their panels so they never cover the beams.
    ax_sky.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)

    ax_size.scatter([b['bmaj'] * 3600 for b in beams], [b['bmin'] * 3600 for b in beams], s=14,
                    facecolors='none', edgecolors=MUTED_COLOUR, linewidths=0.8, label='All beams')
    ax_size.scatter([b['bmaj'] * 3600 for b in kept], [b['bmin'] * 3600 for b in kept], s=14,
                    color=DATA_COLOUR, linewidths=0, label='Kept beams')
    ax_size.axvline(target[0] * 3600, color=MODEL_COLOUR, ls='--', lw=1.2)
    ax_size.axhline(target[1] * 3600, color=MODEL_COLOUR, ls='--', lw=1.2,
                    label=f'Common beam {target[0] * 3600:.2f}" $\\times$ {target[1] * 3600:.2f}", '
                          f'{target[2]:.1f}$^\\circ$')
    ax_size.set_xlabel('BMAJ (arcsec)')
    ax_size.set_ylabel('BMIN (arcsec)')
    ax_size.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)

    save_figure(fig, f'{project_name}_common_beam')


# ============================================================================
# FIT
# ============================================================================

OUTLIER_THRESHOLD = 0.5         # measurements above this outlier probability are shown as outliers
BEAM_FRACTION = 0.1             # reference line at this fraction of the beam FWHM
POINT_SIZE = 5                  # marker size of the inlier points
POINT_ALPHA = 0.15              # their opacity for 10,000 points, scaled as 1 / sqrt(N) and kept in 0.05-0.3


def point_alpha(n_points):
    """Opacity for a cloud of n_points, so its dense core shows structure instead of saturating."""
    return float(np.clip(POINT_ALPHA * np.sqrt(1e4 / max(n_points, 1)), 0.05, 0.3))


def fit_model_figure(results, residual='chi'):
    """
    Astrometric error against SNR in beam units: the data, the fitted error model and the residuals.

    Each measurement's offset along each beam axis is divided by that axis's
    beam scale, so the model applies to all measurements alike. Top: every
    measurement's radial offset r at the posterior median (grey; outliers
    coloured by their outlier probability; r is Rayleigh distributed with
    scale sigma, so the cloud sits above sigma), the per-axis scatter of the
    inliers in SNR bins with standard errors, taken about each posterior draw
    and shown as the median over draws (see fit.realised_scatter), the fitted
    model with its 99% posterior band, the fitted floor B and a line at a
    tenth of the beam FWHM. An absolute B differs between beams and is drawn
    for the median beam, which the legend states. Bottom, each bin against
    the model for its own measurements, draw by draw, as the median over
    draws: with residual 'chi', (realised - model) / standard error, written
    to plots/fit_model_vs_data; with residual 'fraction', model / realised - 1
    in per cent, how far the model sits from the empirical scatter, written
    to plots/fit_model_vs_data_fraction.

    With errors inflated after the fit (MCMC.error_inflation, factor eta > 1),
    the inflated model eta times the fitted one is drawn dashed, and the
    fraction panel marks 100 (1 / eta - 1) per cent: bins at or above that
    line lie within the inflated errors. The chi panel stays against the
    fitted model.
    """
    diag = results['diagnostics']
    snr, radial = results['data']['snr'], diag['radial']
    prob = np.asarray(results['p_outlier'])
    is_outlier = prob > OUTLIER_THRESHOLD
    bins, curve = diag['snr_bins'], diag['model_curve']

    floor = float(np.median(curve['floor']))
    if curve['representative']:
        b_mas = results['parameters']['B']['median'] * 1e3
        floor_label = f'Fitted $B = {b_mas:.1f}$ mas (median beam)'
        msg("WARNING: the floor B is absolute, so its line and the model curve are drawn for the median beam")
    else:
        floor_label = rf'Fitted $B = {floor:.3f}\,\sigma_{{\rm beam}}$'

    fig = plt.figure(figsize=(7.0, 6.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 0.03], hspace=0.0, wspace=0.03)
    ax = fig.add_subplot(grid[0, 0])
    ax_chi = fig.add_subplot(grid[1, 0], sharex=ax)
    cax = fig.add_subplot(grid[0, 1])

    ax.scatter(snr[~is_outlier], radial[~is_outlier], s=POINT_SIZE, color=MUTED_COLOUR,
               alpha=point_alpha(np.count_nonzero(~is_outlier)), linewidths=0, rasterized=True, zorder=1,
               label=r'Radial offset $r$')
    outliers = ax.scatter(snr[is_outlier], radial[is_outlier], s=16, c=prob[is_outlier], cmap='Blues',
                          vmin=OUTLIER_THRESHOLD, vmax=1.0, edgecolors='navy', linewidths=0.4, zorder=3)
    cbar = fig.colorbar(outliers, cax=cax)
    cbar.set_label('Outlier probability', rotation=270, labelpad=14)
    cbar.ax.tick_params(which='both', direction='in')

    ax.errorbar(bins['snr_median'], bins['observed_rms'], yerr=bins['observed_rms_err'], fmt='o', ms=4,
                color=DATA_COLOUR, ecolor=DATA_COLOUR, elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=5,
                label='S/N-binned error')
    lo, mid, hi = np.percentile(curve['rms'], [0.5, 50, 99.5], axis=0)
    ax.fill_between(curve['snr'], lo, hi, color=MODEL_COLOUR, alpha=0.3, linewidth=0, zorder=4)
    ax.plot(curve['snr'], mid, '-', color=MODEL_COLOUR, lw=1.5, zorder=4, label='Fitted error model')
    eta = results['model']['error_inflation']
    if eta > 1:
        ax.plot(curve['snr'], eta * mid, '--', color=MODEL_COLOUR, lw=1.3, zorder=4,
                label=rf'Inflated error model ($\times{eta:.2f}$)')
    ax.axhline(floor, color=MODEL_COLOUR, ls=':', lw=1.3, zorder=4, label=floor_label)
    ax.axhline(curve['beam_line'], color=DATA_COLOUR, ls='--', lw=1.2, zorder=4,
               label=f'{BEAM_FRACTION * 100:.0f}% synthesised beam')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'Astrometric error per axis $\sigma$ (beam units)')
    ax.set_xlim(snr.min() * 0.8, snr.max() * 1.2)
    positive = radial[radial > 0]
    ax.set_ylim(max(positive.min(), 0.1 * mid.min()), positive.max() * 1.2)
    ax.tick_params(labelbottom=False)
    legend = ax.legend(loc='lower left')
    # The offsets are drawn faint to show the density; their legend marker is drawn solid and larger.
    for handle, text in zip(legend.legend_handles, legend.get_texts()):
        if text.get_text() == r'Radial offset $r$':
            handle.set_alpha(1.0)
            handle.set_sizes([20])

    if residual == 'chi':
        value, error = bins['chi'], None
        limit, label, stem = 3.0, r'$\chi$', 'fit_model_vs_data'
    else:
        value, error = bins['fraction'], bins['fraction_err']
        limit, label, stem = 5.0, 'Model / data $-$ 1 (%)', 'fit_model_vs_data_fraction'
    ax_chi.errorbar(bins['snr_median'], value, yerr=error, fmt='o', ms=4, color=DATA_COLOUR, ecolor=DATA_COLOUR,
                    elinewidth=1.0, capsize=2.5, capthick=1.0)
    ax_chi.axhline(0, color=DATA_COLOUR, ls='--', lw=1.0)
    reach = np.abs(value) + (0 if error is None else error)
    covered = 100.0 * (1.0 / eta - 1.0)
    if residual == 'fraction' and eta > 1:
        # The inflated model sits eta times higher, so bins at or above this line lie within the inflated errors.
        ax_chi.axhline(covered, color=MODEL_COLOUR, ls='--', lw=1.3,
                       label=rf'Inflated error model ($\times{eta:.2f}$)')
        ax_chi.legend(loc='best', fontsize='small')
        reach = np.append(reach, abs(covered))
    lim = max(limit, 1.2 * np.nanmax(reach))
    ax_chi.set_ylim(-lim, lim)
    ax_chi.set_xlabel('Signal-to-noise ratio')
    ax_chi.set_ylabel(label)

    save_figure(fig, stem)


def fit_model_figures(results):
    """The model-versus-data figure with chi residuals and with fractional residuals (see fit_model_figure)."""
    fit_model_figure(results, 'chi')
    fit_model_figure(results, 'fraction')


def fit_axes_figure(results):
    """
    The error along the beam's minor and major axes, observed against the model, by SNR.

    Top: the inliers' scatter along each axis in beam units per SNR bin, with
    standard errors, and the model's prediction for each axis with its 68%
    posterior band. Bottom: the scatter of the whitened residuals along each
    axis, which is 1 when the model scales both axes correctly. Written to
    plots/fit_axes.
    """
    bins = results['diagnostics']['snr_bins']
    snr = bins['snr_median']
    axes_style = (('Minor axis', DATA_COLOUR, '-'), ('Major axis', MUTED_COLOUR, '--'))
    style = dict(fmt='.', ms=3, elinewidth=0.7, capsize=0)

    fig, (ax, ax_w) = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 2], 'hspace': 0.0})
    for k, (label, colour, ls) in enumerate(axes_style):
        ax.fill_between(snr, bins['predicted_axes_p16'][:, k], bins['predicted_axes_p84'][:, k],
                        color=MODEL_COLOUR, alpha=0.2, linewidth=0, zorder=1)
        ax.plot(snr, bins['predicted_axes_p50'][:, k], ls, color=POSTERIOR_LINE_COLOUR, lw=1.2, zorder=2,
                label=f'Model, {label.lower()}')
        ax.errorbar(snr, bins['observed_axes'][:, k], yerr=bins['observed_axes_err'][:, k], color=colour,
                    ecolor=colour, zorder=3, label=f'Observed, {label.lower()}', **style)
        ax_w.errorbar(snr, bins['whitened_std_axes'][:, k], yerr=bins['whitened_std_axes_err'][:, k],
                      color=colour, ecolor=colour, zorder=3, **style)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Scatter per axis\n(beam units)')
    ax.tick_params(labelbottom=False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=8, ncol=4)

    ax_w.axhline(1.0, color='0.4', lw=0.8, ls='--', zorder=0)
    spread = np.nanmax(np.abs(bins['whitened_std_axes'] - 1) + bins['whitened_std_axes_err'])
    ax_w.set_ylim(1 - 1.2 * spread, 1 + 1.2 * spread)
    ax_w.set_ylabel('Whitened scatter')
    ax_w.set_xlabel('Signal-to-noise ratio')

    save_figure(fig, 'fit_axes')


def _epoch_axis(trans):
    """Observation date of each epoch in MJD, or the epoch number when times are unavailable."""
    obs_isot = trans.get('obs_isot')
    if obs_isot and all(t not in (None, 'unknown') for t in obs_isot):
        return Time(list(obs_isot), format='isot', scale='utc').mjd, 'Observation epoch (MJD)'
    n = len(trans['median']) if 'median' in trans else len(trans['catalog_epochs'])
    return np.arange(n), 'Epoch'


ADDED_MARKER = '*'               # epochs added after the fit, in figures made by offsets.py
ADDED_LABEL = 'Added epochs'


def _translations_time(trans, stem, added=None):
    """
    Epoch translations against time, right ascension above declination, in mas.

    Posterior medians with 1-sigma errors, a thin line in time order as a
    guide for the eye, and the same symmetric limits on both panels. Epochs
    marked in `added` are drawn as stars. Written to plots/{stem}.
    """
    t, t_label = _epoch_axis(trans)
    median, sd = trans['median'] * 1e3, trans['sd'] * 1e3
    ok = np.all(np.isfinite(median), axis=1)
    added = np.zeros(len(t), dtype=bool) if added is None else np.asarray(added) & ok
    order = np.argsort(np.where(ok, t, np.inf))[:ok.sum()]
    labels = ['Right ascension offset\n' r'$\Delta\alpha\cos\delta$ (mas)',
              'Declination offset\n' r'$\Delta\delta$ (mas)']
    style = dict(color=DATA_COLOUR, ecolor=DATA_COLOUR, elinewidth=0.7, capsize=0, zorder=2)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True, gridspec_kw={'hspace': 0.0})
    for k, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t[order], median[order, k], '-', color='0.35', lw=0.7, zorder=1)
        fitted = ok & ~added
        ax.errorbar(t[fitted], median[fitted, k], yerr=sd[fitted, k], fmt='.', ms=3,
                    label='Fitted epochs' if k == 0 and added.any() else None, **style)
        if added.any():
            ax.errorbar(t[added], median[added, k], yerr=sd[added, k], fmt=ADDED_MARKER, ms=5,
                        label=ADDED_LABEL if k == 0 else None, **style)
        ax.axhline(0, color='0.4', lw=0.8, ls='--', zorder=0)
        ax.set_ylabel(label)
    if added.any():
        axes[0].legend(loc='upper right', fontsize=9, ncol=2)

    limit = 1.08 * np.max(np.abs(median[ok]) + sd[ok])
    for ax in axes:
        ax.set_ylim(-limit, limit)
    axes[-1].set_xlabel(t_label)

    save_figure(fig, stem)


def _translation_plane(trans, stem, added=None):
    """
    Epoch translations in the plane of the sky, with their full covariance, in mas.

    Each epoch is a point coloured by its date with its 1-sigma ellipse from
    the RA-Dec covariance; a thin line joins the epochs in time order.
    Epochs marked in `added` are drawn as stars. Right ascension increases to the left.
    Written to plots/{stem}.
    """
    t, t_label = _epoch_axis(trans)
    median, sd = trans['median'] * 1e3, trans['sd'] * 1e3
    ok = np.all(np.isfinite(median), axis=1)
    added = np.zeros(len(t), dtype=bool) if added is None else np.asarray(added) & ok
    order = np.argsort(np.where(ok, t, np.inf))[:ok.sum()]
    cov = trans['covariance']
    cmap = plt.get_cmap(COLOUR_MAP)
    norm = plt.Normalize(t[ok].min(), t[ok].max())

    fig = plt.figure(figsize=(6.0, 5.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 0.04], wspace=0.04)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])

    for e in np.nonzero(ok)[0]:
        # 1-sigma ellipse of the 2x2 RA-Dec block of this epoch.
        values, vectors = np.linalg.eigh(cov[e, :, e, :] * 1e6)
        angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
        ax.add_patch(Ellipse(median[e], 2 * np.sqrt(values[1]), 2 * np.sqrt(values[0]), angle=angle,
                             facecolor=cmap(norm(t[e])), edgecolor=DATA_COLOUR, linewidth=0.3,
                             alpha=0.35, zorder=2))
    ax.plot(median[order, 0], median[order, 1], '-', color='0.35', lw=0.6, zorder=1)
    fitted = ok & ~added
    ax.scatter(median[fitted, 0], median[fitted, 1], c=t[fitted], cmap=cmap, norm=norm, s=18,
                        edgecolors=DATA_COLOUR, linewidths=0.5, zorder=3, label='Fitted epochs')
    if added.any():
        ax.scatter(median[added, 0], median[added, 1], c=t[added], cmap=cmap, norm=norm, s=80,
                   marker=ADDED_MARKER, edgecolors=DATA_COLOUR, linewidths=0.5, zorder=4, label=ADDED_LABEL)
        ax.legend(loc='upper right', fontsize=9)
    ax.axhline(0, color='0.4', lw=0.8, ls='--', zorder=0)
    ax.axvline(0, color='0.4', lw=0.8, ls='--', zorder=0)

    limit = 1.1 * np.max(np.abs(median[ok]) + sd[ok])
    ax.set_xlim(limit, -limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.set_xlabel(r'Right ascension offset $\Delta\alpha\cos\delta$ (mas)')
    ax.set_ylabel(r'Declination offset $\Delta\delta$ (mas)')

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(t_label, rotation=270, labelpad=14)
    cbar.ax.tick_params(which='both', direction='in')

    save_figure(fig, stem)


def translations_time_figure(results):
    """Epoch translations of the fit against time; written to plots/fit_epoch_translations."""
    if results['epoch_offsets'] is not None:
        _translations_time(results['epoch_offsets'], 'fit_epoch_translations')


def translation_plane_figure(results):
    """Epoch translations of the fit in the plane of the sky; written to plots/fit_translation_plane."""
    if results['epoch_offsets'] is not None:
        _translation_plane(results['epoch_offsets'], 'fit_translation_plane')


# Scalar fit parameters in plotting order: name -> (label, multiplier, unit). B is set per fit.
PARAMETERS = {
    'A': (r'$A$', 1.0, ''),
    'beta': (r'$\beta$', 1.0, ''),
    'snr_break': (r'${\rm SNR}_b$', 1.0, ''),
    'snr_gamma': (r'$\gamma$', 1.0, ''),
    'B': (r'$B$', 1e3, 'mas'),
    'alpha': (r'$\alpha$', 1.0, ''),
    'outlier_fraction': (r'$f_{\rm out}$', 1.0, ''),
    'outlier_scale': (r'$\lambda$', 1.0, ''),
    'sigma_tau': (r'$\sigma_\tau$', 1e3, 'mas'),
    'peak_noise': (r'$k$', 1.0, ''),
    'flux_systematic': (r'$f_{\rm sys}$', 1.0, ''),
    'flux_fractional': (r'$f_{\rm frac}$', 1.0, ''),
}
POSTERIOR_COLOUR = 'steelblue'
POSTERIOR_LINE_COLOUR = 'darkblue'


def _fit_parameters(results):
    """Fitted scalar parameters present in the results, with labels and units for this fit."""
    params = dict(PARAMETERS)
    if results['model']['b_fractional']:
        params['B'] = (r'$B$', 1.0, 'beam')
    pivot = results['model'].get('snr_pivot', 1.0)
    if pivot != 1.0:
        params['A'] = (rf'$A_{{{pivot:g}}}$', 1.0, '')
    draws = results['posterior_draws']
    return [(name, *params[name]) for name in params if name in draws]


def _decimals(err_low, err_high):
    """Decimal places that show the smaller error to two significant figures."""
    err = min(err_low, err_high)
    return int(np.clip(1 - np.floor(np.log10(err)), 0, 6)) if err > 0 else 3


def corner_figure(results):
    """
    Posterior of the scalar fit parameters as a corner plot.

    Posterior draws as points under filled density contours, 1D histograms
    with the 68% interval shaded, and the posterior median marked; for simulations the injected values are
    dashed in the highlight colour, each panel's range widened to include
    them. Each diagonal panel is titled with the median and its 68% errors.
    Written to plots/fit_corner.
    """
    try:
        import corner
    except ImportError:
        msg("corner is not installed (pip install corner); corner plot skipped")
        return

    params = _fit_parameters(results)
    draws = results['posterior_draws']
    samples = np.column_stack([np.ravel(draws[name]) * scale for name, _, scale, _ in params])
    labels = [label + (f' ({unit})' if unit else '') for _, label, _, unit in params]
    medians = np.median(samples, axis=0)
    truth = results.get('truth_parameters') or {}
    truths = [truth[name] * scale if name in truth else None for name, _, scale, _ in params]

    # Each range covers the posterior and, for a simulation, the injected value, so a truth far from the
    # posterior still shows at the edge of its panels.
    ranges = []
    for column, value in zip(samples.T, truths):
        low, high = np.percentile(column, [1, 99])
        pad = 0.1 * (high - low)
        low, high = max(low - pad, column.min()), min(high + pad, column.max())
        if value is not None:
            low, high = min(low, value - pad), max(high, value + pad)
        ranges.append((low, high))

    fig = corner.corner(
        samples, labels=labels, label_kwargs={'fontsize': 14}, quantiles=[0.16, 0.5, 0.84], show_titles=False,
        smooth=2.5, bins=50, hist_kwargs={'alpha': 0.8, 'color': POSTERIOR_COLOUR}, color=POSTERIOR_COLOUR,
        plot_datapoints=True, data_kwargs={'ms': 1.5, 'alpha': 0.35, 'color': POSTERIOR_LINE_COLOUR},
        plot_density=True,
        contour_kwargs={'colors': POSTERIOR_LINE_COLOUR, 'linewidths': 1.2},
        contourf_kwargs={'colors': ['lightblue', POSTERIOR_COLOUR], 'alpha': 0.6},
        range=ranges, truths=medians, truth_color=POSTERIOR_LINE_COLOUR,
        truth_kwargs={'linewidth': 1.5, 'alpha': 0.8})

    axes = np.array(fig.axes).reshape(len(params), len(params))
    for i, (_, label, _, unit) in enumerate(params):
        low, mid, high = np.percentile(samples[:, i], [16, 50, 84])
        p = _decimals(mid - low, high - mid)
        unit_text = f' {unit}' if unit else ''
        axes[i, i].set_title(f'{label}\n${mid:.{p}f}_{{-{mid - low:.{p}f}}}^{{+{high - mid:.{p}f}}}${unit_text}',
                             fontsize=11)
        axes[i, i].axvspan(low, high, alpha=0.2, color='lightblue', zorder=-1)

    # Injected values of a simulation, alongside the posterior medians.
    if truth:
        corner.overplot_lines(fig, truths, color=HIGHLIGHT_COLOUR, linewidth=1.2, linestyle='--')

    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.95, wspace=0.05, hspace=0.05)
    save_figure(fig, 'fit_corner')


def chains_figure(results):
    """
    Sampler chains of the scalar fit parameters.

    One row per parameter: every chain's trace against draw number, and
    beside it each chain's marginal distribution. Divergent draws are marked
    with ticks along the bottom of the trace; for simulations the injected
    value is dashed in the highlight colour. Written to plots/fit_chains.
    """
    params = _fit_parameters(results)
    draws = results['posterior_draws']
    diverging = np.asarray(draws['diverging'])
    n_chains, n_draws = diverging.shape
    colours = plt.get_cmap(COLOUR_MAP)(np.linspace(0.45, 0.95, n_chains))

    fig, axes = plt.subplots(len(params), 2, figsize=(7.0, 1.4 * len(params) + 0.6), sharey='row',
                             gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.0, 'hspace': 0.15}, squeeze=False)
    for row, (name, label, scale, unit) in enumerate(params):
        ax_trace, ax_hist = axes[row]
        values = np.asarray(draws[name]) * scale
        for c in range(n_chains):
            ax_trace.plot(np.arange(n_draws), values[c], lw=0.4, color=colours[c], alpha=0.8)
            ax_hist.hist(values[c], bins=40, orientation='horizontal', histtype='step', density=True,
                         color=colours[c], lw=0.9)
            divergent = np.nonzero(diverging[c])[0]
            if divergent.size:
                ax_trace.plot(divergent, np.full(divergent.size, 0.04), '|', color=HIGHLIGHT_COLOUR, ms=6,
                              transform=ax_trace.get_xaxis_transform())
        truth = (results.get('truth_parameters') or {}).get(name)
        if truth is not None:
            for ax in (ax_trace, ax_hist):
                ax.axhline(truth * scale, color=HIGHLIGHT_COLOUR, lw=1.0, ls='--', zorder=3)
        ax_trace.set_xlim(0, n_draws - 1)
        ax_trace.set_ylabel(label + (f' ({unit})' if unit else ''))
        ax_hist.tick_params(labelleft=False, labelbottom=False)
        if row < len(params) - 1:
            ax_trace.tick_params(labelbottom=False)
    axes[-1, 0].set_xlabel('Draw')

    save_figure(fig, 'fit_chains')


AFFINE_LABELS = (r'Scale $s$', r'Rotation $\omega$', r'Stretch $\gamma_1$', r'Shear $\gamma_2$')


def _affine(affine, trans, stem, added=None):
    """
    Residual field distortion of each epoch against time, in units of 1e-5.

    Scale, rotation, stretch and shear from each epoch's affine fit with
    1-sigma errors: noise-model weights in black, with epochs more than 3
    sigma from zero in the highlight colour, and equal weights in grey for
    comparison. Epochs marked in `added` are drawn as stars. The scale panel
    also shows the time-varying part of the predicted aberration scale, the
    signal the correction removed. Written to plots/{stem}.
    """
    t, t_label = _epoch_axis(trans)
    added = np.zeros(len(t), dtype=bool) if added is None else np.asarray(added)
    order = np.argsort(t)
    style = dict(elinewidth=0.6, capsize=0)

    def draw(ax, mask, v, err, colour, zorder, label=None):
        for marker, ms, sel in (('o', 1.5, mask & ~added), (ADDED_MARKER, 5, mask & added)):
            if sel.any():
                ax.errorbar(t[sel], v[sel], yerr=err[sel], fmt=marker, ms=ms, color=colour, ecolor=colour,
                            zorder=zorder, label=label if marker == 'o' else None, **style)

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 6.2), sharex=True, gridspec_kw={'hspace': 0.0})
    for k, (ax, label) in enumerate(zip(axes, AFFINE_LABELS)):
        limit = 0.0
        for weighting in ('equal', 'model'):
            v, err = affine[weighting]['terms'][:, k] * 1e5, affine[weighting]['errors'][:, k] * 1e5
            ok = np.isfinite(v) & np.isfinite(err)
            limit = max(limit, 1.1 * np.max(np.abs(v[ok]) + err[ok]))
            if weighting == 'equal':
                draw(ax, ok, v, err, MUTED_COLOUR, 1, 'Equal weights' if k == 0 else None)
                continue
            flagged = ok & (np.abs(v) > 3 * err)
            draw(ax, ok & ~flagged, v, err, DATA_COLOUR, 2, 'Noise-model weights' if k == 0 else None)
            draw(ax, flagged, v, err, HIGHLIGHT_COLOUR, 3)
        ax.axhline(0, color='0.4', lw=0.8, ls='--', zorder=0)

        aberration = affine['model']['aberration']
        if k == 0 and aberration is not None:
            varying = (aberration['predicted_scale'] - np.mean(aberration['predicted_scale'])) * 1e5
            ax.plot(t[order], varying[order], '-', color=POSTERIOR_LINE_COLOUR, lw=1.0, zorder=0,
                    label='Removed aberration scale')
            limit = max(limit, 1.1 * np.max(np.abs(varying)))
        if k == 0:
            # Legend above the first panel, outside the plot area.
            if added.any():
                ax.errorbar([], [], fmt=ADDED_MARKER, ms=5, color=DATA_COLOUR, label=ADDED_LABEL)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=8, ncol=4 if added.any() else 3)

        ax.set_ylim(-limit, limit)
        ax.set_ylabel(label + '\n($10^{-5}$)')
    axes[-1].set_xlabel(t_label)

    save_figure(fig, stem)


def _affine_ellipses(affine, trans, stem, added=None):
    """
    Each epoch's residual field distortion drawn as a deformation ellipse.

    The unit circle mapped through I + g L, with L the epoch's fitted 2x2 map
    (noise-model weights) and g the exaggeration in the summary, coloured by
    date. Epochs marked in `added` are drawn dotted. A field with no residual
    distortion collapses onto the dashed unit circle; scale changes the size,
    stretch and shear flatten it, rotation turns it. Right ascension increases to the left.
    Written to plots/{stem}.
    """
    t, t_label = _epoch_axis(trans)
    added = np.zeros(len(t), dtype=bool) if added is None else np.asarray(added)
    L = affine['model']['L']
    ok = np.array([np.all(np.isfinite(m)) for m in L])
    # Blues without its near-white end, so every outline stays visible.
    cmap = ListedColormap(plt.get_cmap(COLOUR_MAP)(np.linspace(0.3, 1.0, 256)))
    norm = plt.Normalize(t[ok].min(), t[ok].max())
    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    circle = np.vstack([np.cos(angle), np.sin(angle)])

    fig = plt.figure(figsize=(6.0, 5.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 0.04], wspace=0.04)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])

    ax.plot(circle[0], circle[1], '--', color=DATA_COLOUR, lw=1.0, zorder=3)
    for e in np.argsort(t):
        if ok[e]:
            ra, dec = (np.eye(2) + affine['ellipse_gain'] * L[e]) @ circle
            ax.plot(ra, dec, ':' if added[e] else '-', color=cmap(norm(t[e])),
                    lw=1.0 if added[e] else 0.8, alpha=0.9, zorder=2)
    if added.any():
        ax.plot([], [], ':', color=cmap(0.8), lw=1.0, label=ADDED_LABEL)
        ax.legend(loc='upper right', fontsize=9)

    limit = 1.0 + 1.5 * affine['ellipse_distortion']
    ax.set_xlim(limit, -limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.set_xlabel('Right ascension')
    ax.set_ylabel('Declination')

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(t_label, rotation=270, labelpad=14)
    cbar.ax.tick_params(which='both', direction='in')

    save_figure(fig, stem)


def affine_figure(results):
    """Residual field distortion of the fitted epochs against time; written to plots/fit_affine."""
    _affine(results['diagnostics']['affine'], results['epochs'], 'fit_affine')


def affine_ellipses_figure(results):
    """Deformation ellipses of the fitted epochs; written to plots/fit_affine_ellipses."""
    _affine_ellipses(results['diagnostics']['affine'], results['epochs'], 'fit_affine_ellipses')


def snr_histograms_figure(results, unit='beam'):
    """
    Residuals about one posterior draw in broad SNR bins, against the model (see fit.snr_histograms).

    One panel per bin: histograms of the residual along the beam's minor axis
    (black) and major axis (grey), outliers included, with the fitted mixture
    expected for that bin's measurements as dashed lines of the same colours.
    Each panel's legend, above it, gives per axis the observed robust
    standard deviation (1.4826 MAD) and, in brackets, the inlier standard
    deviation the fitted model predicts; a caption at the top says so.
    unit is 'beam' (residual over sqrt(v'), beam units) or 'mas'.
    Written to plots/fit_snr_histograms or fit_snr_histograms_mas.
    """
    bins = results['diagnostics'].get('snr_histograms')
    if not bins:
        return
    n_cols = min(4, len(bins))
    n_rows = int(np.ceil(len(bins) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows), squeeze=False,
                             gridspec_kw={'hspace': 0.75, 'wspace': 0.25})
    label = 'Residual (beam units)' if unit == 'beam' else 'Residual (mas)'
    units = 'beam units' if unit == 'beam' else 'mas'
    precision = 3 if unit == 'beam' else 1

    for ax, b in zip(axes.ravel(), bins):
        d = b[unit]
        edges = np.linspace(d['grid'][0], d['grid'][-1], 41)
        width = edges[1] - edges[0]
        for k, (colour, name) in enumerate(((DATA_COLOUR, 'minor'), (MUTED_COLOUR, 'major'))):
            ax.hist(d['data'][:, k], bins=edges, histtype='step', color=colour, lw=1.0,
                    label=f"{name} {d['robust_sd'][k]:.{precision}f} ({d['predicted_sd'][k]:.{precision}f})")
            ax.plot(d['grid'], d['expected'][k] * width, '--', color=colour, lw=0.9)
        ax.tick_params(labelsize=8)
        # Legend above the panel, titled with the bin, so it never covers the histograms.
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7, handlelength=1.2,
                  columnspacing=1.0, title=f"SNR {b['lo']:.0f}-{b['hi']:.0f}, N = {b['n']}", title_fontsize=8)
    for ax in axes.ravel()[len(bins):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(label, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('Measurements', fontsize=9)
    fig.text(0.5, 1.0, f"Legends: beam axis, observed robust standard deviation (1.4826 MAD), and in brackets the "
                       f"inlier standard deviation the fitted model predicts, in {units}.\n"
                       f"Solid: residuals about the fitted true positions; dashed: the fitted mixture for the "
                       f"same measurements.", ha='center', va='bottom', fontsize=8)

    save_figure(fig, 'fit_snr_histograms' if unit == 'beam' else 'fit_snr_histograms_mas')


def peak_chi_figure(results):
    """
    Measured peaks against the fitted flux densities times the epoch flux scales, in units of their
    error sigma_p = sqrt((k rms)^2 + (f_frac S (1 + g_e))^2).

    Left: histogram of chi = (p - S (1 + g_e)) / sigma_p on a log scale against N(0, 1).
    Right: chi against SNR, PyBDSF S_Code S in grey, C or M in red. A Gaussian
    core with isolated extreme points means outliers; a spread that widens
    with SNR means an error in proportion to the peak. Written to
    plots/fit_peak_chi.
    """
    res = results['diagnostics'].get('peak_chi')
    if res is None:
        return
    chi, snr = res['chi'], res['snr']
    fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(10.0, 4.0), gridspec_kw={'wspace': 0.25})

    limit = max(6.0, np.percentile(np.abs(chi), 99.9))
    edges = np.linspace(-limit, limit, 81)
    ax_h.hist(np.clip(chi, -limit, limit), bins=edges, histtype='step', color=DATA_COLOUR, lw=1.0, density=True)
    grid = np.linspace(-limit, limit, 400)
    ax_h.plot(grid, np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi), '-', color=POSTERIOR_LINE_COLOUR, lw=1.2,
              label=r'$\mathcal{N}(0, 1)$')
    ax_h.set_yscale('log')
    ax_h.set_ylim(1e-5, 1.0)
    ax_h.set_xlabel(r'$\chi = [p - S(1 + g_e)] / \sigma_p$')
    ax_h.set_ylabel('Density')
    ax_h.legend(loc='upper right', fontsize=8,
                title=f"beyond 3: {res['beyond_3'] * 100:.2f}%, beyond 5: {res['beyond_5'] * 100:.2f}%",
                title_fontsize=8)

    # Single Gaussians faint in the background, so C and M detections stand out.
    single = res['s_code'] == S_CODES.index('S')
    ax_s.plot(snr[single], chi[single], '.', ms=2, color=MUTED_COLOUR, alpha=point_alpha(single.sum()),
              rasterized=True, label=f"S ({single.sum()})")
    if (~single).any():
        ax_s.plot(snr[~single], chi[~single], '.', ms=4, color=GROUP_COLOURS['C or M'], rasterized=True,
                  label=f"C or M ({(~single).sum()})")
    for level in (-5, 0, 5):
        ax_s.axhline(level, color='0.4', lw=0.8, ls='--' if level else '-', zorder=0)
    ax_s.set_xscale('log')
    ax_s.set_yscale('symlog', linthresh=5)
    ax_s.set_xlabel('Signal-to-noise ratio (fitted)')
    ax_s.set_ylabel(r'$\chi = [p - S(1 + g_e)] / \sigma_p$')
    ax_s.legend(loc='upper left', fontsize=8)

    save_figure(fig, 'fit_peak_chi')


def morphology_figure(results):
    """
    Detections split by PyBDSF S_Code, S against C or M, in broad SNR bins.

    Top: fraction of each group. Middle: whitened position scatter, std
    (filled, solid) and robust sd (open, dashed); 1 means the model fits.
    Bottom: fraction of position outliers (p_outlier > 0.5, solid) and of
    peaks beyond 5 sigma_p (dashed). Written to plots/fit_morphology.
    """
    m = results['diagnostics']['morphology']
    if m is None:
        return
    snr = m['snr_median']
    fig, (ax_f, ax_w, ax_o) = plt.subplots(3, 1, figsize=(6.0, 8.0), sharex=True, gridspec_kw={'hspace': 0.08})
    for c, code in enumerate(m['groups']):
        if not np.nansum(m['count'][c]):
            continue
        colour = GROUP_COLOURS[code]
        ax_f.plot(snr, m['fraction'][c], 'o-', color=colour, ms=4, lw=1.0,
                  label=f"{code}: {m['share_all'][c] * 100:.1f}% of detections")
        ax_w.plot(snr, m['std'][c], 'o-', color=colour, ms=4, lw=1.0, label=f"{code} std")
        ax_w.plot(snr, m['robust_sd'][c], 'o--', color=colour, ms=4, lw=1.0, mfc='none', label=f"{code} robust sd")
        ax_o.plot(snr, m['outlier'][c], 'o-', color=colour, ms=4, lw=1.0, label=f"{code} position outliers")
        ax_o.plot(snr, m['peak_extreme'][c], 'o--', color=colour, ms=4, lw=1.0, mfc='none',
                  label=f"{code} peaks beyond 5")
    ax_w.axhline(1.0, color='0.4', lw=0.8, zorder=0)
    ax_f.set_ylabel('Fraction of detections')
    ax_w.set_ylabel('Whitened position scatter')
    ax_o.set_ylabel('Outlier fraction')
    ax_o.set_xlabel('Signal-to-noise ratio (fitted)')
    ax_o.set_xscale('log')
    ax_f.legend(loc='center left', fontsize=7)
    for ax in (ax_w, ax_o):
        ax.legend(loc='upper left', fontsize=7, ncol=2)

    save_figure(fig, 'fit_morphology')


def snr_histograms_figures(results):
    """The SNR-binned residual histograms in beam units and in mas (see snr_histograms_figure)."""
    snr_histograms_figure(results, 'beam')
    snr_histograms_figure(results, 'mas')


def flux_scales_figure(results):
    """
    Epoch flux scales g_e of the fit against time, in per cent; written to plots/fit_flux_scales.

    Posterior medians with 1-sigma errors, a thin line in time order as a
    guide for the eye, and a band at +/- f_sys, the fitted spread. For a
    simulation the injected scales (grey open squares, in the fitted frame)
    are drawn too, with a panel of chi, recovered minus injected over the
    posterior standard deviation, and its distribution against N(0, 1).
    """
    scale = results['flux_scale']
    if scale is None:
        return
    t, t_label = _epoch_axis(scale)
    order = np.argsort(t)
    median, sd = scale['median'] * 100, scale['sd'] * 100
    spread = results['parameters']['flux_systematic']['median'] * 100
    simulated = 'truth' in scale

    fig = plt.figure(figsize=(7.0, 4.6 if simulated else 3.2))
    grid = fig.add_gridspec(2 if simulated else 1, 2, width_ratios=[4, 1],
                            height_ratios=[2, 1] if simulated else [1], hspace=0.0, wspace=0.0)
    ax = fig.add_subplot(grid[0, 0])
    ax.axhspan(-spread, spread, color='0.9', lw=0, zorder=0, label=r'$\pm f_{\rm sys}$')
    ax.axhline(0, color='0.4', lw=0.8, ls='--', zorder=0)
    ax.plot(t[order], median[order], '-', color='0.35', lw=0.7, zorder=1)
    if simulated:
        ax.plot(t, scale['truth'] * 100, 's', ms=3, mfc='none', mec=MUTED_COLOUR, mew=0.8, zorder=2,
                label='Injected')
    ax.errorbar(t, median, yerr=sd, fmt='.', ms=3, color=DATA_COLOUR, ecolor=DATA_COLOUR, elinewidth=0.7,
                capsize=0, zorder=3, label='Recovered' if simulated else 'Fitted epochs')
    limit = 1.08 * max(np.max(np.abs(median) + sd), spread)
    ax.set_ylim(-limit, limit)
    ax.set_ylabel('Flux scale\n' r'$g_e$ (%)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8, handletextpad=0.3,
              columnspacing=1.0)

    if simulated:
        ax.tick_params(labelbottom=False)
        ax_chi = fig.add_subplot(grid[1, 0], sharex=ax)
        ax_h = fig.add_subplot(grid[1, 1], sharey=ax_chi)
        chi = scale['chi']
        ax_chi.axhspan(-1, 1, color='0.9', lw=0, zorder=0)
        ax_chi.axhline(0, color='0.4', lw=0.8, ls='--', zorder=0)
        ax_chi.plot(t, chi, '.', ms=4, color=DATA_COLOUR, zorder=2)
        chi_limit = max(3.5, 1.1 * np.nanmax(np.abs(chi)))
        ax_chi.set_ylim(-chi_limit, chi_limit)
        ax_chi.set_ylabel(r'$\chi$')
        ax_chi.set_xlabel(t_label)
        ax_h.hist(chi, bins=np.linspace(-chi_limit, chi_limit, 25), orientation='horizontal', histtype='step',
                  density=True, color=DATA_COLOUR, lw=1.0)
        grid_chi = np.linspace(-chi_limit, chi_limit, 200)
        ax_h.plot(np.exp(-0.5 * grid_chi ** 2) / np.sqrt(2 * np.pi), grid_chi, '-', color=POSTERIOR_LINE_COLOUR,
                  lw=1.2)
        ax_h.tick_params(labelleft=False, labelbottom=False)
    else:
        ax.set_xlabel(t_label)

    save_figure(fig, 'fit_flux_scales')


def fit_diagnostics(results):
    """
    Draw every fit figure from the results of fit.py (see fit.load_results).

    Parameters
    ----------
    results : dict
        Saved fit results; no trace is needed.
    """
    set_style()
    figures = [fit_model_figures, fit_axes_figure, translations_time_figure, translation_plane_figure,
               flux_scales_figure, corner_figure, chains_figure, affine_figure, affine_ellipses_figure, snr_histograms_figures,
               peak_chi_figure, morphology_figure]
    for figure in figures:
        figure(results)
    msg(f"Fit figures written to {PLOTS_DIR}/" if figures else "No fit figures defined in plot.py")


def epoch_histograms(offsets):
    """
    One figure per epoch: its residuals against the translation and the counts the model expects.

    Right ascension and declination panels share one set of bins across all epochs. Each shows a
    histogram of the epoch's residuals from the fitted true positions
    (inliers in blue, outliers stacked in navy), the model's expected count
    in each bin (solid steps), and the epoch's translation (dash-dot) with
    its 1-sigma band. Added epochs are named as such in the title. Written to
    plots/offsets_epoch_histograms/epoch_{n}.
    """
    trans, res = offsets['translations'], offsets['residuals']
    data, meta = offsets['data'], offsets['epochs']
    epoch = np.asarray(data['epoch'])
    inlier = np.asarray(trans['p_outlier']) < 0.5
    edges = res['edges'] * 1e3
    labels = (r'Right ascension offset $\Delta\alpha\cos\delta$ (mas)', r'Declination offset $\Delta\delta$ (mas)')
    obs_isot = meta.get('obs_isot') or [''] * len(trans['median'])
    n_drawn = 0

    for e in np.nonzero(np.isfinite(trans['median'][:, 0]))[0]:
        m = epoch == e
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={'wspace': 0.25})
        for k, (ax, label) in enumerate(zip(axes, labels)):
            r = res['residual'][m, k] * 1e3
            ax.hist([r[inlier[m]], r[~inlier[m]]], bins=edges, stacked=True, histtype='bar',
                    color=[plt.get_cmap(COLOUR_MAP)(0.45), 'navy'], edgecolor=DATA_COLOUR, linewidth=0.4,
                    label=['Inliers', 'Outliers'])
            ax.stairs(res['expected'][e, k], edges, color=POSTERIOR_LINE_COLOUR, lw=1.2, label='Model')
            tau, sd = trans['median'][e, k] * 1e3, trans['sd'][e, k] * 1e3
            ax.axvspan(tau - sd, tau + sd, color='lightblue', alpha=0.5, lw=0, zorder=0)
            ax.axvline(tau, color=POSTERIOR_LINE_COLOUR, lw=1.0, ls='-.', label='Translation')
            ax.axvline(0, color='0.4', lw=0.8, ls='--', zorder=0)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_xlabel(label)
        axes[0].set_ylabel('Measurements')
        axes[1].legend(loc='upper right', fontsize=7)
        status = 'added epoch' if not trans['in_fit'][e] else 'fitted epoch'
        fig.suptitle(f'Epoch {e}, {obs_isot[e][:10]}, {int(m.sum())} sources, {status}', fontsize=10)
        save_figure(fig, f'offsets_epoch_histograms/epoch_{e:03d}', formats=EPOCH_FORMATS, quiet=True)
        n_drawn += 1
    msg(f"Per-epoch histograms written for {n_drawn} epochs to {PLOTS_DIR / 'offsets_epoch_histograms'}/")


def epoch_flux_histograms(offsets):
    """
    One figure per fitted epoch: its measured peaks over the fitted flux densities against the flux scale g_e.

    A histogram of p / S_j - 1 in per cent for the epoch's peaks at or
    above the censoring threshold, the model's expected count in each bin
    (solid steps; each detection's normal truncated at the threshold, see
    offsets.epoch_flux_residuals), and g_e (dash-dot) with its 1-sigma band.
    The bins are shared across epochs. Written to
    plots/offsets_epoch_flux_histograms/epoch_{n}.
    """
    res = offsets['flux_residuals']
    if res is None:
        return
    epoch = np.asarray(offsets['data']['epoch'])
    obs_isot = offsets['epochs'].get('obs_isot') or [''] * len(res['scale'])
    edges = res['edges'] * 100
    n_drawn = 0
    for e in np.nonzero(np.isfinite(res['scale']))[0]:
        m = res['measured'] & (epoch == e)
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.hist(res['residual'][m] * 100, bins=edges, histtype='bar', color=plt.get_cmap(COLOUR_MAP)(0.45),
                edgecolor=DATA_COLOUR, linewidth=0.4, label='Measured peaks')
        ax.stairs(res['expected'][e], edges, color=POSTERIOR_LINE_COLOUR, lw=1.2, label='Model')
        g, sd = res['scale'][e] * 100, res['scale_sd'][e] * 100
        ax.axvspan(g - sd, g + sd, color='lightblue', alpha=0.5, lw=0, zorder=0)
        ax.axvline(g, color=POSTERIOR_LINE_COLOUR, lw=1.0, ls='-.', label=r'Flux scale $g_e$')
        ax.axvline(0, color='0.4', lw=0.8, ls='--', zorder=0)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xlabel(r'Peak over fitted flux density $p / S_j - 1$ (%)')
        ax.set_ylabel('Measurements')
        ax.legend(loc='upper right', fontsize=7)
        fig.suptitle(f'Epoch {e}, {obs_isot[e][:10]}, {int(m.sum())} measured peaks', fontsize=10)
        save_figure(fig, f'offsets_epoch_flux_histograms/epoch_{e:03d}', formats=EPOCH_FORMATS, quiet=True)
        n_drawn += 1
    msg(f"Per-epoch flux histograms written for {n_drawn} fitted epochs to "
        f"{PLOTS_DIR / 'offsets_epoch_flux_histograms'}/")


def truth_figure(offsets):
    """
    Recovered epoch translations against the offsets a simulation injected.

    Left column, against time: recovered right ascension and declination
    translations (black points with 1-sigma errors, stars for added epochs)
    with the injected offsets (grey open squares, shifted to the fitted
    frame), then chi, recovered minus injected over the posterior standard
    deviation, right ascension in red and declination in blue, with a band at
    +/-1. Right of the chi panel: their distribution against N(0, 1). The
    legend sits above the figure. Written to plots/offsets_truth.
    """
    trans, truth = offsets['translations'], offsets['truth']
    meta = {k: v for k, v in offsets['epochs'].items() if v is not None}
    t, t_label = _epoch_axis(dict(trans, **meta))
    ok = np.isfinite(trans['median'][:, 0])
    added = ~np.asarray(trans['in_fit']) & ok
    fitted = ok & ~added
    style = dict(elinewidth=0.7, capsize=0, zorder=2)

    fig = plt.figure(figsize=(7.0, 6.4))
    grid = fig.add_gridspec(3, 2, width_ratios=[4, 1], hspace=0.0, wspace=0.0)
    ax_ra = fig.add_subplot(grid[0, 0])
    ax_dec = fig.add_subplot(grid[1, 0], sharex=ax_ra)
    ax_chi = fig.add_subplot(grid[2, 0], sharex=ax_ra)
    ax_h = fig.add_subplot(grid[2, 1], sharey=ax_chi)

    labels = ('Right ascension\n' r'$\Delta\alpha\cos\delta$ (mas)', 'Declination\n' r'$\Delta\delta$ (mas)')
    for k, (ax, label) in enumerate(zip((ax_ra, ax_dec), labels)):
        rec, sd, inj = trans['median'][:, k] * 1e3, trans['sd'][:, k] * 1e3, truth['truth'][:, k] * 1e3
        ax.plot(t[ok], inj[ok], 's', ms=3, mfc='none', mec=MUTED_COLOUR, mew=0.8, zorder=1,
                label='Injected' if k == 0 else None)
        ax.errorbar(t[fitted], rec[fitted], yerr=sd[fitted], fmt='.', ms=3, color=DATA_COLOUR,
                    ecolor=DATA_COLOUR, label='Recovered' if k == 0 else None, **style)
        if added.any():
            ax.errorbar(t[added], rec[added], yerr=sd[added], fmt=ADDED_MARKER, ms=5, color=DATA_COLOUR,
                        ecolor=DATA_COLOUR, label=ADDED_LABEL if k == 0 else None, **style)
        ax.axhline(0, color='0.4', lw=0.8, ls='--', zorder=0)
        ax.set_ylabel(label)
        ax.tick_params(labelbottom=False)

    chi = truth['chi']
    ax_chi.axhspan(-1, 1, color='0.9', lw=0, zorder=0)
    ax_chi.axhline(0, color='0.4', lw=0.8, ls='--', zorder=0)
    for k, (colour, name) in enumerate(zip(AXIS_COLOURS, AXIS_NAMES)):
        ax_chi.plot(t[fitted], chi[fitted, k], '.', ms=4, color=colour, zorder=2, label=rf'$\chi$, {name}')
        ax_chi.plot(t[added], chi[added, k], ADDED_MARKER, ms=5, color=colour, zorder=2)
    limit = max(3.5, 1.1 * np.nanmax(np.abs(chi)))
    ax_chi.set_ylim(-limit, limit)
    ax_chi.set_ylabel(r'$\chi$')
    ax_chi.set_xlabel(t_label)

    values = chi[ok].ravel()
    ax_h.hist(values, bins=np.linspace(-limit, limit, 25), orientation='horizontal', histtype='step',
              density=True, color=DATA_COLOUR, lw=1.0)
    grid_chi = np.linspace(-limit, limit, 200)
    ax_h.plot(np.exp(-0.5 * grid_chi ** 2) / np.sqrt(2 * np.pi), grid_chi, '-', color=POSTERIOR_LINE_COLOUR,
              lw=1.2, label=r'$\mathcal{N}(0, 1)$')
    ax_h.tick_params(labelleft=False, labelbottom=False)

    # One legend above the figure for the markers of every panel.
    handles, names = [], []
    for ax in (ax_ra, ax_chi, ax_h):
        h, n = ax.get_legend_handles_labels()
        handles, names = handles + h, names + n
    ax_ra.legend(handles, names, loc='lower center', bbox_to_anchor=(0.625, 1.02), ncol=len(names), fontsize=8,
                 handletextpad=0.3, columnspacing=1.0)

    save_figure(fig, 'offsets_truth')


def truth_sources_figure(offsets):
    """
    Fitted sources against the sources a simulation injected.

    Both SNR axes are the injected SNR, so a source's place on them does not
    depend on its own fit. Top left: recovered minus injected position against SNR, right
    ascension in red and declination in blue, with 1-sigma errors. Top
    right: chi of both components, recovered minus injected over the
    posterior standard deviation, against N(0, 1). Bottom left: the residual of each source as
    an arrow at its position, exaggerated to be visible, with a key arrow.
    Bottom right: recovered over injected peak flux density against SNR,
    with 1-sigma errors (fitted, or median detected values with MCMC.snr_from_median_flux), and the
    mean injected flux scale of the fitted epochs, where the ratio should sit (1 without injected scales).
    Written to plots/offsets_truth_sources.
    """
    src = offsets['truth_sources']
    snr, diff, sd = src['injected_snr'], src['diff'] * 1e3, src['sd'] * 1e3
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.6), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    ax_r, ax_h, ax_q, ax_f = axes.ravel()
    style = dict(fmt='.', ms=3, elinewidth=0.6, capsize=0)

    for k, (colour, name) in enumerate(zip(AXIS_COLOURS, AXIS_NAMES)):
        ax_r.errorbar(snr, diff[:, k], yerr=sd[:, k], color=colour, ecolor=colour, label=name, **style)
    ax_r.axhline(0, color='0.4', lw=0.8, ls='--', zorder=0)
    ax_r.set_xscale('log')
    ax_r.set_xlabel('Signal-to-noise ratio (injected)')
    ax_r.set_ylabel('Recovered $-$ injected (mas)')
    ax_r.legend(loc='upper right', fontsize=8)

    limit = max(3.5, 1.1 * np.nanmax(np.abs(src['chi'])))
    bins = np.linspace(-limit, limit, 25)
    for k, (colour, name) in enumerate(zip(AXIS_COLOURS, AXIS_NAMES)):
        ax_h.hist(src['chi'][:, k], bins=bins, histtype='step', density=True, color=colour, lw=1.0, label=name)
    z = np.linspace(-limit, limit, 200)
    ax_h.plot(z, np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi), '-', color=POSTERIOR_LINE_COLOUR, lw=1.2,
              label=r'$\mathcal{N}(0, 1)$')
    ax_h.set_xlabel(r'$\chi$')
    ax_h.set_ylabel('Density')
    ax_h.legend(loc='upper right', fontsize=8)

    # Arrows are drawn in arcmin on the sky with the residuals scaled up by a common factor.
    xy = src['true_xy'] / 60.0
    radial = np.hypot(*diff.T)
    key = float(np.percentile(radial, 90))
    gain = 0.08 * np.ptp(xy) / key if key > 0 else 1.0
    arrows = ax_q.quiver(xy[:, 0], xy[:, 1], diff[:, 0] * gain, diff[:, 1] * gain, angles='xy',
                         scale_units='xy', scale=1.0, width=0.004, color=DATA_COLOUR)
    ax_q.quiverkey(arrows, 0.82, 1.04, key * gain, f'{key:.0f} mas', labelpos='E', coordinates='axes',
                   fontproperties={'size': 8})
    ax_q.set_xlim(ax_q.get_xlim()[::-1])
    ax_q.set_aspect('equal', adjustable='datalim')
    ax_q.set_xlabel('Right ascension offset (arcmin)')
    ax_q.set_ylabel('Declination offset (arcmin)')

    ax_f.errorbar(snr, src['flux_ratio'], yerr=src['flux_ratio_err'], **style, color=DATA_COLOUR,
                  ecolor=DATA_COLOUR)
    ax_f.axhline(1.0, color='0.4', lw=0.8, ls='--', zorder=0)
    ax_f.axhline(src['flux_scale_mean'], color=MODEL_COLOUR, lw=1.3, ls='--', zorder=10,
                 label=r'Mean injected $1 + g_e$')
    ax_f.legend(loc='upper right', fontsize=8)
    ax_f.set_xscale('log')
    ax_f.set_xlabel('Signal-to-noise ratio (injected)')
    ax_f.set_ylabel(f"Recovered / injected peak\nflux density ({src['flux_method']})")

    save_figure(fig, 'offsets_truth_sources')


def completeness_figure(offsets):
    """
    Detected fraction of source-epoch pairs against true SNR, the step the fit assumes, and where misses arise.

    Black points are the fraction measured, detected with a peak at or above
    the censoring threshold T; grey points the fraction PyBDSF detected at
    all; both with binomial 1-sigma errors. The solid line is the step the fit
    assumes for the measured fraction, (1 - e) Phi((SNR - T) / w), and the
    dash-dotted line its step for detection, at t = MCMC.detection_censor_snr. Dashed lines
    give the fraction of pairs lost at each stage: no PyBDSF source, or a
    PyBDSF source left unmatched.
    Written to plots/offsets_completeness.
    """
    c = offsets['truth_sources']['completeness']
    ok = c['n'] > 0
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.errorbar(c['snr_centre'][ok], c['completeness'][ok], yerr=c['completeness_err'][ok], fmt='.', ms=4,
                color=MUTED_COLOUR, ecolor=MUTED_COLOUR, elinewidth=0.7, capsize=0, label='Detected by PyBDSF',
                zorder=3)
    ax.errorbar(c['snr_centre'][ok], c['measured'][ok], yerr=c['measured_err'][ok], fmt='.', ms=4,
                color=DATA_COLOUR, ecolor=DATA_COLOUR, elinewidth=0.7, capsize=0,
                label=f"Measured, peak $\\geq$ {c['threshold']:g} rms", zorder=4)
    ax.plot(c['model_snr'], c['model'], '-', color=MODEL_COLOUR, lw=1.3, zorder=2,
            label=rf"$(1 - {c['floor']:g})\,\Phi((\mathrm{{SNR}} - {c['threshold']:g}) / \sigma_p)$, "
                  rf"$k = {c['width']:.2f}$, $f_{{\rm frac}} = {c['flux_fractional']:.3f}$")
    ax.plot(c['model_snr'], c['model_detected'], '-.', color=MODEL_COLOUR, lw=1.3, zorder=2,
            label=rf"Detection step, $t = {c['detect_threshold']:g}$ rms")
    styles = {'pybdsf': ('Lost: no PyBDSF source', '0.6'), 'matching': ('Lost: matching', HIGHLIGHT_COLOUR)}
    for name, lost in c['lost'].items():
        ax.plot(c['snr_centre'][ok], lost[ok], '--', marker='.', ms=3, color=styles[name][1], lw=1.0,
                label=styles[name][0], zorder=1)
    ax.axvline(c['pybdsf_threshold'], color='0.4', lw=0.8, ls=':', zorder=0)
    ax.axvline(c['threshold'], color='0.4', lw=0.8, ls='--', zorder=0)
    ax.set_xscale('log')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('True signal-to-noise ratio')
    ax.set_ylabel('Fraction of source-epoch pairs')
    ax.legend(loc='center right', fontsize=8)
    save_figure(fig, 'offsets_completeness')


# ============================================================================
# TARGET
# ============================================================================

def _ellipse(ax, centre, cov, colour, scale=1.0):
    """1-sigma ellipse of a 2x2 RA-Dec covariance at centre, both multiplied by scale."""
    values, vectors = np.linalg.eigh(cov * scale ** 2)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    ax.add_patch(Ellipse(centre, 2 * np.sqrt(values[1]), 2 * np.sqrt(values[0]), angle=angle,
                         facecolor=colour, edgecolor=DATA_COLOUR, linewidth=0.3, alpha=0.5, zorder=2))


def target_figure(target):
    """
    Corrected target track, with the true track of a simulation when given.

    Left: the track on the sky relative to the mean corrected position,
    corrected positions coloured by date with 1-sigma ellipses from their
    RA-Dec covariance, and measured positions before correction as faint
    points joined to them. Right: right ascension, declination and the
    distance from the first corrected position against time, with 1-sigma
    errors from the full covariance, and, for a simulation, the true track as
    a line and chi, corrected minus true over the error, right ascension in
    red and declination in blue. Right ascension increases to the left.
    Written to plots/target_{stem}.
    """
    set_style()
    n = len(target['p'])
    diag = np.arange(n)
    block = target['cov'][diag, :, diag, :]
    sd = np.sqrt(block[:, [0, 1], [0, 1]])
    centre = target['p'].mean(axis=0)
    p, measured = target['p'] - centre, target['measured'] - centre
    t, truth = target['mjd'], target.get('truth')
    true = truth['true_xy'] - centre if truth else None
    order = np.argsort(t)
    cmap = plt.get_cmap(COLOUR_MAP)
    norm = plt.Normalize(t.min(), t.max() if t.max() > t.min() else t.min() + 1)

    both = np.r_[p, measured]
    span = 1.15 * max(np.ptp(both[:, 0]), np.ptp(both[:, 1]), 1e-3) / 2

    n_rows = 4 if truth else 3
    fig = plt.figure(figsize=(12.0, 5.6))
    grid = fig.add_gridspec(n_rows, 4, width_ratios=[1.0, 0.04, 0.32, 1.3], wspace=0.05, hspace=0.0)
    ax = fig.add_subplot(grid[:, 0])
    cax = fig.add_subplot(grid[:, 1])
    side = [fig.add_subplot(grid[0, 3])]
    side += [fig.add_subplot(grid[k, 3], sharex=side[0]) for k in range(1, n_rows)]

    for i in range(n):
        ax.plot([measured[i, 0], p[i, 0]], [measured[i, 1], p[i, 1]], '-', color=MUTED_COLOUR, lw=0.4, zorder=1)
        _ellipse(ax, p[i], block[i], cmap(norm(t[i])))
    ax.plot(measured[:, 0], measured[:, 1], '.', ms=2, color=MUTED_COLOUR, zorder=1, label='Measured')
    ax.scatter(p[:, 0], p[:, 1], c=t, cmap=cmap, norm=norm, s=10, edgecolors=DATA_COLOUR, linewidths=0.5,
               zorder=3, label=r'Corrected, 1$\sigma$ ellipses')
    if truth:
        ax.plot(true[order, 0], true[order, 1], '-', color=HIGHLIGHT_COLOUR, lw=0.8, zorder=2, label='True')
    mid = (both.max(axis=0) + both.min(axis=0)) / 2
    ax.set_xlim(mid[0] + span, mid[0] - span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_aspect('equal')
    ax.set_xlabel(r'Right ascension offset $\Delta\alpha\cos\delta$ (arcsec)')
    ax.set_ylabel(r'Declination offset $\Delta\delta$ (arcsec)')
    ax.legend(loc='upper right', fontsize=8)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label('Observation epoch (MJD)', rotation=270, labelpad=14)
    cbar.ax.tick_params(which='both', direction='in')

    series = [(p[:, 0], sd[:, 0], true[:, 0] if truth else None, r'$\Delta\alpha\cos\delta$ (arcsec)'),
              (p[:, 1], sd[:, 1], true[:, 1] if truth else None, r'$\Delta\delta$ (arcsec)'),
              (target['radial'], target['radial_err'], truth['true_radial'] if truth else None, 'Radial (arcsec)')]
    for a, (value, err, true_value, label) in zip(side, series):
        a.errorbar(t, value, yerr=err, fmt='.', ms=3, color=DATA_COLOUR, ecolor=DATA_COLOUR,
                   elinewidth=0.7, capsize=0, zorder=2)
        if true_value is not None:
            a.plot(t[order], true_value[order], '-', color=HIGHLIGHT_COLOUR, lw=0.8, zorder=1)
        a.set_ylabel(label)
    if truth:
        limit = max(3.5, 1.1 * np.max(np.abs(truth['chi'])))
        side[3].axhspan(-1, 1, color='0.9', lw=0, zorder=0)
        side[3].axhline(0, color='0.4', lw=0.8, ls='--', zorder=0)
        for k, (colour, name) in enumerate(zip(AXIS_COLOURS, AXIS_NAMES)):
            side[3].plot(t, truth['chi'][:, k], '.', ms=4, color=colour, label=name)
        side[3].set_ylim(-limit, limit)
        side[3].set_ylabel(r'$\chi$')
        side[3].legend(loc='upper right', fontsize=8, ncol=2)
    for a in side:
        a.tick_params(labelbottom=False)
    side[-1].tick_params(labelbottom=True)
    side[-1].set_xlabel('Observation epoch (MJD)')

    save_figure(fig, f"target_{target['stem']}")


# ============================================================================
# SIMULATION
# ============================================================================

def simulation_noise_figure(check):
    """
    Autocorrelation of the simulated noise against the beam, from simulate.py noise mode.

    First panel: the first epoch's noise autocorrelation coefficient rho on
    the sky (right ascension increasing to the left), with the 50% contours
    of the measurement, the beam and the beam self-convolution (white noise
    convolved with the full beam). Second and third: rho along that epoch's
    beam major and minor axes against the same two expectations. Fourth: for
    every epoch, the fitted noise correlation FWHM over the beam FWHM along
    each axis, 1 when the noise autocorrelation is the beam and sqrt(2) for
    the beam self-convolution. Written to plots/simulate_noise_correlation.

    Parameters
    ----------
    check : dict
        From simulate.noise_correlation_check for the first epoch, with
        'epochs' (n,) and 'ratio' (n, 2), major and minor, added for all epochs.
    """
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.4), gridspec_kw={'wspace': 0.45})

    # Autocorrelation map; columns run West, so the left edge is the largest right ascension offset.
    ax = axes[0]
    d_ra, d_dec, rho = check['d_ra'], check['d_dec'], check['rho']
    half = 0.5 * check['pixel_scale']
    extent = [d_ra.max() + half, d_ra.min() - half, d_dec.min() - half, d_dec.max() + half]
    image = ax.imshow(rho, origin='lower', extent=extent, cmap=COLOUR_MAP, vmin=min(0.0, rho.min()), vmax=1.0)
    ax.contour(d_ra, d_dec, rho, levels=[0.5], colors=DATA_COLOUR, linewidths=1.2)
    ax.contour(d_ra, d_dec, check['beam_rho'], levels=[0.5], colors=HIGHLIGHT_COLOUR, linewidths=1.2)
    ax.contour(d_ra, d_dec, check['self_rho'], levels=[0.5], colors=MUTED_COLOUR, linewidths=1.2,
               linestyles='--')
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r'$\rho$', rotation=270, labelpad=14)
    cbar.ax.tick_params(which='both', direction='in')
    ax.set_xlabel('Right ascension offset (arcsec)')
    ax.set_ylabel('Declination offset (arcsec)')

    for ax, name in zip(axes[1:3], ('major', 'minor')):
        cut = check['cuts'][name]
        ax.axhline(0.0, color='0.4', lw=0.6, ls=':', zorder=0)
        ax.plot(cut['offset'], cut['beam'], '-', color=HIGHLIGHT_COLOUR, lw=1.3, zorder=2, label='Beam')
        ax.plot(cut['offset'], cut['self'], '--', color=MUTED_COLOUR, lw=1.3, zorder=2,
                label='Beam self-convolution')
        ax.plot(cut['offset'], cut['measured'], '.', ms=4, color=DATA_COLOUR, zorder=3, label='Simulated noise')
        ax.set_xlim(cut['offset'][0], cut['offset'][-1])
        ax.set_xlabel(f'Offset along beam {name} axis (arcsec)')
        ax.set_ylabel(r'$\rho$')

    ax = axes[3]
    ax.axhline(1.0, color='0.4', lw=0.8, ls='--', zorder=0)
    ax.axhline(np.sqrt(2.0), color='0.4', lw=0.8, ls=':', zorder=0)
    ax.plot(check['epochs'], check['ratio'][:, 0], '.', ms=4, color=DATA_COLOUR, zorder=2, label='Major axis')
    ax.plot(check['epochs'], check['ratio'][:, 1], '.', ms=4, color=MUTED_COLOUR, zorder=2, label='Minor axis')
    ax.set_ylim(0.8, 1.55)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Noise correlation FWHM / beam FWHM')

    # One legend above all panels, outside the plot areas.
    handles, labels = axes[1].get_legend_handles_labels()
    more_handles, more_labels = axes[3].get_legend_handles_labels()
    fig.legend(handles + more_handles, labels + more_labels, loc='lower center', bbox_to_anchor=(0.5, 0.98),
               ncol=len(handles) + len(more_handles))

    save_figure(fig, 'simulate_noise_correlation')


# ============================================================================
# OFFSETS
# ============================================================================

def offsets_figures(offsets):
    """
    Draw every figure for the translations of all epochs (see offsets.py).

    Epochs added after the fit are drawn as stars (dotted in the ellipse
    figure). Written to plots/offsets_epoch_translations,
    offsets_translation_plane, offsets_affine and offsets_affine_ellipses,
    plus one histogram per epoch in plots/offsets_epoch_histograms/, one flux histogram per fitted
    epoch in plots/offsets_epoch_flux_histograms/ and, for
    simulations, plots/offsets_truth, offsets_truth_sources and offsets_completeness.

    Parameters
    ----------
    offsets : dict
        Results saved by offsets.py.
    """
    set_style()
    trans = dict(offsets['translations'], **{k: v for k, v in offsets['epochs'].items() if v is not None})
    added = ~np.asarray(trans['in_fit'])
    _translations_time(trans, 'offsets_epoch_translations', added)
    _translation_plane(trans, 'offsets_translation_plane', added)
    _affine(offsets['affine'], trans, 'offsets_affine', added)
    _affine_ellipses(offsets['affine'], trans, 'offsets_affine_ellipses', added)
    epoch_histograms(offsets)
    epoch_flux_histograms(offsets)
    if offsets.get('truth'):
        truth_figure(offsets)
    if offsets.get('truth_sources'):
        truth_sources_figure(offsets)
        completeness_figure(offsets)
    msg(f"Offsets figures written to {PLOTS_DIR}/")
