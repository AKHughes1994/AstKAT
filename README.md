# AstKAT

Astrometric calibration for multi-epoch radio images.

AstKAT finds the sources common to a set of epoch images and fits all of their positions at once. It measures how far each epoch is shifted and how large the position errors really are. You then use the per-epoch shifts to correct the positions of a target, such as a moving jet knot, and get honest uncertainties on them.

## Install

Requires Python 3.10 or later.

```bash
pip install "astkat[extraction] @ git+https://github.com/AKHughes1994/AstKAT.git"
```

Or clone the repository and run `pip install ".[extraction]"` inside it. Clone it if you want the [example script](#try-it-on-simulated-data), as pip doesn't install `examples/`.

`[extraction]` adds [PyBDSF](https://pybdsf.readthedocs.io), which finds the sources in the images. It compiles Fortran and C code, so it needs `gfortran` and Boost first:

```bash
sudo apt install gfortran libboost-python-dev libboost-numpy-dev   # Debian / Ubuntu
```

If PyBDSF won't build, `conda install -c conda-forge bdsf` usually works. You can also install without `[extraction]`; every step except `astkat extract` still runs.

## Quickstart

Put your epoch images (FITS, one per epoch) in a folder. Then, from a new, empty project directory:

```bash
astkat init -i /path/to/images -n myfield   # set up the project and stack a deep image
astkat extract                              # find sources in every epoch (PyBDSF)
astkat filter                               # keep point sources
astkat match                                # cross-match sources between epochs
astkat fit                                  # joint fit (several minutes)
astkat offsets                              # shift of every epoch, with uncertainties
astkat target my_target.txt                 # correct a target's positions
```

Run every step from the project directory. Each step reads what the previous one wrote there.

| Step | What it does |
|---|---|
| `init` | Creates the project folders and `astkat_files/astkat.yaml`, and stacks the epochs into a deep image. That image is the reference list of sources. |
| `extract` | Runs PyBDSF on every epoch and on the deep image. |
| `filter` | Keeps sources that look like point sources. |
| `match` | Matches each epoch's sources to the reference list, then keeps sources that are seen often enough, point-like and not variable. |
| `fit` | Fits every position at once: the true source positions, a shift for each epoch, and how the position errors depend on SNR. |
| `offsets` | Works out the shift of every epoch, with its uncertainty. |
| `target` | Applies those shifts to your target's positions, with full uncertainties. |

Useful options (`astkat <command> -h` lists them all):
- `astkat init -d deep.fits` uses a deep image you already have instead of stacking.
- `astkat init --no-stack` builds no deep image and uses one epoch as the reference instead.
- `astkat fit -y` replaces an existing fit without asking.
- `astkat fit -n` keeps the existing fit and only redraws its figures.
- `astkat stack` rebuilds the deep image of an existing project.

### Target file

`astkat target` takes a text table of your target's measurements, one row per epoch, separated by spaces, tabs or commas:

```
flux rms ra dec bmaj bmin bpa time
0.35 0.05 190.325439973721 38.116708592094 8.622653 6.326764 -5.7641 1994-09-20T03:00:00.000
0.35 0.05 190.324719135707 38.116964333025 10.457242 8.292177 -65.4843 1994-09-21T03:00:00.000
```

| Column | Meaning |
|---|---|
| `flux` | Peak flux density, in the same units as `rms` |
| `rms` | Image noise at the target |
| `ra`, `dec` | Measured position, in decimal degrees or sexagesimal (`17h27m43.3s -16d12m19.3s`) |
| `bmaj`, `bmin`, `bpa` | *Optional.* The beam for that measurement: major and minor axes in arcsec (or set `--beam-unit arcmin` or `deg`), position angle in degrees. Leave them out and each row uses its epoch's beam. |
| `time` | Observation time, ISO-8601 or MJD. It must be within 12 hours of an epoch in the project; change that with `--match-tolerance-hours`. |

- The header row is optional. Without one, the columns must come in the order above, with either all 8 or just `flux rms ra dec time`.
- Common header names also work, such as `peak_flux`, `local_rms`, `maj`/`min`/`pa` or `mjd`.

## Outputs

| Folder | Contents |
|---|---|
| `results/` | `fit_summary.txt`; `offsets_translations_allepochs.csv` (the shift of each epoch); `target_<name>.csv` (corrected target positions) and its covariance; `matched_catalog_<project>.npz` (the matched sources) |
| `plots/` | Diagnostic figures (PNG and PDF) for the fit, the offsets and the target |
| `pybdsf_files/` | PyBDSF source catalogue for each epoch and the deep image |
| `astkat_files/` | `astkat.yaml` (the project settings) and intermediate catalogues |
| `logs/` | A log of each step |

## Settings

`astkat init` writes `astkat_files/astkat.yaml`, and a comment in it explains every setting. The ones most often changed are below. The defaults assume MeerKAT; for another telescope, change `MCMC.observatory_location`.

**Stacking** (`init`, `stack`)
- `BOOKKEEPING.stack_min_epochs`: stacking starts from a core of this many epochs whose beams are most alike. All epochs are stacked if there are fewer.
- `BOOKKEEPING.stack_beam_growth`: further epochs are added while they widen the stack's common beam by at most this fraction of its area (default 0.15). Raise it for a deeper but less sharp image.

**Source finding** (`extract`)
- `PYBDSF.snr_threshold`: the detection threshold, in sigma.

**Matching** (`match`)
- `THRESHOLDS.match_threshold`: the matching region, an ellipse shaped like the beam with semi-axes this many times BMAJ and BMIN. The default, 0.75, is 1.5 times the beam.
- `THRESHOLDS.ambiguous_fraction`: a source that's ambiguous or blended with a neighbour (marked `S_Code A`) in more than this share of its epochs is dropped, as it's probably two sources that can't be told apart.
- `THRESHOLDS.epoch_min`: the number of epochs a source must appear in to be used.

**Fit** (`fit`)
- `MCMC.snr_term`: how the position error depends on SNR. The default, `fixed`, is the textbook error, which falls as A / SNR. Two more flexible forms are there for when the data need them:
  - `power`, A SNR^−β, fits the slope β.
  - `broken` follows 1 / SNR above a fitted break SNR and lets the slope change below it.

  A full description of the noise model and the SNR terms is coming soon in a separate document.
- `MCMC.flux_censor_snr` and `MCMC.detection_censor_snr`: peaks at or above `flux_censor_snr` (in rms) are fitted as measured. Fainter detections are only known to lie between the two thresholds, and non-detections below `detection_censor_snr`. That keeps faint sources from biasing the fitted fluxes high.
- `MCMC.correct_differential_aberration`: on by default for real data, and always off for simulated data.

## Try it on simulated data

`astkat simulate` makes epoch images with known errors, so you can check that AstKAT recovers them. Choose what it makes with `--mode`:

| Mode | Makes | Tests |
|---|---|---|
| `noise` (default) | Realistic images with beam-correlated noise. The position errors come from PyBDSF's own fitting. | The whole pipeline on realistic data |
| `flat` | Noise-free images with every error injected directly: SNR term, floor, outliers and flux errors | Extraction, matching and the fit, against known values |
| `catalogue` | The matched catalogue directly, skipping extraction and matching | The fit on its own |
| `target` | A moving target in an existing project's epochs, written for `astkat target`. Run it after `astkat offsets`, from the project directory. | Target correction |

```bash
astkat simulate --mode flat -i 100 -o images   # 100 epochs into images/
astkat simulate -h                             # every option: injected errors, seeds, target motion, ...
```

[`examples/run_simulation.sh`](examples/run_simulation.sh) runs the whole pipeline end to end and also recovers a simulated moving target. The settings at its top choose the mode and the injected errors. Run it from an empty directory:

```bash
mkdir sim && cd sim && bash /path/to/AstKAT/examples/run_simulation.sh
```

To stop partway, add `exit 0` after the step you want to end on. The script deletes earlier results when it starts, so to carry on afterwards, run the remaining `astkat` commands by hand from the same folder.

### Comparing fitted and injected values

On simulated data, AstKAT finds the simulator's truth files and compares them with what it fitted, with nothing extra to run. The injected values appear in these plots, as dashed lines or grey open squares:

| Plot (`plots/`) | Fitted against injected |
|---|---|
| `fit_corner`, `fit_chains` | Noise-model parameters (A, B, outlier fraction, flux errors, ...) |
| `fit_flux_scales` | Flux scale of each epoch |
| `offsets_truth` | Shift of each epoch, with χ = (fitted − injected) / σ |
| `offsets_truth_sources` | Source positions and peak flux densities, against injected SNR |
| `offsets_completeness` | Fraction of sources detected against SNR, with the detection model the fit assumes |
| `target_<name>` | Corrected target track against the true track |

The same comparison appears as numbers:
- the `truth` column in `results/fit_summary.txt`;
- `results/offsets_summary.txt`;
- `results/target_<name>_summary.txt`.
