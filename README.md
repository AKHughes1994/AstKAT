# AstKAT

Astrometric calibration for multi-epoch radio images.

AstKAT finds the sources common to a set of epoch images and fits all of their positions at once. It measures how far each epoch is shifted and how large the position errors really are. You then use the per-epoch shifts to correct the positions of a target, such as a moving jet knot, and get honest uncertainties on them.

## Install

```bash
pip install "astkat[extraction] @ git+https://github.com/AKHughes1994/AstKAT.git"
```

Or clone the repository and run `pip install ".[extraction]"` inside it.

`[extraction]` adds [PyBDSF](https://pybdsf.readthedocs.io), which finds the sources in the images. It compiles Fortran and C code, so it needs `gfortran` and Boost first:

```bash
sudo apt install gfortran libboost-python-dev libboost-numpy-dev   # Debian / Ubuntu
```

If PyBDSF won't build, `conda install -c conda-forge bdsf` usually works. You can also install without `[extraction]`. Every step except `astkat extract` will still run.

Requires Python 3.10+.

## Quickstart

Put your epoch images (FITS, one per epoch) in a folder. Then, from a new, empty project directory:

```bash
astkat init -i /path/to/images -n myfield   # set up the project and stack a deep image
astkat extract                              # find sources in every epoch (PyBDSF)
astkat filter                               # keep point sources
astkat match                                # cross-match sources between epochs
astkat fit                                  # joint fit (several minutes; -y replaces an existing fit)
astkat offsets                              # shift of every epoch, with uncertainties
astkat target my_target.txt                 # correct a target's positions
```

Run every step from the project directory. Each step reads what the previous one wrote there. `astkat <command> -h` lists a command's options.

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
- `astkat target -h` lists every option.

## Outputs

| Folder | Contents |
|---|---|
| `results/` | `fit_summary.txt`, `offsets_translations_allepochs.csv` (per-epoch shifts), `target_<name>.csv` (corrected target positions) and their covariances |
| `plots/` | Diagnostic figures (PNG and PDF) for the fit and the offsets |
| `pybdsf_files/` | Source catalogue for each epoch |
| `astkat_files/` | `astkat.yaml` (the project settings) and intermediate catalogues |
| `logs/` | Logs from each step |

## Settings

`astkat init` writes `astkat_files/astkat.yaml`, and every comment in it explains the setting. The ones most often changed:

- `PYBDSF.snr_threshold`: the detection threshold.
- `THRESHOLDS.match_threshold`: the cross-matching radius, as a fraction of the beam.
- `THRESHOLDS.epoch_min`: the number of epochs a source must appear in to be used.
- `MCMC.snr_term`: the shape of the position error against SNR (`fixed`, `power` or `broken`).
- `MCMC.correct_differential_aberration`: on by default for real data.

The defaults assume MeerKAT. For another telescope, change `MCMC.observatory_location`.

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

[`examples/run_simulation.sh`](examples/run_simulation.sh) runs the whole pipeline end to end and also recovers a simulated moving target. Run it from an empty directory:

```bash
mkdir sim && cd sim && bash /path/to/examples/run_simulation.sh
```

Compare `results/target_target_trajectory.csv` with the truth in `sim_target/target_trajectory_truth.csv`.
