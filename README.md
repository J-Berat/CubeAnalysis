# CubeAnalysis

Automated analysis of 3D FITS and HDF5 cubes with Julia.

CubeAnalysis generates projections, histograms, phase diagrams, power spectra,
structure functions, and magnetic-field alignment diagnostics.
All figures share a publication-ready visual style with LaTeX typography,
consistent scientific colors, quiet legends, and high-resolution exports.

## Installation

```bash
cd "/Users/jb270005/Desktop"
git clone https://github.com/J-Berat/CubeAnalysis.git
cd CubeAnalysis
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

To update an existing installation:

```bash
cd "/Users/jb270005/Desktop/CubeAnalysis"
git pull origin main
```

## Required files

Each RAMSES simulation must contain:

```text
density.fits
temperature.fits
Bx.fits
By.fits
Bz.fits
Vx.fits
Vy.fits
Vz.fits
```

The RAMSES batch script expects `256 x 256 x 256` cubes and uses a
`50 x 50 x 50 pc` box.

## Analyze all simulations

```bash
julia --startup-file=no --project=. scripts/run_simu_ramses.jl \
  "/Users/jb270005/Desktop/simu_RAMSES" \
  "/Users/jb270005/Desktop/simu_RAMSES/results/CubeAnalysis_final"
```

Add `1` to process only the first simulation:

```bash
julia --startup-file=no --project=. scripts/run_simu_ramses.jl \
  "/Users/jb270005/Desktop/simu_RAMSES" \
  "/Users/jb270005/Desktop/simu_RAMSES/results/CubeAnalysis_test" \
  1
```

Completed simulations are skipped. Because batch runs use `overwrite=false`,
choose a new output directory when regenerating figures after a code change.

## Analyze one cube

Set `input_dir` and `output_dir` in `config/cube_analysis.toml`, then run:

```bash
julia --project=. bin/analyze_cube.jl config/cube_analysis.toml
```

## Main outputs

- slices and projections, including column density;
- grouped `Bx, By, Bz` and `Vx, Vy, Vz` histograms;
- phase diagrams with thermal equilibrium and isotherms;
- scalar, magnetic, kinetic, velocity, and vorticity spectra;
- solenoidal and compressive spectral components;
- separated injection, inertial, and numerical-dissipation ranges;
- spectral fits and Kolmogorov/Burgers references restricted to the inertial range;
- HRO and alignment parameters;
- first-, second-, and third-order structure-function figures with inertial-range fits;
- autocorrelations and physical diagnostics.

Figures have no titles or grid lines. Text, ticks, legends, and colorbar labels
use LaTeX. Only figures are saved by default.

Enable CSV and TOML outputs explicitly with:

```toml
[run]
save_data = true
```

## Memory settings

Recommended settings for large cubes:

```toml
[run]
field_by_field = true
memory_budget_gb = 12
sample_stride = 16
atomic_directory = true
```

`field_by_field=true` avoids keeping every field in memory. Set
`memory_budget_gb = 0` to disable the memory limit.

## Alignment outputs

- `alignment_*.png` contains the configurable HRO and alignment parameter.
- `xi_*.png` uses 40 cosine bins, with `|cos(phi)| >= 0.75` for parallel
  orientations and `|cos(phi)| <= 0.25` for perpendicular orientations.

## Tests

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```
