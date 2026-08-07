# CubeAnalysis

Automated analysis of 3D FITS and HDF5 cubes with Julia.

CubeAnalysis generates projections, histograms, phase diagrams, power spectra,
structure functions, and magnetic-field alignment diagnostics.
All figures share a publication-ready visual style with LaTeX typography,
consistent scientific colors, quiet legends, and high-resolution exports.

## Installation

```bash
git clone https://github.com/J-Berat/CubeAnalysis.git
cd CubeAnalysis
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Start Julia with several threads to use the parallel stages:

```bash
export JULIA_NUM_THREADS=auto
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

The batch script reads the cube shape from the FITS header, and the physical
box size from `CDELT`/`CD` when present. Pass `--box-pc` when the header carries
no WCS scale.

## Analyze all simulations

```bash
julia --startup-file=no --project=. scripts/run_simu_ramses.jl <input_root> <output_root>
```

`<input_root>` holds one subdirectory per simulation. Options:

```text
--limit=N        only process the first N simulations
--pattern=REGEX  only keep subdirectories whose name matches REGEX
--box-pc=L       box size in parsecs when the FITS header has no WCS scale
--config=PATH    base parameter file (default: config/cube_analysis.toml)
--free-gib=X     stop when less than X GiB remain on the output volume
```

Run `scripts/run_simu_ramses.jl` with no argument to print this list.

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
- phase-conditioned velocity ESS exponents compared with turbulence models;
- a summary table of the sonic and Alfvenic Mach numbers, the plasma beta and
  the turbulent-to-ordered field ratios, volume- and mass-weighted;
- the density PDF with its lognormal fit and the driving parameter `b`;
- volume and mass filling factors per thermal phase;
- the column-density PDF with a maximum-likelihood power-law tail fit;
- the four Minkowski functionals of the density excursion sets;
- autocorrelations and physical diagnostics.

The ESS comparison uses the classical Kolmogorov scaling, the intermittent
[She--Leveque model](https://doi.org/10.1103/PhysRevLett.72.336), and the
supersonic [Boldyrev model](https://doi.org/10.1086/340758).

Figures have no titles or grid lines. Text, ticks, legends, and colorbar labels
use LaTeX. Only figures are saved by default.

Enable CSV and TOML outputs explicitly with:

```toml
[run]
save_data = true
```

## Spectra

`spectra.quantity` selects the plotted normalisation. Bins are logarithmic in
`k`, so the bin width grows like `k`:

| value | quantity | Kolmogorov slope |
| --- | --- | --- |
| `energy` (default) | `E(k) = shell sum / dk` | `-5/3` |
| `shell` | raw sum over the bin | `-2/3` |
| `average` | power per Fourier mode | `-11/3` |

Only `energy` is directly comparable with the Kolmogorov and Burgers reference
lines. Fits are weighted by `sqrt(modes)`, and each point carries its Poisson
error bar `E/sqrt(modes)`.

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

Cubes read from disk go through a least-recently-used cache, since one stage
indexes the same field several times. It holds half of `memory_budget_gb` by
default; `run.cache_gb` overrides that, and `0` disables caching.

## Alignment outputs

- `alignment_*.png` contains the configurable HRO and alignment parameter.
- `xi_*.png` uses 40 cosine bins, with `|cos(phi)| >= 0.75` for parallel
  orientations and `|cos(phi)| <= 0.25` for perpendicular orientations.

## Tests

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```
