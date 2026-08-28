# CubeAnalysis

Automated analysis of 3-D FITS and HDF5 cubes with Julia: projections,
histograms, phase diagrams, power spectra, structure functions, PDFs and
magnetic-field alignment diagnostics, all in one publication-ready style.

## Installation and use

```bash
git clone https://github.com/J-Berat/CubeAnalysis.git
cd CubeAnalysis
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Several stages are threaded, so start Julia with more than one thread:

```bash
export JULIA_NUM_THREADS=auto
```

To analyse one cube, set `input_dir` and `output_dir` in the parameter file and
run:

```bash
julia --startup-file=no --project=. bin/analyze_cube.jl config/cube_analysis.toml
```

Only figures are written by default; set `run.save_data = true` for the CSV and
TOML tables. Tests:

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

## Several simulations

```bash
julia --startup-file=no --project=. scripts/run_simu_ramses.jl <input_root> <output_root>
```

`<input_root>` holds one subdirectory per simulation, each containing
`density.fits`, `temperature.fits`, `Bx.fits`, `By.fits`, `Bz.fits`,
`Vx.fits`, `Vy.fits` and `Vz.fits`. Incomplete directories are skipped with a
warning, and simulations already analysed are left untouched.

| Option | Effect |
| --- | --- |
| `--limit=N` | only process the first `N` simulations |
| `--pattern=REGEX` | only keep subdirectories whose name matches `REGEX` |
| `--box-pc=L` | box size in parsecs when the FITS header carries no WCS scale |
| `--config=PATH` | base parameter file (default `config/cube_analysis.toml`) |
| `--free-gib=X` | stop when less than `X` GiB remain on the output volume |

Run the script without arguments to print this list. The cube shape comes from
the FITS header, and the physical box size from `CDELT`/`CD` when present.

Batch runs use `overwrite = false`, so pick a new output directory when
regenerating figures after a code change. Per-simulation tables are also
concatenated into `ensemble_*.csv` files at the root of `<output_root>`.

## Parameter file

All relative paths resolve from the parameter file itself. `[run]`, `[fields]`
and at least one plot are required; everything else has defaults.

### `[run]`

| Key | Default | Meaning |
| --- | --- | --- |
| `input_dir`, `output_dir` | — | required |
| `formats` | `["png"]` | any of `png`, `pdf`, `svg` |
| `overwrite` | `true` | fail instead of replacing existing outputs when `false` |
| `atomic_directory` | `false` | build in a staging directory and publish at the end |
| `save_data` | `false` | also write the CSV and TOML tables |
| `sample_stride` | `1` | keep one cell in `N` for statistics |
| `field_by_field` | `false` | read cubes on demand instead of all at once |
| `memory_budget_gb` | `0` | `0` disables the limit; stages fail early above it |
| `cache_gb` | half the budget | LRU cache for cubes read from disk; `0` disables it |

For large cubes, `field_by_field = true` with `memory_budget_gb = 12`,
`sample_stride = 16` and `atomic_directory = true` is a sane starting point.

### `[grid]`

`box_size` (scalar or three values), `axis_order` (default `["x","y","z"]`),
`box_unit` (`pc`, `kpc`, `au`, `m`, `cm` — used for column density), and
`periodic` (default `true`), which selects periodic gradients and the default
spectral window.

### `[render]`

`width`, `height`, `fontsize`, `colormap`, and `quantile_low` / `quantile_high`
for the colour range of slices and projections (defaults `0.005` / `0.995`).

### `[plots]`

One boolean per stage. Enabled by default: `physics_table`, `density_pdf`,
`phase_budget`, `column_density_pdf`, plus `summary` and `quality`, which write
tables and therefore only run when `save_data = true`.

Off by default: `slices`, `projections`, `histograms`, `phase_diagrams`,
`relations`, `power_spectra`, `vector_spectra`, `cross_spectra`, `alignments`,
`advanced_diagnostics`, `topology`, `anisotropic_spectra`,
`directional_structure_functions`.

Stages needing fields the cube does not have are skipped with a message rather
than failing.

### `[fields.<name>]` and `[derived]`

Each field points at one cube:

```toml
[fields.density]
file = "density.fits"     # required
label = "number density"
unit = "cm^-3"
log = true                 # logarithmic colour scale and histograms
colormap = "magma"
```

`permutation` (e.g. `[3, 2, 1]`), `scale` and `offset` apply to both formats;
`dataset` and `chunk_mb` are HDF5-only.

`[[derived.vectors]]` builds a magnitude from three components
(`name`, `components`, `factor`); `[[derived.products]]` multiplies fields
(`name`, `factors`, `factor`). Both accept the same display keys as a field.

### `[spectra]`

`fields`, `bins`, `remove_mean`, `compensation`, `window` (`none` or `hann`),
`fit_range`, `reference_range`, `velocity_fields`,
`velocity_reference_slopes`, plus `injection_modes` and `dissipation_cells`,
which set the shaded injection and dissipation ranges and clip every fit to the
inertial range.

`quantity` selects the normalisation. Bins are logarithmic in `k`, so the bin
width grows like `k`:

| Value | Quantity | Kolmogorov slope |
| --- | --- | --- |
| `energy` (default) | `E(k) = shell sum / dk` | `-5/3` |
| `shell` | raw sum over the bin | `-2/3` |
| `average` | power per Fourier mode | `-11/3` |

Only `energy` is comparable with the Kolmogorov and Burgers reference lines.
Fits are weighted by `sqrt(modes)` and each point carries a Poisson error bar.

### `[diagnostics]`

Field names (`density`, `temperature`, `velocity`, `magnetic`) and the unit
conversions used by the physics table: `mean_molecular_weight`,
`velocity_to_cms`, `magnetic_to_gauss`, `gamma`.

Structure functions: `structure_fields`, `structure_orders`,
`structure_samples`, `seed`. Phase-conditioned ESS: `phase_ess`,
`phase_ess_orders`, `phase_ess_samples`, `phase_ess_velocity`,
`phase_ess_temperature`. PDFs: `density_pdf_bins`, `column_density_axes`,
`column_density_bins`. Autocorrelation: `correlation_fields`,
`correlation_bins`.

`[[diagnostics.phases]]` defines the thermal phases (`name`,
`temperature_min`, `temperature_max`) used by both the ESS and the phase
budget; the default is CNM / UNM / WNM.

### Array-of-table sections

Each entry below is a `[[...]]` block with a `name`.

| Section | Key keys |
| --- | --- |
| `[[vector_spectra]]` | `components`, `bins`, `transform` (`none`/`curl`), `weight_field`, `weight_power`, `reference_slopes`, `reference_range` |
| `[[cross_spectra]]` | `first`, `second`, `bins` |
| `[[phase_diagrams]]` | `x`, `y`, `bins`, `log_x`, `log_y`, `thermal_overlay` (`none`/`temperature`/`pressure`), `isotherms`, `isotherm_colors` |
| `[[relations]]` | `x`, `y`, `bins`, `minimum_per_bin`, `log_x`, `log_y`, `stride`, `mask_field`, `weight_field` |
| `[[alignments]]` | `scalar`, `vector`, `condition`, `condition_bins`, `angle_bins`, `weighting` (`uniform`/`gradient`), `angle_coordinate` (`cosine`/`degrees`), `bootstrap_replicates` |
| `[[alignments]]` (xi) | `xi_plot`, `velocity_vector`, `xi_bins`, `xi_density_range`, `xi_blocks_per_axis`, `xi_cosine_bins`, `xi_parallel_cosine`, `xi_perpendicular_cosine` |
| `[[topology]]` | `field`, `quantiles` or `thresholds`, `above` |
| `[[anisotropic_spectra]]` | `field`, `direction` or `guide_fields`, `parallel_bins`, `perpendicular_bins`, `radial_bins` |
| `[[directional_structure_functions]]` | `components`, `orders`, `lags`, `fit_range` |
| `[[histograms.groups]]` | `fields`, `xlabel` |
| `[[snapshots]]` | `input_dir`, `time`, `output_subdir` — analyses a time series and writes `snapshot_evolution.csv` |
