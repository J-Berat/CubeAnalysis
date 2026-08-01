# CubeAnalysis

Configurable Julia pipeline for systematic analysis of three-dimensional FITS
and HDF5 cubes. It produces quality reports, CSV summaries, maps, distributions,
phase diagrams, conditional relations, scalar/vector spectra, HRO alignment,
structure functions, autocorrelations and physical plasma diagnostics.

## Run

```bash
julia --project=. bin/analyze_cube.jl config/cube_analysis.toml
```

Relative paths are resolved from the TOML file. Outputs are written atomically
and inputs are never modified. `overwrite=false` applies to figures, CSV and
TOML products.

## Memory modes

```toml
[run]
field_by_field = true
memory_budget_gb = 32
sample_stride = 37
atomic_directory = true
```

`field_by_field=true` reloads only the fields needed by the active analysis
instead of retaining the full dataset. Derived norms and products are lazy.
HDF5 conversion is performed in bounded hyperslabs; control their temporary
size with `chunk_mb` on each field. Scalar FFTs use reusable `Float32` real-FFT
buffers. A stage stops before allocating a known working set above
`memory_budget_gb`; zero disables the limit.

With `atomic_directory=true`, the complete run is built in a sibling temporary
directory and replaces the previous output only after every stage succeeds.

## Geometry and physical spectra

```toml
[grid]
box_size = [50.0, 50.0, 100.0]
box_unit = "pc"
axis_order = ["x", "y", "z"]
periodic = true

[spectra]
fields = ["density", "Bmag"]
quantity = "shell"       # or "average"
compensation = 1.6666667  # plot k^(5/3) P(k)
fit_range = [0.2, 10.0]
window = "none"          # or "hann"
```

Exported wavenumbers are physical. CSV files contain both the mean power per
mode and integrated shell energy, while the companion fit TOML records the
log-log slope and standard error.

Vector spectra support Helmholtz decomposition:

```toml
[[vector_spectra]]
name = "kinetic"
components = ["Vx", "Vy", "Vz"]
weight_field = "density"
weight_power = 0.5
```

This example analyzes `sqrt(density) * velocity`. Omitting the weight gives a
plain velocity or magnetic spectrum. `[[cross_spectra]]` entries accept
`name`, `first`, `second`, and `bins`, and export the complex cross-spectrum
and spectral coherence.

## HRO

```toml
[[alignments]]
name = "density_gradient_vs_B"
scalar = "density"
vector = ["Bx", "By", "Bz"]
condition = "density"
weighting = "uniform"       # or "gradient"
angle_coordinate = "cosine" # or "degrees"
bootstrap_replicates = 200
bootstrap_seed = 1234
```

The CSV contains the alignment parameter and its bootstrap uncertainty.

## Masks and weights

Every phase-diagram or conditional-relation entry may include:

```toml
mask_field = "temperature"
mask_min = 100.0
mask_max = 8000.0
weight_field = "density"
```

## Advanced diagnostics

```toml
[plots]
advanced_diagnostics = true

[diagnostics]
structure_fields = ["density", "Vmag"]
structure_orders = [1, 2, 3]
structure_samples = 100000
correlation_fields = ["density", "Bmag"]
physics = true
```

Physics mode derives sonic Mach number, Alfvén Mach number, and plasma beta.
Its unit conversion and mean molecular weight are configurable; see the sample
configuration.

## Diagnostics complementary to LesHouchesGit

CubeAnalysis also provides three generic diagnostics intentionally absent from
the LesHouchesGit figure registry:

- excursion-set topology: filling factor, physical interface area, connected
  components, largest-component fraction, and boundary spanning;
- a cylindrical spectrum in `(k_parallel, k_perpendicular)` relative to an
  explicit direction or the measured mean magnetic field;
- longitudinal and transverse vector structure functions, including skewness,
  flatness and fitted intermittency exponents for each grid direction.

Enable them with `plots.topology`, `plots.anisotropic_spectra`, and
`plots.directional_structure_functions`. Complete `[[topology]]`,
`[[anisotropic_spectra]]`, and `[[directional_structure_functions]]` examples
are provided in the default configuration. They are disabled there because
topological connectivity and directional increments can be expensive on a
1024³ cube.

## Snapshot series

Add snapshots at top level to run the same analysis over time:

```toml
[[snapshots]]
name = "output_00010"
time = 0.5
input_dir = "/data/run/output_00010"

[[snapshots]]
name = "output_00020"
time = 1.0
input_dir = "/data/run/output_00020"
```

Each snapshot gets an isolated subdirectory and `snapshot_evolution.csv`
combines its field statistics for temporal comparison.

## Reproducibility

`analysis_manifest.toml` embeds the resolved configuration, Julia and package
version information, generated files, and per-stage timing. `input_quality.csv`
reports NaN, infinities and non-positive values. The committed `Manifest.toml`
allows `Pkg.instantiate()` and `Pkg.test()` without the former unregistered
`JuliaCommon` dependency.
