module CubeAnalysis

using CairoMakie
using FFTW
using FITSIO
using HDF5
using LaTeXStrings
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML


export run_analysis, load_config, load_fields, read_cube, periodic_gradient, isotropic_spectrum

function __init__()
    # Start Julia with -t auto (or JULIA_NUM_THREADS) to use this.
    FFTW.set_num_threads(Threads.nthreads())
    return nothing
end

struct FieldMeta
    label::String
    unit::String
    logscale::Bool
    colormap::Symbol
end

struct AnalysisResult
    output_dir::String
    files::Vector{String}
    fields::Vector{String}
end

include("plot_style.jl")
include("geometry.jl")
include("lazy_fields.jl")
include("streaming.jl")
include("io.jl")
include("thermal_physics.jl")
include("plots.jl")
include("spectra.jl")
include("quality.jl")
include("diagnostics.jl")
include("physics_table.jl")
include("pdf_statistics.jl")
include("phase_diagnostics.jl")
include("synthetic_observations.jl")
include("topology.jl")
include("anisotropy.jl")
include("directional_structure.jl")
include("ibanez_alignment.jl")
include("ensemble.jl")
include("runner.jl")

end
