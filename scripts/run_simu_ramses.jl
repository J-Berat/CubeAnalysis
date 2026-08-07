using CubeAnalysis
using Dates
using FITSIO
using Printf
using TOML

const REQUIRED_FIELDS = ("density", "temperature", "Bx", "By", "Bz", "Vx", "Vy", "Vz")
const DEFAULT_PATTERN = r"."
const DEFAULT_BOX_PC = 50.0

"""
Read the cube shape from the FITS header, and the physical box size from the
header when it carries a usable WCS scale.

`CDELT`/`CD` are taken as the cell size along each axis; the unit comes from
`CUNIT` and defaults to parsecs. When the header says nothing, `fallback_pc` is
used for all three axes.
"""
function cube_geometry(path::AbstractString; fallback_pc=DEFAULT_BOX_PC)
    header = FITSIO.read_header(path)
    shape = ntuple(d -> Int(header["NAXIS$d"]), 3)
    all(>=(2), shape) || error("Expected a 3-D cube with at least 2 cells; got $shape in $path")
    scales = ntuple(3) do d
        for key in ("CDELT$d", "CD$(d)_$(d)")
            haskey(header, key) || continue
            value = abs(Float64(header[key]))
            isfinite(value) && value > 0 && return value
        end
        return NaN
    end
    unit = haskey(header, "CUNIT1") ? lowercase(strip(string(header["CUNIT1"]))) : "pc"
    if all(isfinite, scales) && unit in ("pc", "parsec")
        return shape, collect(scales .* shape), "pc"
    end
    return shape, fill(Float64(fallback_pc), 3), "pc"
end

function available_gib(path::AbstractString)
    stat = Base.Filesystem.diskstat(path)
    return stat.bavail * stat.bsize / 1024^3
end

function simulation_directories(root::AbstractString; pattern::Regex=DEFAULT_PATTERN)
    candidates = filter(readdir(root; join=true)) do path
        isdir(path) && occursin(pattern, basename(path))
    end
    simulations = String[]
    for path in sort(candidates)
        missing = filter(REQUIRED_FIELDS) do field
            !isfile(joinpath(path, "$field.fits"))
        end
        if isempty(missing)
            push!(simulations, path)
        else
            @warn "Skipping incomplete simulation" path missing
        end
    end
    return simulations
end

function configure(base, input_dir, output_dir; fallback_box_pc=DEFAULT_BOX_PC)
    cfg = deepcopy(base)
    cfg["run"]["input_dir"] = input_dir
    cfg["run"]["output_dir"] = output_dir
    cfg["run"]["formats"] = ["png"]
    cfg["run"]["overwrite"] = false
    cfg["run"]["atomic_directory"] = true
    cfg["run"]["field_by_field"] = true
    cfg["run"]["memory_budget_gb"] = 12.0
    cfg["run"]["sample_stride"] = 16

    shape, box_size, box_unit = cube_geometry(joinpath(input_dir, "density.fits");
        fallback_pc=fallback_box_pc)
    cfg["grid"]["box_size"] = box_size
    cfg["grid"]["box_unit"] = box_unit
    cfg["grid"]["axis_order"] = ["x", "y", "z"]
    cfg["grid"]["periodic"] = true

    cfg["render"]["width"] = 1000
    cfg["render"]["height"] = 700

    plots = cfg["plots"]
    for name in ("summary", "quality", "slices", "projections", "histograms",
            "phase_diagrams", "relations", "power_spectra", "vector_spectra",
            "cross_spectra", "alignments", "advanced_diagnostics", "physics_table")
        plots[name] = true
    end
    for name in ("density_pdf", "phase_budget", "column_density_pdf")
        plots[name] = true
    end
    for name in ("topology", "anisotropic_spectra", "directional_structure_functions")
        plots[name] = false
    end

    selected = ["density", "temperature", "Bmag", "Vmag"]
    cfg["slices"]["fields"] = selected
    cfg["slices"]["axes"] = ["x", "y", "z"]
    cfg["slices"]["positions"] = [0.5]
    cfg["projections"]["fields"] = selected
    cfg["projections"]["axes"] = ["x", "y", "z"]
    cfg["projections"]["statistics"] = ["mean"]
    cfg["projections"]["column_density_fields"] = ["density"]
    cfg["histograms"]["fields"] = selected

    lengths = Float64.(box_size)
    kmin = minimum(2pi ./ lengths)
    kmax = minimum(pi .* collect(shape) ./ lengths)
    cfg["spectra"]["fields"] = selected
    cfg["spectra"]["bins"] = 64
    cfg["spectra"]["quantity"] = "energy"
    cfg["spectra"]["fit_range"] = [4kmin, 0.5kmax]
    cfg["spectra"]["reference_range"] = [4kmin, 0.5kmax]
    cfg["spectra"]["injection_modes"] = 4.0
    cfg["spectra"]["dissipation_cells"] = 4.0
    cfg["spectra"]["velocity_fields"] = ["Vmag"]
    cfg["spectra"]["velocity_reference_slopes"] = [-5 / 3, -2.0]
    for spectrum in get(cfg, "vector_spectra", Any[])
        lowercase(string(get(spectrum, "name", ""))) == "velocity" || continue
        spectrum["reference_slopes"] = [-5 / 3, -2.0]
        spectrum["reference_range"] = [4kmin, 0.5kmax]
    end

    cfg["cross_spectra"] = [Dict(
        "name" => "density_magnetic_magnitude",
        "first" => "density",
        "second" => "Bmag",
        "bins" => 64,
    )]

    for alignment in get(cfg, "alignments", Any[])
        alignment["weighting"] = "uniform"
        alignment["angle_coordinate"] = "cosine"
        alignment["bootstrap_replicates"] = 30
        alignment["bootstrap_max_samples"] = 5_000
    end

    diagnostics = cfg["diagnostics"]
    diagnostics["structure_fields"] = ["density", "Vmag"]
    diagnostics["structure_orders"] = [1, 2, 3]
    diagnostics["structure_samples"] = 25_000
    diagnostics["correlation_fields"] = ["density", "Bmag"]
    diagnostics["correlation_bins"] = 60
    return cfg, shape, box_size
end

function write_report(path, rows)
    temporary = "$path.tmp"
    open(temporary, "w") do io
        println(io, "simulation,status,elapsed_seconds,nx,ny,nz,Lx_pc,Ly_pc,Lz_pc,output,message")
        for row in rows
            escaped = replace(string(row.message), ',' => ';', '\n' => ' ')
            println(io, join((row.simulation, row.status, row.elapsed, row.shape...,
                row.box_size..., row.output, escaped), ','))
        end
    end
    mv(temporary, path; force=true)
end

function aggregate_csv(output_root, simulations, source_name, destination_name)
    destination = joinpath(output_root, destination_name)
    temporary = "$destination.tmp"
    wrote_header = false
    open(temporary, "w") do io
        for input_dir in simulations
            simulation = basename(input_dir)
            source = joinpath(output_root, simulation, source_name)
            isfile(source) || continue
            lines = readlines(source)
            isempty(lines) && continue
            if !wrote_header
                println(io, "simulation,$(lines[1])")
                wrote_header = true
            end
            for line in lines[2:end]
                println(io, "$simulation,$line")
            end
        end
    end
    mv(temporary, destination; force=true)
    return destination
end

function write_ensemble_tables(output_root, simulations)
    outputs = String[]
    push!(outputs, aggregate_csv(output_root, simulations, "cube_summary.csv",
        "ensemble_cube_summary.csv"))
    push!(outputs, aggregate_csv(output_root, simulations, "physics_diagnostics.csv",
        "ensemble_physics_diagnostics.csv"))
    push!(outputs, aggregate_csv(output_root, simulations, "field_fluctuations.csv",
        "ensemble_field_fluctuations.csv"))
    push!(outputs, aggregate_csv(output_root, simulations, "phase_budget.csv",
        "ensemble_phase_budget.csv"))
    push!(outputs, aggregate_csv(output_root, simulations,
        "alignment_density_gradient_vs_b.csv", "ensemble_alignment.csv"))

    fit_path = joinpath(output_root, "ensemble_spectral_fits.csv")
    temporary = "$fit_path.tmp"
    open(temporary, "w") do io
        println(io, "simulation,field,slope,slope_std,points,k_min,k_max,quantity,window")
        for input_dir in simulations, field in ("density", "temperature", "bmag", "vmag")
            simulation = basename(input_dir)
            source = joinpath(output_root, simulation, "spectrum_$(field)_fit.toml")
            isfile(source) || continue
            fit = TOML.parsefile(source)
            println(io, join((simulation, field, fit["slope"], fit["slope_std"],
                fit["points"], fit["k_min"], fit["k_max"], fit["quantity"],
                fit["window"]), ','))
        end
    end
    mv(temporary, fit_path; force=true)
    push!(outputs, fit_path)
    return outputs
end

const USAGE = """
Usage: julia --project=. scripts/run_simu_ramses.jl <input_root> [output_root] [options]

  <input_root>        directory holding one subdirectory per simulation
  [output_root]       defaults to <input_root>/results/CubeAnalysis_<date>

Options:
  --limit=N           only process the first N simulations
  --pattern=REGEX     only keep subdirectories matching REGEX (default: all)
  --box-pc=L          box size in parsecs when the FITS header has no WCS scale
                      (default: $DEFAULT_BOX_PC)
  --config=PATH       base parameter file (default: config/cube_analysis.toml)
  --free-gib=X        stop when less than X GiB remain (default: 1.0)
"""

function parse_options(args)
    positional = String[]
    options = Dict{String,String}()
    for argument in args
        if startswith(argument, "--")
            key, _, value = partition_option(argument)
            options[key] = value
        else
            push!(positional, argument)
        end
    end
    return positional, options
end

function partition_option(argument)
    body = argument[3:end]
    index = findfirst(==('='), body)
    isnothing(index) && return body, true, ""
    return body[1:index-1], true, body[index+1:end]
end

function main(args)
    positional, options = parse_options(args)
    if isempty(positional) || haskey(options, "help")
        println(USAGE)
        isempty(positional) && error("An input root directory is required")
        return
    end
    root = abspath(expanduser(positional[1]))
    isdir(root) || error("Input root is not a directory: $root")
    output_root = length(positional) >= 2 ? abspath(expanduser(positional[2])) :
        joinpath(root, "results", "CubeAnalysis_$(Dates.format(today(), "yyyy-mm-dd"))")
    base_path = haskey(options, "config") ? abspath(expanduser(options["config"])) :
        normpath(joinpath(@__DIR__, "..", "config", "cube_analysis.toml"))
    fallback_box_pc = haskey(options, "box-pc") ?
        parse(Float64, options["box-pc"]) : DEFAULT_BOX_PC
    minimum_free = haskey(options, "free-gib") ?
        parse(Float64, options["free-gib"]) : 1.0
    pattern = haskey(options, "pattern") ? Regex(options["pattern"]) : DEFAULT_PATTERN
    base = CubeAnalysis.load_config(base_path)
    save_outputs = CubeAnalysis.save_data(base)
    simulations = simulation_directories(root; pattern)
    isempty(simulations) && error("No complete simulation found under $root")
    if haskey(options, "limit")
        limit = parse(Int, options["limit"])
        limit > 0 || error("--limit must be positive")
        simulations = simulations[1:min(limit, length(simulations))]
    end
    mkpath(output_root)
    rows = NamedTuple[]
    report_path = joinpath(output_root, "batch_report.csv")
    @info "CubeAnalysis batch" simulations=length(simulations) output_root available_gib=available_gib(output_root)

    for (index, input_dir) in enumerate(simulations)
        name = basename(input_dir)
        output_dir = joinpath(output_root, name)
        if isdir(output_dir) && (!save_outputs || isfile(joinpath(output_dir, "analysis_manifest.toml")))
            shape, box_size, _ = cube_geometry(joinpath(input_dir, "density.fits");
                fallback_pc=fallback_box_pc)
            @info "Already complete; skipping" index total=length(simulations) name
            push!(rows, (; simulation=name, status="skipped_complete", elapsed=0.0,
                shape, box_size, output=output_dir, message="manifest already present"))
            save_outputs && write_report(report_path, rows)
            continue
        end
        free = available_gib(output_root)
        free >= minimum_free ||
            error(@sprintf("Only %.2f GiB remain; stopping before %s", free, name))
        cfg, shape, box_size = configure(base, input_dir, output_dir; fallback_box_pc)
        @info "Starting simulation" index total=length(simulations) name shape box_size free_gib=free
        started = time()
        status, message = "complete", ""
        try
            CubeAnalysis.run_analysis_config(cfg)
        catch exception
            status = "failed"
            message = sprint(showerror, exception, catch_backtrace())
            @error "Simulation failed" name exception=(exception, catch_backtrace())
            for staging in filter(path -> startswith(basename(path), "$name.tmp-"),
                    readdir(output_root; join=true))
                isdir(staging) && rm(staging; recursive=true)
            end
        end
        elapsed = round(time() - started; digits=3)
        push!(rows, (; simulation=name, status, elapsed, shape, box_size,
            output=output_dir, message))
        save_outputs && write_report(report_path, rows)
        GC.gc(true)
    end
    complete = count(row -> row.status in ("complete", "skipped_complete"), rows)
    xi_files = String[]
    _, reference_box, _ = cube_geometry(joinpath(first(simulations), "density.fits");
        fallback_pc=fallback_box_pc)
    CubeAnalysis.plot_xi_vs_sigma_v!(xi_files, simulations, output_root, ["png"];
        overwrite=false, box_size=Tuple(reference_box), axis_order=("x", "y", "z"))
    ensemble = save_outputs ? write_ensemble_tables(output_root, simulations) : String[]
    @info "Batch finished" complete total=length(rows) report=(save_outputs ? report_path : "disabled") available_gib=available_gib(output_root)
    save_outputs && @info "Ensemble comparison tables" files=ensemble
    complete == length(rows) || error("$(length(rows) - complete) simulation(s) failed")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
