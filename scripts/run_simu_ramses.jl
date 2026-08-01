using CubeAnalysis
using Dates
using FITSIO
using Printf
using TOML

const REQUIRED_FIELDS = ("density", "temperature", "Bx", "By", "Bz", "Vx", "Vy", "Vz")

function cube_geometry(path::AbstractString)
    header = FITSIO.read_header(path)
    shape = ntuple(d -> Int(header["NAXIS$d"]), 3)
    shape == (256, 256, 256) || error("Expected a 256^3 cube; found $shape in $path")
    return shape, [50.0, 50.0, 50.0], "pc"
end

function available_gib(path::AbstractString)
    stat = Base.Filesystem.diskstat(path)
    return stat.bavail * stat.bsize / 1024^3
end

function simulation_directories(root::AbstractString)
    candidates = filter(readdir(root; join=true)) do path
        isdir(path) && occursin(r"^d.*nograv1024$", basename(path))
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

function configure(base, input_dir, output_dir)
    cfg = deepcopy(base)
    cfg["run"]["input_dir"] = input_dir
    cfg["run"]["output_dir"] = output_dir
    cfg["run"]["formats"] = ["png"]
    cfg["run"]["overwrite"] = false
    cfg["run"]["atomic_directory"] = true
    cfg["run"]["field_by_field"] = true
    cfg["run"]["memory_budget_gb"] = 12.0
    cfg["run"]["sample_stride"] = 16

    shape, box_size, box_unit = cube_geometry(joinpath(input_dir, "density.fits"))
    cfg["grid"]["box_size"] = box_size
    cfg["grid"]["box_unit"] = box_unit
    cfg["grid"]["axis_order"] = ["x", "y", "z"]
    cfg["grid"]["periodic"] = true

    cfg["render"]["width"] = 1000
    cfg["render"]["height"] = 700

    plots = cfg["plots"]
    for name in ("summary", "quality", "slices", "projections", "histograms",
            "phase_diagrams", "relations", "power_spectra", "vector_spectra",
            "cross_spectra", "alignments", "advanced_diagnostics")
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
    cfg["spectra"]["quantity"] = "shell"
    cfg["spectra"]["fit_range"] = [4kmin, 0.5kmax]

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
    diagnostics["physics"] = true
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

function main(args)
    root = length(args) >= 1 ? abspath(expanduser(args[1])) :
        "/Users/jb270005/Desktop/simu_RAMSES"
    output_root = length(args) >= 2 ? abspath(expanduser(args[2])) :
        joinpath(root, "results", "CubeAnalysis_$(Dates.format(today(), "yyyy-mm-dd"))")
    base_path = normpath(joinpath(@__DIR__, "..", "config", "cube_analysis.toml"))
    base = CubeAnalysis.load_config(base_path)
    save_outputs = CubeAnalysis.save_data(base)
    simulations = simulation_directories(root)
    isempty(simulations) && error("No complete simulation found under $root")
    if length(args) >= 3
        limit = parse(Int, args[3])
        limit > 0 || error("The optional simulation limit must be positive")
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
            shape, box_size, _ = cube_geometry(joinpath(input_dir, "density.fits"))
            @info "Already complete; skipping" index total=length(simulations) name
            push!(rows, (; simulation=name, status="skipped_complete", elapsed=0.0,
                shape, box_size, output=output_dir, message="manifest already present"))
            save_outputs && write_report(report_path, rows)
            continue
        end
        free = available_gib(output_root)
        free >= 1.0 || error(@sprintf("Only %.2f GiB remain; stopping before %s", free, name))
        cfg, shape, box_size = configure(base, input_dir, output_dir)
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
    CubeAnalysis.plot_xi_vs_sigma_v!(xi_files, simulations, output_root, ["png"];
        overwrite=false, box_size=(50.0, 50.0, 50.0), axis_order=("x", "y", "z"))
    ensemble = save_outputs ? write_ensemble_tables(output_root, simulations) : String[]
    @info "Batch finished" complete total=length(rows) report=(save_outputs ? report_path : "disabled") available_gib=available_gib(output_root)
    save_outputs && @info "Ensemble comparison tables" files=ensemble
    complete == length(rows) || error("$(length(rows) - complete) simulation(s) failed")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
