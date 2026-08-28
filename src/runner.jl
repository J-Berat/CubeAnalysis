sanitize(name::AbstractString) = lowercase(replace(name, r"[^A-Za-z0-9]+" => "_"))

function write_manifest!(files, cfg, output_dir, fields; overwrite=true, timings=Dict{String,Float64}())
    path = joinpath(output_dir, "analysis_manifest.toml")
    manifest = Dict(
        "source_config" => cfg["_config_path"],
        "input_dir" => cfg["run"]["input_dir"],
        "output_dir" => cfg["run"]["output_dir"],
        "cube_size" => collect(analysis_field_size(fields, first(keys(fields)))),
        "fields" => sort!(collect(keys(fields))),
        "generated_files" => sort!(basename.(files)),
        "julia_version" => string(VERSION),
        "cubeanalysis_version" => string(pkgversion(CubeAnalysis)),
        "package_versions" => Dict(
            "CairoMakie" => string(pkgversion(CairoMakie)),
            "FFTW" => string(pkgversion(FFTW)),
            "FITSIO" => string(pkgversion(FITSIO)),
            "HDF5" => string(pkgversion(HDF5)),
        ),
        "timings_seconds" => timings,
        "config" => Dict(k => v for (k, v) in cfg if !startswith(string(k), "_")),
    )
    write_text_output(path; overwrite) do io
        TOML.print(io, manifest; sorted=true)
    end
    push!(files, path)
end

function run_analysis_config(cfg::AbstractDict)
    final_output_dir = cfg["run"]["output_dir"]
    atomic_directory = Bool(get(cfg["run"], "atomic_directory", false))
    overwrite = Bool(get(cfg["run"], "overwrite", true))
    atomic_directory && !overwrite && ispath(final_output_dir) &&
        error("Output exists and overwrite=false: $final_output_dir")
    if atomic_directory
        mkpath(dirname(final_output_dir))
        output_dir = mktempdir(dirname(final_output_dir);
            prefix="$(basename(final_output_dir)).tmp-")
    else
        output_dir = final_output_dir
        mkpath(output_dir)
    end
    formats = string.(get(cfg["run"], "formats", ["png"]))
    stride = Int(get(cfg["run"], "sample_stride", 1))
    fields, metadata = load_fields(cfg)
    files = String[]
    timings = Dict{String,Float64}()

    @info "Prepared cube fields" input_dir=cfg["run"]["input_dir"] size=analysis_field_size(fields, first(keys(fields))) fields=sort!(collect(keys(fields))) field_by_field=(fields isa LazyFieldStore)
    stages = Tuple{String,Function}[]
    save_data(cfg) && requested(cfg, "quality"; default=true) && push!(stages,
        ("quality", () -> write_quality_report!(files, fields, metadata, output_dir; overwrite)))
    save_data(cfg) && requested(cfg, "summary"; default=true) && push!(stages,
        ("summary", () -> write_summary!(files, fields, metadata, output_dir, stride; overwrite)))
    requested(cfg, "slices") && push!(stages,
        ("slices", () -> plot_slices!(files, fields, metadata, cfg, output_dir, formats, overwrite)))
    requested(cfg, "projections") && push!(stages,
        ("projections", () -> plot_projections!(files, fields, metadata, cfg, output_dir, formats, overwrite)))
    requested(cfg, "histograms") && push!(stages,
        ("histograms", () -> plot_histograms!(files, fields, metadata, cfg, output_dir, formats, overwrite, stride)))
    requested(cfg, "phase_diagrams") && push!(stages,
        ("phase_diagrams", () -> plot_phase_diagrams!(files, fields, metadata, cfg, output_dir, formats, overwrite, stride)))
    requested(cfg, "relations") && push!(stages,
        ("relations", () -> plot_relations!(files, fields, metadata, cfg, output_dir, formats, overwrite)))
    requested(cfg, "power_spectra") && push!(stages,
        ("power_spectra", () -> plot_spectra!(files, fields, metadata, cfg, output_dir, formats, overwrite)))
    requested(cfg, "vector_spectra") && push!(stages,
        ("vector_spectra", () -> plot_vector_spectra!(files, fields, metadata, cfg,
            output_dir, formats, overwrite)))
    requested(cfg, "cross_spectra") && push!(stages,
        ("cross_spectra", () -> plot_cross_spectra!(files, fields, cfg,
            output_dir, formats, overwrite)))
    requested(cfg, "alignments") && push!(stages,
        ("alignments", () -> plot_alignments!(files, fields, metadata, cfg, output_dir, formats, overwrite)))
    requested(cfg, "physics_table"; default=true) && push!(stages,
        ("physics_table", () -> plot_physics_table!(files, fields, cfg, output_dir,
            formats, overwrite)))
    requested(cfg, "density_pdf"; default=true) && push!(stages,
        ("density_pdf", () -> plot_density_pdf!(files, fields, cfg, output_dir,
            formats, overwrite)))
    requested(cfg, "phase_budget"; default=true) && push!(stages,
        ("phase_budget", () -> plot_phase_budget!(files, fields, cfg, output_dir,
            formats, overwrite)))
    requested(cfg, "column_density_pdf"; default=true) && push!(stages,
        ("column_density_pdf", () -> plot_column_density_pdf!(files, fields, metadata,
            cfg, output_dir, formats, overwrite)))
    requested(cfg, "advanced_diagnostics") && push!(stages,
        ("advanced_diagnostics", () -> plot_advanced_diagnostics!(files, fields, metadata,
            cfg, output_dir, formats, overwrite)))
    requested(cfg, "phase_diagnostics") && push!(stages,
        ("phase_diagnostics", () -> plot_phase_diagnostics!(files, fields, metadata,
            cfg, output_dir, formats, overwrite)))
    requested(cfg, "synthetic_observations") && push!(stages,
        ("synthetic_observations", () -> plot_synthetic_observations!(files, fields,
            metadata, cfg, output_dir, formats, overwrite)))
    requested(cfg, "topology") && push!(stages,
        ("topology", () -> plot_topology!(files, fields, metadata, cfg,
            output_dir, formats, overwrite)))
    requested(cfg, "anisotropic_spectra") && push!(stages,
        ("anisotropic_spectra", () -> plot_anisotropic_spectra!(files, fields, metadata,
            cfg, output_dir, formats, overwrite)))
    requested(cfg, "directional_structure_functions") && push!(stages,
        ("directional_structure_functions", () -> plot_directional_structure!(files, fields,
            metadata, cfg, output_dir, formats, overwrite)))
    for (index, (name, action)) in enumerate(stages)
        timed_stage!(timings, index, length(stages), name, action)
        fields isa LazyFieldStore && GC.gc(false)
    end
    save_data(cfg) && write_manifest!(files, cfg, output_dir, fields; overwrite, timings)
    if atomic_directory
        relative_files = relpath.(files, Ref(output_dir))
        publish_output_directory(output_dir, final_output_dir; overwrite)
        files = joinpath.(final_output_dir, relative_files)
        output_dir = final_output_dir
    end
    return AnalysisResult(output_dir, files, sort!(collect(keys(fields))))
end

function run_snapshot_series(cfg::AbstractDict)
    snapshots = cfg["snapshots"]
    parent_output = cfg["run"]["output_dir"]
    mkpath(parent_output)
    overwrite = Bool(get(cfg["run"], "overwrite", true))
    results = AnalysisResult[]
    series_files = String[]
    rows = String[]
    for (index, snapshot) in enumerate(snapshots)
        name = string(get(snapshot, "name", @sprintf("snapshot_%04d", index)))
        @info "Running snapshot" index total=length(snapshots) name
        child = deepcopy(cfg)
        delete!(child, "snapshots")
        child["run"]["input_dir"] = resolve_path(cfg["_config_dir"], string(snapshot["input_dir"]))
        subdirectory = string(get(snapshot, "output_subdir", sanitize(name)))
        child["run"]["output_dir"] = joinpath(parent_output, subdirectory)
        result = run_analysis_config(child)
        push!(results, result); append!(series_files, result.files)
        summary = joinpath(result.output_dir, "cube_summary.csv")
        if isfile(summary)
            summary_lines = readlines(summary)
            time = get(snapshot, "time", index)
            for line in summary_lines[2:end]
                push!(rows, "$(replace(name, ',' => ';')),$time,$line")
            end
        end
    end
    if save_data(cfg)
        evolution = joinpath(parent_output, "snapshot_evolution.csv")
        write_text_output(evolution; overwrite) do io
            println(io, "snapshot,time,field,label,unit,nx,ny,nz,finite_samples,minimum,q01,median,mean,std,q99,maximum")
            foreach(line -> println(io, line), rows)
        end
        push!(series_files, evolution)
    end
    fields = sort!(unique!(reduce(vcat, [result.fields for result in results]; init=String[])))
    return AnalysisResult(parent_output, series_files, fields)
end

function run_analysis(config_path::AbstractString)
    cfg = load_config(config_path)
    snapshots = get(cfg, "snapshots", Any[])
    return isempty(snapshots) ? run_analysis_config(cfg) : run_snapshot_series(cfg)
end
