#!/usr/bin/env julia

using CubeAnalysis

config_path = isempty(ARGS) ?
    normpath(joinpath(@__DIR__, "..", "config", "cube_analysis.toml")) :
    abspath(ARGS[1])

isfile(config_path) || error("Parameter file not found: $config_path")
result = run_analysis(config_path)
@info "Systematic cube analysis complete" output_dir=result.output_dir files=length(result.files)
