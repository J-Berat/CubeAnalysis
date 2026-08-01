function write_quality_report!(files, fields, metadata, output_dir; overwrite=true)
    path = joinpath(output_dir, "input_quality.csv")
    write_text_output(path; overwrite) do io
        println(io, "field,cells,finite,nan,posinf,neginf,nonpositive,finite_fraction")
        for name in sort!(collect(keys(fields)))
            finite = 0; nan = 0; posinf = 0; neginf = 0; nonpositive = 0
            inspect = function(raw)
                value = Float64(raw)
                if isnan(value)
                    nan += 1
                elseif value == Inf
                    posinf += 1
                elseif value == -Inf
                    neginf += 1
                else
                    finite += 1
                    value <= 0 && (nonpositive += 1)
                end
            end
            streamed = false
            if fields isa LazyFieldStore
                streamed = stream_aligned_hdf5(fields, [name]) do row
                    inspect(row[1])
                end
            end
            if !streamed
                for raw in fields[name]
                    inspect(raw)
                end
            end
            total = finite + nan + posinf + neginf
            @printf(io, "%s,%d,%d,%d,%d,%d,%d,%.8f\n", name, total, finite, nan,
                posinf, neginf, nonpositive, finite / total)
            if metadata[name].logscale && nonpositive > 0
                @warn "Log-scaled field contains non-positive cells" field=name cells=nonpositive
            end
        end
    end
    push!(files, path)
end

function timed_stage!(timings, index, total, name, action)
    @info "[$index/$total] Starting analysis stage" stage=name
    started = time_ns()
    action()
    elapsed = (time_ns() - started) / 1e9
    timings[name] = elapsed
    @info "[$index/$total] Completed analysis stage" stage=name seconds=round(elapsed; digits=3)
end
