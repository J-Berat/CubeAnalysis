function latex_escape_text(value)
    characters = collect(string(value))
    output = IOBuffer()
    index = 1
    while index <= length(characters)
        character = characters[index]
        if character == '^'
            stop = index + 1
            stop <= length(characters) && characters[stop] in ('+', '-') && (stop += 1)
            first_digit = stop
            while stop <= length(characters) && (isdigit(characters[stop]) || characters[stop] == '.')
                stop += 1
            end
            if stop > first_digit
                print(output, "^{", String(characters[index + 1:stop - 1]), "}")
                index = stop
                continue
            end
            print(output, "\\wedge ")
        elseif character == '\\'
            print(output, "\\backslash ")
        elseif character in ('{', '}', '_', '%', '&', '#', '$')
            print(output, '\\', character)
        elseif character == '~'
            print(output, "\\sim ")
        elseif character == ' '
            print(output, "\\ ")
        else
            print(output, character)
        end
        index += 1
    end
    return String(take!(output))
end

latex_text(value::LaTeXString) = value
latex_text(value) = latexstring("\\mathrm{", latex_escape_text(value), "}")

function latex_number(value::Real)
    !isfinite(value) && return latexstring(string(value))
    value == 0 && return latexstring("0")
    rendered = @sprintf("%.4g", Float64(value))
    matched = match(r"^([+-]?[0-9.]+)e([+-]?[0-9]+)$", rendered)
    isnothing(matched) && return latexstring(rendered)
    exponent = parse(Int, matched.captures[2])
    return latexstring(matched.captures[1], "\\times 10^{", exponent, "}")
end

latex_tickformat(values) = latex_number.(values)

function latex_axis(parent; kwargs...)
    options = Dict{Symbol,Any}(kwargs)
    pop!(options, :title, nothing)
    for key in (:xlabel, :ylabel)
        haskey(options, key) && (options[key] = latex_text(options[key]))
    end
    get!(options, :xtickformat, latex_tickformat)
    get!(options, :ytickformat, latex_tickformat)
    return Axis(parent; options...)
end

function latex_colorbar(parent, plot; kwargs...)
    options = Dict{Symbol,Any}(kwargs)
    haskey(options, :label) && (options[:label] = latex_text(options[:label]))
    get!(options, :tickformat, latex_tickformat)
    return Colorbar(parent, plot; options...)
end

latex_layout_label(parent, text; kwargs...) = nothing
latex_legend_label(text) = latex_text(text)
