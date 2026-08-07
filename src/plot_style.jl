const PLOT_INK = RGBf(31 / 255, 41 / 255, 55 / 255)
const PLOT_MUTED = RGBf(107 / 255, 114 / 255, 128 / 255)
const PLOT_BLUE = RGBf(35 / 255, 99 / 255, 151 / 255)
const PLOT_ORANGE = RGBf(221 / 255, 107 / 255, 32 / 255)
const PLOT_TEAL = RGBf(42 / 255, 133 / 255, 120 / 255)
const PLOT_GOLD = RGBf(226 / 255, 174 / 255, 54 / 255)
const PLOT_PURPLE = RGBf(120 / 255, 86 / 255, 148 / 255)
const PLOT_RED = RGBf(190 / 255, 62 / 255, 62 / 255)
const PLOT_SERIES = (PLOT_BLUE, PLOT_ORANGE, PLOT_TEAL, PLOT_PURPLE, PLOT_GOLD)
const PLOT_LINESTYLES = (:solid, :dash, :dot, :dashdot)
const PLOT_MARKERS = (:circle, :rect, :utriangle, :diamond, :cross)

series_color(index::Integer) = PLOT_SERIES[mod1(index, length(PLOT_SERIES))]
series_linestyle(index::Integer) = PLOT_LINESTYLES[mod1(index, length(PLOT_LINESTYLES))]
series_marker(index::Integer) = PLOT_MARKERS[mod1(index, length(PLOT_MARKERS))]

function publication_figure(; size=(900, 620), fontsize=18, kwargs...)
    return Figure(; size, fontsize, backgroundcolor=:white, figure_padding=18, kwargs...)
end

function latex_escape_text(value)
    characters = collect(string(value))
    output = IOBuffer()
    index = 1
    while index <= length(characters)
        if index + 4 <= length(characters) && String(characters[index:index + 4]) == "log10"
            print(output, "\\log_{10}")
            index += 5
            continue
        end
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
        elseif character in ('{', '}', '%', '&', '#', '$')
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

function latex_log_ticks(vmin, vmax)
    isfinite(vmin) && isfinite(vmax) && 0 < vmin < vmax ||
        return (Float64[], LaTeXString[])
    lower, upper = log10(vmin), log10(vmax)
    target_step = (upper - lower) / 4
    candidates = (0.25, 0.5, 1.0, 2.0, 5.0)
    step = candidates[argmin(abs.(log.(collect(candidates) ./ max(target_step, eps()))))]
    first_exponent = ceil(lower / step) * step
    last_exponent = floor(upper / step) * step
    exponents = collect(first_exponent:step:last_exponent)
    values = 10.0 .^ exponents
    labels = map(exponents) do exponent
        rounded = round(exponent; digits=8)
        rendered = isinteger(rounded) ? string(Int(rounded)) : @sprintf("%.2g", rounded)
        latexstring("10^{", rendered, "}")
    end
    return values, labels
end

function latex_axis(parent; kwargs...)
    options = Dict{Symbol,Any}(kwargs)
    xlogcoordinates = Bool(pop!(options, :xlogcoordinates, false))
    ylogcoordinates = Bool(pop!(options, :ylogcoordinates, false))
    for key in (:xlabel, :ylabel)
        haskey(options, key) && (options[key] = latex_text(options[key]))
    end
    options[:xgridvisible] = false
    options[:ygridvisible] = false
    options[:xminorgridvisible] = false
    options[:yminorgridvisible] = false
    get!(options, :backgroundcolor, :white)
    get!(options, :leftspinecolor, PLOT_INK)
    get!(options, :bottomspinecolor, PLOT_INK)
    get!(options, :rightspinevisible, false)
    get!(options, :topspinevisible, false)
    get!(options, :spinewidth, 1.2)
    get!(options, :xtickcolor, PLOT_INK)
    get!(options, :ytickcolor, PLOT_INK)
    get!(options, :xminortickcolor, PLOT_MUTED)
    get!(options, :yminortickcolor, PLOT_MUTED)
    get!(options, :xticklabelcolor, PLOT_INK)
    get!(options, :yticklabelcolor, PLOT_INK)
    get!(options, :xlabelcolor, PLOT_INK)
    get!(options, :ylabelcolor, PLOT_INK)
    get!(options, :xtickwidth, 1.1)
    get!(options, :ytickwidth, 1.1)
    get!(options, :xminortickwidth, 0.8)
    get!(options, :yminortickwidth, 0.8)
    get!(options, :xticksize, 7)
    get!(options, :yticksize, 7)
    get!(options, :xminorticksize, 4)
    get!(options, :yminorticksize, 4)
    get!(options, :xtickalign, 1)
    get!(options, :ytickalign, 1)
    get!(options, :xlabelpadding, 10)
    get!(options, :ylabelpadding, 12)
    get!(options, :xticklabelpad, 6)
    get!(options, :yticklabelpad, 6)
    native_xlog = get(options, :xscale, identity) === log10
    native_ylog = get(options, :yscale, identity) === log10
    if xlogcoordinates || native_xlog
        options[:xminorticksvisible] = true
        get!(options, :xminorticks, IntervalsBetween(xlogcoordinates ? 10 : 9))
    end
    if ylogcoordinates || native_ylog
        options[:yminorticksvisible] = true
        get!(options, :yminorticks, IntervalsBetween(ylogcoordinates ? 10 : 9))
    end
    native_xlog && get!(options, :xticks, latex_log_ticks)
    native_ylog && get!(options, :yticks, latex_log_ticks)
    get!(options, :xtickformat, latex_tickformat)
    get!(options, :ytickformat, latex_tickformat)
    return Axis(parent; options...)
end

function latex_colorbar(parent, plot; kwargs...)
    options = Dict{Symbol,Any}(kwargs)
    logcoordinates = Bool(pop!(options, :logcoordinates, false))
    haskey(options, :label) && (options[:label] = latex_text(options[:label]))
    if logcoordinates
        options[:minorticksvisible] = true
        get!(options, :minorticks, IntervalsBetween(10))
    end
    get!(options, :width, 16)
    get!(options, :spinewidth, 1.0)
    get!(options, :ticksize, 6)
    get!(options, :tickwidth, 1.0)
    get!(options, :tickcolor, PLOT_INK)
    get!(options, :ticklabelcolor, PLOT_INK)
    get!(options, :labelcolor, PLOT_INK)
    get!(options, :ticklabelpad, 6)
    get!(options, :labelpadding, 12)
    get!(options, :tickformat, latex_tickformat)
    return Colorbar(parent, plot; options...)
end

latex_legend_label(text) = latex_text(text)

function publication_legend!(axis; kwargs...)
    options = Dict{Symbol,Any}(kwargs)
    get!(options, :framevisible, false)
    get!(options, :backgroundcolor, (:white, 0.88))
    get!(options, :labelcolor, PLOT_INK)
    get!(options, :labelsize, 16)
    get!(options, :padding, (7, 7, 5, 5))
    get!(options, :patchsize, (34, 18))
    get!(options, :rowgap, 4)
    return axislegend(axis; options...)
end
