struct VectorMagnitude{A<:AbstractArray{Float32,3}} <: AbstractArray{Float32,3}
    x::A
    y::A
    z::A
    factor::Float32
end

Base.size(field::VectorMagnitude) = size(field.x)
Base.IndexStyle(::Type{<:VectorMagnitude}) = IndexLinear()
@inline Base.getindex(field::VectorMagnitude, index::Int) = field.factor * sqrt(
    field.x[index]^2 + field.y[index]^2 + field.z[index]^2)

struct ProductField <: AbstractArray{Float32,3}
    factors::Vector{AbstractArray{Float32,3}}
    factor::Float32
    shape::NTuple{3,Int}
end

mutable struct LazyFieldStore <: AbstractDict{String,AbstractArray{Float32,3}}
    cfg::Any
    names::Vector{String}
    reference_size::Union{Nothing,NTuple{3,Int}}
end

Base.length(store::LazyFieldStore) = length(store.names)
Base.keys(store::LazyFieldStore) = store.names
Base.haskey(store::LazyFieldStore, name) = string(name) in store.names
function Base.iterate(store::LazyFieldStore, state=1)
    state > length(store.names) && return nothing
    name = store.names[state]
    return (name => store[name], state + 1)
end

function Base.getindex(store::LazyFieldStore, raw_name)
    name = string(raw_name)
    name in store.names || throw(KeyError(name))
    cfg = store.cfg
    value = if haskey(cfg["fields"], name)
        spec = cfg["fields"][name]
        path = resolve_path(cfg["run"]["input_dir"], string(spec["file"]))
        read_cube(path, spec, name)
    else
        derived = get(cfg, "derived", Dict())
        vector_index = findfirst(spec -> string(spec["name"]) == name,
            get(derived, "vectors", Any[]))
        product_index = findfirst(spec -> string(spec["name"]) == name,
            get(derived, "products", Any[]))
        if !isnothing(vector_index)
            spec = derived["vectors"][vector_index]
            components = string.(spec["components"])
            arrays = [store[component] for component in components]
            VectorMagnitude(arrays[1], arrays[2], arrays[3], Float32(get(spec, "factor", 1.0)))
        elseif !isnothing(product_index)
            spec = derived["products"][product_index]
            arrays = AbstractArray{Float32,3}[store[factor] for factor in string.(spec["factors"])]
            ProductField(arrays, Float32(get(spec, "factor", 1.0)), size(first(arrays)))
        else
            throw(KeyError(name))
        end
    end
    if isnothing(store.reference_size)
        store.reference_size = size(value)
    elseif size(value) != store.reference_size
        error("Field '$name' has size $(size(value)); expected $(store.reference_size)")
    end
    budget_gb = Float64(get(cfg["run"], "memory_budget_gb", 0.0))
    if budget_gb > 0
        bytes = value isa Array ? sizeof(eltype(value)) * length(value) : 0
        bytes <= budget_gb * 1024^3 || error(@sprintf(
            "Field '%s' alone requires %.2f GiB, above memory_budget_gb=%.2f",
            name, bytes / 1024^3, budget_gb))
    end
    return value
end

Base.size(field::ProductField) = field.shape
Base.IndexStyle(::Type{ProductField}) = IndexLinear()
@inline function Base.getindex(field::ProductField, index::Int)
    value = field.factor
    @inbounds for source in field.factors
        value *= source[index]
    end
    return value
end

function resident_bytes(fields)
    fields isa LazyFieldStore && return 0
    seen = IdSet{Any}()
    total = 0
    function account(array)
        array in seen && return
        push!(seen, array)
        if array isa Array
            total += sizeof(eltype(array)) * length(array)
        elseif array isa VectorMagnitude
            account(array.x); account(array.y); account(array.z)
        elseif array isa ProductField
            foreach(account, array.factors)
        end
    end
    foreach(account, values(fields))
    return total
end

function lazy_field_metadata(cfg)
    default_colormap = Symbol(get(get(cfg, "render", Dict()), "colormap", "viridis"))
    metadata = Dict{String,FieldMeta}()
    for (name, spec) in cfg["fields"]
        metadata[string(name)] = field_metadata(string(name), spec, default_colormap)
    end
    derived = get(cfg, "derived", Dict())
    for spec in [get(derived, "vectors", Any[]); get(derived, "products", Any[])]
        name = string(spec["name"])
        haskey(metadata, name) && error("Derived field '$name' duplicates an input field")
        metadata[name] = field_metadata(name, spec, default_colormap)
    end
    return metadata
end

function load_fields_lazy(cfg)
    input_dir = cfg["run"]["input_dir"]
    isdir(input_dir) || error("Input directory does not exist: $input_dir")
    for (name, spec) in cfg["fields"]
        haskey(spec, "file") || error("Missing fields.$name.file")
        path = resolve_path(input_dir, string(spec["file"]))
        isfile(path) || error("Missing cube file for '$name': $path")
    end
    metadata = lazy_field_metadata(cfg)
    names = sort!(collect(keys(metadata)))
    return LazyFieldStore(cfg, names, nothing), metadata
end

function enforce_memory_budget(fields, cfg; extra_bytes=0, context="loaded fields")
    budget_gb = Float64(get(cfg["run"], "memory_budget_gb", 0.0))
    budget_gb <= 0 && return
    required = resident_bytes(fields) + extra_bytes
    budget = round(Int, budget_gb * 1024^3)
    required <= budget || error(@sprintf(
        "Memory budget exceeded for %s: %.2f GiB required, %.2f GiB allowed",
        context, required / 1024^3, budget / 1024^3))
end

function enforce_working_set(cfg, required_bytes; context="analysis")
    budget_gb = Float64(get(cfg["run"], "memory_budget_gb", 0.0))
    budget_gb <= 0 && return
    budget = budget_gb * 1024^3
    required_bytes <= budget || error(@sprintf(
        "Memory budget exceeded for %s: %.2f GiB required, %.2f GiB allowed",
        context, required_bytes / 1024^3, budget / 1024^3))
end
