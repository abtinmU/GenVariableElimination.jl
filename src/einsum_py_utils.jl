using PyCall

ENV["JAX_ENABLE_X64"]="1"

for (name, init) in [
    (:_OE,         :(Ref{PyObject}())),                         # opt_einsum
    (:_NP,         :(Ref{PyObject}())),                         # numpy
    (:_CP,         :(Ref{Union{PyObject,Nothing}}(nothing))),   # cupy (optional)
    (:_JAX,        :(Ref{Union{PyObject,Nothing}}(nothing))),   # jax (optional)
    (:_JNP,        :(Ref{Union{PyObject,Nothing}}(nothing))),   # jax.numpy (optional)
    (:_EXPR_CACHE, :(IdDict{Any,PyObject}())),                  # compiled expressions
    (:_EPB_INIT,   :(Ref{Bool}(false)))                         # init guard
]
    if !isdefined(@__MODULE__, name)
        @eval const $(name) = $(init)
    end
end

if !isdefined(@__MODULE__, :_LABELS_SUBS_CACHE)
    const _LABELS_SUBS_CACHE = Dict{Any, Tuple{Dict{Int,Char}, Vector{String}}}()
end

if !isdefined(@__MODULE__, :_ARRAY_CACHE)
    const _ARRAY_CACHE = Dict{Tuple{Symbol,Vector{Tuple}}, Vector{PyObject}}()
end

# accepts a real NumPy ndarray PyObject
function _np_to_julia(host_np::PyObject, ::Type{T}) where {T<:AbstractFloat}
    # 0-D numpy scalar
    is0d = try Int(host_np[:ndim]) == 0 catch; false end
    if is0d
        return reshape([T(host_np[:item]())], ())
    end
    # zero-copy wrap then copy into a Julia Array
    A = Array(PyArray(host_np))
    return T.(A)
end

# also accept Julia arrays directly
function _np_to_julia(host_np::AbstractArray, ::Type{T}) where {T<:AbstractFloat}
    if ndims(host_np) == 0
        return reshape([T(host_np[])], ())
    else
        return T.(host_np)
    end
end


function _epb_py_to_array(x; dtype::Symbol = :float64, debug::Bool = true)
    T = dtype === :float32 ? Float32 : Float64

    dbg(msg) = debug ? (println("[epb] ", msg); flush(stdout)) : nothing
    pytype_str(o) = (try string(pytype(o)) catch; "unavailable" end)
    np_ndim(a)    = (try Int(a[:ndim]) catch; -1 end)
    np_shape(a)   = (try string(a[:shape]) catch; "?" end)
    np_dtype(a)   = (try string(a[:dtype]) catch; "n/a" end)
    np_info(a)    = "(dtype=$(np_dtype(a)), ndim=$(np_ndim(a)), shape=$(np_shape(a)))"

    if x isa AbstractArray
        if ndims(x) == 0
            return reshape([T(x[])], ())
        else
            return T.(x)
    end

    elseif x isa Number
        return reshape(fill(T(x), 1), ())
    end

    # CuPy -> host NumPy -> Julia
    try
        cp = pyimport("cupy")
        is_cp = try pycall(pybuiltin("isinstance"), Bool, x, cp[:ndarray]) catch; false end
        if is_cp
            np_host = cp[:asnumpy](x)
            return _np_to_julia(np_host, T)
        end
    catch e
    end

    # JAX device_get then force to real NumPy ndarray
    try
        jax = pyimport("jax")
        np  = pyimport("numpy")
        host = jax[:device_get](x)
        host_np = np[:array](host, copy=true)
        return _np_to_julia(host_np, T)
    catch e
    end

    # DLPack fallback
    try
        np = pyimport("numpy")
        host_np = np[:from_dlpack](x)
        return _np_to_julia(host_np, T)
    catch e
    end

    # anything NumPy can view -> Julia
    try
        np = pyimport("numpy")

        host_np = nothing
        has_array = try pyhasattr(x, "__array__") catch; false end

        if has_array
            try
                host_np = x[:__array__]()
            catch e
            end
        end

        if host_np === nothing
            try
                host_np = np[:asarray](x)
            catch e
            end
        end

        if host_np === nothing
            host_np = np[:array](x, copy=true)
        end

        return _np_to_julia(host_np, T)
    catch e
    end

    # scalar-like
    try
        pyf = pybuiltin("float")(x)
        return reshape(T(pyf), ())
    catch e
        xtype = try string(pytype(x)) catch; "unknown Python type" end
        msg = "Could not convert to a Julia array. Got $(repr(x)) of type $(xtype)"
        error(msg)
    end
end



function _epb_exp(x; backend::Symbol)
    _epb_init!()
    if backend == :jax && _JNP[] !== nothing
        return _JNP[][:exp](x)
    elseif backend == :cupy && _CP[] !== nothing
        return _CP[][:exp](x)
    else
        return _NP[][:exp](x)
    end
end

function _epb_log(x; backend::Symbol)
    _epb_init!()
    if backend == :jax && _JNP[] !== nothing
        return _JNP[][:log](x)
    elseif backend == :cupy && _CP[] !== nothing
        return _CP[][:log](x)
    else
        return _NP[][:log](x)
    end
end



# Import python libs
function _epb_init!()
    _EPB_INIT[] && return
    _OE[] = pyimport("opt_einsum")
    _NP[] = pyimport("numpy")
    try
        _CP[] = pyimport("cupy")
    catch
        _CP[] = nothing
    end
    try
        _JAX[] = pyimport("jax")
        _JNP[] = _JAX[][:numpy]
    catch
        _JAX[] = nothing
        _JNP[] = nothing
    end
    _EPB_INIT[] = true
    return nothing
end

function _epb_reset!()
    _EPB_INIT[] = false
    try _OE[]  = PyNULL() catch end
    try _NP[]  = PyNULL() catch end
    _CP[]  = nothing
    _JAX[] = nothing
    _JNP[] = nothing
    empty!(_EXPR_CACHE)
    nothing
end

# Choose backend
function _epb_choose_backend(sym::Symbol)
    _epb_init!()
    sym == :auto || return sym
    # Prefer JAX on GPU
    if _JAX[] !== nothing
        try
            gpus = _JAX[][:devices]("gpu")
            if length(gpus) > 0
                return :jax
            end
        catch
        end
    end
    # Fall back to CuPy GPU, then JAX CPU, then NumPy
    if _CP[] !== nothing
        return :cupy
    end
    if _JAX[] !== nothing
        return :jax
    end
    return :numpy
end


# Julia array -> backend array
function _epb_to_backend_array(A::AbstractArray; backend::Symbol=:numpy, dtype::Symbol=:float64)
    _epb_init!()
    Tstr = dtype === :float32 ? "float32" : "float64"
    if backend == :jax
        has64 = false
        try
            has64 = Bool(_JAX[][:config][:read]("jax_enable_x64"))
        catch
            has64 = false
        end
        Tstr = has64 ? Tstr : "float32"
        arr = _JNP[][:array](A, dtype=_JNP[][:dtype](Tstr))
        # Pin to GPU if available
        try
            gpus = _JAX[][:devices]("gpu")
            if length(gpus) > 0
                arr = _JAX[][:device_put](arr, gpus[1])
            end
        catch
        end
        return arr

    elseif backend == :cupy
        return _CP[] === nothing ? _NP[][:array](A, dtype=_NP[][:dtype](Tstr)) :
                                   _CP[][:asarray](A, dtype=_CP[][:dtype](Tstr))
    else
        return _NP[][:array](A, dtype=_NP[][:dtype](Tstr))
    end
end

# Get (and cache) a compiled opt_einsum contract expression; JIT with JAX if requested
function _epb_get_expr(spec::String, shapes::Vector{Tuple{Vararg{Int}}},
                       backend::Symbol, optimize; cache::Bool=true, jit::Bool=true)
    _epb_init!()
    key = (spec, shapes, backend, optimize, jit)
    if cache && haskey(_EXPR_CACHE, key)
        return _EXPR_CACHE[key]
    end
    expr = _OE[][:contract_expression](spec, shapes...; optimize=optimize)
    if backend == :jax && jit && _JAX[] !== nothing
        expr = _JAX[][:jit](expr)
    end
    cache && (_EXPR_CACHE[key] = expr)
    return expr
end

# Read shapes from Python arrays (NumPy/CuPy/JAX all have .shape)
_epb_shapes(xs::Vector{PyObject}) =
    [ Tuple(pyconvert.(Int, x[:shape])) for x in xs ]


# Single-call contract
function _epb_contract(spec::String, xs::Vector{PyObject};
                       backend::Symbol, optimize, cache::Bool, jit::Bool)
    _epb_init!()
    try
        shapes = _epb_shapes(xs)
        expr = _epb_get_expr(spec, shapes, backend, optimize; cache=cache, jit=jit)
        return expr(xs...)
    catch
        return _OE[][:contract](spec, xs...; optimize=optimize)
    end
end

# Manual one-step contract for a subset (used in explicit-order modes)
function _epb_contract_subset(subs_in::Vector{String}, xs_in::Vector{PyObject}, out_sub::String;
                              backend::Symbol, optimize, cache::Bool, jit::Bool)
    spec = join(subs_in, ",") * "->" * out_sub
    return _epb_contract(spec, xs_in; backend=backend, optimize=optimize, cache=cache, jit=jit)
end
