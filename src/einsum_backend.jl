using PyCall


struct EinsumContractionResult{T<:AbstractFloat}
    kept_addrs::Vector{Any}
    array::Array{T}
end

# Internal helper routines

# Give each variable index a label usable in einsum strings.
function _build_label_map(fg::FactorGraph{N}) where N
    oe = pyimport("opt_einsum")
    labels = Dict{Int,Char}()
    for i in 1:N
        s = String(oe[:get_symbol](i-1))
        labels[i] = s[1]
    end
    return labels
end

# Collect unique factor nodes from the graph.
function _unique_factors(fg::FactorGraph{N}) where N
    seen = Set{Int}()
    out = FactorNode{N}[]
    for vnode in values(fg.var_nodes)
        for fn in factor_nodes(vnode)
            if !(fn.id in seen)
                push!(seen, fn.id)
                push!(out, fn)
            end
        end
    end
    return out
end

# Convert a FactorNode's global N-dim log array into a compact dense tensor (linear domain)
# whose axes correspond exactly to node.vars in that order.
function _small_tensor_linear(fn::FactorNode{N}) where N
    A = fn.log_factor
    keep = sort(fn.vars)
    drop = setdiff(collect(1:N), keep)
    small = dropdims(A; dims=Tuple(drop))
    if keep != fn.vars
        perm = [findfirst(==(v), keep) for v in fn.vars]
        small = permutedims(small, perm)
    end
    return small   # return LOG values here
end



# Build "a b, b c, ... -> out" style einsum string.
function _make_einsum_str(factors, labels::Dict{Int,Char}, kept_idxs::Vector{Int})
    inputs = String[]
    for fn in factors
        # concatenate the labels for this factor's vars
        s = String([labels[i] for i in fn.vars])
        push!(inputs, s)
    end
    out = String([labels[i] for i in kept_idxs])
    return isempty(out) ? string(join(inputs, ","), "->") :
                          string(join(inputs, ","), "->", out)
end


# Public API
function contract_factor_graph_einsum(fg::FactorGraph{N};
        kept_addrs = Any[],
        optimize::Union{String,Symbol} = "auto",
        order::Union{Nothing,AbstractVector} = nothing,
        backend::Symbol = :auto,
        jit::Bool = true,
        cache::Bool = true,
        dtype::Symbol = :float64) where {N}

    # normalize inputs
    kept_addrs = Any[kept_addrs...]
    optimize   = String(optimize)

    # init Python backends & select device
    _epb_init!()
    backend = _epb_choose_backend(backend)

    # labels and factor tensors
    facs = _unique_factors(fg)
    key_struct = (fg.num_factors, map(fn -> (fn.id, fn.vars), facs))
    (labels, subs) = get!(_LABELS_SUBS_CACHE, key_struct) do
        local L = _build_label_map(fg)
        local S = String[ String([L[i] for i in fn.vars]) for fn in facs ]
        (L, S)
    end

    # _small_tensor_linear now returns LINEAR tensors
    tensJ_log = [ _small_tensor_linear(fn) for fn in facs ]  # LOG tensors aligned to fn.vars
    shapesJ = [ size(t) for t in tensJ_log ]
    akey = (backend, shapesJ)

    if backend == :jax
        xs = PyObject[
            _epb_exp(_epb_to_backend_array(t; backend=backend, dtype=dtype); backend=backend)
            for t in tensJ_log
        ]
        _ARRAY_CACHE[akey] = xs
    else
        if haskey(_ARRAY_CACHE, akey)
            xs = _ARRAY_CACHE[akey]
            for (x, tlog) in zip(xs, tensJ_log)
                x[:] = exp.(tlog)   # refresh cached arrays in linear space
            end
        else
            xs = PyObject[
                _epb_to_backend_array(exp.(t); backend=backend, dtype=dtype)
                for t in tensJ_log
            ]
            _ARRAY_CACHE[akey] = xs
        end
    end


    # Helper to turn kept_addrs into output subscript
    kept_idxs = isempty(kept_addrs) ? Int[] : [ addr_to_idx(fg, a) for a in kept_addrs ]
    out_sub = String([labels[i] for i in kept_idxs])

    if order === nothing
        # single-shot contraction with opt_einsum planner 
        spec = _make_einsum_str(facs, labels, kept_idxs)
        pres = _epb_contract(spec, xs; backend=backend, optimize=optimize, cache=cache, jit=jit)
        return EinsumContractionResult(kept_addrs, _epb_py_to_array(pres; dtype=dtype))
    else
        # explicit elimination order (manual VE via einsum) 
        # Copy so we can mutate
        subs_work = copy(subs)
        xs_work   = copy(xs)

        # Map addr -> label char
        function label_for_addr(a)
            i = addr_to_idx(fg, a)
            return labels[i]
        end

        # At each step: gather all tensors that contain the label, multiply them,
        # and sum out that label.
        for a in order
            lab = label_for_addr(a)
            # Find factors containing this label
            hit = findall(s -> occursin(string(lab), s), subs_work)
            isempty(hit) && continue  # already eliminated
            # Subscripts and arrays to contract
            # Output subscript = union of input labels minus the eliminated one,
            # in a stable left-to-right order
            # find factors that contain this label
            sub_in = subs_work[hit]
            xs_in  = xs_work[hit]


            seen = Set{Char}()
            out_chars = Char[]
            for s in sub_in
                for c in s
                    if c != lab && !(c in seen)
                        push!(seen, c)
                        push!(out_chars, c)
                    end
                end
            end
            sub_out = String(out_chars)

            # Perform subset contraction
            y = _epb_contract_subset(sub_in, xs_in, sub_out; backend=backend, optimize=optimize, cache=cache, jit=jit)
            for i in sort(hit; rev=true)
                splice!(subs_work, i)
                splice!(xs_work, i)
            end
            push!(subs_work, sub_out)
            push!(xs_work, y)
        end

        spec_final = isempty(out_sub) ? join(subs_work, ",") : (join(subs_work, ",") * "->" * out_sub)
        pres = _epb_contract(spec_final, xs_work; backend=backend, optimize=optimize, cache=cache, jit=jit)
        return EinsumContractionResult(kept_addrs, _epb_py_to_array(pres; dtype=dtype))

    end
end