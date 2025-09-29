using FunctionalCollections: PersistentSet, PersistentHashMap, dissoc, assoc, conj, disj
using StatsFuns: logsumexp

####################################################
# factor graph, variable elimination, and sampling #
####################################################

# TODO performance optimize?

struct VarNode{T,V} # T would be FactorNode, but for https://github.com/JuliaLang/julia/issues/269
    addr::Any
    factor_nodes::PersistentSet{T}
    idx_to_domain::Vector{V}
    domain_to_idx::Dict{V,Int}
end

addr(node::VarNode) = node.addr
factor_nodes(node::VarNode) = node.factor_nodes
num_values(node::VarNode) = length(node.idx_to_domain)
idx_to_value(node::VarNode{T,V}, idx::Int) where {T,V} = node.idx_to_domain[idx]::V
value_to_idx(node::VarNode{T,V}, value::V) where {T,V} = node.domain_to_idx[value]

function remove_factor_node(node::VarNode{T,V}, factor_node::T) where {T,V}
    return VarNode{T,V}(
        node.addr, disj(node.factor_nodes, factor_node),
        node.idx_to_domain, node.domain_to_idx)
end

function add_factor_node(node::VarNode{T,V}, factor_node::T) where {T,V}
    return VarNode{T,V}(
        node.addr, conj(node.factor_nodes, factor_node),
        node.idx_to_domain, node.domain_to_idx)
end

struct FactorNode{N} # N is the number of variables in the (original) factor graph
    id::Int
    vars::Vector{Int} # immutable
    log_factor::Array{Float64,N} # immutable
end

vars(node::FactorNode) = node.vars
get_log_factor(node::FactorNode) = node.log_factor

struct FactorGraph{N}
    num_factors::Int
    var_nodes::PersistentHashMap{Int,VarNode}

    # NOTE: when variables get eliminated from a factor graph, they don't get reindex]
    # (i.e. these fields are unchanged)
    addr_to_idx::Dict{Any,Int}
end

# just for testing purposes:
function factor_value(fg::FactorGraph, node::FactorNode{N}, values::Dict) where {N}
    idxs = (idx in vars(node) ? value_to_idx(idx_to_var_node(fg, idx), values[idx]) : 1 for idx in 1:N)
    return exp.(node.log_factor[idxs...])
end

function draw_factor_graph(fg::FactorGraph, graphviz, fname, addr_to_name)
    dot = graphviz.Digraph()
    factor_idx = 1
    for node in values(fg.var_nodes)
        shape = "ellipse"
        color = "white"
        name = addr_to_name(addr(node))
        dot[:node](name, name, shape=shape, fillcolor=color, style="filled")
        for factor_node in factor_nodes(node)
            shape = "box"
            color = "gray"
            factor_name = string(factor_node.id)
            dot[:node](factor_name, factor_name, shape=shape, fillcolor=color, style="filled")
            dot[:edge](name, factor_name)
        end
    end
    dot[:render](fname, view=true)
end


idx_to_var_node(fg::FactorGraph, idx::Int) = fg.var_nodes[idx]
addr_to_idx(fg::FactorGraph, addr) = fg.addr_to_idx[addr]

# variable elimination
# - generates a sequence of factor graphs
# - multiply all factors that mention the variable, generating a product factor, which replaces the other factors
# - then sum out the product factor, and remove the variable
# ( we could break these into two separate operations -- NO )

# all factors are of the same dimension, but with singleton dimensions for
# variables that are eliminated

function multiply_and_sum(log_factors::Vector{Array{Float64,N}}, idx_to_sum_over::Int) where {N}
    result = broadcast(+, log_factors...)
    m = maximum(result, dims=idx_to_sum_over)
    return m .+ log.(sum(exp.(result .- m), dims=idx_to_sum_over))
end

function eliminate(fg::FactorGraph{N}, addr::Any) where{N}
    eliminated_var_idx = addr_to_idx(fg, addr)
    eliminated_var_node = idx_to_var_node(fg, eliminated_var_idx)
    log_factors_to_combine = Vector{Array{Float64,N}}()
    other_involved_var_nodes = Dict{Int,VarNode{FactorNode{N}}}()
    for factor_node in factor_nodes(eliminated_var_node)
        push!(log_factors_to_combine, get_log_factor(factor_node))

        # remove the reference to this factor node from its variable nodes
        for other_var_idx::Int in vars(factor_node)
            if other_var_idx == eliminated_var_idx
                continue
            end
            if !haskey(other_involved_var_nodes, other_var_idx)
                other_var_node = idx_to_var_node(fg, other_var_idx)
                other_involved_var_nodes[other_var_idx] = other_var_node
            else
                other_var_node = other_involved_var_nodes[other_var_idx]
            end
            @assert factor_node in factor_nodes(other_var_node)
            other_var_node = remove_factor_node(other_var_node, factor_node)
            other_involved_var_nodes[other_var_idx] = other_var_node
        end
    end

    # compute the new factor
    new_log_factor = multiply_and_sum(log_factors_to_combine, eliminated_var_idx)

    # add the new factor node
    new_factor_node = FactorNode{N}(
        fg.num_factors+1, collect(keys(other_involved_var_nodes)), new_log_factor)
    for (other_var_idx, other_var_node) in other_involved_var_nodes
        other_involved_var_nodes[other_var_idx] = add_factor_node(other_var_node, new_factor_node)
    end

    # remove the eliminated var node
    new_var_nodes = dissoc(fg.var_nodes, eliminated_var_idx)

    # replace old other var nodes with new other var nodes
    for (other_var_idx, other_var_node) in other_involved_var_nodes
        new_var_nodes = assoc(new_var_nodes, other_var_idx, other_var_node)
    end

    return FactorGraph{N}(fg.num_factors+1, new_var_nodes, fg.addr_to_idx)
end

function conditional_dist(fg::FactorGraph{N}, values::Vector{Any}, addr::Any) where {N}

    # other_values must contain a value for all variables that have a factor in
    # common with variable addr in fg
    var_idx = addr_to_idx(fg, addr)
    var_node = idx_to_var_node(fg, var_idx)
    n = num_values(var_node)
    #println("conditional_dist, addr: $addr, var_idx: $var_idx, num_values: $n")
    log_probs = zeros(n)
    # TODO : writing the slow version first..
    # LATER: use generated function to generate a version that is specialized
    # to N? (unroll this loop, and inline the indices..)
    indices = Vector{Int}(undef, N)
    for i in 1:n
        for factor_node in factor_nodes(var_node)
            log_factor::Array{Float64,N} = get_log_factor(factor_node)
            fill!(indices, 1)
            for other_var_idx in vars(factor_node)
                if other_var_idx != var_idx
                    other_var_node = idx_to_var_node(fg, other_var_idx)
                    indices[other_var_idx] = value_to_idx(other_var_node, values[other_var_idx])
                end
            end
            indices[var_idx] = i
            log_probs[i] += log_factor[CartesianIndex{N}(indices...)]
        end
    end
    return exp.(log_probs .- logsumexp(log_probs))
end

struct VariableEliminationResult{N}
    elimination_order::Any
    intermediate_fgs::Vector{FactorGraph{N}}
end

#function variable_elimination(fg::FactorGraph{N}, elimination_order) where {N}
#    intermediate_fgs = Vector{FactorGraph{N}}(undef, N)
#    for addr in elimination_order
#        var_idx = addr_to_idx(fg, addr)
#        intermediate_fgs[var_idx] = fg
#        fg = eliminate(fg, addr)
#    end
#    return VariableEliminationResult(elimination_order, intermediate_fgs)
#end

# NEW: engine-aware variable elimination
function variable_elimination(fg::FactorGraph{N}, elimination_order;
                              engine::Union{Symbol,String} = :native,
                              backend::Union{Symbol,String} = :auto,
                              optimize::Union{String,Symbol} = "auto",
                              dtype::Union{Symbol,String} = :float64,
                              jit::Bool = true,
                              cache::Bool = true) where {N}
    engine_sym = Symbol(engine)
    backend_sym = Symbol(backend)
    optimize_str = String(optimize)
    dtype_sym = Symbol(dtype)

    intermediate_fgs = Vector{FactorGraph{N}}(undef, N)
    for addr in elimination_order
        var_idx = addr_to_idx(fg, addr)
        intermediate_fgs[var_idx] = fg

        if engine_sym == :native
            fg = eliminate(fg, addr)
        elseif engine_sym == :einsum
            fg = eliminate_einsum(fg, addr;
                                  backend=backend_sym, optimize=optimize_str,
                                  dtype=dtype_sym, jit=jit, cache=cache)
        elseif engine_sym == :auto
            chosen = _epb_choose_backend(backend_sym)
            if chosen == :numpy
                fg = eliminate(fg, addr)
            else
                fg = eliminate_einsum(fg, addr;
                                      backend=chosen, optimize=optimize_str,
                                      dtype=dtype_sym, jit=jit, cache=cache)
            end
        else
            error("Unknown engine: $(engine)")
        end
    end
    return VariableEliminationResult(elimination_order, intermediate_fgs)
end



# NEW: einsum-based elimination of a single variable
function eliminate_einsum(fg::FactorGraph{N}, addr::Any;
                          backend::Union{Symbol,String} = :auto,
                          optimize::Union{String,Symbol} = "auto",
                          dtype::Union{Symbol,String} = :float64,
                          jit::Bool = true,
                          cache::Bool = true) where {N}
    _epb_init!()
    backend = _epb_choose_backend(Symbol(backend))
    optimize = String(optimize)
    dtype = Symbol(dtype)

    eliminated_var_idx = addr_to_idx(fg, addr)
    eliminated_var_node = idx_to_var_node(fg, eliminated_var_idx)

    factors_to_combine = FactorNode{N}[fn for fn in factor_nodes(eliminated_var_node)]
    isempty(factors_to_combine) && return fg

    other_involved_var_nodes = Dict{Int,VarNode{FactorNode{N}}}()
    for fn in factors_to_combine
        for other_var_idx::Int in vars(fn)
            other_var_idx == eliminated_var_idx && continue
            if !haskey(other_involved_var_nodes, other_var_idx)
                other_involved_var_nodes[other_var_idx] = idx_to_var_node(fg, other_var_idx)
            end
            other_involved_var_nodes[other_var_idx] =
                remove_factor_node(other_involved_var_nodes[other_var_idx], fn)
        end
    end

    labels = _build_label_map(fg)              # Int -> Char
    elim_label = labels[eliminated_var_idx]
    label_to_idx = Dict(v => k for (k,v) in labels)

    subs_in = String[ String([labels[i] for i in fn.vars]) for fn in factors_to_combine ]


    xs_backend = PyObject[
        _epb_exp(_epb_to_backend_array(_small_tensor_linear(fn);
                                      backend=backend, dtype=dtype);
                backend=backend)
        for fn in factors_to_combine
    ]

    # Decide output labels (stable left-to-right union minus the eliminated label)
    seen = Set{Char}()
    out_chars = Char[]
    for s in subs_in
        for c in s
            if c != elim_label && !(c in seen)
                push!(seen, c)
                push!(out_chars, c)
            end
        end
    end
    sub_out  = String(out_chars)
    out_idxs = [ label_to_idx[c] for c in out_chars ]

    # Contract once, then take log on host
    y = _epb_contract_subset(subs_in, xs_backend, sub_out;
                            backend=backend, optimize=optimize, cache=cache, jit=jit)
    small_log = log.( _epb_py_to_array(y; dtype=dtype) )


    @assert size(small_log) == Tuple(num_values(idx_to_var_node(fg, i)) for i in out_idxs)

    dimsN = ntuple(i -> (i in out_idxs ? num_values(idx_to_var_node(fg, i)) : 1), N)
    new_log_factor = Array{Float64,N}(undef, dimsN...)
    view_inds = ntuple(i -> (i in out_idxs ? Colon() : 1), N)

    val = (small_log isa AbstractArray && ndims(small_log) == 0) ? small_log[] : small_log
    new_log_factor[view_inds...] = val



    # After computing out_idxs and small_log
    sorted_out = sort(out_idxs)
    if out_idxs != sorted_out
        # permutation that maps current axis order -> ascending index order
        perm = [findfirst(==(i), out_idxs) for i in sorted_out]
        small_log = permutedims(small_log, perm)
        out_idxs = sorted_out
    end

    dimsN = ntuple(i -> (i in out_idxs ? num_values(idx_to_var_node(fg, i)) : 1), N)
    new_log_factor = Array{Float64,N}(undef, dimsN...)
    view_inds = ntuple(i -> (i in out_idxs ? Colon() : 1), N)
    new_log_factor[view_inds...] = small_log




    new_factor_node = FactorNode{N}(fg.num_factors + 1, out_idxs, new_log_factor)

    for (other_var_idx, other_var_node) in other_involved_var_nodes
        other_involved_var_nodes[other_var_idx] = add_factor_node(other_var_node, new_factor_node)
    end

    new_var_nodes = dissoc(fg.var_nodes, eliminated_var_idx)
    for (other_var_idx, other_var_node) in other_involved_var_nodes
        new_var_nodes = assoc(new_var_nodes, other_var_idx, other_var_node)
    end

    return FactorGraph{N}(fg.num_factors + 1, new_var_nodes, fg.addr_to_idx)
end
