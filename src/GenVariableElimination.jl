module GenVariableElimination

ENV["JAX_ENABLE_X64"] = "1" 

# Import
include("factor_graph.jl")
include("compiler.jl")
include("gen_fns.jl")
include("einsum_py_utils.jl")
include("einsum_backend.jl")

# Public exports
export FactorGraph
export VariableEliminationResult, variable_elimination
export Latent, Observation, compile_trace_to_factor_graph
export draw_factor_graph
export factor_graph_analysis
export generate_backwards_sampler_fixed_trace, generate_backwards_sampler_fixed_structure
export backwards_sampler_dml, backwards_sampler_sml
export EinsumContractionResult, contract_factor_graph_einsum, eliminate_einsum

# Discrete support annotations used by compiler.jl
function is_finite_discrete end
function discrete_finite_support_overapprox end

is_finite_discrete(::Gen.Bernoulli) = true
discrete_finite_support_overapprox(::Gen.Bernoulli, ::Real) = (true, false)

is_finite_discrete(::Gen.Categorical) = true
discrete_finite_support_overapprox(::Gen.Categorical, probs) = (1:length(probs)...,)

is_finite_discrete(::Gen.UniformDiscrete) = true
discrete_finite_support_overapprox(::Gen.UniformDiscrete, low::Integer, high::Integer) = (low:high...,)

export is_finite_discrete
export discrete_finite_support_overapprox

end # module