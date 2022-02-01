using Pkg
Pkg.activate(".")
using OptimalTransport
using PyPlot
using LinearAlgebra
using Random
using Distributions
using Distances
using Graphs
using SimpleWeightedGraphs
using GraphRecipes
# using NearestNeighborDescent
using NearestNeighbors
using SparseArrays
using ManifoldLearning
using MultivariateStats
using Roots
using StatsBase
using ProgressMeter
pyplot()

include("ssl_util.jl")
Random.seed!(0)
ENV["JULIA_DEBUG"] = ""
include("util.jl")

function gen_data(; N = 100, N_branch = 10)
    X, T, labels_all = spiral(fill(N, N_branch))
    label_frac = 0.01
    sample_frac(x, f) = x[randperm(length(x))[1:ceil(Int, f*length(x))]]
    label_idx = vcat([sample_frac((1:length(labels_all))[labels_all .== l], label_frac) for l = 1:maximum(labels_all)]...)
    labels_obs = zero(labels_all)
    labels_obs[label_idx] .= labels_all[label_idx]
    return X, labels_all, label_idx
end

N_vals = round.(Int, range(50, 250; length = 10) )
X_all = [gen_data(N = N, N_branch = 10) for N in N_vals]
ε_all_quad = exp10.(range(-3, 1; length = 10))
k_all = 

l_quad_all = [@showprogress [LLGC(kernel_ot_quad(X[1]', ε), X[2], X[3], 1.0)[1] for ε in ε_all_quad] for X in X_all]

l_knn_all = [@showprogress [LLGC(norm_kernel(1.0*(form_kernel(X[1]', 1.0; k = k) .> 0), :sym), X[2], X[3], 1.0)[1] for k in k_all] for X in X_all]

errs_quad_all = [[1-err(X[2], x) for x in l_quad_all[i]] for (i, X) in enumerate(X_all)]
errs_knn_all = [[1-err(X[2], x) for x in l_knn_all[i]] for (i, X) in enumerate(X_all)]

heatmap(hcat(errs_quad_all...))

plot(plot(ε_all_quad, errs_quad_all; xaxis = :log10), 
     plot(k_all, errs_knn_all; xaxis = :log10))

