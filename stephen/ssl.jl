using Pkg
Pkg.activate(".")
using OptimalTransport
using Plots
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
pyplot()

include("ssl_util.jl")

Random.seed!(0)

ENV["JULIA_DEBUG"] = ""
include("util.jl")

N_branch = 10
# round.(Int, N_tot * (0.1/N_branch .+ 0.9rand(Dirichlet(1e6*ones(N_branch)))))
# nums = vcat([[125, 125, ] for _ in 1:3]...)
nums = vcat([[150, 150, ] for _ in 1:N_branch÷2]...)
N_tot = sum(Int, nums)

# compute embedding and noise
X_orig, T, labels_all = spiral(nums)

m = 100
R = qr(randn(m, m)).Q[:, 1:2]
η = randn(N_tot, m)
η = η ./ [norm(x) for x in eachrow(η)]
η = reshape(1 .- sin.(3T).^4, :, 1).*η

X = X_orig*R'
X .+= 0.5*η 

label_frac = 0.025
# label_idx = randperm(length(labels_all))[1:
sample_frac(x, f) = x[randperm(length(x))[1:ceil(Int, f*length(x))]]
label_idx = vcat([sample_frac((1:length(labels_all))[labels_all .== l], label_frac) for l = 1:maximum(labels_all)]...)
labels_obs = zero(labels_all)
labels_obs[label_idx] .= labels_all[label_idx]

#=
l_sym, _, Y = LLGC(norm_kernel(form_kernel(X', 1e-3; k = 6), :sym), labels_all, label_idx, 1.0)
l_quad, _, _ = LLGC(kernel_ot_quad(X', 0.0316), labels_all, label_idx, 1.0)
l_ent, _, _ = LLGC(kernel_ot_ent(X', 0.001), labels_all, label_idx, 1.0)
=#

using ColorSchemes
cmap_inc_grey = pushfirst!([RGBA(i) for i in cgrad(:seaborn_bright)], RGBA(0.5, 0.5, 0.5, 0.5))
cmap = cmap_inc_grey[2:end]

using MultivariateStats
pca = MultivariateStats.fit(PCA, X'; maxoutdim = 3)
X_pca = transform(pca, X')'

scatter(X_pca[:, 1], X_pca[:, 2])

#=
plot(scatter(X_pca[:, 1], X_pca[:, 2]; markerstrokewidth = 0, group = labels_all, palette = cmap, title = "True"), 
     scatter(X_pca[:, 1], X_pca[:, 2]; markerstrokewidth = 0, group = labels_obs, palette = cmap_inc_grey, title = "Input"), 
     scatter(X_pca[:, 1], X_pca[:, 2]; markerstrokewidth = 0, group = l_sym, palette = cmap, title = "Symmetric, err = $(round(err(l_sym, labels_all); digits = 3))"),
     scatter(X_pca[:, 1], X_pca[:, 2]; markerstrokewidth = 0, group = l_quad, palette = cmap, title = "Quad, err = $(round(err(l_quad, labels_all); digits = 3))"),
     scatter(X_pca[:, 1], X_pca[:, 2]; markerstrokewidth = 0, group = l_ent, palette = cmap, title = "Ent, err = $(round(err(l_ent, labels_all); digits = 3))"),
     markersize = 4, alpha = 0.25, legend = nothing)
=#

## grid search for this example

using ProgressMeter
ε_all_quad = exp10.(range(-2, 0; length = 25))
ε_all = exp10.(range(-3, -1; length = 25))
k_all = round.(Int, range(1, 10; length = 10))
acc_knn = @showprogress [map(ε -> 1-err_norm(LLGC(norm_kernel(form_kernel(X', ε; k = k), :row), labels_all, label_idx, 1.0)[1], labels_all), ε_all) for k in k_all]
acc_quad = @showprogress map(ε -> 1-err_norm(LLGC(kernel_ot_quad(X', ε), labels_all, label_idx, 1.0)[1], labels_all), ε_all_quad)

unlabelled_idx = symdiff(label_idx, 1:length(labels_all))
plt_labels = scatter(X_pca[unlabelled_idx, 1], X_pca[unlabelled_idx, 2]; markercolor = cmap[labels_all[unlabelled_idx]], markeralpha = 0.5, markersize = 2.5, markerstrokewidth = 0, legend = nothing, aspectratio = :equal)
scatter!(X_pca[label_idx, 1], X_pca[label_idx, 2]; markerstrokewidth = 0, markercolor = cmap[labels_all[label_idx]], title = "Data", markersize = 5)

# plt_labels = scatter(X_pca[unlabelled_idx, 1], X_pca[unlabelled_idx, 2]; markeralpha = 0.175, markerstrokewidth = 0, legend = nothing, aspectratio = :equal)
# scatter!(X_pca[label_idx, 1], X_pca[label_idx, 2]; markerstrokewidth = 0, title = "Data", markersize = 5)

plt = plot(plt_labels, 
    plot(ε_all_quad, acc_quad; xaxis = :log10, ylim = (0., 1.05), title = "Quadratic OT", m = "o", legend = nothing, xlabel = "ε", ylabel = "avg class accuracy"), 
     plot(ε_all, acc_knn; xaxis = :log10, ylim = (0., 1.05), title = "kNN + Gaussian", palette = :tab10, m = "o", legend = :outerright, label = reshape(map(k -> "k = $k", k_all), 1, :), xlabel = "ε", ylabel = "avg class accuracy"), markerstrokewidth = 0, layout = (1, 3), size = (750, 250))
savefig(plt, "ssl_density_uneven.pdf")

using Graphs, GraphPlot
using Cairo, Fontconfig
using Gadfly, Compose

function edgecols(A; q = 0.95, cmax = 0.8)
    B = A - Diagonal(diag(A))
    q = map(x -> quantile(x[x .> 0], q), eachrow(B))
    vcat([[RGBA(0, 0, 0, cmax*f/r) for f in x[x .> 0]] for (x, r) in zip(eachrow(B), q)]...)
end

function thresh_quantile(x, q)
    x[x .< quantile(vec(x), q)] .= 0
    return x
end
q_degree(x, q) = degree(SimpleWeightedGraph(thresh_quantile(x, q)))

W_quad_best = kernel_ot_quad(X', ε_all_quad[argmax(acc_quad)]; diag_inf = false)
l = LLGC(W_quad_best, labels_all, label_idx, 1.0)[1]
g = SimpleWeightedGraph(symm(W_quad_best))
A_thresh = sparse(Matrix(adjacency_matrix(g)))
draw(PDF("ssl_graph_best_quad.pdf", 5cm, 5cm), gplot(g, X_pca[:, 1], X_pca[:, 2]; NODESIZE = 0.015, nodefillc = cmap[l], edgestrokec = edgecols(A_thresh), edgelinewidth = 0.5))


W_knn_best = norm_kernel(form_kernel(X', 0.01; k = 5), :row)
l = LLGC(W_knn_best, labels_all, label_idx, 1.0)[1]
g = SimpleWeightedGraph(symm(W_knn_best))
A_thresh = sparse(Matrix(adjacency_matrix(g)))
draw(PDF("ssl_graph_best_knn.pdf", 5cm, 5cm), gplot(g, X_pca[:, 1], X_pca[:, 2]; NODESIZE = 0.015, nodefillc = cmap[l], edgestrokec = edgecols(A_thresh), edgelinewidth = 0.5))

scatter(X[:, 1], X[:, 2], X[:, 3]; group = labels_all)


#= grid search part

function gen_data(λ; N_tot = 1000, N_branch = 5)
    X, T, labels_all = spiral(round.(Int, N_tot * (0.1/N_branch .+ 0.9rand(Dirichlet(λ*ones(N_branch))))))
    label_frac = 0.01
    sample_frac(x, f) = x[randperm(length(x))[1:ceil(Int, f*length(x))]]
    label_idx = vcat([sample_frac((1:length(labels_all))[labels_all .== l], label_frac) for l = 1:maximum(labels_all)]...)
    labels_obs = zero(labels_all)
    labels_obs[label_idx] .= labels_all[label_idx]
    return X, labels_all, label_idx
end


N_vals = [100, 200, 300, 400, 500].*10
X_all = [gen_data(1e6; N_tot = N, N_branch = 10) for N in N_vals]
ε_all_quad = exp10.(range(-3, 1; length = 25))
k_all = round.(Int, range(5, 50; length = 25))

l_quad_all = [@showprogress [LLGC(kernel_ot_quad(X[1]', ε), X[2], X[3], 1.0)[1] for ε in ε_all_quad] for X in X_all]

l_knn_all = [@showprogress [LLGC(norm_kernel(1.0*(form_kernel(X[1]', 1.0; k = k) .> 0), :sym), X[2], X[3], 1.0)[1] for k in k_all] for X in X_all]

errs_quad_all = [[1-err(X[2], x) for x in l_quad_all[i]] for (i, X) in enumerate(X_all)]
errs_knn_all = [[1-err(X[2], x) for x in l_knn_all[i]] for (i, X) in enumerate(X_all)]

plot(plot(ε_all_quad, errs_quad_all; xaxis = :log10), 
     plot(k_all, errs_knn_all; xaxis = :log10))

=# 
