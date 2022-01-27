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
using NearestNeighborDescent
using SparseArrays
using ManifoldLearning
using MultivariateStats
pyplot()

ENV["JULIA_DEBUG"] = ""

include("util.jl")

## Manifold learning?
Random.seed!(42)

using ForwardDiff
using QuadGK
using Roots

m = 100
N = 1_000
f(t) = [cos(t)*(0.5cos(6t)+1) sin(t)*(0.4cos(6t)+1) 0.4sin(6t)]
arclength(t::Real) = quadgk(t -> norm(ForwardDiff.derivative(f, t)), 0, t, rtol = 1e-8)[1]
L_tot = arclength(2π)
t_range_scaled = range(0, 1; length = N)
θ_range = map(x -> find_zero(t -> arclength(t)/L_tot - x, (0., 2π)), t_range_scaled)

R = qr(randn(m, m)).Q[:, 1:3]
X = vcat(f.(θ_range)...)' # ManifoldLearning.spirals(1_000, 0.)[1]
# 6 segments
ρ(θ) = 0.05 + 0.95*(1 + cos(6θ))/2
η = hcat([x/norm(x)*ρ(θ) for (θ, x) in zip(θ_range, eachcol(randn(m, size(X, 2))))]...)

scatter(X[1, :], X[2, :], X[3, :])
X_embed = R * X + η
X_embed_orig = R * X 

pca = fit(PCA, X_embed; maxoutdim = 25)
X_pca = transform(pca, X_embed)
Plots.scatter(collect(eachrow(X_pca[1:2, :]))...; marker_z = θ_range, color = :gist_rainbow, alpha = 0.25)

arclength(X) = cumsum([norm(x-y) for (x, y) in zip(eachcol(X[:, 1:end-1]), eachcol(X[:, 2:end]))])

l = [quadgk(t -> norm(ForwardDiff.derivative(f, t)), 0, tfinal, rtol = 1e-8)[1] for tfinal in θ_range]
l[2:end].-l[1:end-1] # check we are sampling w.r.t. arclength
histogram(l)

scatter(collect(eachcol(vcat(f.(θ_range)...)))...)

k_vals = [5, 10, 15, 20, 25,]

using PyCall
magic = pyimport("magic")
magic_op_all = [magic.MAGIC() for _ in k_vals]
[op.set_params(t = 5, knn = k) for (k, op) in zip(k_vals, magic_op_all)]
X_embed_magic_all = [op.fit_transform(X_embed', genes = "all_genes")' for op in magic_op_all]
# ε values
ε_all_quad = 10f0.^range(-1.5, 1.5; length = 25)
ε_all = 10f0.^range(-2, 1; length = 25)

W_all = Dict(k => (x, [f(X_embed, ε) for ε in x]) for (k, f, x) in zip(["quad", "ent", ["knn_$k" for k in k_vals]..., "row"],
                                                                       [kernel_ot_quad, kernel_ot_ent,
                                                                       [(x, ε) -> norm_kernel(form_kernel(x, ε; k = k), :row) for k in k_vals]...,
                                                                        (x, ε) -> norm_kernel(form_kernel(x, ε; k = Inf), :row)],
                                                                       [ε_all_quad, [ε_all for _ in k_vals]..., ε_all, ε_all]))

W_all_decomp = Dict(k => [real.(eigen(y).vectors) for y in x] for (k, (_, x)) in W_all)

W_ref = norm_kernel(form_kernel(X_embed_orig, 0.025; k = 10), :row)
W_ref_decomp = real.(eigen(W_ref).vectors)

scatter(collect(eachcol(W_ref_decomp[:, end-2:end-1]))...; aspect_ratio = :equal, title = "Reference", alpha = 0.25, markersize = 4, marker_z = θ_range, color = :gist_rainbow, legend = nothing, markerstrokewidth = 0)

Plots.plot([scatter(collect(eachcol(x[:, end-2:end-1]))...; alpha = 0.1) for x in W_all_decomp["quad"]]...; axis = nothing, legend = nothing, markersize = 1, ylim = (-0.1, 0.1), xlim = (-0.1, 0.1), plot_title = "Quadratic", size = (500, 500))

Plots.plot([scatter(collect(eachcol(x[:, end-2:end-1]))...; alpha = 0.1) for x in W_all_decomp["ent"]]...; axis = nothing, legend = nothing, markersize = 1, ylim = (-0.1, 0.1), xlim = (-0.1, 0.1), plot_title = "Entropic", size = (500, 500))

Plots.plot([scatter(collect(eachcol(x[:, end-2:end-1]))...; alpha = 0.1) for x in W_all_decomp["knn_15"]]...; axis = nothing, legend = nothing, markersize = 1, ylim = (-0.1, 0.1), xlim = (-0.1, 0.1), plot_title = "kNN + Gaussian", size = (500, 500))

Plots.plot([scatter(collect(eachcol(x[:, end-2:end-1]))...; alpha = 0.1) for x in W_all_decomp["row"]]...; axis = nothing, legend = nothing, markersize = 1, ylim = (-0.1, 0.1), xlim = (-0.1, 0.1), plot_title = "Gaussian", size = (500, 500))

Plots.plot([Plots.plot(θ_range, x[:, end-3:end-1];) for x in W_all_decomp["quad"]]...; axis = nothing, legend = nothing,  linewidth = 1, plot_title = "Quadratic")

Plots.plot([Plots.plot(θ_range, x[:, end-3:end-1];) for x in W_all_decomp["ent"]]...; axis = nothing, legend = nothing, markersize = 1, plot_title = "Entropic")

Plots.plot([Plots.plot(θ_range, x[:, end-3:end-1];) for x in W_all_decomp["knn_15"]]...; axis = nothing, legend = nothing, markersize = 1, plot_title = "kNN + Gaussian")

Plots.plot([Plots.plot(θ_range, x[:, end-3:end-1];) for x in W_all_decomp["row"]]...; axis = nothing, legend = nothing, markersize = 1, plot_title = "Gaussian")

Plots.plot(θ_range, W_ref_decomp[:, end-3:end-1];)

# geterr(x, y) = min(norm(x - y)^2, norm(x + y)^2)
# # geterr(x::AbstractMatrix, y::AbstractMatrix) = sqrt(mean([geterr(u, v) for (u, v) in zip(eachcol(x), eachcol(y))]))
# 
# function geterr(x::AbstractMatrix, y::AbstractMatrix)
#     # for periodic BCs, eigenfunctions of the same frequency occur in pairs sin/cos
#     # but the order is arbitrary, so for each pair need to test both cases
#     _err(a, b) = min(geterr(a[:, 1], b[:, 1]) + geterr(a[:, 2], b[:, 2]),
#                      geterr(a[:, 1], b[:, 2]) + geterr(a[:, 2], b[:, 1]))
#     @assert (size(x, 2) % 2 == 0) & (size(x) == size(y))
#     sqrt(mean([_err(x[:, 2(i-1)+1:2i], y[:, 2(i-1)+1:2i]) for i = 1:(size(x, 2)÷2)]))
# end

# try instead subspace angles
using PyCall
scipy = pyimport("scipy")
function geterr(x::AbstractMatrix, y::AbstractMatrix)
    return mean(scipy.linalg.subspace_angles(x, y))
end


# check eigenfunctions 
# scatter(eigen(W_ref).values[end-6:end-1])
# geterr(W_ref_decomp[:, end-6:end-1], W_ref_decomp[:, [995, 994, 997, 996, 998, 999]])
# scatter(eigen(W_all["quad"][2][15]).values[end-6:end-1])

err_all = Dict(k => map(W -> geterr(W_ref_decomp[:, end-10:end-1], W[:, end-10:end-1]), x) for (k, x) in W_all_decomp)
err_magic = Dict(k => geterr(W_ref_decomp[:, end-10:end-1], real.(eigen(op.diff_op.todense()).vectors[:, end-10:end-1])) for (k, op) in zip(k_vals, magic_op_all))

Plots.plot(W_all["quad"][1], err_all["quad"], xscale = :log10, label = "Quad", palette = :Set1_9, legend = :outerright, xlabel = "ε", ylabel = "⟨ cos(θ) ⟩", color = :blue, m = "o", markerstrokewidth = 0, size = (500, 250))
Plots.plot!(W_all["ent"][1], err_all["ent"]; label = "Ent", color = :green, m = "o", markerstrokewidth = 0)
Plots.plot!(W_all["row"][1], err_all["row"]; label = "Gauss", color = :orange, m = "o", markerstrokewidth = 0)
# [Plots.plot!(W_all["knn_$k"][1], err_all["knn_$k"], label = "kNN_$k") for k in k_vals]
Plots.plot!(W_all["knn_5"][1], minimum(hcat([err_all["knn_$k"] for k in k_vals]...); dims = 2); label = "kNN_best", color = :red, m = "o", markerstrokewidth = 0)
# [hline!([err_magic[k], ], label = "MAGIC_$k") for k in k_vals]
hline!([last(minimum(values(err_magic))), ]; label = "MAGIC_best", color = :black)

# q-degree
function thresh_quantile(x, q)
    x[x .< quantile(vec(x), q)] .= 0
    return x
end
q_degree(x, q) = degree(SimpleWeightedGraph(thresh_quantile(x, q)))

scatter(X_pca[1, :], X_pca[2, :], marker_z = q_degree(W_all["ent"][2][argmin(err_all["ent"])], 0.75), markerstrokewidth = 0)

scatter(X_pca[1, :], X_pca[2, :], marker_z = q_degree(W_all["quad"][2][argmin(err_all["quad"])], 0.75), markerstrokewidth = 0)

Plots.plot(Plots.plot(θ_range, W_all_decomp["quad"][argmin(err_all["quad"])][:, end-3:end-1]; alpha = 1, markerstrokewidth = 0, title = "Quadratic"),
     Plots.plot(θ_range, W_all_decomp["ent"][argmin(err_all["ent"])][:, end-3:end-1]; alpha = 1, markerstrokewidth = 0, title = "Entropic"), 
     [Plots.plot(θ_range, W_all_decomp["knn_$k"][argmin(err_all["knn_$k"])][:, end-3:end-1]; alpha = 1, markerstrokewidth = 0, title = "kNN_$k") for k in k_vals]..., 
     Plots.plot(θ_range, W_all_decomp["row"][argmin(err_all["row"])][:, end-3:end-1]; alpha = 1, markerstrokewidth = 0, title = "Gaussian"),
     [Plots.plot(θ_range, real.(eigen(op.diff_op.todense()).vectors[:, end-3:end-1]); alpha = 1, markerstrokewidth = 0, title = "MAGIC_$k") for (k, op) in zip(k_vals, magic_op_all)]...,
     Plots.plot(θ_range, real.(eigen(W_ref).vectors[:, end-3:end-1]); alpha = 1, markerstrokewidth = 0, title = "True"); legend = nothing)

k=10
Plots.plot(Plots.scatter(collect(eachcol(W_all_decomp["quad"][argmin(err_all["quad"])][:, end-2:end-1]))...; alpha = 0.05, title = "Quadratic", marker_z = θ_range, color = :gist_rainbow),
     Plots.scatter(collect(eachcol(W_all_decomp["ent"][argmin(err_all["ent"])][:, end-2:end-1]))...; alpha = 0.05, title = "Entropic", marker_z = θ_range, color = :gist_rainbow), 
     Plots.scatter(collect(eachcol(W_all_decomp["knn_$k"][argmin(err_all["knn_$k"])][:, end-2:end-1]))...; alpha = 0.05, title = "kNN_$k", marker_z = θ_range, color = :gist_rainbow), 
     Plots.scatter(collect(eachcol(W_all_decomp["row"][argmin(err_all["row"])][:, end-2:end-1]))...; alpha = 0.05, title = "Gaussian", marker_z = θ_range, color = :gist_rainbow),
     Plots.scatter(collect(eachcol(real.(eigen(magic_op_all[k_vals .== k][1].diff_op.todense()).vectors[:, end-2:end-1])))...; alpha = 0.05, title = "MAGIC_$k", marker_z = θ_range, color = :gist_rainbow),
     Plots.scatter(collect(eachcol(real.(eigen(W_ref).vectors[:, end-2:end-1])))...; alpha = 0.05, title = "True", marker_z = θ_range, color = :gist_rainbow); legend = nothing, markersize = 4, markerstrokewidth = 0, colorbar = nothing, color = :viridis, xaxis = nothing, yaxis = nothing)

using Graphs, GraphPlot
using Cairo, Fontconfig
using Gadfly, Compose

using GLMakie, SGtSNEpi, UMAP

function edgecols(A; q = 0.95, cmax = 0.5)
    q = quantile(vec(A[A .> 0]), q)
    [RGBA(0, 0, 0, cmax*f/q) for f in A[A .> 0]]
end

g = SimpleWeightedGraph(symm(W_all["quad"][2][argmin(err_all["quad"])]))
A_thresh = sparse(thresh_quantile(Matrix(adjacency_matrix(g)), 0.9))
# show_embedding(X_pca[1:2, :]', A = A_thresh, edge_alpha = 0.5)
# draw(PDF("graph_quad_best.pdf", 5cm, 5cm), gplot(g, X_pca[1, :], X_pca[2, :]; nodefillc = fill(colorant"red", size(X_pca, 2)), edgestrokec = fill(RGBA(0.5, 0.5, 0.5, 0.25), sum(A_thresh .> 0)), edgelinewidth = 0.5))
draw(PDF("graph_quad_best.pdf", 5cm, 5cm), gplot(g, X_pca[1, :], X_pca[2, :]; NODESIZE = 0.01, nodefillc = fill(colorant"red", size(X_pca, 2)), edgestrokec = edgecols(A_thresh), edgelinewidth = 0.5))

# Plots.scatter(X_pca[1, :], X_pca[2, :]; marker_z = degree(g), markerstrokewidth = 0, cmap = :viridis)

g = SimpleWeightedGraph(symm(W_all["ent"][2][argmin(err_all["ent"])]))
A_thresh = sparse(thresh_quantile(Matrix(adjacency_matrix(g)), 0.9))
draw(PDF("graph_ent_best.pdf", 5cm, 5cm), gplot(SimpleWeightedGraph(A_thresh), X_pca[1, :], X_pca[2, :]; nodefillc = fill(colorant"red", size(X_pca, 2)), edgestrokec = edgecols(A_thresh; cmax = 0.1), edgelinewidth = 0.5, NODESIZE = 0.01))

g = SimpleWeightedGraph(symm(W_all["knn_10"][2][argmin(err_all["knn_10"])]))
A_thresh = sparse(thresh_quantile(Matrix(adjacency_matrix(g)), 0.9))
draw(PDF("graph_knn_best.pdf", 5cm, 5cm), gplot(SimpleWeightedGraph(A_thresh), X_pca[1, :], X_pca[2, :]; nodefillc = fill(colorant"red", size(X_pca, 2)), edgestrokec = edgecols(A_thresh), edgelinewidth = 0.5, NODESIZE = 0.01))

g = SimpleWeightedGraph(symm(magic_op_all[k_vals .== 10][1].diff_op.todense()))
A_thresh = sparse(thresh_quantile(Matrix(adjacency_matrix(g)), 0.9))
draw(PDF("graph_magic_best.pdf", 5cm, 5cm), gplot(SimpleWeightedGraph(A_thresh), X_pca[1, :], X_pca[2, :]; nodefillc = fill(colorant"red", size(X_pca, 2)), edgestrokec = edgecols(A_thresh; cmax = 0.1), NODESIZE = 0.01))

hstack(gplot(SimpleWeightedGraph(A_thresh), X_pca[1, :], X_pca[2, :]; nodefillc = fill(colorant"red", size(X_pca, 2)), edgestrokec = fill(RGBA(0.5, 0.5, 0.5, 0.25), sum(A_thresh .> 0))),
       gplot(SimpleWeightedGraph(A_thresh), X_pca[1, :], X_pca[2, :]; nodefillc = fill(colorant"red", size(X_pca, 2)), edgestrokec = fill(RGBA(0.5, 0.5, 0.5, 0.25), sum(A_thresh .> 0))))



# Plots.plot([Plots.heatmap(thresh_quantile(W, 0.75) .> 0) for W in W_all["quad"][2]]...; colorbar = nothing, xaxis = nothing, yaxis = nothing)
