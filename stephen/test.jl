using OptimalTransport
using Plots
using LinearAlgebra
using Random
using Distributions
using Distances
using Graphs
using SimpleWeightedGraphs
using GraphRecipes
using Makie
using GraphMakie
using GLMakie

symm(A) = 0.5*(A.+A')
kernel_ot_ent = (X, ε) -> symm(sinkhorn(ones(size(X, 2)), ones(size(X, 2)), pairwise(SqEuclidean(), X, X), ε; maxiter = 5_000, rtol = 1e-9*size(X, 2)))
kernel_ot_quad = (X, ε) -> symm(quadreg(ones(size(X, 2)), ones(size(X, 2)), pairwise(SqEuclidean(), X, X), ε; maxiter = 100, rtol = 1e-9*size(X, 2)))

function form_kernel(X, ε)
    C = pairwise(SqEuclidean(), X, X)
    K = exp.(-C/ε)
    K[diagind(K)] .= 0
    K
end

function norm_kernel(K, type)
    W = K
    if type == :unnorm
        # do nothing
    elseif type == :row
        W .= K ./ reshape(sum(K; dims = 2), :, 1)
    elseif type == :sym
        r = sum(K; dims = 2)
        W .= K .* sqrt.(1f0 ./reshape(r, :, 1)) .* sqrt.(1f0 ./reshape(r, 1, :))
    end
    W
end

## denoising
# generate R
m = 25
# u = randn(m)
# v = randn(m)
# u = u .- dot(u, v/norm(v))*(v/norm(v))
# R = hcat(u/norm(u), v/norm(v))
R = qr(randn(m, m)).Q[:, 1:2]

arm(t; ω = 4) = if (t < 0.2)
    [t, 0]
else
    [t, 0.1 * sin(ω*π * 1.25*(t - 0.2))]
end

function kspiral(k, R; ns = fill(100, k), σs = fill(0.025, k), ωs = fill(4, k))
    Rot(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    hcat([R * Rot(θ) * (hcat(arm.(range(0, 1; length = n); ω = ω)...)) + randn(size(R, 1), n)*σ for (θ, n, σ, ω) in zip(range(0, 2*π*(1-1/k); length = k), ns, σs, ωs)]...)
end

N1, N2, N3 = [800, 400, 200]
X_embed = kspiral(3, R; ns = [N1, N2, N3], ωs = [12, 8, 4], σs = [0.015, 0.03, 0.06]*0.75)
id = vcat(fill(1, N1), fill(2, N2), fill(3, N3))
X_embed_orig = kspiral(3, R; ns = [N1, N2, N3], ωs = [12, 8, 4], σs = [0.015, 0.03, 0.06]*0)


using MultivariateStats
pca = fit(PCA, Matrix(X_embed_orig))
X_embed_orig_pca = MultivariateStats.predict(pca, Matrix(X_embed_orig))

Plots.scatter(collect(eachcol(predict(pca, X_embed)'[:, 1:2]))...; alpha = 0.25, marker_z = id, markersize = 1, legend = nothing)

ε, ε_quad = 0.001, 0.1
W_row = norm_kernel(form_kernel(X_embed, ε), :row)
W_ent = kernel_ot_ent(X_embed, ε)
W_quad = kernel_ot_quad(X_embed, ε_quad)
Y_denoised_all = [X_embed * (W')^5 for W in [W_row, W_ent, W_quad]]

for i in 1:3
    @info "i = $i"
    @info [norm(X_embed_orig[:, id .== i] - X[:, id .== i]) for X in Y_denoised_all]
end

Plots.plot(Plots.scatter(Y_denoised_all[1][1, :], Y_denoised_all[1][2, :]; alpha = 0.1, markersize = 2, color = :purple, label = "denoised - row", marker_z = id, colorbar = false),
     Plots.scatter(Y_denoised_all[2][1, :], Y_denoised_all[2][2, :]; alpha = 0.1, markersize = 2, color = :purple, label = "denoised - ent", marker_z = id, colorbar = false),
     Plots.scatter(Y_denoised_all[3][1, :], Y_denoised_all[3][2, :]; alpha = 0.1, markersize = 2, color = :purple, label = "denoised - quad", marker_z = id, colorbar = false), 
     Plots.scatter(X_embed_orig[1, :], X_embed_orig[2, :]; alpha = 0.1, markersize = 2, color = :green, label = "original"), 
     Plots.scatter(X_embed[1, :], X_embed[2, :]; alpha = 0.1, markersize = 2, color = :red, label = "noisy"), size = (1250, 750))


##
ε_all = 10f0.^range(-3, 0; length = 25)
ε_quad_all = 10f0.^range(-2, 1; length = 25)

W_row_all = [norm_kernel(form_kernel(X_embed, ε), :row) for ε in ε_all]
Y_denoised_all = [X_embed * (W')^5 for W in W_row_all]
Plots.plot([Plots.scatter(Y[1, :], Y[2, :]; alpha = 0.1, markersize = 2, marker_z = id, colorbar = false) for Y in Y_denoised_all]...; legend = nothing, size = (1250, 750))
errs_row = hcat([[norm(X_embed_orig[:, id .== i] - X[:, id .== i]) for X in Y_denoised_all] for i = 1:3]...)'

W_ent_all = [kernel_ot_ent(X_embed, ε) for ε in ε_all]
Y_denoised_all = [X_embed * (W')^5 for W in W_ent_all]
Plots.plot([Plots.scatter(Y[1, :], Y[2, :]; alpha = 0.1, markersize = 2, marker_z = id, colorbar = false) for Y in Y_denoised_all]...; legend = nothing, size = (1250, 750))
errs_ent = hcat([[norm(X_embed_orig[:, id .== i] - X[:, id .== i]) for X in Y_denoised_all] for i = 1:3]...)'

W_quad_all = [kernel_ot_quad(X_embed, ε) for ε in ε_quad_all]
Y_denoised_all = [X_embed * (W')^5 for W in W_quad_all]
Plots.plot([Plots.scatter(Y[1, :], Y[2, :]; alpha = 0.1, markersize = 2, marker_z = id, colorbar = false) for Y in Y_denoised_all]...; legend = nothing, size = (1250, 750))
errs_quad = hcat([[norm(X_embed_orig[:, id .== i] - X[:, id .== i]) for X in Y_denoised_all] for i = 1:3]...)'

Plots.plot(ε_all, vec(sum(errs_row; dims = 1))/sqrt(size(errs_row, 2)), label = "Row", yaxis = :log10, xaxis = :log10, marker = :square, color = :blue, ylabel = "Denoising error (RMS)", xlabel = "Parameter value (ε)")
Plots.plot!(ε_all, vec(sum(errs_ent; dims = 1))/sqrt(size(errs_ent, 2)), label = "Ent", marker = :triangle, color = :purple)
Plots.plot!(ε_quad_all, vec(sum(errs_quad; dims = 1))/sqrt(size(errs_quad, 2)), label = "Quad", marker = :circle, color = :green)

# Plots.plot(Plots.plot(ε_all, errs_row', xaxis = :log10, yaxis = :log10, ylim = (1e-2, 0.5), label = "Row"), 
#      Plots.plot(ε_all, errs_ent', xaxis = :log10, yaxis = :log10, ylim = (1e-2, 0.5), label = "Ent"),
#      Plots.plot(ε_quad_all, errs_quad', xaxis = :log10, yaxis = :log10, ylim = (1e-2, 0.5), label = "Quad"))

Plots.plot(Plots.scatter(collect(eachcol(predict(pca, X_embed)'))...; marker_z = id, alpha = 0.1, title = "Noisy"), 
    Plots.scatter(collect(eachcol(predict(pca, X_embed * (W_row_all[argmin(sum(errs_row; dims = 1))[2]]')^5)'))...; marker_z = id, alpha = 0.1, title = "Row"),
     Plots.scatter(collect(eachcol(predict(pca, X_embed * (W_ent_all[argmin(sum(errs_ent; dims = 1))[2]]')^5)'))...; marker_z = id, alpha = 0.1, title = "Ent"),
     Plots.scatter(collect(eachcol(predict(pca, X_embed * (W_quad_all[argmin(sum(errs_quad; dims = 1))[2]]')^5)'))...; marker_z = id, alpha = 0.1, title = "Quad");
     markersize = 2, legend = false, layout = (2, 2), colorbar = false)

# what about MAGIC?
using PyCall
magic = pyimport("magic")
magic_op = magic.MAGIC()
magic_op.set_params(t = 5)
X_embed_magic = magic_op.fit_transform(X_embed, genes = "all_genes")
errs_magic = [norm(X_embed_orig[:, id .== i] - X_embed_magic[:, id .== i]) for i = 1:3]

Plots.scatter(collect(eachcol(predict(pca, X_embed_magic)'))...; marker_z = id, alpha = 0.1, title = "MAGIC")
