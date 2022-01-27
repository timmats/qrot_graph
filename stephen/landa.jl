

## Try the example of Landa et al.
n = 1_000
θ_vals = rand(Uniform(-π, π), n)
X = hcat([[cos(θ), sin(θ)] for θ in θ_vals]...)
# ϕ_vals = rand(Uniform(-π, π), n)
# Y = hcat([[cos(θ), sin(θ)] for θ in ϕ_vals]...)
# X = hcat(X, 0.75*Y .+ 0.5*0.25/sqrt(2))
scatter(X[1, :], X[2, :])
m = 10_000
# too slow
# X_embed = qr(randn(m, m)).Q[:, 1:2] * X
# generate R 
u = randn(m)
v = randn(m)
u = u .- dot(u, v/norm(v))*(v/norm(v))
R = hcat(u/norm(u), v/norm(v))
X_embed = R * X

scatter(X_embed[1, :], X_embed[2, :])

## Example 1
α = rand(Uniform(0.05, 0.5), m)
β = rand(Uniform(0.05, 0.5), n)

σ = @. sqrt(α * β' / m)
η = randn(m, n).*σ

X_embed_noisy = X_embed + η

scatter(X_embed_noisy[1, :], X_embed_noisy[2, :])

ε = 1.5
ε_ent = ε
ε_quad = 2.5
K = form_kernel(X_embed, ε)
K_noisy = form_kernel(X_embed_noisy, ε)

W_noisy_ent = kernel_ot_ent(X_embed_noisy, ε_ent)
W_noisy_quad = kernel_ot_quad(X_embed_noisy, ε_quad)
W_quad = kernel_ot_quad(X_embed, ε_quad)

W = norm_kernel(K, :row)
W_noisy = norm_kernel(K_noisy, :sym)

norm(W_noisy - W)
norm(W_noisy_ent - kernel_ot_ent(X, ε))

idx = randperm(length(vec(W)))[1:10_000]
plot(scatter(vec(W)[idx], vec(W_noisy_ent)[idx], alpha = 0.1, title = "Entropic"),
     scatter(vec(W)[idx], vec(W_noisy)[idx], alpha = 0.1, title = "Alternative"),
     scatter(vec(W_quad)[idx], vec(W_noisy_quad)[idx], alpha = 0.1, title = "Quadratic"))

mean(W_noisy_quad .> 0)

X_eigmap = eigen(W_noisy_ent).vectors[:, end-2:end-1]
scatter(X_eigmap[:, 1], X_eigmap[:, 2])

## Example 2
ρ = θ -> 0.01 + 0.99*(1 + cos(2*θ))/2
η = hcat([x/norm(x)*ρ(θ) for (θ, x) in zip(θ_vals, eachcol(randn(size(X_embed))))]...)
X_embed_noisy = X_embed + η

scatter(X_embed_noisy[1, :], X_embed_noisy[3, :], markersize = 1, alpha = 0.25)

mean(pairwise(SqEuclidean(), X_embed_noisy, X_embed_noisy))

ε_all = 10f0.^collect(range(-2, 1; length = 10))
W_simple_all = [norm_kernel(form_kernel(X_embed_noisy, ε), :row) for ε in ε_all]
W_ent_all = [kernel_ot_ent(X_embed_noisy, ε)  for ε in ε_all]

plt_simple = plot([scatter(collect(eachcol(real.(eigen(W).vectors[:, end-2:end-1])))...; title = "ε = $(round(ε; digits = 3))", markersize = 1) for (W, ε) in zip(W_simple_all, ε_all)]...)

plt_ent = plot([scatter(collect(eachcol(real.(eigen(W).vectors[:, end-2:end-1])))...; title = "ε = $(round(ε; digits = 3))", markersize = 1) for (W, ε) in zip(W_ent_all, ε_all)]...)

ε_quad_all = 10f0.^collect(range(0, 1; length = 10))
W_quad_all = [kernel_ot_quad(X_embed_noisy, ε) for ε in ε_quad_all]

plt_quad = plot([scatter(collect(eachcol(real.(eigen(W).vectors[:, end-2:end-1])))...; title = "ε = $(round(ε; digits = 3))", markersize = 1) for (W, ε) in zip(W_quad_all, ε_quad_all)]...)

scatter(collect(eachcol(eigen(W_quad_all[end-2]).vectors[:, end-2:end-1]))...; markersize = 1)

Plots.plot(
    scatter(θ_vals, eigen(W_quad_all[end-2]).vectors[:, end-3:end-1]; markersize = 1, markerstrokewidth = 0), 
    scatter(θ_vals, eigen(W_ent_all[end-3]).vectors[:, end-3:end-1]; markersize = 1, markerstrokewidth = 0), 
    scatter(θ_vals, eigen(W_simple_all[end-3]).vectors[:, end-3:end-1]; markersize = 1, markerstrokewidth = 0))

plot([Plots.heatmap(W, clim = (0, 1e-2)) for W in W_quad_all]...; colorbar = false)
plot([Plots.heatmap(W, clim = (0, 1e-2)) for W in W_ent_all]...; colorbar = false)

