using Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/zsteve/OptimalTransport.jl", rev="symmetric_quad")

using OptimalTransport
using Distances
using Distances: pairwise
using Plots
using StatsBase
using GLM
using DataFrames
using LinearAlgebra

d = 2

function get_plot(N)
    # θs = range(-π, π; length = N)
    # X = mapreduce(x -> [cos(x), sin(x)], hcat, θs)
    X = randn(d, N)
    X ./= reshape(map(norm, eachcol(X)), 1, :)
    @info size(X)
    C = pairwise(SqEuclidean(), X)
    # C /= mean(C)
    εvals = 10 .^range(3, -3; length = 100)
    μ = fill(1/size(X, 2), size(X, 2))
    function dualpotentials(ε)
        # @info ε
        solver = OptimalTransport.build_solver(μ, C, ε, OptimalTransport.SymmetricQuadraticOTNewton(δ = 1e-5); maxiter = 100)
        OptimalTransport.solve!(solver)
        solver.cache.u
    end
    uvals = map(x -> dualpotentials(x), εvals)
    umeans = map(mean, uvals)
    df1 = DataFrame(logu = log10.(umeans[εvals .< 0.05]), logeps = log10.(εvals[εvals .< 0.05]))
    fit1 = lm(@formula(logu ~ logeps), df1)
    df2 = DataFrame(logu = log10.(umeans[εvals .> 5]), logeps = log10.(εvals[εvals .> 5]))
    fit2 = lm(@formula(logu ~ logeps), df2)
    plt=scatter(log10.(εvals), log10.(umeans); markersize = 1, alpha = 0.25, label = "data", legend = :topleft, title = "N = $N")
    plot!(plt, df1.logeps, predict(fit1); label = "α = $(coef(fit1)[2])")
    plot!(plt, df2.logeps, predict(fit2); label = "α = $(coef(fit2)[2])")
    # hline!(plt, [log10.(Cthresh), ]; label = "u_thresh")
    plt
end

Ns = [500, ]

plots_all = map(x -> get_plot(x), Ns)

# plot(plots_all...; size = (1_000, 1_000))

plots_all[1]
