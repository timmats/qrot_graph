using Pkg
Pkg.activate(".")
using OptimalTransport
using Distances
using StatsBase
using LinearAlgebra
using Plots
using Laplacians

ENV["JULIA_DEBUG"] = ""

x = reshape(collect(range(-1, 1; length = 250)), 1, :)
l = x[2]-x[1]
L_discrete = SymTridiagonal(-2*ones(length(x)), ones(length(x)-1))/l^2


function get_cost(X; diag_inf = false)
    C = pairwise(Euclidean(), X, X)
    # C = min.(C, maximum(C) .- C)
    C = C.^2
    if diag_inf
        C[diagind(C)] .= Inf
    end
    mean_norm(C)
end

heatmap(get_cost(x))

mean_norm(x) = x ./ mean(x)

symm(A) = 0.5*(A.+A')
kernel_ot_ent = (X, ε; diag_inf = false, rtol = 1e-6, atol = 1e-9) -> symm(sinkhorn(ones(size(X, 2)), get_cost(X; diag_inf = diag_inf), ε, OptimalTransport.SymmetricSinkhornGibbs(); maxiter = 5_000, rtol = rtol, atol = atol))
kernel_ot_quad = (X, ε; diag_inf = false, rtol = 1e-6, atol = 1e-9) -> symm(quadreg(ones(size(X, 2)), get_cost(X; diag_inf = diag_inf), ε, OptimalTransport.SymmetricQuadraticOTNewton(); maxiter = 100, atol = atol, rtol = rtol))

function knn_adj(X, k)
    indices, _ = knn_matrices(nndescent(X, k, Euclidean())); 
    A = spzeros(size(X, 2), size(X, 2));
    @inbounds for i = 1:size(A, 1)
        @inbounds for j in indices[:, i]
            A[i, j] = 1
        end
    end
    return A
end

function form_kernel(X, ε; k = Inf)
    C = get_cost(X) 
    K = exp.(-C/ε)
    if k < Inf
        K .= K .* knn_adj(X, k)
    end
    # K[diagind(K)] .= 0
    symm(K)
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

geterr(x, y) = min(norm(x - y)^2, norm(x + y)^2)
geterr(x::AbstractMatrix, y::AbstractMatrix) = sqrt(mean([geterr(u, v) for (u, v) in zip(eachcol(x), eachcol(y))]))

function eigfn2(x; k = 1, L = 2)
    if k == 1
        return 1/sqrt(L)
    else
        return sqrt(2/L)*cos((k-1)*π*(x-L/2)/L)
    end
end

l2_normalize(x) = x/norm(x)
N = 5
spec_ref = hcat([l2_normalize(eigfn2.(x; k = k))' for k = 1:N]...)
plt_ref = plot(spec_ref)

function get_spec(f, ε, N)
    L = (I - f(x, ε))
    spec = real.(eigen(L).vectors[:, 1:N])
    return spec
end

ε_all_ent = 10f0.^range(-4, 2; length = 25)
ε_all_quad = 10f0.^range(-4, 2; length = 25)
ε_all_sym = 10f0.^range(-4, 2; length = 25)
spec_ent = [get_spec((x, ε) -> kernel_ot_ent(x, ε; atol = 0, rtol = 0), z, N) for z in ε_all_ent]
spec_quad = [get_spec((x, ε) -> kernel_ot_quad(x, ε; atol = 0, rtol = 0), z, N) for z in ε_all_quad]
spec_sym = [get_spec((x, ε) -> norm_kernel(form_kernel(x, ε), :sym), z, N) for z in ε_all_sym]
spec_row = [get_spec((x, ε) -> norm_kernel(form_kernel(x, ε), :row), z, N) for z in ε_all_sym]

scatter(log10.(ε_all_ent), [geterr(s, spec_ref) for s in spec_ent], label = "Ent", xlabel = "log10(ε)", yscale = :log)
scatter!(log10.(ε_all_quad), [geterr(s, spec_ref) for s in spec_quad], label = "Quad")
scatter!(log10.(ε_all_sym), [geterr(s, spec_ref) for s in spec_sym], label = "Sym")
scatter!(log10.(ε_all_sym), [geterr(s, spec_ref) for s in spec_row], label = "Row")
