# An implemetation of Laplacian-Optimized Classifier
# Reference: Laplacian-optimized diffusion for semi-supervised learning
# https://www-sciencedirect-com.ezproxy.library.ubc.ca/science/article/pii/S0167839620300510


using LinearAlgebra, SparseArrays, StatsBase, Distances, IterativeSolvers
using NearestNeighborDescent

include("util.jl")

## load kSpiral data
N = 400
K = 4
X, tlabel = kSpiral(K, Int(N/K); st=0.1)


# set up

ink_idx = sample(1:N, 10, replace = false)
# ink_idx = [1, 101, 201, 301, 99, 199, 299, 399]
# ink_idx = [1, 101, 201, 301]

sort!(ink_idx)
ink_color = tlabel[ink_idx]

# label selection matrix
S = sparse(ink_idx, ink_idx, 1.0, N, N)
P = sparse(ink_idx, ink_color, 1.0, N, K)
Q = spzeros(N, K)

# construct knn

k=20

kdtree = nndescent(X', k, Euclidean())
knn_idxs, knn_dists = knn_matrices(kdtree)

# adjacency matrix
inds = vcat(Int.(ones(k)*collect(1:N)')...)
G = sparse(inds, vcat(knn_idxs...), 1, N, N)

d_out = sparse( sparse(vcat(G...)).nzind, inds, 1, N^2, N)
d_in  = sparse(sparse(vcat(G'...)).nzind, inds, 1, N^2, N)

d = d_out -d_in

w̄ = sparse(inds, vcat(knn_idxs...), vcat(knn_dists...), N, N).^4

# parameters
v = α = β = γ = 1

# spmtx 
dx = sparse(d*X)
A = 2*(α*(d*d').* (dx*dx')) + β * abs.(d)*abs.(d)'

##

# DEC Laplacian
function L(w, d)
    return d'* spdiagm(0=> w)*d
end

function optimize_w(w, Q, maxiter; tol = 1e-10)
    κ = 0.8
    converged = false
    h = 0.5
    
    # use B as vector, stored as spmtx
    B = sparse(γ* w̄ + v * pairwise(SqEuclidean(), Q, Q, dims=1)) 
    B = sparse(vcat(B...))
    E = Energy(w, B)
    # w_ = copy(w)

    for i in 1:maxiter
        h = κ*h
        w = w.*ExpClip(-(A*w + B .-4β), h)
        E_ = Energy(w, B)
        # println("h = $h, Ediff = ", E - E_)
        if (E - E_) < tol #|| h < 1e-10
            converged = true
            println("@ i = $i, optimizing weights converged.")
            break
        end
        E = E_
    end

    if !converged
        @warn "optimizing weights did not converge after $maxiter iterations."
    end 


    return w
end


function ExpClip(z, h)
    z = exp.(z)
    z[z .< 1-h] .= 1-h
    z[z .> 1+h] .= 1+h
    return z
end

function Energy(w, B)
   return 0.5 * w'*A*w + (B.-4β)'*w 
end

## w = sparse(vcat(w̄...))
w = sparse(vcat(G...))

maxiter = 50

# takes ~103 sec for 10 iterations
@time for i in 1:maxiter
    @time global w = optimize_w(w, Q, 200; tol=1e-5)
    cg!(Q, S + L(w, d), P)
end