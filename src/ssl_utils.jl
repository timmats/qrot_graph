using Distributed, SharedArrays

using LinearAlgebra, SparseArrays, StatsBase, Distances, IterativeSolvers
using NearestNeighbors, OptimalTransport, MultivariateStats
using Distributed, SharedArrays
# include("util.jl")
using Plots

function rotate_point(x, θ)
    r = @. [[cos(θ) -sin(θ)]; 
            [sin(θ)  cos(θ)]] 

    return (r*x)'
end



#Make a spiral with k arms and n points per arm
# PARAM: 
# arms: array of integers, n-th entry is the number of points on n-th arm
# todo: take a function f as input for sscaling along arm
function general_kSpiral(arms; st = 0.5, ed = 5)
    total_so_far = 0
    total_pts = sum(arms)
    class_num = length(arms)
    
    #Divide a full rotation into k intervals
    θ = range(0, 2π, length = class_num+1)
    
    #Make a matrix of labels
    inds = ones(Int32, total_pts) 
    
    X = zeros(total_pts, 2)
    T = zeros(total_pts)


    for i in 1:class_num

        t = ed * rand(arms[i]).^2 .+ st

#         t = ed * LinRange(0, 1, arms[i]).^2 .+ st
        
        sort!(t)
        
        #Plug into the parametrization for the spiral
        x = @. [cos(t) * t sin(t) * t] 

        #Rotate to give spiral effect
        X[total_so_far+1 : total_so_far + arms[i], :] = rotate_point(x', θ[i]) 
        T[total_so_far+1 : total_so_far + arms[i], :] = t

        #Label arm i as i
        inds[total_so_far+1 : total_so_far + arms[i]] .= i 

        total_so_far += arms[i]

    end
    
    return X, T, inds
end


function SwissRoll(N)
    
    T = rand(N)
    t = (3 * pi / 2) * (1 .+ 2 * T)
    # t = 2* (1 .+ 2 * rand(N)).^2
    h = 30 * rand(N)
    coords = [t .* cos.(t) h t .* sin.(t)] 
    # + noise * randn(N, 3)
    
    arc = @. 0.5 * (t*sqrt(t^2 +1) + asinh(t))
    
    geodesic = @. sqrt( (arc- arc')^2 + (h-h')^2) 

    params = hcat(T, h./30)
    
   return coords, geodesic, params
end



function sample_labels(all_labels, prop)
    N = length(all_labels)
    #Sample nodes for which we assume labels are known
    label_idx = sample(1:N, Int(floor(prop*N)), replace = false) #Get the indices of the nodes
#     sort!(label_idx)
    labels = all_labels[label_idx]  #Get the corresponding labels

    return label_idx, labels
end

function initialize_SSL(label_idx, labels, N, K)
    #Mark selected nodes as true, and all others false (indicator)
    given_labels = zeros(Bool, N) 
    given_labels[label_idx] .= true

    # label selection matrix
    S = sparse(label_idx, label_idx, 1.0, N, N) #A diag matrix with ones on labelled nodes
    P = sparse(label_idx, labels, 1.0, N, K) #A selection matrix where labelled cells are marked in their corresponding color colum
#     Q = rand(N, K) #A likelihood matrix for nodes being in a class
    Q = ones(N, K) #A likelihood matrix for nodes being in a class

    return given_labels, S, P, Q
end



function solve_SSL!(γ, Q, P, S, η = 8)
    #Solve the linear system for likelihood
    cg!(Q, S+η*(I-γ), P, maxiter=1000)
#     Q = Matrix(S+η*(I-γ)) \ Matrix(P) .+ 1e-10
    
    return Q
end

function infer_labels(Q)
    _, class_num = size(Q)
    soft_label = Q*collect(1:class_num)
    inferred_label = getindex.( argmax(Q, dims=2), 2)[:]
    
    return soft_label, inferred_label
end

function accuracy(all_labels, ink_ind, infer_label, N)
    acc = 1- sum(abs.(labels_all[.!ink_ind]-infer_label[.!ink_ind]) .>0)/(N-sum(ink_ind))

    return acc
end

function predict_label(γ, label_idx, labels, N, K)

    given_labels, S, P, Q = initialize_SSL(label_idx, labels, N, K)
    
    Q = solve_SSL!(γ, Q, P, S)
    
    _, inferred_label = infer_labels(Q)    
    
    return Q, inferred_label
end


function kNN_Weight(X, k, σ = 0.5)
    
    N = size(X)[1]
    knn_idxs, knn_dists = knn( KDTree(X'), X', k, true)

    # adjacency matrix
    inds = vcat(Int.(ones(k)*collect(1:N)')...)
    A = sparse(inds, vcat(knn_idxs...), 1, N, N)

    inds_sym = A+A'.>0
    K = spzeros(N, N)
#     σ = 0.5
#     TODO: vary \sigma
    C = pairwise(SqEuclidean(), X, X, dims=1)
    K[inds_sym] = exp.(-C[inds_sym].^2 / σ)

    d = spdiagm(0 => sum(K, dims=1)[1, :].^(-1))

    return d*K
end


function QOT_weight(X, ε; maxiter=15)
    N, class_num = size(X)
    C = pairwise(SqEuclidean(), X, X, dims=1) #Euclidean distance between nodes
    C[diagind(C)] .= Inf #No self edges allowed
    
    return quadreg(ones(N), ones(N), C, ε, maxiter=maxiter)
end

function ENT_weight(X, ε)
    N, class_num = size(X)
    C = pairwise(SqEuclidean(), X, X, dims=1) #Euclidean distance between nodes
    C[diagind(C)] .= Inf #No self edges allowed
    
    return sinkhorn(ones(N), ones(N), C, ε)
end
