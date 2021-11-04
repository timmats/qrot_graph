# Rotate each point in x by theta about the x-axis
function rotate(x, θ)
    r = @. [[cos(θ) -sin(θ)]; 
            [sin(θ)  cos(θ)]] 

    return (r*x)'
end

#Make a spiral with k arms and n points per arm
function kSpiral(k, n; st = 0.5, ed = 5, noise = 0.0, unf=true)
    θ = range(0, 2π, length = k+1) #Divide a full rotation into k intervals
    inds = ones(Int32, k*n) #Make a matrix of labels
    y = zeros(n*k, 2) 
    
    for i in 1:k
        if unf
            #Make n random points along the curve
            t =  sort(ed * sqrt.(rand(n)) .+ st) 
        else
            t =  ed * rand(n) .+ st       
        end 
        
        x = @. [cos(t) * t sin(t) * t] #Plug into the parametrization for the spiral
        y[(i-1)*n+1:i*n, :] = rotate(x', θ[i]) #Rotate to give spiral effect
        y += noise * rand(n*k, 2) #Optionally add noise to the position
        inds[(i-1)*n+1:i*n] .= i #Label arm i as i
    end
    
    return y, Int.(inds)
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



function compute_NS(P_, sinks)
    P = copy(P_)
    P[sinks, :] .= 0
    P[sinks, sinks] .= I(sum(sinks))
    P = P./sum(P, dims = 2)
    Q = P[.!sinks, .!sinks]
    S = P[.!sinks, sinks]
    N = inv(I(size(Q, 1)) .- Q)
    return N, S
end

function compute_B(P, sinks)
    N, S = compute_NS(P, sinks)
    return N*S
end


#include("debugQROT.jl")


function fit_ssl(x, λ0, ϵ, sink_idx)
    C = pairwise(SqEuclidean(), x, x, dims=1)
    C[diagind(C)] .= Inf
    C[C.> 0.7*mean(C)] .= Inf

    μ = ones(size(x, 1))
    ν = ones(size(x, 1))
    λ = λ0/sum(.!sink_idx)

    μ[.!sink_idx] .+= λ
    ν[sink_idx] .+= λ*sum(.!sink_idx)/sum(sink_idx)

    γ = Matrix(dSSN(μ, ν, C, ϵ; maxiter = 1000, θ = 0.3))
    # γ = sinkhorn(μ, ν, C, 0.1);
    return γ
end


function BistochasticKernel(mu, nu, C, ϵ, reg; k=15)        
    
    if reg == "ent"
        
        K = @. exp(-C / ϵ)
        K = sparse(K)
        
        u, v = OptimalTransport.sinkhorn_gibbs(mu, nu, K,  maxiter=5000)
        
        P = Diagonal(u) * K * Diagonal(v)
    elseif  reg == "quad"
        # P = dSSN(mu, nu, C, ϵ; θ = 0.1, maxiter = 500)
        P = quadreg(mu, nu, C, ϵ)
    else
        println("Invalud input: ", reg)
    end

    if any(isnan.(P))
        print("\nError: Bi-normalized matrix contains NaN \n")
        n = size(C, 1)
        return spzeros(k), spzeros(n, k), spzeros(n, n) 
    end
    
    # D, U = eigen(Matrix(P))
    # return reverse(real.(D)), reverse(real.(U), dims=2), P
    

    # P[P.<1e-16] .=0
    # P = sparse(P)
    
    D, U, _ = eigsolve(P, k, :LR ;krylovdim=2*k )
    return real.(D), hcat(real.(U)...), P
end


function Diffusion_Distance(U, Λ, t)
    
    # if Λ[1] > 1 - 1e-20
    #     print("\n Warning: eigenvalue larger than 1\n")
    #     # return spzeros(size(U, 1), size(U, 1))
    # end

    π0 = normalize(abs.(U[:,1]), 1) # noemalize
    ψ = π0.^(-1/2).*U # check this line

    if t > 0
        # D = diagm(Λ.t)
        ϕ = ψ.*(Λ.^t)'
    else
        # D = diagm(1 / (1-Λ))
        ϕ = @. ψ[:, 2:end]*(1 / (1-Λ[2:end]))'
    end
    # ϕ = ψ*D
    
    diff = pairwise(Euclidean(), ϕ, ϕ, dims=1)
    # return sparse(diff)
    return diff
end


function FW(D::AbstractMatrix{W}) where {W} 
    # D: minimum distance matrix (initialized to edge distances)

    # argument checking

    n = size(D, 1)
    if size(D, 2) != n
        throw(ArgumentError("D should be a square matrix."))
    end

    rows, cols, vals = findnz(D)
    m = length(rows)

    # initialize
    D[D.==0] .= Inf
    D[diagind(D)] .= 0
    # main loop

    for k = 1 : m, i = 1 : n, j = 1 : n
        # d = D[i,k] + D[k,j]
        # if d < D[i,j]
        #     D[i,j] = d
        # end
        D[i,j] = min(D[i,cols[k]] + D[rows[k],j], D[i,j])

    end

    return Matrix(D)
end




#Given the number of nodes, the known labels,
#and the proportion of labels to sample, returns
#prop*N selected indices and labels.
#
#PARAMS:
#N: The number of nodes
#labels_all: A label corresponding to each node
#prop: The proportion of nodes to sample #NOT IMPLEMENTED
#
#RETURNS:
#ink_idx: An array of indices of selected nodes
#ink_color: Colors corresponding to the selected nodes
function sample_labels(N, labels_all, prop)
    #Sample nodes for which we assume labels are known
    ink_idx = sample(1:N, Int(floor(prop*N)), replace = true) #Get the indices of the nodes
    sort!(ink_idx)
    ink_color = labels_all[ink_idx]  #Get the corresponding labels

    return ink_idx, ink_color
end

#Makes selection matrices S, P, Q, and an indicator vector
#
#PARAMS:
#ink_idx: The indices of the selected nodes
#ink_color: The label of the selected notes
#N: The number of nodes
#K: The number of labels
#
#RETURNS:
#ink_ind: A indicator vector of all nodes with 1 when a node is selected
#S: A diagonal matrix of all nodes with 1 when a node is selected
#P: A selection matrix where labelled cells are marked in their corresponding color column
#Q: A likelihood matrix for nodes being in a class (initialized to 0)
function make_selection_matrices(ink_idx, ink_color, N, K)
    #Mark selected nodes as true, and all others false (indicator)
    ink_ind = zeros(Bool, N) 
    ink_ind[ink_idx] .= true

    # label selection matrix
    S = sparse(ink_idx, ink_idx, 1.0, N, N) #A diag matrix with ones on labelled nodes
    P = sparse(ink_idx, ink_color, 1.0, N, K) #A selection matrix where labelled cells are marked in their corresponding color colum
    Q = spzeros(N, K) #A likelihood matrix for nodes being in a class
    
    return ink_ind, S, P, Q
end

#Creates a cost matrix for optimal transport and
#target and source histograms.
#
#PARAMS:
#X: Coordinates of points in R^2
#
#RETURNS:
#C: A euclidean cost matrix b/w nodes
#μ: A uniform source histogram on nodes
#ν: A uniform target histogram on nodes
function make_ot_params(X)
    C = pairwise(SqEuclidean(), X, X, dims=1) #Euclidean distance between nodes
#     C ./= mean(C)
    C[diagind(C)] .= Inf #No self edges allowed

    μ = ones(size(X, 1)) #Uniform distribution on nodes
    ν = ones(size(X, 1))
    
    return C, μ, ν
end

function knnWeight(k, C, X, N, σ = 0.5)

    knn_idxs, knn_dists = knn( KDTree(X'), X', k, true)

    # adjacency matrix
    inds = vcat(Int.(ones(k)*collect(1:N)')...)
    A = sparse(inds, vcat(knn_idxs...), 1, N, N)

    inds_sym = A+A'.>0
    K = spzeros(N, N)
#     σ = 0.5
#     TODO: vary \sigma

    K[inds_sym] = exp.(-C[inds_sym].^2 / σ)

    d = spdiagm(0 => sum(K, dims=1)[1, :].^(-1))

    return d*K
end

#Solve the linear system for likelihood given
#a transition matrix.
#
#PARAM:
#γ: A transition matrix between cells
#η: A regularization term
#Q: A initial guess for the likelihood matrix
#P: A selection matrix where labelled cells are marked in their corresponding color column
#
#RETURNS:
#Q: A likelihood matrix for nodes belonging to a class (label)
function solve_for_likelihood!(γ, Q, P, η = 8)
    #Solve the linear system for likelihood
    cg!(Q, I+η*(I-γ), P, maxiter=1000)
    Q = Q./sum(Q, dims=2)
    
    return Q
end

#Solve the linear system for likelihood given
#a transition matrix.
#
#PARAM:
#γ: A transition matrix between cells
#η: A regularization term
#Q: A initial guess for the likelihood matrix
#P: A selection matrix where labelled cells are marked in their corresponding color column
#
#RETURNS:
#Q: A likelihood matrix for nodes belonging to a class (label)
function solve_for_likelihood_KNN!(γ, Q, P, S, η = 8)
    #Solve the linear system for likelihood
    cg!(Q, S+η*(I-γ), P, maxiter=1000)
    Q = Q./sum(Q, dims=2)
    
    return Q
end

#Get expected labels and inferred labels
#
#PARAMS:
#Q: A likelihood matrix on nodex and labels
#K: The number of labels
#
#RETURNS:
#soft_label: Expectation of labels
#infer_label: The label infered by the largest probability
function get_labels(Q, K)
    soft_label = Q*collect(1:K)
    inferred_label = getindex.( argmax(Q, dims=2), 2)[:]
    
    return soft_label, inferred_label
end

#Get the accuracy of the inferred labels as a fraction
#of total (unknown) labels
#
#PARAMS:
#labels_all: The complete labels for all nodes
#ink_ind: The indices of known labels
#infer_label: The inferred labels
#N: The number of nodes
#
#RETURNS:
#accuracy: The fraction of labels infered correctly
function get_accuracy(labels_all, ink_ind, infer_label, N)
    accuracy = 1- sum(abs.(labels_all[.!ink_ind]-infer_label[.!ink_ind]) .>0)/(N-sum(ink_ind))

    return accuracy
end

#Performs semi-supervised label learning on a k-armed spiral
#using a quadratic OT transition matrix.
#
#PARAMS:
#N: The number of nodes in the spiral
#ϵ: The regularization parameter for quadratic OT
#K: The number of arms in the spiral
#η: The weight of the energy term in the system of 
#   equations solving for likelihood
#
#RETURNS:
#Q: A likelihood matrix on the nodes and labels
#accuracy: The accuracy of the prediction
#X: The locations of the nodes
#labels_all: The actual labels of the nodes
function simulate_QOT(X, N, K, labels_all, ink_idx, ink_color, ϵ, prop, η=8)
    
    #Sample nodes and get their labels
#     ink_idx, ink_color = sample_labels(N, labels_all, prop)
    ink_ind, S, P, Q = make_selection_matrices(ink_idx, ink_color, N, K)
    
    #Perform quad OT
    C, μ, ν = make_ot_params(X)
#     γ = Matrix(OptimalTransport.quadreg(μ, ν, C, ϵ; maxiter = 100, θ = 0.3)) #Get a transport map based on node-node distances
    γ = quadreg(μ, ν, C, ϵ)
    
    #Solve for the likelihood matrix
    Q = solve_for_likelihood!(γ, Q, P)
    
    soft_label, infer_label = get_labels(Q, K)

    #Check what percentage of inferred labels match 
    accuracy = get_accuracy(labels_all, ink_ind, infer_label, N)
    println("Solved for $N nodes, K= $K, ϵ= $ϵ, η= $η, accuracy= ", accuracy)
    
    return Q, accuracy, γ
end

#Performs semi-supervised label learning on a k-armed spiral
#using a simple weighted KNN transition matrix.
#
#PARAMS:
#N: The number of nodes in the spiral
#K: The number of arms in the spiral
#η: The weight of the energy term in the system of 
#   equations solving for likelihood
#
#RETURNS:
#Q: A likelihood matrix on the nodes and labels
#accuracy: The accuracy of the prediction
#X: The locations of the nodes
#labels_all: The actual labels of the nodes
function simulate_KNN(X, N, K, labels_all, ink_idx, ink_color, neighbours, prop, σ, η=8)
    
    #Sample nodes and get their labels
    
    ink_ind, S, P, Q = make_selection_matrices(ink_idx, ink_color, N, K)
    
    #Perform quad OT
    C, _, _ = make_ot_params(X)
    γ = knnWeight(neighbours, C, X, N, σ)
    
    #Solve for the likelihood matrix
    Q = solve_for_likelihood_KNN!(γ, Q, P, S)
    
    soft_label, infer_label = get_labels(Q, K)

    #Check what percentage of inferred labels match 
    accuracy = get_accuracy(labels_all, ink_ind, infer_label, N)
    println("Solved for $N nodes, K= $K, neighbours= $neighbours, η= $η, accuracy= $accuracy ")
    
    return Q, accuracy, γ
end