using Distances 

function get_cost(X; diag_inf = false)
    C = pairwise(SqEuclidean(), X, X)
    C = mean_norm(C)
    if diag_inf
        C[diagind(C)] .= Inf
    end
    C
end

mean_norm(x) = x ./ mean(x)

symm(A) = 0.5*(A.+A')
kernel_ot_ent = (X, ε; diag_inf = false, rtol = 1e-6, atol = 1e-9) -> symm(sinkhorn(ones(size(X, 2)), get_cost(X; diag_inf = diag_inf), ε, OptimalTransport.SymmetricSinkhornGibbs(); maxiter = 5_000, rtol = rtol, atol = atol))
kernel_ot_quad = (X, ε; diag_inf = false, rtol = 1e-6, atol = 1e-9) -> symm(quadreg(ones(size(X, 2)), get_cost(X; diag_inf = diag_inf), ε, OptimalTransport.SymmetricQuadraticOTNewton(); maxiter = 100, atol = atol, rtol = rtol))
kernel_epanech = (X, ε; diag_inf = false, rtol = 1e-6, atol = 1e-9) -> symm(relu.(1 .- get_cost(X; diag_inf = diag_inf)/ε))


function knn_adj(X, k)
    # indices, _ = knn_matrices(nndescent(X, k, Euclidean())); 
    indices, _ = knn(KDTree(X), X, k);
    A = spzeros(size(X, 2), size(X, 2));
    @inbounds for i = 1:size(A, 1)
        A[i, i] = 1
        @inbounds for j in indices[i]
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
