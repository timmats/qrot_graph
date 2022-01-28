function rotate_point(x, θ)
    r = @. [[cos(θ) -sin(θ)]; 
            [sin(θ)  cos(θ)]] 
    return (r*x)'
end

function spiral(arms; st = 1, ed = 5)
    L(t) = 0.5(t*sqrt(t^2 + 1) + asinh(t))
    Linv(s) = find_zero(t -> L(t) - s, (st, ed))
    total_so_far = 0
    total_pts = sum(arms)
    class_num = length(arms)
    #Divide a full rotation into k intervals
    θ = range(0, 2π, length = class_num+1)
    #Make a matrix of labels
    inds = ones(Int, total_pts) 
    X = zeros(total_pts, 2)
    T = zeros(total_pts)
    for i in 1:class_num
        # t = ed * rand(arms[i]).^2 .+ st
        # sample w.r.t arclength
        unitrange = collect(range(0., 1.; length = arms[i]))
        # unitrange = rand(arms[i])
        t = Linv.((L(ed)-L(st))*unitrange .+ L(st))
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

function LLGC(S, labels_all, label_idx, μ)
    Y = fill!(similar(labels_all, length(labels_all), maximum(labels_all)), 0)
    for i in label_idx
        Y[i, labels_all[i]] = 1
    end
    β = μ/(1+μ)
    α = 1-β
    F = β*((I - α*S) \ Y)
    labels_infer = map(argmax, eachrow(F))
    return labels_infer, F, Y
end

err(l, l_true) = mean(l .!= l_true)

err_norm(l, l_true) = mean([mean(l[l_true .== x] .!= x) for x in unique(l_true)])

