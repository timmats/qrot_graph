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
using GraphPlot
using Cairo, Fontconfig, Gadfly, Compose
using GraphRecipes
pyplot()

θ = rand(VonMises(1.5), 250)
x = hcat(map(θ -> [cos(θ),sin(θ)], θ)...)
scatter(x[1, :], x[2, :]; markerstrokewidth = 0)

include("util.jl")

# kNN
gplot(SimpleGraph(symm(knn_adj(x, 10))), x[1, :], x[2, :])
# quadOT
bgplot(SimpleGraph(kernel_ot_quad(x, 1.5)), x[1, :], x[2, :])

#=
"""
Author: Piotr A. Zolnierczuk (zolnierczukp at ornl dot gov)

Based on a paper by:
Drawing an elephant with four complex parameters
Jurgen Mayer, Khaled Khairy, and Jonathon Howard,
Am. J. Phys. 78, 648 (2010), DOI:10.1119/1.3254017
"""
import numpy as np
import pylab

# elephant parameters
p1, p2, p3, p4 = (50 - 30j, 18 +  8j, 12 - 10j, -14 - 60j )
p5 = 40 + 20j # eyepiece

def fourier(t, C):
    f = np.zeros(t.shape)
    A, B = C.real, C.imag
    for k in range(len(C)):
        f = f + A[k]*np.cos(k*t) + B[k]*np.sin(k*t)
    return f

def elephant(t, p1, p2, p3, p4, p5):
    npar = 6
    Cx = np.zeros((npar,), dtype='complex')
    Cy = np.zeros((npar,), dtype='complex')

    Cx[1] = p1.real*1j
    Cx[2] = p2.real*1j
    Cx[3] = p3.real
    Cx[5] = p4.real

    Cy[1] = p4.imag + p1.imag*1j
    Cy[2] = p2.imag*1j
    Cy[3] = p3.imag*1j

    x = np.append(fourier(t,Cx), [-p5.imag])
    y = np.append(fourier(t,Cy), [p5.imag])

    return x,y

x, y = elephant(np.linspace(0,2*np.pi,1000), p1, p2, p3, p4, p5)
pylab.plot(y,-x,'.')
pylab.show()
=#

p1, p2, p3, p4 = (50 - 30im, 18 +  8im, 12 - 10im, -14 - 60im )

function fourier(t, C)
    f = zeros(size(t)...)
    A, B = real(C), imag(C)
    for k = 1:length(C)
        f .= f .+ A[k]*cos.(k*t) .+ B[k]*sin.(k*t)
    end
    return f
end

function elephant(t, p1, p2, p3, p4, p5)
    npar = 6
    Cx = zeros(Complex, npar) 
    Cy = zeros(Complex, npar) 
    Cx[1] = real(p1)*1im
    Cx[2] = real(p2)*1im
    Cx[3] = real(p3)
    Cx[5] = real(p4)
    Cy[1] = imag(p4) + imag(p1)*1im
    Cy[2] = imag(p2)*1im
    Cy[3] = imag(p3)*1im
    x = fourier(t,Cx)
    y = fourier(t,Cy)
    return x, y
end

Random.seed!(42)
θ = rand(VonMises(1.1), 200)
x = hcat(elephant(θ, p1, p2, p3, p4, 0)...)' .+ 2.5*randn(2, length(θ))

function edgecols(A)
    q = quantile(vec(A[A .> 0]), 0.95)
    [RGBA(0, 0, 0, f/q) for f in A[A .> 0]]
end

# kNN
A = symm(knn_adj(x, 15))
draw(PDF("elephant_knn.pdf", 5cm, 5cm),
     gplot(SimpleWeightedGraph(A), x[2, :], x[1, :]; edgestrokec = edgecols(A ./ sum(A; dims = 2)), nodefillc = fill(colorant"red", size(x, 2))))
# quadOT
A = kernel_ot_quad(x, 1.0)
draw(PDF("elephant_quad.pdf", 5cm, 5cm),
     gplot(SimpleWeightedGraph(A), x[2, :], x[1, :]; edgestrokec = edgecols(A ./ sum(A; dims = 2)), nodefillc = fill(colorant"red", size(x, 2))))

