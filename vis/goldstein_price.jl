using Plots
using ExactPenaltyMirrorDescent
using LaTeXStrings
using ExactPenaltyMirrorDescent.Objectives: objectives, atan2
using LinearAlgebra

input("general.jl")

Random.seed!(0)
#
#
objective_name = "goldstein_price",
objective, projection, penalty, _, bounds, x₀, camera = objectives[objective_name]
# Oracle
function oracle(f, x₀, snr)
    α = 1/norm(f'(x₀))
    function temp(x)
        g = α * f'(x)
        n = randn(size(x))
        return g + (norm(g) / norm(n) / snr) * randn(size(x)) 
    end
    return temp
end
objective_penalty = x -> objective(x) + 50*penalty(x)
xs, vs = mirror_descent(
    oracle(objective_penalty, x₀, 0.5),
    0.1randn(2) + float.(x₀),
    γ=n->1.0/(n^0.6),
    λ=n->0.01,
    iterations=100000,
)
# top
view_top(objective_penalty, xs, bounds=bounds, n=100, levels=256)
title!(transform_name(objective_name))
plot!(size=(400, 400))
plot!(fontfamily="Computer Modern")
# savefig("img/goldstein_price/$(objective_name)_top.png")
# 3d
view_3d(objective_penalty, xs, bounds=bounds, camera=camera, n=1000);
plot!(fontfamily="Computer Modern")
# savefig("img/goldstein_price/$(objective_name)_3d.png");
savefig("img/tmp.png");
