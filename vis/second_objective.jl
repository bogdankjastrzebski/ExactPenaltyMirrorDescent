using Plots
using ExactPenaltyMirrorDescent
using LaTeXStrings
using ExactPenaltyMirrorDescent.Objectives: objectives, atan2
using LinearAlgebra
using Random
using Zygote

include("vis/general.jl")

Random.seed!(1)
objective_name = "second_objective"
objective, projection, penalty, mirror, bounds, x₀, camera = objectives[objective_name]
mirror = Mirror(mirror[1], mirror[2])
# mirror = identity_mirror
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
function objective_penalty(x)
    return objective(x) + 100*penalty(x)
end
xs, vs = mirror_descent(
    oracle(objective_penalty, x₀, 0.5),
    0.1randn(2) + float.(x₀),
    γ=n->1.0/(n^0.5),
    λ=n->0.1,
    iterations=10000,
    mirror=mirror,
    project=projection,
)
# top
view_top(objective_penalty, xs, bounds=bounds, n=100, levels=64);
title!(transform_name(objective_name));
plot!(size=(400, 400));
plot!(fontfamily="Computer Modern");
plot!(10*[-1, 1, 1, -1, -1], 10*[-1, -1, 1, 1, -1], color="red", label="projection bounds");
plot!([-10, 10], [-10, 10], color="green");
savefig("img/goldstein_price/$(objective_name)_top_penalty.png");

# 3d
# view_3d(objective_penalty, xs, bounds=bounds, camera=camera, n=1000);
# plot!(fontfamily="Computer Modern")
# zlims!(0, 500);
# plot!(scale=(1 ,1, 0.5));
# savefig("img/goldstein_price/$(objective_name)_3d.png");
