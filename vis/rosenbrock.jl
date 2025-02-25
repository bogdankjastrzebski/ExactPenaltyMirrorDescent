using Plots
using ExactPenaltyMirrorDescent
using LaTeXStrings
using ExactPenaltyMirrorDescent.Objectives: rosenbrock, rosenbrock_random, rosenbrock_projection, rosenbrock_through, rosenbrock_back, rosenbrock_penalty
using LinearAlgebra
using Random
using Zygote


include("vis/general.jl")

objective_name = "rosenbrock"
function oracle(n, p=0.1)
    function temp(x)
        rosenbrock_random(x, n=n) + p * rosenbrock_penalty(x)
    end
    return temp' 
end
mirror = Mirror(rosenbrock_through, rosenbrock_back)
projection = rosenbrock_projection

results = Dict()
for n in [10, 100, 1000]
    xss = []
    k = 10
    for seed in 1:10
        Random.seed!(seed)
        iter = 10000
        x₀ = -ones(n)
        xs, vs = mirror_descent(
            oracle(n),
            float.(x₀),
            γ=n->0.0001/(n^0.2),
            λ=n->0.01,
            iterations=iter,
            mirror=mirror,
            project=projection,
        )
        push!(xss, xs)
    end
    results[n] = xss
end

plot(1:iter, [rosenbrock(results[100][1][:, k]) for k in 1:iter]);
savefig("img/temp.png")
