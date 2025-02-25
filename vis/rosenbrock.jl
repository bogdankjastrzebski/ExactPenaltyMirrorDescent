# using P] activate .
using ExactPenaltyMirrorDescent

using Plots
using LaTeXStrings
using ExactPenaltyMirrorDescent.Objectives: rosenbrock, rosenbrock_random, rosenbrock_projection, rosenbrock_through, rosenbrock_back, rosenbrock_penalty
using LinearAlgebra
using ProgressBars
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
for k in [1, 10, 100]
for lr in [0.001, 0.01, 0.05]
    xss = []
    for seed in 1:1
        println
        Random.seed!(seed)
        iter = 1000
        x₀ = zeros(n)
        xs, vs = mirror_descent(
            oracle(k),
            float.(x₀),
            γ=n->lr/(n^0.5),
            λ=n->0.1,
            iterations=iter,
            mirror=mirror,
            project=projection,
        )
        push!(xss, xs)
    end
    results[(n, k)] = xss
    serialize("results.ser", results)
end
end
end
serialize("results.ser", results)


# plot(1:iter, [rosenbrock(results[1000][1][:, k]) for k in 1:iter]);
# savefig("img/temp.png")

