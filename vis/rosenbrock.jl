#!/home/bodo/.local/bin/julia
using Pkg
Pkg.activate(".")
using ExactPenaltyMirrorDescent

using Plots
using LaTeXStrings
using ExactPenaltyMirrorDescent.Objectives: rosenbrock, rosenbrock_random, rosenbrock_projection, rosenbrock_through, rosenbrock_back, rosenbrock_penalty
using LinearAlgebra
using ProgressBars
using Random
using Zygote
using Serialization


include("general.jl")

objective_name = "rosenbrock"
function oracle(n, p=0.0001)
    function temp(x)
        rosenbrock_random(x, n=n) + p * rosenbrock_penalty(x)
    end
    return temp' 
end
mirror = Mirror(rosenbrock_through, rosenbrock_back)
projection = rosenbrock_projection
# Finding Best Parameters
results = Dict()
iter = 10_000
for n in [4, 8, 16, 32]
for k in [1]
for γ₀ in [0.1, 0.01, 0.001]
for γ₁ in [0.5, 0.1, 0.2, 0.3]
for λ in [0.5, 0.1, 0.01, 0.001]
    print("n : $n , ")
    print("k : $k , ")
    print("γ₀: $γ₀, ")
    print("γ₁: $γ₁, ")
    print("λ : $λ \n")
    try
        xss = []
        for seed in 1:1
            Random.seed!(seed)
            x₀ = zeros(n)
            # x₀ = 0.99 * ones(n)
            xs, vs = mirror_descent(
                oracle(k),
                float.(x₀),
                γ=n->γ₀/(n^(0.5 + γ₁)),
                λ=n->λ,
                iterations=iter,
                mirror=mirror,
                project=projection,
            )
            push!(xss, xs)
        end
        results[(n, k, γ₀, γ₁, λ)] = xss
        serialize("results/results.ser", results)
    catch e             
        println(e)
    end
end
end
end
end
end

function get_best(results)
    best = Dict()
    for key in keys(results)
        res = results[key]
        val = sum([rosenbrock(r[:, end]) for r in res])/length(res)
        yek = (key[1], key[2])
        if !(yek in keys(best)) || best[yek][1] > val
            best[yek] = (val, key)
        end
    end
    return best
end


best = get_best(results)

# key = rand(keys(best))
# res = results[best[key][2]][1]
# plot(1:size(res, 2), [rosenbrock(res[:, k]) for k in 1:size(res, 2)]);
# title!("Rosenbrock (n=$(key[1]))");
# plot!(fontfamily="Computer Modern");
# plot!(xscale=:log);
# plot!(yscale=:log);
# savefig("img/temp.png")



# Calculate for Best Parameters
best_results = Dict()
iter = 1_000_000
for (_, (_, key)) in best
    n, k, γ₀, γ₁, λ = key
    print("n : $n , ")
    print("k : $k , ")
    print("γ₀: $γ₀, ")
    print("γ₁: $γ₁, ")
    print("λ : $λ \n")
    try
        xss = []
        for seed in 1:10
            Random.seed!(seed)
            x₀ = zeros(n)
            # x₀ = 0.99 * ones(n)
            xs, vs = mirror_descent(
                oracle(k),
                float.(x₀),
                γ=n->γ₀/(n^(0.5 + γ₁)),
                λ=n->λ,
                iterations=iter,
                mirror=mirror,
                project=projection,
            )
            push!(xss, xs)
        end
        best_results[(n, k, γ₀, γ₁, λ)] = xss
        serialize("results/best_results.ser", best_results)
    catch e             
        println(e)
    end
end


# key = rand(keys(best_results))
# res = best_results[key][1]
# plot(1:size(res, 2), [rosenbrock(res[:, k]) for k in 1:size(res, 2)]);
# title!("Rosenbrock (n=$(key[1]))");
# plot!(fontfamily="Computer Modern");
# # plot!(xscale=:log);
# # plot!(yscale=:log);
# savefig("img/temp.png")


# key = rand(keys(results))
# res = results[key][1]
# plot(1:size(res, 2), [rosenbrock(res[:, k]) for k in 1:size(res, 2)]);
# savefig("img/temp.png")
# 
# # serialize("results.ser", results)
# for key in keys(results)
#     res = results[key][1]
#     print("Press enter to continue> ")
#     readline()
# end
# 
# results[(100, 10)][1][:, end]
# rosenbrock(0.99 * ones(100))
