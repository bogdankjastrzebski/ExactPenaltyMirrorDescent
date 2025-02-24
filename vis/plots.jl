using Plots
using ExactPenaltyMirrorDescent
using ExactPenaltyMirrorDescent.Objectives: objectives, atan2
using LinearAlgebra
using LaTeXStrings


objective_names = [
#      "first_objective",
#      "second_objective",
#      "rastrigin",
#      "beale",
     "goldstein_price",
#      "booth",
#      "bukin",
#      "matyas",
#      "himmelblau",
#      "three_hump_camel",
#      "easom",
#      "mccormick",
#      "styblinski_tang",
#      "sphere",
#      "ackley",
]
plot_dict = Dict()
objective = Nothing
xs = Nothing
Random.seed!(0)
for objective_name in objective_names
  try
    println("Objective: $objective_name")
    objective, projection, penalty, _, bounds, x₀, camera = objectives[objective_name]
    # Top
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
        # γ=n->1.0/(n^0.6),
        γ=n->1.0/(n^0.6),
        # λ=n->0.01,
        λ=n->0.01,
        iterations=100000,
    )
    fig = view_top(objective_penalty, xs, bounds=bounds, n=100, levels=256)
    title!(transform_name(objective_name))
    plot!(size=(400, 400))
    plot!(fontfamily="Computer Modern")
    savefig("img/$(objective_name)_top.png")
    # savefig("img/top/$(objective_name)_top.png")
    # plot_dict[objective_name] = fig
    # 3d
    view_3d(objective_penalty, xs, bounds=bounds, camera=camera, n=1000);
    #savefig("img/3d/$(objective_name)_3d.png")
    savefig("img/$(objective_name)_3d.png")
  catch e
    println("Error: $objective_name")
    rethrow(e)
  end
end

gr()

plot(
    [plot_dict[name] for name in objective_names]...,
    layout=(5, 3),
    size=(3000, 5000),
    fontfamily="Computer Modern",
); savefig("img/top/all.png")


plot(1:10, cumsum(randn(10)), fontfamily="Computer Modern")
title!("Hello World");
savefig("img/test.png")

