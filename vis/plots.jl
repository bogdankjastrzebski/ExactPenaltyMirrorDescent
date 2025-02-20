using Plots
using ExactPenaltyMirrorDescent
using LaTeXStrings
using ExactPenaltyMirrorDescent.Objectives: objectives


function bound_range(bound, xs, n)
    l, r = bound
    l = min(minimum(xs), l)
    r = max(maximum(xs), r)
    rng = l:((r-l)/n):r
    return rng
end


function bounds_ranges(bounds, xs, n)
    return Tuple( 
        bound_range(bounds[k], xs[k, :], n)
        for k in 1:size(xs, 1)
    )
end


function view_top(
            f, xs;
            bounds=((-2, 2), (-2, 2)), n=64,
            levels=16,
            ϵ=1e-1,
        )
    rng_x, rng_y = bounds_ranges(bounds[1:2], xs, n)
    # contour(
    #     rng_x, rng_y,
    #     (x, y) -> f([x, y]),
    #     levels=levels,
    #     fill=true,
    # )
    tv = 1:levels
    tl = [L"e^{$i}" for i in tv]
    contourf(
        rng_x, rng_y,
        # (x, y) -> log(f([x, y]) + 1),
        # (x, y) -> f([x, y]),
        (x, y) -> clamp(f([x, y]), bounds[3][1] - ϵ,  bounds[3][2] + ϵ),
        aspect_ratio=:equal,
        # colorbar_ticks=(tv, tl),
        # color_scale=:log,
        dpi=300,
        fill=true,
        color=:turbo,
        grid=false,
        levels=levels,
        # axis=false,
    )
    plot!(
        xs[1, :], xs[2, :],
        legend=false,
        aspect_ratio=:equal,
    )
end


function view_3d(
            f, xs;
            bounds=((-2, 2), (-2, 2)), n=64,
            camera=(150, 30),
            ϵ=1e-1,
        )
    rng_x, rng_y = bounds_ranges(bounds, xs, n)
    plot(
        rng_x, rng_y,
        # (x, y) -> f([x, y]),
        (x, y) -> clamp(f([x, y]), bounds[3][1] - ϵ,  bounds[3][2] + ϵ),
        st=:surface,
        # zscale=:ln,
        zlim=bounds[3],
        camera=camera,
    )
    #plot!(
    #    xs[1, :], xs[2, :],
    #    [f(xs[:, i]) for i in 1:size(xs, 2)],
    #)
end


function convergence_plot(f, xss)
    m = size(xss, 1)
    n = size(xss[1], 2)
    plot(1:n, [
        sum(f(xss[j][:, i]) for j in 1:m) / m
        for i in 1:n
    ], scale=:ln)
end



view_3d(second_objective, [0, 0]);
savefig("img/second_objective_3d.pdf")


view_3d(rosenbrock, xs[:, 1:1000])
savefig("img/rosenbrock_3d.pdf")

objective_names = [
#    "first_objective",
#    "second_objective",
#     "rastrigin",
#     "beale",
#     "goldstein_price",
#     "booth",
#     "bukin",
#     "matyas",
#     "himmelblau",
#     "three_hump_camel",
     "easom",
#     "sphere",
#     "mccormick",
#     "second",
#     "first",
#     "ackley",
#     "styblinski_tang",
]
for objective_name in objective_names
    println("Objective: $objective_name")
    objective, _, _, _, bounds, x₀, camera = objectives[objective_name]
    # Top
    # view_top(objective, [3 3; 3 3], bounds=bounds)
    # savefig("img/$(objective_name)_top.pdf")
    # 3d
    view_3d(objective, [3 3; 3 3], bounds=bounds, camera=camera, n=1000);
    savefig("img/$(objective_name)_3d.pdf")
end
