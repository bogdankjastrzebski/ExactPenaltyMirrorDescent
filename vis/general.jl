using Plots
using LaTeXStrings


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


function my_palette(n=400)
    p = deepcopy(palette(:turbo))
    for i in 1:30
        pop!(p.colors.colors)
    end
    e = p.colors.colors[end] 
    for i in 1:n
        τ = i / n
        push!(p.colors.colors, (1 - τ) * e + τ * RGB(1,1,1))
    end
    return p
end


 """view_top(f, xs; bounds=((-2, 2), (-2, 2), :auto), n=64, levels=32, ϵ=1e-1)

Creates a contour plot of a function `f` with overlaid data points.

# Arguments
- `f::Function`: Objective function (ℝ² -> ℝ).
- `xs::Matrix{Float64}`: Data points to overlay (2 x N matrix).
- `bounds::Tuple`: Axis ranges ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
- `n::Int`: Number of grid points per dimension.
- `levels::Int`: Number of contour levels.
- `ϵ::Float64`: Clamp range extension for z-values.
"""
function view_top(
            f, xs;
            bounds=((-2, 2), (-2, 2)), n=64,
            levels=32,
            ϵ=1e-1,
        )
    rng_x, rng_y = bounds_ranges(bounds[1:2], xs, n)
    contour(
        rng_x, rng_y,
        (x, y) -> f([x, y]),
        levels=levels,
        fill=true,
    )
    tv = 1:levels
    tl = [L"e^{$i}" for i in tv]
    contourf(
        rng_x, rng_y,
        # (x, y) -> log(f([x, y])),
        # (x, y) -> f([x, y]),
        (x, y) -> clamp(f([x, y]), bounds[3][1] - ϵ,  bounds[3][2] + ϵ),
        aspect_ratio=:equal,
        # figure_size=(10, 10),
        framestyle=:semi,
        # colorbar_ticks=(tv, tl),
        # color_scale=:log,
        dpi=300,
        fill=true,
        # linewidth=-1,
        lw=0,
        color=my_palette(),
        grid=false,
        levels=levels,
        xlims=bounds[1],
        ylims=bounds[2],
        # axis=false,
    )
    plot!(
        xs[1, :], xs[2, :],
        legend=false,
    )
end



"""view_3d(f, xs; bounds=((-2, 2), (-2, 2), :auto), n=64, camera=(150, 30), ϵ=1e-1)

Creates a 3D surface plot of a function `f` with overlaid data points.

# Arguments
- `f::Function`: The objective function (ℝ² -> ℝ).
- `xs::Matrix{Float64}`: Data points to overlay (2 x N matrix).
- `bounds::Tuple`: Axis ranges ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
- `n::Int`: Number of grid points per dimension for surface.
- `camera::Tuple`: Camera angle (azimuth, elevation).
- `ϵ::Float64`: Clamp range extension for z-values.
"""
function view_3d(
            f, xs;
            bounds=((-2, 2), (-2, 2)),
            n=64,
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
    plot!(
        xs[1, :], xs[2, :],
        [f(xs[:, i]) for i in 1:size(xs, 2)],
        legend=false,
    )
end


"""convergence_plot(f, xss)

Plots average objective value (log scale) vs. iteration for
optimization trajectories `xss` with objective function `f`.

# Arguments
- `f::Function`: Objective function (scalar output).
- `xss::Array{Matrix{Float64}, 1}`: Trajectories (d x n matrices).

# Returns
A Plots.Plot object.
"""
function convergence_plot(f, xss)
    m = size(xss, 1)
    n = size(xss[1], 2)
    plot(1:n, [
        sum(f(xss[j][:, i]) for j in 1:m) / m
        for i in 1:n
    ], scale=:ln)
end


"""transform_name(input_string::String)
Transforms a snake_case string (e.g., not_the_camel_case) into a
space-separated string of capitalized words (e.g., Not The Camel Case).
# Args:
  input_string: The string to transform (in snake_case).
# Returns:
  A string with each word capitalized and separated by spaces.
"""
function transform_name(input_string::String)
    words = split(input_string, "_")
    capitalized_words = map(x -> titlecase(x), words)
    transformed_string = join(capitalized_words, " ")
    return transformed_string
end
