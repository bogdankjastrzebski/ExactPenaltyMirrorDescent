using Plots

function view_top(f, xs, a=-2, b=2, n=64, l=32)
    rng = a:((b-a)/n):b
    contour(rng, rng, (x, y) -> f([x, y]), levels=l)
    plot!(xs[1, :], xs[2, :])
end

function view_3d(f, xs, a=-2, b=2, n=64)
    a = min(minimum(xs), a)
    b = max(maximum(xs), b)
    rng = a:((b-a)/n):b
    plot(
        rng, rng,
        (x, y) -> f([x, y]),
        st=:surface,
        camera=(150, 30),
    )
    plot!(xs[1, :], xs[2, :], [f(xs[:, i]) for i in 1:size(xs, 2)])
end

function convergence_plot(f, xss)
    m = size(xss, 1)
    n = size(xss[1], 2)
    plot(1:n, [
        sum(f(xss[j][:, i]) for j in 1:m) / m
        for i in 1:n
    ], scale=:ln)
end

view_3d(rosenbrock, xs[:, 1:1000])

savefig("rosenbrock_3d.pdf")
