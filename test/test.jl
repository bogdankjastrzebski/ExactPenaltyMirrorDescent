using Base.Iterators

function test(
            f, x, α=[0.0001], β=[0.55], λ=[0.1],
            mirror=identity_mirror,
            project=identity,
            iterations=10000,
            times=5,
        )
    function inner(α, β, λ)
        return Tuple(
            mirror_descent(
                f, x,
                n -> (α / n)^(0.5 + β),
                n -> λ,
                iterations=iterations,
                mirror=identity_mirror,
                project=identity,
            )
            for _ in 1:times
        )
    end
    return Dict(
        (α, β, λ) => inner(α, β, λ)
        for (α, β, λ) in Iterators.product(α, β, λ)
    )
end

setups = [
    (first_objective,   -ones(2), identity_mirror, first_projection),
    (second_objective,  -ones(2), identity_mirror, identity),
    (rosenbrock,        -ones(2), identity_mirror, second_projection),
]


# f = x -> sum(x.^2)
f = rosenbrock
xs, vs = mirror_descent(
    f,
    -ones(2),
    n->(0.0001/n)^0.55, # 0.001/n^0.6
    zeros(2),
    n->0.1,
    iterations=100000,
)
view_top(f, xs[:, 1:10:end], -2, 2)

f(xs[:, end-1])

plot(1:1000:1000000, cumsum(1 ./ (1:1000000).^1.1)[1:1000:1000000]);
savefig("./plot.png")
run(`kitty icat ./plot.png`)



