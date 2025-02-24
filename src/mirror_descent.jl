# Implementation of SMD

struct Mirror{F, B}
    through::F
    back::B
end

identity_mirror = Mirror(identity, identity)

"""mirror_descent_update(M, g, γ, x, λ=0, v=0)

# Arguments
* M: mirror map.
* g: first order oracle.
* γ: learning rate.
* x: current state.
* λ: dual averaging parameter.
* v: dual average.

# Returns
* x: new state
* v: new dual average
"""
function mirror_descent_update(
        g, x, γ, v=0, λ=1;
        mirror=identity_mirror,
        project=identity,)
    y = mirror.through(x)
    v = (1 - λ) * v .+ λ * g(x)
    y -= γ * v
    x = project(mirror.back(y))
    return x, v
end


"""mirror_descent(g, x, γ=n->1/n, λ=n->1;
                  mirror=identity_mirror,
                  project=identity,
                  iterations=1000)

Minimize function using mirror descent with first order oracle `g`.

# Arguments
*   `g`: First order oracle function (gradient/subgradient).
*   `x`: Initial point.
*   `γ`: Step size function (default: `n->1/n`).
*   `λ`: Regularization parameter function (default: `n->1`).

# Keywords
*   `mirror`: Mirror map (default: `identity_mirror`).
*   `project`: Projection function (default: `identity`).
*   `iterations`: Number of iterations (default: 1000).

# Returns
Tuple: `(xs, vs)` - history of `x` and `v` states as matrices.

# Example
```julia
# using Zygote
# f(x) = [2*x[1], 4*x[2]] # Gradient of x[1]^2 + 2*x[2]^2
# x0 = [1.0, 1.0]
# xs, vs = mirror_descent(f', x0, iterations=100)
"""
function mirror_descent(
        g, x;
        γ=n->1.0/n, λ=n->1,
        mirror=identity_mirror,
        project=identity,
        iterations=1000,)
    v = zero(x)
    xs = [x]
    vs = [v]
    for n in 1:iterations
        x, v = mirror_descent_update(
            g, x, γ(n), v, λ(n),
            mirror=mirror,
            project=project,
        )
        push!(xs, x)
        push!(vs, v)
    end
    return stack(xs), stack(vs)
end
