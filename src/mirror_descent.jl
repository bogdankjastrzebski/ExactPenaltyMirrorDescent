# Implementation of SMD
using Zygote

struct Mirror{F, B}
    through::F
    back::B
end

identity_mirror = Mirror(identity, identity)

"""`mirror_descent_update(M, f, γ, x, λ=0, v=0)`

# Arguments
* M: mirror map.
* f: function being optimized.
* γ: learning rate.
* x: current state.
* λ: dual averaging parameter.
* v: dual average.

# Returns
* x: new state
* v: new dual average
"""
function mirror_descent_update(
        f, x, γ, v=0, λ=1;
        mirror=identity_mirror,
        project=identity,)
    y = mirror.through(x)
    v = (1 - λ) * v + λ * f'(x)
    y -= γ * v
    x = project(mirror.back(y))
    return x, v
end


function mirror_descent(
        f, x, γ=n->1.0/n, λ=n->1;
        mirror=identity_mirror,
        project=identity,
        iterations=1000)
    v = zero(x)
    xs = [x]
    vs = [v]
    for n in 1:iterations
        x, v = mirror_descent_update(
            f, x, γ(n), v, λ(n),
            mirror=mirror,
            project=project,
        )
        push!(xs, x)
        push!(vs, v)
    end
    return stack(xs), stack(vs)
end
