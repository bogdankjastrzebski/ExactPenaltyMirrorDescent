
function first_objective(x)
    r, θ = x[1], x[2]
    return (3 + sin(5θ) + cos(3θ)) * r^2 * (5/3) * r
end

function first_objective_random(x, σ=0.1)
    return first_objective(x) + σ * randn()
end

function first_projection(x)
    r, θ = x[1], x[2]
    return [
        clamp(r, 0.0, 1.0),
        θ,
        # clamp(θ, 0.0, 2π), # ?
    ]
end

function second_objective(x)
    return prod(x.^2)
end

function second_objective_random(x, σ=0.1)
    return second_objective(x) + σ * randn()
end

function second_projection(x)
    return clamp.(x, -1.0, 1.0)
end


function rosenbrock(x; a=1, b=100)
    # return sum(rb(i, x, a=a, b=b) for i in 1:length(x)-1)
    return sum(
        b * (x[2:end] - x[1:end-1].^2).^2
        + (a .- x[1:end-1]).^2
    )
end


function rosenbrock_random(x; a=1, b=100, n=1)
    return sum(
        b * (x[i+1] + x[i]^2)^2 + (a - x[i])^2
        for i in rand(1:length(x)-1, n)
    )
end


function rosenbrock_projection(x)
    """rosenbrock_projection
    ||x|| <= 1
    """
    return x / sqrt(x'x)
end


function rastrigin(x, a=10)
    """
    argmin = (0, 0)
    min = 0
    """
    return a * length(x) + sum(x.^2 - a * cos.(2π * x))
end


function ackley(x)
    """
    argmin = (0, 0)
    min = 0
    """
    return (
       - 20 * exp(-0.2sqrt(0.5*x'x))
       - exp(0.5sum(cos.(2π*x)))
       + exp(1) + 20
    )
end


"""
argmin = (0, 0)
min = 0
"""
function sphere(x)
    return x'x
end


function beale(x)
    h(x, y) = (
          (1.50  - x + x * y).^2
        + (2.25  - x + x * y ^ 2).^2
        + (2.625 - x + x * y ^ 3).^2
    )
    return h(x[1], x[2])
end


function goldstein_price(x)
    h(x, y) = (
        1 + (x + y + 1)^2
          * (19 - 14x + 3x^2 - 14y + 6x*y + 3y^2)
    ) * (
        30 + (2x - 3y)^2
           * (18 - 32x + 12x^2 + 48y - 36x*y + 27y^2)
    )
    return h(x[1], x[2])
end


function booth(x)
    h(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2
    return h(x[1], x[2])
end


function bukin(x)
    h(x, y) = 100sqrt(abs(y - 0.01x^2)) + 0.01abs(x + 10)
    return h(x[1], x[2])
end


function matyas(x)
    h(x, y) = 0.26 * (x^2 + y^2) - 0.48x*y
    return h(x[1], x[2])
end


function himmelblau(x)
    h(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2 
    return h(x[1], x[2])
end


function three_hump_camel(x)
    h(x, y) = 2x^2 - 1.05x^4 + x^6 / 6 + x*y + y^2
    return h(x[1], x[2])
end


function easom(x)
    """f(π, π) = -1
    """
    h(x, y) = -cos(x)*cos(y)*exp(- (x - π)^2 - (y - π)^2)
    return h(x[1], x[2])
end


function mccormick(x)
    """
    f(-0.54719, -1.54719) = -1.9133
    """
    h(x, y) = sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1
    return h(x[1], x[2])
end


function styblinski_tang(x)
    return 0.5sum(x.^4 - 16x.^2 + 5x)
end



function binary_embedding(x, L)
    return sum((L'x) .* x)
end


function binary_embedding_penalty(x)
    return -x'x
end


function project(x, y, ϵ=1e-4)
    y = y / max(norm(y), ϵ)
    return (x'y) * y
end


"""
projection into intersection of:
A: to [-1 1]
B: to be orthogonal
"""
function binary_embedding_projection(x, n=10, ϵ=1e-4)
    # Fit to Cube
    @inline function pa(v)
        clamp.(v, -one(eltype(v)), one(eltype(v)))
    end
    # Orthogonalize
    """
    Remember, x has to orthogonal!
    """
    @inline function pb(v, x)
        d = max.(sum(x -> x^2, x, dims=1), ϵ)
        return v - sum((v'x ./ d) .* x, dims=2)
    end
    # Alphabet
    ret = zero(x)
    p = zeros(eltype(x), size(x, 1))
    q = zeros(eltype(x), size(x, 1))
    v = zeros(eltype(x), size(x, 1))
    y = zeros(eltype(x), size(x, 1))
    vp = zeros(eltype(x), size(x, 1))
    yq = zeros(eltype(x), size(x, 1))
    for i in 1:size(x, 2)
        p .= 0
        q .= 0
        v .= x[:, i]
        for _ in 1:n
            @. vp = v + p
            y .= pb(vp, view(ret, :, 1:i-1))
            @. p = vp - y
            @. yq = y + q
            v .= pa(yq)
            @. q = yq - v
        end
        ret[:, i] .= v
    end
    return ret
end
