
objectives = Dict()

function atan2(y, x)
    if x > 0
        return atan(y / x)
    elseif x < 0 && y >= 0
        return atan(y / x) + π
    elseif x < 0 && y < 0
        return atan(y / x) - π 
    elseif x == 0 && y > 0
        return π / 2
    elseif x == 0 && y < 0
        return -π / 2
    else # x == 0 && y == 0
        return 0.0
    end
end

# First
function first_objective(x)
    r = sqrt(x[1]^2 + x[2]^2)
    θ = atan2(x[2], x[1])
    # r, θ = x[1], x[2]
    return (3 + sin(5θ) + cos(3θ)) * r^2 * (5/3 - r)
end

function first_objective_projection(x)
    r, θ = x[1], x[2]
    return [
        clamp(r, 0.0, 1.0),
        θ,
        # clamp(θ, 0.0, 2π), # ?
    ]
end

objectives["first_objective"] = (
    first_objective,
    first_objective_projection,
    Nothing,
    Nothing, # mirror
    ((-1, 1), (-1, 1), (0, 5)),
    [1., 1.],
    (45, 45), # camera
)


# Second
function second_objective(x)
    return prod(x.^2)
end

function second_objective_projection(x)
    return clamp.(x, -10.0, 10.0)
end

function second_objective_penalty(x)
    # return abs(x[1] + x[2] - 5)
    # return abs(x[1] * x[2] - 5)
    return abs(x[1] - x[2])
end


function second_objective_through(x)
    return (x'x)^(-1/3) * x
    # return cbrt.([x[1]^2/x[2], x[2]^2/x[1]])
end

function second_objective_back(x)
    return x'x * x
    # return [x[2]*x[1]^2, x[1]*x[2]^2]
end

objectives["second_objective"] = (
    second_objective,
    second_objective_projection,
    second_objective_penalty,
    (second_objective_through, second_objective_back),
    ((-12, 12), (-12, 12), (0, 50000)),
    [5., 10.],
    (45, 45), # camera
)


# Rastrigin
"""
argmin = (0, 0)
min = 0
"""
function rastrigin(x, a=10)
    return a * length(x) + sum(x.^2 - a * cos.(2π * x))
end

function rastrigin_projection(x)
    return    
end

objectives["rastrigin"] = (
    rastrigin,
    Nothing, # projection,
    Nothing, 
    Nothing, # mirror
    ((-5, 5), (-5, 5), (0, 100)),
    ones(2),
    (45, 45), # camera
#    beale_projection,
#     rastrigin_projection,
#     rastrigin_penalty,
)


# Ackley
"""
argmin = (0, 0)
min = 0
"""
function ackley(x)
   return (
       - 20 * exp(-0.2sqrt(0.5*x'x))
       - exp(0.5sum(cos.(2π*x)))
       + exp(1) + 20
    )
end
objectives["ackley"] = (
    ackley,
#    ackley_projection,
    Nothing, # projection,
    Nothing,
    Nothing, # mirror
    ((-5, 5), (-5, 5), (0, 20)),
    ones(2),
    (45, 45), # camera
#    beale_projection,
)

# Sphere
"""
argmin = (0, 0)
min = 0
"""
function sphere(x)
    return x'x
end

objectives["sphere"] = (
    sphere,
#    sphere_projection,
    Nothing, # projection,
    Nothing,
    Nothing, # mirror
    ((-5, 5), (-5, 5), (-1, 19)),
    ones(2),
    (45, 45), # camera
#    beale_projection,
)


# Beale
function beale(x)
    h(x, y) = (
          (1.50  - x + x * y).^2
        + (2.25  - x + x * y ^ 2).^2
        + (2.625 - x + x * y ^ 3).^2
    )
    return h(x[1], x[2])
end

function beale_projection(x)
    return [max(x[1], 0.0), min(x[2], 1.0)]
end

function beale_penalty(x)
    return abs(sqrt(x'x) - 3)
end

function beale_through(x)
    # return x'x * x
    return x
end

function beale_back(x)
    # return (x'x)^(-1/3) * x
    return x
end

objectives["beale"] = (
    beale,
    beale_projection,
    beale_penalty,
    (beale_through, beale_back),
    ((-5, 5), (-5, 5), (0, 1000)),
    [0.0, -2.0],
    (45, 45), # camera
)

# Goldstein
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


"""
Bound the goldstein price argument (in norm) for
numerical stability. 
"""
function goldstein_price_projection(x, A=-1.0, B=1.0)
    # return norm(x) > B ? B * x / norm(x) : x
    return clamp.(x, A, B)
end


"""goldstein_price_penalty(x, x₂=-0.5)
Exact penalty, which keeps x₂ near the
desired value.
# Args:
* x - argument in R².
* x₂ - desired value of the second. (default -0.5)
    For -0.5 the problem is two modal, so it is
    interesting, if it will be able to find the
    proper minimum.
"""
function goldstein_price_penalty(x, x₂=-0.5, k=1)
    return (abs(x[2] - x₂) + k)^2 / k^2
end


function goldstein_price_through(x)
    return x'x * x
end


function goldstein_price_back(y)
    return (y'y)^(-1/3) * y
end


objectives["goldstein_price"] = (
    goldstein_price,
    goldstein_price_projection,
    goldstein_price_penalty,
    (goldstein_price_through, goldstein_price_back),
    ((-2, 2), (-3, 1), (0, 10000)),
    # ((-1, 1), (-1, 1), (0, 10000)),
    [1., 0.],
    (45, 45), # camera
)


# Booth
function booth(x)
    h(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2
    return h(x[1], x[2])
end

objectives["booth"] = (
    booth,
    Nothing, # booth_projection,
    Nothing,
    Nothing, # mirror
    ((-12, 15), (-11, 16), (0, 1000)),
    ones(2),
    (45, 45), # camera
)


function bukin(x)
    h(x, y) = 100sqrt(abs(y - 0.01x^2)) + 0.01abs(x + 10)
    return h(x[1], x[2])
end

function bukin_projection(x, s=[5.0, 2.0])
    return clamp.(x ./ s, -1, 1) .* s
end

function bukin_penalty(x)
    return 0.0 * sum(x)
end

function bukin_through(x, s=[5.0, 2.0])
    return x ./ s
end

function bukin_back(x, s=[5.0, 2.0])
    return s .* x
end

objectives["bukin"] = (
    bukin,
    bukin_projection,
    bukin_penalty,
    (bukin_through, bukin_back),
    ((-5, 5), (-4, 6), (0, 250)),
    ones(2),
    (45, 45), # camera
)


function matyas(x)
    h(x, y) = 0.26 * (x^2 + y^2) - 0.48x*y
    return h(x[1], x[2])
end

objectives["matyas"] = (
    matyas,
    Nothing,
    Nothing,
    Nothing, # mirror
    ((-10, 10), (-10, 10), (0, 10)),
    ones(2),
    (45, 45), # camera
)

function himmelblau(x)
    h(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2 
    return h(x[1], x[2])
end

objectives["himmelblau"] = (
    himmelblau,
    Nothing,#    himmelblau_projection,
    Nothing,
    Nothing,
    ((-5, 5), (-5, 5), (0, 500)),
    ones(2),
    (45, 45), # camera
)


function three_hump_camel(x)
    h(x, y) = 2x^2 - 1.05x^4 + x^6 / 6 + x*y + y^2
    return h(x[1], x[2])
end
objectives["three_hump_camel"] = (
    three_hump_camel,
    Nothing,#    himmelblau_projection,
    Nothing,
    Nothing,
    ((-3, 3), (-3, 3), (0, 50)),
    ones(2),
    (45, 45), # camera
)


function easom(x)
    """f(π, π) = -1
    """
    h(x, y) = -cos(x)*cos(y)*exp(- (x - π)^2 - (y - π)^2)
    return h(x[1], x[2])
end
objectives["easom"] = (
    easom,
    Nothing, # projection,
    Nothing, # Penalty
    Nothing,
    ((1, 5), (1, 5), (-1, 0.1)),
    [2, 2], # x₀
    (45, 45),
)

function mccormick(x)
    """
    f(-0.54719, -1.54719) = -1.9133
    """
    h(x, y) = sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1
    return h(x[1], x[2])
end

objectives["mccormick"] = (
    mccormick,
    Nothing, # projection,
    Nothing, # Penalty
    Nothing,
    ((-3, 4), (-3, 4), (-2, 70)),
    [2, 2], # x₀
    (45, 45),
#     mccormick_projection,
)


function styblinski_tang(x)
    return 0.5sum(x.^4 - 16x.^2 + 5x)
end


objectives["styblinski_tang"] = (
    styblinski_tang,
    Nothing, # easom_projection,
    Nothing, # Penalty
    Nothing,
    ((-5, 5), (-5, 5), (-100, 250)),
    [2, 2], # x₀
    (45, 45),
#    styblinski_tang_projection,
)
