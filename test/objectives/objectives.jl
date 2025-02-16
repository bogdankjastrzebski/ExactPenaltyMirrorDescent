
objectives = Dict()

# First
function first_objective(x)
    r, θ = x[1], x[2]
    return (3 + sin(5θ) + cos(3θ)) * r^2 * (5/3) * r
end

function first_objective_projection(x)
    r, θ = x[1], x[2]
    return [
        clamp(r, 0.0, 1.0),
        θ,
        # clamp(θ, 0.0, 2π), # ?
    ]
end

objectives["first"] = (
    first_objective,
    first_objective_projection,
)


# Second
function second_objective(x)
    return prod(x.^2)
end

function second_objective_projection(x)
    return clamp.(x, -1.0, 1.0)
end

objectives["second"] = (
    second_objective,
    second_objective_projection,
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
    rastrigin_projection,
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
    ackley_projection,
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
    sphere_projection,
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

objectives["beale"] = (
    beale,
    beale_projection,
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

objectives["goldstein_price"] = (
    goldstein_price,
    goldstein_price_projection,
)


# Booth
function booth(x)
    h(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2
    return h(x[1], x[2])
end

objectives["booth"] = (
    booth,
    booth_projection,
)


function bukin(x)
    h(x, y) = 100sqrt(abs(y - 0.01x^2)) + 0.01abs(x + 10)
    return h(x[1], x[2])
end

objectives["bukin"] = (
    bukin,
    bukin_projection,
)


function matyas(x)
    h(x, y) = 0.26 * (x^2 + y^2) - 0.48x*y
    return h(x[1], x[2])
end

objectives["matyas"] = (
    matyas,
    matyas_projection,
)

function himmelblau(x)
    h(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2 
    return h(x[1], x[2])
end
objectives["himmelblau"] = (
    himmelblau,
    himmelblau_projection,
)


function three_hump_camel(x)
    h(x, y) = 2x^2 - 1.05x^4 + x^6 / 6 + x*y + y^2
    return h(x[1], x[2])
end
objectives["three_hump_camel"] = (
    three_hump_camel,
    three_hump_camel_projection,
)


function easom(x)
    """f(π, π) = -1
    """
    h(x, y) = -cos(x)*cos(y)*exp(- (x - π)^2 - (y - π)^2)
    return h(x[1], x[2])
end
objectives["easom"] = (
    easom,
    easom_projection,
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
    mccormick_projection,
)


function styblinski_tang(x)
    return 0.5sum(x.^4 - 16x.^2 + 5x)
end

objectives["styblinski_tang"] = (
    styblinski_tang,
    styblinski_tang_projection,
)
