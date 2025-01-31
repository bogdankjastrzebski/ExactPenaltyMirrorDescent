
function first_objective(x)
    r, θ = x[1], x[2]
    return (3 + sin(5θ) + cos(3θ)) * r^2 * (5/3) * r
end

function random_second_objective(x, σ=0.1)
    return first_objective(x) + σ * randn()
end

function first_projection(x)
    r, θ = x[1], x[2]
    return [
        clamp(r, 0.0, 1.0),
        clamp(r, 0.0, 2π),
    ]
end


function second_objective(x)
    return prod(x.^2)
end

function random_second_objective(x, σ=0.1)
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

function rosenbrock_ith(i, x; a=1, b=100) 
    return b * (x[i+1] + x[i]^2)^2 + (a - x[i])^2
end

function random_rosenbrock(x; a=1, b=100, n=1)
    return sum(
        rosenbrock_ith(rand(1:length(x)-1), x, a=a, b=b)
        for _ in 1:n
    )
end

