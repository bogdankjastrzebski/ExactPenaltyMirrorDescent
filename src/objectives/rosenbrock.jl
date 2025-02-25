# Rosenbrock
function rosenbrock(x; a=1, b=100)
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


function rosenbrock_projection(x, k=1.0)
    norm = sqrt(x'x)
    # return norm > k ? x/norm : x
    return x
end


function rosenbrock_through(x)
    # return x'x * x
    return x
end


function rosenbrock_back(x)
    # return (x'x)^(-1/3) * x
    return x
end


function rosenbrock_penalty(x)
    return 0.0 * sum(x)
end


