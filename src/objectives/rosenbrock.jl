# Rosenbrock
function rosenbrock(x; a=1, b=100)
    return sum(
        b * (x[2:end] - x[1:end-1].^2).^2
        + (a .- x[1:end-1]).^2
    )
end


function rosenbrock_random(x; a=1, b=100, n=1)
    return sum(
        b * (x[i+1] - x[i]^2)^2 + (a - x[i])^2
        for i in rand(1:size(x,1)-1, n)
    )
end


function rosenbrock_oracle(x; a=1, b=100, n=1)
    ret = zero(x)
    i = rand(1:size(x, 1)-1, n)
    alpha =  b * 2 * (x[i .+ 1] - x[i]^2)
    x[i .+ 1] .+= alpha
    x[i] .+= -alpha * 2 * x[i] + 2 * (x[i] - a)
    return ret
end


function rosenbrock_projection(x, k=2.0)
    k = k * sqrt(size(x, 1))
    norm = sqrt(x'x)
    return norm > k ? k * x / norm : x
    # return x
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
    norm = sqrt(x'x)
    return abs(norm - sqrt(size(x, 1)))
end


