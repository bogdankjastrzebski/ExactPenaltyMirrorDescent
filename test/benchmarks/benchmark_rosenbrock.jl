using BenchmarkTools
using Zygote

function bench(n=100, N=1000)
    function rosenbrock_random_1(x; a=1, b=100, n=1)
        return sum(
            b * (x[i+1] + x[i]^2)^2 + (a - x[i])^2
            for i in rand(1:length(x)-1, n)
        )
    end
    function rosenbrock_random_2(x; a=1, b=100, n=1)
        i = rand(1:length(x)-1, n)
        return sum(b * (x[i .+ 1] + x[i] .^ 2).^2 + (a .- x[i]) .^2)
    end
    r = randn(N)
    v1(r) = rosenbrock_random_1(r, n=n)
    v2(r) = rosenbrock_random_2(r, n=n)
    g1 = v1'
    g2 = v2'
    v1 = @benchmark $v1($r)
    v2 = @benchmark $v2($r)
    z1 = @benchmark $g1($r)
    z2 = @benchmark $g2($r)
    return (v1, z1, v2, z2)
end

