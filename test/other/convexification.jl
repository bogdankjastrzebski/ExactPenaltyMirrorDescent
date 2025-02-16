using LinearAlgebra
using Zygote
using Plots



function maximize(f, x; α=0.1, n=100, Δ=0.1)
    for _ in 1:n
        x = x + α * (f(x+Δ) - f(x)) / Δ  
    end
    return x
end

function dual(f; α=0.1, n=100)
    function temp(y)
        g(x) = y * x - f(x)
        g(maximize(g, randn(), α=α, n=n))
    end
    return temp
end


f(x) = ((x-1)^2 * (x+1)^2 + 1)^(0.3) - 0.2x

plot(f, dpi=300); savefig("img/test.png")

plot(x -> -1 * x - f(x), dpi=300); savefig("img/test.png")

plot(dual(f, n=100)); savefig("img/test.png")

plot(f, dpi=300);
plot!(x -> sum([max(dual(dual(f))(x), 0) for _ in 1:15])/15);
savefig("img/test.png")

plot(-5:0.1:5, y -> maximize(x -> -dual(dual(f))(x), y, n=100, Δ=0.1)); savefig("img/test.png")

maximize(x -> -dual(dual(f))(x), 1.0, n=100)


contour(
    -3:0.1:3,
    -3:0.1:3,
    (x, y) -> log(x^2 + x^4 + 1 + abs(y) + y^2),
    dpi=300,
); savefig("img/test.png")
