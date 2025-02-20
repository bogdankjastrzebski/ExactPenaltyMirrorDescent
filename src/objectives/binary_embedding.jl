# BINARY EMBEDDING
function binary_embedding(x, L)
    return sum((L'x).* x) / size(x, 1)
end


function binary_embedding_penalty(x)
    return -x'x
end


@inline function project(x, y, ϵ=1e-4)
    return (x'y / max(sum(x->x^2, y), ϵ)) * y
end

@inline function pb(v, x)
    for j in 1:iter_orthogonal
        for i in 1:size(x, 2)
            v .-= project(v, x[:, i])
        end
    end
    return v
end


"""orthogonalize(v, x, orthogonal=true)
# Args:
* v - vector to be projected.
* x - basis, non orthogonal.
# Returns:
* v orthogonalized to x.
"""
@inline function orthogonalize(v, x; orthogonal=false)
    M = orthogonal ?
        Diagonal(sum(x->x^2, x, dims=1)[:]) : (x'x)
    return v - x * (M \ (x'v))
end


# Fit to Cube
"""project_cube(v)
Projects vector v into cube [-1 1]^n
# Args:
* v -- array.
# Returns:
* projected vector.
"""
@inline function project_cube(v)
    clamp.(v, -one(eltype(v)), one(eltype(v)))
end


"""
Args:
* v - vector to be projected
* x - system of orthogonal directions. In case x is not orthogonal, use iter_orthogonal > 1.
# Keyword Arguments
* iter -- number of interations
* iter_orthogonal -- iterations for
    orthogonal projection into subspace.
"""
function binort_projection(
        v, x;
        iter=10,
        orthogonal=false,
        ϵ=1e-4)
    # Project
    v = copy(v)
    y = zeros(eltype(v), size(v, 1))
    p = zeros(eltype(v), size(v, 1))
    q = zeros(eltype(v), size(v, 1))
    vp = zeros(eltype(v), size(v, 1))
    yq = zeros(eltype(v), size(v, 1))
    for _ in 1:iter
        @. vp = v + p
        y .= orthogonalize(vp, x, orthogonal=orthogonal)
        @. p = vp - y
        @. yq = y + q
        v .= project_cube(yq)
        @. q = yq - v
    end
    return v
end


# Test 1
function test_binort_projection(n=1024, m=10)
    x = rand([-1., 1.], n, m)
    v = randn(n)
    vp = binort_projection(v, x)
    println(vp'v)
    println(maximum(abs, vp'x))
    println(maximum(abs, vp))
    @benchmark binort_projection($v, $x)
end

# Test regularization
using Zygote

function opt(f, v, x; α=0.01, λ=0.99, iter=100)
    # d = zero(v)
    p = zero(v)
    for _ in 1:iter
        # d = λ * d + (1 - λ) * f'(v)
        # p = x_n+1 - x_n
        p = binort_projection(
             v - α * (1 - λ) * f'(v) + λ * p,
             x
        ) - v
        v += p
    end
    return v, p
end

function test_binort_penalty(; α=0.1, iter=100, λ=0.9)
    # k = 1024
    # l = 64
    k = 4
    l = 1
    v = 0.01randn(k) # zeros(k) #
    # v = 0.01*rand([-1, 1], k)
    # On chyba nie może się zdecydować
    x = rand([-1, 1], k, l)
    x = ones(k, l)
    f(x) = -(x'x)
    vp, d = opt(f, v, x, α=α, iter=iter, λ=λ)
    println("vp'v: ", vp'v)
    println(maximum(abs, vp'x))
    println(vp'x)
    println(minimum(abs, vp))
    println(d' * f'(vp))
    vn = vp - 0.1 * f'(vp)
    vn = binort_projection(vn, x, iter=100)
    return vp, vn, x
end

vp, vn, x = test_binort_penalty(iter=100)

histogram(
    abs.(vp),
    title="Asdf",
    nbins=100,
    size=(800, 400),
);
savefig("img/histogram_binort_penalty.pdf")


# titlefontsize=30,
# tickfontsize=30,
# guidefontsize=30,
# legendfontsize=30,

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

function test_projection(k)
    for i in 1:1000
        a = rand([-1, 1], k)
        b = rand([-1, 1], k)
        c = rand([-1, 1], k)
        project(x, y) = x - x'y / norm(y)^2 * y
        aa = project(a, b)
        cc = project(c, b)
        println(norm(aa) - norm(cc))
        if abs(norm(aa) - norm(cc)) > 0.5
            return a, b, c, aa, cc
        end
    end
    return true
end




