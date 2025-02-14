using Plots

contour(
    -5:0.1:5,
    -5:0.1:5,
    (x, y) -> ackley([x, y]),
    aspect_ratio=:equal,
    dpi=300,
    fill=:true,
    color=:turbo,
)
savefig("img/ackley_contour.png")


contour(
    -3:0.01:3,
    -3:0.01:3,
    (x, y) -> log(beale([x,y]) + 1),
    dpi=300,
    levels=16,
    fill=true,
    aspect_ratio=:equal,
    # color=:turbo,
    color=cgrad(:turbo, scale=x->log(x + 1))
); savefig("img/beale_contour.png")


contour(
    -2:0.01:2,
    -3:0.01:1,
    (x, y) -> log(goldstein_price([x,y]) + 1),
    dpi=300,
    levels=16,
    fill=true,
    aspect_ratio=:equal,
    # color=:turbo,
    color=cgrad(:turbo, scale=x->log(x + 1))
); savefig("img/goldstein_price_contour.png")

contour(
    -10:0.01:10,
    -10:0.01:10,
    (x, y) -> log(booth([x,y]) + 1),
    dpi=300,
    levels=16,
    fill=true,
    aspect_ratio=:equal,
    # color=:turbo,
    color=cgrad(:turbo, scale=x->log(x + 1))
); savefig("img/booth.png")


contour(
    -15:0.01:5,
    -4:0.01:6,
    (x, y) -> log(bukin([x,y]) + 100),
    dpi=300,
    levels=10,
    fill=true,
    aspect_ratio=:equal,
    # color=:turbo,
    color=cgrad(:turbo)
); savefig("img/bukin.png")


contour(
    -10:0.01:10,
    -10:0.01:10,
    (x, y) -> matyas([x, y]) - 0.05(x^2 + y^2) ,
    dpi=300,
    levels=10,
    fill=true,
    aspect_ratio=:equal,
    # color=:turbo,
    color=cgrad(:turbo)
); savefig("img/matyas.png")


W = rand(2,2); Q = W'W

plot(
    -10:0.01:10,
    -10:0.01:10,
    (x, y) -> [x, y]'Q*[x, y] - 1.0(x^2 + y^2),
    dpi=300,
    st=:surface,
    # color=:turbo,
    color=cgrad(:turbo)
); savefig("img/matyas.png")
