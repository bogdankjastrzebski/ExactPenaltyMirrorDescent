module ExactPenaltyMirrorDescent

export  mirror_descent,
        mirror_descent_update,
        Mirror,
        identity_mirror,
        first_objective,
        random_second_objective,
        first_projection,
        second_objective,
        random_second_objective,
        second_projection,
        rosenbrock,
        random_rosenbrock,
        view_3d,
        view_top

include("mirror_descent.jl")
include("objectives.jl")
include("plots.jl")

end # module ExactPenaltyMirrorDescent
