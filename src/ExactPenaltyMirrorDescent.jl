module ExactPenaltyMirrorDescent

export  mirror_descent,
        mirror_descent_update,
        Mirror,
        identity_mirror,
        rosenbrock_oracle


include("mirror_descent.jl")
include("objectives/Objectives.jl")

end # module ExactPenaltyMirrorDescent
