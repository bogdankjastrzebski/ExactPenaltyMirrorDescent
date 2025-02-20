
module Objectives

include("elementary.jl")
# include("rosenbrock.jl")
# include("binary_embedding")

export objectives,
       first_objective,
       second_objective,
       booth
       easom,
       sphere,
       mccormick,
       bukin,
       rastrigin,
       beale,
       matyas,
       himmelblau,
       ackley,
       goldstein_price,
       three_hump_camel,
       styblinski_tang

end
