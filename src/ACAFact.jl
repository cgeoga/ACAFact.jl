
module ACAFact

  using LinearAlgebra

  export aca, aca!, aca_psvd, aca_pqr, kernelmatrix

  include("extmatrix.jl")

  include("aca.jl")

  include("kernelmatrix.jl")

end

