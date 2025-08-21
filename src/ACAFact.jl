
module ACAFact

  using LinearAlgebra

  export aca, aca!, aca_psvd, kernelmatrix

  include("extmatrix.jl")

  include("aca.jl")

  include("kernelmatrix.jl")

end

