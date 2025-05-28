
module ACAFact

  using LinearAlgebra

  export aca, aca!

  include("extmatrix.jl")

  include("aca.jl")

  include("kernelmatrix.jl")

end

