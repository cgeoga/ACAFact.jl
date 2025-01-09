
using ACAFact

gauss(x, y, r) = exp(-(norm(x-y)^2)/r)

function aca_fgt(sources::Vector{T}, target_locations::Vector{T}, 
                 values; range=1.0, maxrank=100, tol=eps()) where{T}
  fgt = ACAFact.kernelmatrix(sources, target_locations, (x,y)->gauss(x, y, range))
  (U, V, err) = aca(fgt, tol)
  U*(V'*values)
end


# Quick test:

using BenchmarkTools

function slowgausstransform(sources::Vector{T}, target_locations::Vector{T}, 
                            values; range=1.0) where{T}
  [exp(-norm(x-y)^2/range) for x in sources, y in target_locations]*values
end

xv = sort(rand(5000))
yv = sort(rand(6000))
va = randn(length(yv))

# Timings on my machine (intel 12th gen laptop with power saving CPU governor)
@btime aca_fgt($xv, $yv, $va);            # 2 ms,   17 alloc (10 MiB)
@btime slowgausstransform($xv, $yv, $va); # 265 ms, 4 alloc, (229 MiB)

@show maximum(abs, aca_fgt(xv, yv, va) - slowgausstransform(xv, yv, va)) # 6e-13

