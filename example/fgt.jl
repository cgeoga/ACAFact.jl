
using ACAFact

struct FastGaussSpec{T}
  sources::Vector{T}
  target_locations::Vector{T}
  range::Float64
end

Base.size(fg::FastGaussSpec{T}) where{T} = (length(fg.sources), length(fg.target_locations))
Base.eltype(fg::FastGaussSpec{T}) where{T} = Float64

function ACAFact.col!(buf, fg::FastGaussSpec{T}, j::Int) where{T}
  x = fg.target_locations[j]
  @inbounds for k in eachindex(fg.sources)
    buf[k] = exp(-norm(x - fg.sources[k])^2/fg.range)
  end
  nothing
end

function ACAFact.row!(buf, fg::FastGaussSpec{T}, j::Int) where{T}
  x = fg.sources[j]
  @inbounds for k in eachindex(fg.target_locations)
    buf[k] = exp(-norm(x - fg.target_locations[k])^2/fg.range)
  end
  nothing
end

function aca_fgt(sources::Vector{T}, target_locations::Vector{T}, 
                 values; range=1.0, maxrank=100, tol=eps()) where{T}
  spec = FastGaussSpec(sources, target_locations, range)
  (U, V) = aca(spec, maxrank, tol=tol)
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

