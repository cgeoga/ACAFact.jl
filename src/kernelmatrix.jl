
struct KernelMatrix{T,P,F}
  x1::Vector{P}
  x2::Vector{P}
  kernel::F
end

function kernelmatrix(x1::AbstractVector{P}, x2::AbstractVector{P}, fn::F) where{P,F}
  T = typeof(fn(x1[1], x2[1]))
  KernelMatrix{T,P,F}(x1, x2, fn)
end

function Base.size(km::KernelMatrix{T,P,F})::Tuple{Int64, Int64} where{T,P,F} 
  (length(km.x1), length(km.x2))
end
Base.size(km::KernelMatrix{T,P,F}, j) where{T,P,F} = size(km)[j]
Base.eltype(km::KernelMatrix{T,P,F}) where{T,P,F} = T

LinearAlgebra.issymmetric(km::KernelMatrix) = (km.x1 == km.x2)

function ACAFact.col!(buf::AbstractVector{T}, km::KernelMatrix{T,P,F}, j::Int) where{T,P,F}
  length(buf) == size(km, 1) || throw(error("Input buffer not the right size!"))
  for k in eachindex(buf)
    kval::T = km.kernel(km.x1[k], km.x2[j])
    @inbounds buf[k] = kval
  end
  nothing
end

function ACAFact.row!(buf::AbstractVector{T}, km::KernelMatrix{T,P,F}, j::Int) where{T,P,F}
  length(buf) == size(km, 2) || throw(error("Input buffer not the right size!"))
  for k in eachindex(buf)
    kval::T = km.kernel(km.x1[j], km.x2[k])
    @inbounds buf[k] = kval
  end
  nothing
end

# (This is a debug that will obviously be slow)
function LinearAlgebra.Matrix(km::KernelMatrix)
  [km.kernel(x1, x2) for x1 in km.x1, x2 in km.x2]
end

