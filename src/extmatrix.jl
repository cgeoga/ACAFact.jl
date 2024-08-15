
mutable struct ExtendableMatrix{T}
  n::Int64
  m::Int64
  buf::Vector{T}
end

ExtendableMatrix(::Type{T}, n, m) where{T} = ExtendableMatrix(n, m, Vector{T}(undef, n*m))

Base.eltype(em::ExtendableMatrix{T}) where{T} = T
Base.size(em::ExtendableMatrix) = (em.n, em.m)
Base.size(em::ExtendableMatrix, j::Int) = j <= 2 ? (em.n, em.m)[j] : 1

function Base.getindex(em::ExtendableMatrix{T}, j::Int, k::Int) where{T}
  @inbounds em.buf[(k-1)*em.n + j]
end

function add_cols!(em::ExtendableMatrix{T}, k::Int) where{T}
  resize!(em.buf, length(em.buf) + em.n*k)
  em.m += k
  nothing
end

function delete_cols!(em::ExtendableMatrix{T}, k::Int) where{T}
  resize!(em.buf, length(em.buf) - em.n*k)
  em.m -= k
  nothing
end

function Base.view(em::ExtendableMatrix{T}, ::Colon, j::Int) where{T}
  view(em.buf, (1+(j-1)*em.n):(j*em.n))
end

Base.conj!(em::ExtendableMatrix{T}) where{T} = conj!(em.buf)

Base.Matrix(em::ExtendableMatrix{T}) where{T} = reshape(em.buf, (em.n, em.m))

