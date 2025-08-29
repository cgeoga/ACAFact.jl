
struct ACACache{T}
  rbuf::Vector{T}
  cbuf::Vector{T}
  rix::Vector{Int64}
  cix::Vector{Int64}
end

function ACACache(::Type{T}, su, sv, maxrank) where{T}
  rbuf  = Vector{T}(undef, sv)
  cbuf  = Vector{T}(undef, su)
  ixbuf = zeros(Int64, maxrank)
  ACACache(rbuf, cbuf, ixbuf, copy(ixbuf))
end

ACACache(M, maxrank) = ACACache(eltype(M), size(M,1), size(M,2), maxrank)

function resetcache!(cache::ACACache{T}) where{T}
  fill!(cache.rbuf, NaN)
  fill!(cache.cbuf, NaN)
  fill!(cache.rix, 0)
  fill!(cache.cix, 0)
  nothing
end

function next_pivot(rbuf, used_ixs)
  (val, ix) = (zero(eltype(rbuf)), 0)
  @inbounds for j in eachindex(rbuf)
    rj = rbuf[j]
    if abs(val) < abs(rj) && !in(j, used_ixs)
      val = rj
      ix  = j
    end
  end
  iszero(ix) && @warn "Could not effectively select a pivot, defaulting to next available index." maxlog=1
  (!iszero(ix), val, ix)
end

row!(buf, M, j) = (buf .= view(M, j, :))
col!(buf, M, k) = (buf .= view(M, :, k))

function initialize_row!(rbuf, M)
  rowj = 1
  row!(rbuf, M, rowj)
  any(isnan, rbuf) && throw(error("NaN found in row $rowj, aborting..."))
  while iszero(maximum(abs, rbuf)) && rowj < size(M,1)
    rowj += 1
    row!(rbuf, M, rowj)
    any(isnan, rbuf) && throw(error("NaN found in row $rowj, aborting..."))
  end
  (maximum(abs, rbuf), rowj)
end

function check_resume(start, rix, cix, z)
  isone(start) && return
  ix_leq_checks = any(<(1), view(rix, 1:(start-1))) ||  any(<(1), view(cix, 1:(start-1)))
  if ix_leq_checks || iszero(z)
    @warn "You appear to be resuming a factorization, but your index trackers have suspicious values. Proceed with caution."
  end
  nothing
end

"""
`(rank, z, err) = aca!(M, U, V, tol=0.0; [cache=ACACache(M, size(V,2)), kwargs...])`

compute an ACA factorization `M ≈ U*V'` by in-place modifying `U` and `V`. If tol>0.0, this population may terminate early, and **the unused columns of `U` and `V` are NOT zeroed out**, so you should be sure to look at the `rank` output and zero out the `(rank+1):end` columns of `U` and `V`. The `rank` output gives you the number of columns used (which may be less than `size(U,2)` if `tol > 0`), and the `z` output is an internal quantity that you only need to keep if you intend to resume the factorization later.

This function is designed to be non-allocating and resumable: for non-allocating factorizations, you just need to pass in a pre-allocated `ACACache`. But note: **`ACACache` objects are stateful**, and when starting a new factorization (as opposed to resuming an existing one) you should `ACAFact.resetcache!(cache)`. 

For resuming, you can allocate bigger `U` and `V` than you need to achieve a factorization at tolerance `tol1`, and then with careful use of returned values resume the computation to factorize at a higher precision `tol2`. See the source for `aca(M, ::Float64; kwargs...)` and the tests for examples of resuming. 

See also: the docs for `aca` and the README outline the necessary bridge methods for you to write to pass in an arbitrary matrix-like object that gives access to individual rows and columns.
"""
function aca!(M, U, V, tol=0.0; start=1, z=0.0,
              cache=ACACache(M, size(V,2)))::Tuple{Int64, Float64, Float64}
  if eltype(M) != eltype(U) || eltype(M) != eltype(V)
    throw(error("eltype(M) does not agree with the type of provided buffers!"))
  end
  # unpack cache struct:
  (;rbuf, cbuf, rix, cix) = cache
  # imperfect check for bug avoidance:
  check_resume(start, rix, cix, z)
  # allocate the err float just so it can be returned outside the loop:
  err = 0.0
  # Now loop over the rest of the rank of U and V and fill in the rows/cols:
  for l in start:size(U,2)
    # get the next row index and put that buffer in place:
    if isone(l)
      (maxval, rowj) = initialize_row!(rbuf, M)
    else
      (success, maxval, rowj) = next_pivot(cbuf, view(rix, 1:l))
      success || return (l-1, z, err)
      row!(rbuf, M, rowj)
      conj!(rbuf)
    end
    any(isnan, rbuf) && throw(error("NaN found in row $rowj, aborting..."))
    rix[l] = rowj
    # adjustment:
    for t in 1:(l-1)
      axpy!(-conj(U[rowj,t]), view(V, :, t), rbuf)
    end
    # get the next column index and put that buffer in place:
    (success, maxval, colk) = next_pivot(rbuf, view(cix, 1:l))
    (iszero(maxval) || !success) && return (l-1, z, err)
    cix[l] = colk
    col!(cbuf, M, colk)
    any(isnan, cbuf) && throw(error("NaN found in column $colk, aborting..."))
    # adjustment:
    rbuf ./= maxval
    for t in 1:(l-1)
      axpy!(-conj(V[colk,t]), view(U, :, t), cbuf)
    end
    # add the new row and column:
    view(U, :, l) .= cbuf
    view(V, :, l) .= rbuf
    # convergence check:
    for t in 1:(l-1)
      z += 2*abs(dot(view(U, :, t), cbuf))*abs(dot(view(V, :, t), rbuf))
    end
    z  += sum(abs2, cbuf)*sum(abs2, rbuf)
    err = sqrt(sum(abs2, cbuf)*sum(abs2, rbuf))/sqrt(z)
    if sqrt(sum(abs2, cbuf)*sum(abs2, rbuf)) < tol*sqrt(z) && l > 1
      return (l, z, err)
    end
  end
  (size(U,2), z, err)
end

"""
`aca(M, rank::Int64)`

`aca(M, tol::Float64; [maxrank, rankstart])`

takes a matrix and produces matrices `U` and `V` such that `M ≈ U*V`' to either a prescribed rank (`rank::Int64`) or tolerance (`tol::Float64`).

- `rank`: the maximum allowed rank of the approximation

- `tol`:  the tolerance for the stopping criterion of || M - U*V' || < tol

- `maxrank`: when the second argument is `tol::Float64`, `maxrank=s` means that the factorization will terminate when `size(U,2)=s` regardless of whether ||M - U*V'|| < tol`.

- `rankstart`: when the second argument is `tol::Float64`, this routine works by adding chunks of columns at a time and resuming the factorization. `rankstart` dictates the initial guess for the `tol`-rank of `M`. The higher it is, the fewer pauses from heap allocations you are likely to have---but if you overshoot, you will still have to pay the price of discarding unused but pre-allocated rows. I recommend keeping this reasonably low.




You are welcome to provide an arbitrary object `M::MyObject` that is not an `::AbstractMatrix`. The object `M` must implement `ACAFact.{row!, col!}`, which both look like
```
ACAFact.{col!,row!}(buf, M::MyObject, j)
```
and populate `buf` with the row (or column) of index `j`. Additionally, it must implement
```
Base.eltype(M::MyObject)
Base.size(M::MyObject)
```
But with those methods, you can pass the object in to `aca` just as you can pass in a standard matrix.
"""
function aca(M, rank::Int64; sz=size(M))
  rank = min(rank, sz[1], sz[2])
  U    = Array{eltype(M)}(undef, sz[1], rank)
  V    = Array{eltype(M)}(undef, sz[2], rank)
  (rank, z, err) = aca!(M, U, V, 0.0)
  (U, V, err)
end

function aca(M, tol::Float64; sz=size(M), maxrank=min(sz[1], sz[2]), rankstart=20)
  rank  = min(rankstart, sz[1], sz[2])
  U     = ExtendableMatrix(eltype(M), sz[1], rank)
  V     = ExtendableMatrix(eltype(M), sz[2], rank)
  cache = ACACache(eltype(U), size(U,1), size(V,1), size(V,2))
  (rank, z, err) = aca!(M, U, V, tol, cache=cache)
  while err > tol && rank < maxrank
    added_rank = min(rank, maxrank - rank)
    add_cols!(U, added_rank)
    add_cols!(V, added_rank)
    cache_ix_len = length(cache.cix)
    resize!(cache.cix, cache_ix_len+added_rank)
    resize!(cache.rix, cache_ix_len+added_rank)
    fill!(view(cache.cix, (cache_ix_len+1):length(cache.cix)), 0)
    fill!(view(cache.rix, (cache_ix_len+1):length(cache.rix)), 0)
    (rank, z, err) = aca!(M, U, V, tol; start=rank+1, z=z, cache=cache)
  end
  if rank < size(U, 2)
    s = size(U, 2)
    delete_cols!(U, s-rank)
    delete_cols!(V, s-rank)
  end
  (Matrix(U), Matrix(V), err)
end

"""
`aca_psvd(M, tol_or_rank; kwargs...)`

Computes a partial SVD of a matrix (or an abstract object implementing the necessary methods, see the docstrings for `aca` for details) based on the ACA with given specifications.

- `M`: your matrix or matrix-like object.
- `tol_or_rank`: if `::Float64`, then this is interpreted as an `rtol`-like parameter. If `::Int64`, then it is a fixed rank. This is the same as the second argument passed to `aca`, so consult those docstrings for details.
- `kwargs...`: keyword arguments that will be provided to `aca`. See the docstrings for `aca` for details.

This function returns a standard `SVD` object, at least for now.
"""
function aca_psvd(M, tol_or_rank; kwargs...)
  (_U, _V) = aca(M, tol_or_rank; kwargs...)
  qrU      = qr!(_U)
  D        = qrU.R*_V'
  svdD     = svd!(D)
  U        = qrU.Q*svdD.U
  mul!(U, qrU.Q, svdD.U)
  ishermitian(M) && return SVD(U, svdD.S, U')
  SVD(U, svdD.S, svdD.Vt)
end

