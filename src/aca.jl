
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
  if iszero(ix)
    @warn "Could not effectively select a pivot, defaulting to next available index." maxlog=1
    for t in (used_ixs[1]+1):length(rbuf)
      in(t, used_ixs) && continue
      (val, ix) = (rbuf[t], t)
      break
    end
  end
  (val, ix)
end

row!(buf, M, j) = (buf .= view(M, j, :))
col!(buf, M, k) = (buf .= view(M, :, k))

function initialize_row!(rbuf, M)
  rowj = 1
  row!(rbuf, M, rowj)
  any(isnan, rbuf) && throw(error("NaN found in row $rowj, aborting..."))
  while iszero(maximum(abs, rbuf)) && rowj < size(U,1)
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
`(rank, z, err) = aca!(M, U, V, tol=0.0; [kwargs...])`: given an object `M` that implements `ACAFact.{row!, col!}` (see also the docstring for `ACAFact.aca`), compute an ACA factorization `M ≈ U*V'` by in-place modifying `U` and `V`. If tol>0.0, this population may terminate early, _and the unused columns of `U` and `V` are NOT zeroed out_, so you should be sure to look at the `rank` output and zero out the `(rank+1):end` columns of `U` and `V`. The `rank` output gives you the number of columns used (which may be less than `size(U,2)` if tol > 0`), and the `z` output is an internal quantity that you only need to keep if you intend to resume the factorization later.

This function is designed to be non-allocating and resumable. In particular, you can allocate bigger `U` and `V` than you need to achieve a factorization at tolerance `tol1`, and then with careful use of returned values resume the computation to factorize at a higher precision `tol2`. See the README and tests for examples.

WARNING: this function does _not_ check compatibility of array dimensions. So if your `U` and `V` are not the right size, you might waste time waiting for that error to occur or simply get incorrect answers.
"""
function aca!(M, U, V, tol=0.0; start=1, z=0.0,
              cache=ACACache(eltype(U), size(U,1), size(V,1), size(V,2)))
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
      (maxval, rowj) = next_pivot(cbuf, view(rix, 1:l))
      row!(rbuf, M, rowj)
    end
    any(isnan, rbuf) && throw(error("NaN found in row $rowj, aborting..."))
    rix[l] = rowj
    # adjustment:
    for t in 1:(l-1)
      axpy!(-U[rowj,t], view(V, :, t), rbuf)
    end
    # get the next column index and put that buffer in place:
    (maxval, colk) = next_pivot(rbuf, view(cix, 1:l))
    iszero(maxval) && break
    cix[l] = colk
    col!(cbuf, M, colk)
    any(isnan, cbuf) && throw(error("NaN found in column $colk, aborting..."))
    # adjustment:
    rbuf ./= maxval
    for t in 1:(l-1)
      axpy!(-V[colk,t], view(U, :, t), cbuf)
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
      eltype(M) <: Complex && conj!(V)
      return (l, z, err)
    end
  end
  # if M is complex, then conj! the V:
  eltype(M) <: Complex && conj!(V)
  (size(U,2), z, err)
end

"""
`aca(M, rank::Int64; [tol=0.0, sz=size(M)])`: takes any object representing a matrix that implements `ACAFact.{row!, col!}` and produces matrices `U` and `V` such that `aM ≈ U*V`'.

- `rank`: the maximum allowed rank of the approximation

- `tol`:  the tolerance for the stopping criterion of || M - U*V' || < tol

- `sz`:   a `Tuple{Int64, Int64}` providing the size of M (which may not have a method for size(x::typeof{M}))

The object `M` must implement `ACAFact.{row!, col!}`, which both look like

```
ACAFact.row!(buf, M, j)
```

and populate `buf` with the row (or column) of index `j`. Feel free to write your own methods for your special object.
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

