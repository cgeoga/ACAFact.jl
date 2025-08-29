# ACAFact.jl

A simple and dependency-free [adaptive cross approximation (ACA)](https://link.springer.com/article/10.1007/s00607-002-1469-6) 
factorization in Julia. 

# Usage

Let's create a matrix with low numerical rank:
```julia
using LinearAlgebra
pts1 = range(0.0, 1.0, length=100)
pts2 = range(0.0, 1.0, length=110)
K    = [exp(-abs2(pj - pk)) for pj in pts1, pk in pts2]
@show rank(K) # 10
```

This package has two primary convenience methods:
```julia
aca(K, rank::Int64)
aca(K, tol::Float64; maxrank::Int64, rankstart::Int64=20)
```
As you might expect, `(U, V, _) = aca(K, rank::Int64)` gives you an exactly rank
`rank` approximation `U*V'` for `K`, with `size(U, 2) = rank`. The second option
is _adaptive_, and will add or remove columns in `U` and `V` to give you
something which (ideally) achieves `||U*V' - K|| < tol`. This is _not_
guaranteed and this factorization can be tricked! Greedy partial pivoting
approximations are not infallible. But for low rank matrices that come from
smooth kernels and a few other geometric hypotheses, there is some level of
theory to trust. Even in more general cases, though, I have found this routine
to work. This is all to say: hopefully this will work for you! But I would
recommend sanity checking this factorization or the downstream products of using
it before dropping this in to a critical workflow and walking away.

Here is a demonstration of the rank-based method:
```julia
for rk in (5, 10, 15, 20, 25)
  (U, V) = aca(K, rk)
  @show (rk, opnorm(K - U*V'))
end
```

And the analog adaptive tolerance-based method:
```julia
for tol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
  (U, V) = aca(K, tol; maxrank=100)
  @show (tol, size(U, 2), opnorm(K - U*V'))
end
```


# Lower-level allocation-free methods

If you want something that is completely non-allocating, you can pre-allocate an
`ACACache`. Note that if you also provide a `tol` here, you will terminate early
but the function won't remove the unused columns in `U` and `V`. The `rnk`
return parameter gives you the rank of the approximation, so you should work
with `view(U, :, 1:rnk)`, for example. Here is a demonstration:
```julia
cache       = ACAFact.ACACache(Float64, length(pts1), length(pts2), 50) # max rank 50
(U, V)      = (zeros(length(pts1), 50), zeros(length(pts2), 50))
(rnk, _, _) = ACAFact.aca!(K, U, V, cache=cache) # zero allocations
```
**Note**: this `ACACache` does carry state that gets used in the factorizations,
so if you want to factorize a new matrix that is the same size as `K` with the
same cache, be sure to `ACAFact.resetcache!(cache)`.

# Working with arbitrary matrix-like objects

To me, the best reason to use a greedy partially-pivoted factorization like the
ACA is to build low-rank approximations for matrices where you cannot afford to
pass over every entry even once. To accommodate this use case, `ACAFact.jl` has
an interface
```julia
  ACAFact.col!(buf, K, j)
  ACAFact.row!(buf, K, j)
```
that expects `buf` to be filled with the corresponding column/row of `K`. So if
you have some cool operator that is defined implicitly or whatever, all you need
to do add special methods 
```julia
  ACAFact.col!(buf, K::MyCoolObject, j) = [ ... ]
  ACAFact.row!(buf, K::MyCoolObject, j) = [ ... ]
```
and you can create low-rank approximations that never touch rows/columns of your
matrix that aren't selected as pivots. Your object needs the following
additional methods:
```julia
Base.eltype(K::MYCoolObject)
Base.size(K::MyCoolObject)
```

As an example, `ACAFact.jl` has a non-exported demo struct called a
`KernelMatrix`. You can look at `./src/kernelmatrix.jl` for a full demonstration
of creating this abstract interface. The analog to the first usage demo with a
`ACAFact.KernelMatrix` would look like this:
```julia
pts1 = range(0.0, 1.0, length=100)
pts2 = range(0.0, 1.0, length=110)
K      = ACAFact.kernelmatrix(pts1, pts2, (x,y)->exp(-abs2(x-y)))
(U, V) = aca(K, 1e-12, maxrank=100)
```
As you can see, nothing really changes...except for the fact that you never have
to fully assemble a potentially huge matrix. If you knew the rank of your matrix
was O(1), for example, that would change the runtime of your code from O(n^2) to
O(n) basically for free. Not bad!

# Cheap conversions to partial factorizations

This package now also offers simple extension functions `aca_psvd` and `aca_pqr`
to convert the obtained approximation `K \approx U*V'` into truncated low-rank
factorizations:
```julia
(U, S, Vt) = aca_psvd(K, 1e-12, maxrank=100)
(Q, R)     = aca_pqr(K,  1e-12, maxrank=100)
```
Both of these methods simply compute an ACA and then do manipulations on the
small matrices to give the more standard-form factorizations.

