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
One easy way to use `ACAFact.jl` is to just provide your matrix and a maximum
allowed rank for the approximation:
```julia
for rk in (5, 10, 15, 20, 25)
  (U, V) = aca(K, rk)
  @show (rk, opnorm(K - U*V'))
end
```

Another thing you can do is provide a desired `opnorm` error for `||K -
U*V'||`...but I wouldn't bet the farm that the error control is all that
guaranteed. Greedy deterministic pivoting schemes can be tricked, or at least
made very inefficient! It does work pretty well in toy applications at least
though, so it isn't useless.
```julia
for tol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
  (U, V) = aca(K, 100, tol=tol)
  @show tol, size(U, 2), opnorm(K - U*V')
end
```

If you want something that is completely non-allocating, you can pre-allocate a
cache. Note that if you also provide a `tol` here, you will terminate early but
the function won't remove the unused columns in `U` and `V`. The `rnk` return
parameter gives you the rank of the approximation, so you should work with
`view(U, :, 1:rnk)`, for example.
```julia
cache    = ACAFact.ACACache(Float64, length(pts1), length(pts2), 50) # max rank 50
(U, V)   = (zeros(length(pts1), 50), zeros(length(pts2), 50))
(rnk, _) = ACAFact.aca!(K, U, V, cache=cache) # zero allocations
```
One note here though: this `ACACache` does carry state that gets used in the
factorizations, so if you want to factorize a new matrix that is the same size
as `K` with the same cache, be sure to `ACAFact.resetcache!(cache)`.

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
  ACAFact.col!(buf, K::MyCoolObject, j) = # ...
  ACAFact.row!(buf, K::MyCoolObject, j) = # ...
```
and you can create low-rank approximations that never touch rows/columns of your
matrix that aren't selected as pivots. It would be nice if you could also
implement `Base.size(K::MyCoolObject)`, but if for some reason you don't want to
you can compute the ACA of `K` with `aca(K, [...], sz=(size_K_1, size_K_2))`.

