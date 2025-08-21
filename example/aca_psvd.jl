
using Printf, LinearAlgebra, ACAFact, BenchmarkTools

# kernel function for testing:
fn(x, y)  = sinc(2*100*(x-y))

# make the KernelMatrix object:
pts = sort(rand(2_000))
km  = kernelmatrix(pts, pts, fn)

# compute the ACA-accelerated PSVD:
psvd = aca_psvd(km, 1e-14; maxrank=250)
@show opnorm(Matrix(km) - Matrix(psvd))/opnorm(Matrix(km)) # ≈ [...]e-14 or [small]e-13.

# compare with the exact svd:
kmsvd = svd(Matrix(km))
for j in vcat(collect(1:50:length(psvd.S)), (length(psvd.S)-3:length(psvd.S)))
  @printf "Exact  σ(%3i): %1.15e\n"   j kmsvd.S[j]
  @printf "Approx σ(%3i): %1.15e\n\n" j psvd.S[j]
end

