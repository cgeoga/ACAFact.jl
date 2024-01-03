
using Test, LinearAlgebra, ACAFact

pts1 = range(0.0, 1.0, length=100)
pts2 = range(0.0, 1.0, length=110)
K    = [exp(-abs2(pj - pk)) for pj in pts1, pk in pts2]

@testset begin "out-of-place"
  (U1, V1) = aca(K, 50)
  @test opnorm(K - U1*V1') < 1e-14
  (U2, V2) = aca(K, 50, tol=1e-6)
  @test opnorm(K - U2*V2') < 1e-5
end

@testset begin "in-place and resume"
  (U3, V3)  = (zeros(length(pts1), 50), zeros(length(pts2), 50))
  cache     = ACAFact.ACACache(Float64, length(pts1), length(pts2), 50)
  (rank, z) = ACAFact.aca!(K, U3, V3, 1e-6, cache=cache)
  (U4, V4)  = (view(U3, :, 1:rank), view(V3, :, 1:rank))
  @test 1e-14 < opnorm(K - U4*V4') < 1e-5
  ACAFact.aca!(K, U3, V3, 0.0; start=rank+1, z=z, cache=cache)
  (U5, V5)  = aca(K, 50)
  @test U3 ≈ U5
  @test V3 ≈ V5
  @test opnorm(K - U3*V3') < 1e-14
end

