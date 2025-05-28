
using ACAFact, BenchmarkTools

pts = sort(rand(10_000))
km  = ACAFact.kernelmatrix(pts, pts, (x,y)->exp(-abs2(x-y)))

@btime aca($km, $(1e-13), rankstart=$(50))

