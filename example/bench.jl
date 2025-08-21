
using ACAFact, BenchmarkTools

pts = sort(rand(12_000))
fn(x, y) = sinc(2*100*(x-y))

const km  = ACAFact.kernelmatrix(pts, pts, fn)
@time (U, V) = aca(km, 1e-13; rankstart=50)

