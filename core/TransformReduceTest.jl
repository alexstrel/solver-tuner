push!(LOAD_PATH, joinpath(@__DIR__, "../", "core"))

using Base.Threads
using GenericBlas
using LinearAlgebra

@inline function transform(policy::Any, range::Int64, transformer::Any) 

@threads for i in 1:range
           transformer(i)
         end
end #transform

@inline function transform_reduce(policy::Any, range::Int64, init::Any, reducer::Any, transformer::Any)
         local res::Float32 = init
@threads for i in 1:range
           local val = transformer(i)
           res       = reducer(res, val)
         end
         println("Res :: ", res)
         return res
end #transform_reduce

data_type   = Float32
scalar_type = Float32

len = 8192

x = Vector{data_type}(undef, len)
y = Vector{data_type}(undef, len)
z = Vector{data_type}(undef, len)

x .= rand(data_type, len)
y .= rand(data_type, len)
z .= rand(data_type, len)

xnr2 = BLAS.nrm2(len, x, 1)
println("Norm check : ", xnr2)

a = scalar_type(1.1)

axpyXmaz_ = GenericBlas.axpyXmaz{scalar_type, data_type}(a,x,y,z)

policy = missing

@time transform(policy, len, axpyXmaz_)

xnr2 = BLAS.nrm2(len, x, 1)
println("Norm check : ", xnr2)

a1 = scalar_type(1.1)
a2 = scalar_type(1.1)

plus  = GenericBlas.plus 

#overload Base.:+
#Base.:+(x1::Float32, x2::Float32) = 2.0*x1 - x2
#define alias
plus2 = Base.:+


println(" \n", plus2(a1,a2), "  ", a1)


norm2_x = GenericBlas.norm2{data_type}(x)


@time res = transform_reduce(policy, len, 0.0, plus, norm2_x)

println("Norm 2 : ", res)






























