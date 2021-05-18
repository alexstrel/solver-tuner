module TransformReduce

push!(LOAD_PATH, joinpath(@__DIR__, "../", "core"))

using Base.Threads
using GenericBlas

@inline function transform(policy::Any, range::Int64, transformer::Any) 

@threads for i in 1:range
           transformer(i)
         end
end #transform

@inline function transform_reduce(policy::Any, range::Int64, init::Any, reducer::Any, transformer::Any)
         local res = init
@threads for i in 1:range
           local val = transformer(i)
           res       = reducer(res, val)
         end
end #transform_reduce

end #TrnasformReduce


