module GenericBlas

import BlasIntrinsics

using Base.Threads

#https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects-1

#create function/type alias
#SSE
double2   = BlasIntrinsics.double2
float4    = BlasIntrinsics.float4
#AVX/AVX2
double4   = BlasIntrinsics.double4
float8    = BlasIntrinsics.float8
#AVX512
double8   = BlasIntrinsics.double8
float16   = BlasIntrinsics.float16

mm_mul    = BlasIntrinsics.mm_mul
mm_add    = BlasIntrinsics.mm_add
mm_sub    = BlasIntrinsics.mm_sub
mm_mad    = BlasIntrinsics.mm_mad

#collection of generic blas operations (e.g., on vector registers etc.)

abstract type GenericMultiblasFunctor{T1<:Any, T2<:Any} end

mutable struct ConvertC2R{DstTp <: Any, SrcScalarTp <: AbstractFloat} <: GenericMultiblasFunctor{DstTp, SrcScalarTp}
  #scalars:
  #vectors
  dst::Vector{DstTp}
  src::Vector{Complex{SrcScalarTp}}

  ConvertC2R{DstTp, SrcScalarTp}(y::Vector{DstTp}, x::Vector{Complex{SrcScalarTp}}) where {DstTp <: Any} where {SrcScalarTp <: AbstractFloat} = new(y, x)

end # struct ConvertC2R

# Helper methods
#function register_t1(f::GenericMultiblasFunctor{T1<:Any, T2<:Any}) where T1 where T2; return T1; end
#function register_t2(f::GenericMultiblasFunctor{T1<:Any, T2<:Any}) where T1 where T2; return T2; end

@inline function (f::ConvertC2R)(i::Int64)
             dst_t = Float32 #register_t1(f)
             if     dst_t == double4 || dst_t == float4
               c=[real(f.src[2i-1]), imag(f.src[2i-1]), real(f.src[2i]), imag(f.src[2i]) ] 
               f.dst[i] = dst_t(ntuple(i->c[i],4))
             elseif dst_t == double8 || dst_t == float8
               c=[real(f.src[4i-3]), imag(f.src[4i-3]), real(f.src[4i-2]), imag(f.src[4i-2]), real(f.src[4i-1]), imag(f.src[4i-1]), real(f.src[4i]), imag(f.src[4i]) ] 
               f.dst[i] = dst_t(ntuple(i->c[i],8))
             elseif dst_t ==float16
               c=[real(f.src[8i-7]), imag(f.src[8i-7]), 
                  real(f.src[8i-6]), imag(f.src[8i-6]), 
                  real(f.src[8i-5]), imag(f.src[8i-5]), 
                  real(f.src[8i-4]), imag(f.src[8i-4]),
                  real(f.src[8i-3]), imag(f.src[8i-3]), 
                  real(f.src[8i-2]), imag(f.src[8i-2]), 
                  real(f.src[8i-1]), imag(f.src[8i-1]), 
                  real(f.src[8i  ]), imag(f.src[8i  ]) ] 
               f.dst[i] = dst_t(ntuple(i->c[i],16))
             elseif dst_t <: AbstractFloat
               f.dst[2i-1] = real(src[i])
               f.dst[2i  ] = real(src[i]) 
             else
               println("Register type is not supported.")
             end
             
end

#mutable struct ConvertR2C{DstScalarTp <: AbstractFloat, SrcTp <: Any} <: GenericMultiblasFunctor{DstScalarTp, SrcTp}
  #scalars:
  #vectors
#  dst::Vector{Complex{DstScalarTp}}
#  src::Vector{SrcTp}

#  ConvertC2R{DstScalarTp, SrcTp}(y_::Vector{Complex{DstScalarTp}}, x_::Vector{SrcTp}) where {DstScalarTp <: AbstractFloat} where {SrcTp <: Any} = new(y_, x_)

#end # 

#@inline function (f::ConvertR2C)(i::Int64)
#          f.dst[i] = f.src[2i-1] + f.src[2i]*im;
#end

mutable struct cpy{DstTp <: AbstractArray, SrcTp <: AbstractArray} <: GenericMultiblasFunctor{DstTp, SrcTp}
  #scalars:
  #vectors
  dst::Vector{DstTp}
  src::Vector{SrcTp}

  cpy{DstTp, SrcTp}(y_::DstTp, x_::SrcTp) where {DstTp <: AbstractArray} where {SrcTp <: AbstractArray}  = new(y_, x_)

end #

@inline function (f::cpy)(i::Int64)
          f.dst[i] = f.src[i]
end

#@inline function cpy(y::AbstractArray, x::AbstractArray) 
#  if pointer_from_objref(y) == pointer_from_objref(x); return; end
#@threads for i in 1:length(x);  y[i] = x[i]; end
#end #cpy

mutable struct axpy{ScalarTp <: AbstractFloat, VectorTp <: Any} <: GenericMultiblasFunctor{ScalarTp, VectorTp}
  #scalars:
  a::ScalarTp
  #vectors
  x::Vector{VectorTp}
  y::Vector{VectorTp}

  axpy{ScalarTp, VectorTp}(a_::ScalarTp, x_::Vector{VectorTp}, y_::Vector{VectorTp}) where {ScalarTp <: AbstractFloat} where {VectorTp <: Any}  = new(a_, x_, y_)

end #

@inline function (f::axpy)(i::Int64)
          f.y[i] = mm_mad(f.a, f.x[i], f.y[i])
end

mutable struct xpay{VectorTp <: Any, ScalarTp <: AbstractFloat} <: GenericMultiblasFunctor{VectorTp, ScalarTp}

  #vectors
  x::Vector{VectorTp}
  y::Vector{VectorTp}
  #scalars:
  a::ScalarTp

  xpay{VectorTp, ScalarTp}(x_::Vector{VectorTp}, y_::Vector{VectorTp}, a_::ScalarTp) where {VectorTp <: Any} where {ScalarTp <: AbstractFloat}  = new(x_, y_, a_)

end #

@inline function (f::xpay)(i::Int64)
          f.y[i] = mm_mad(f.a, f.y[i], f.x[i])
end


mutable struct xmy{VectorTp <: Any} <: GenericMultiblasFunctor{VectorTp, VectorTp}

  #vectors
  x::Vector{VectorTp}
  y::Vector{VectorTp}

  xmy{VectorTp}(x_::Vector{VectorTp}, y_::Vector{VectorTp}) where {VectorTp <: Any}  = new(x_, y_)

end #

@inline function (f::xmy)(i::Int64)
          f.y[i] = mm_sub(f.x[i], f.y[i])
end

mutable struct xpy{VectorTp <: Any} <: GenericMultiblasFunctor{VectorTp, VectorTp}

  #vectors
  x::Vector{VectorTp}
  y::Vector{VectorTp}

  xpy{VectorTp}(x_::Vector{VectorTp}, y_::Vector{VectorTp}) where {VectorTp <: Any}  = new(x_, y_)

end #

@inline function (f::xpy)(i::Int64)
          f.y[i] = mm_add(f.x[i], f.y[i])
end


#First performs the operation y[i] += a*x[i]
#Second performs the operator x[i] -= a*z[i]

mutable struct axpyXmaz{ScalarTp <: AbstractFloat, VectorTp <: Any} <: GenericMultiblasFunctor{ScalarTp, VectorTp}
  #scalar
  a::ScalarTp
  #vectors
  x::Vector{VectorTp}
  y::Vector{VectorTp}
  z::Vector{VectorTp}

  axpyXmaz{ScalarTp, VectorTp}(a_::ScalarTp, x_::Vector{VectorTp}, y_::Vector{VectorTp}, z_::Vector{VectorTp}) where {ScalarTp <: AbstractFloat} where {VectorTp <: Any}  = new(a_, x_, y_, z_)

end #

@inline function (f::axpyXmaz)(i::Int64)
          f.y[i] = mm_mad(+f.a, f.x[i], f.y[i])
          f.x[i] = mm_mad(-f.a, f.z[i], f.x[i])
end


#First performs the operation x[i] = x[i] + a*p[i]
#Second performs the operator p[i] = u[i] + b*p[i]

mutable struct axpyZpbx{ScalarTp <: AbstractFloat, VectorTp <: Any} <: GenericMultiblasFunctor{ScalarTp, VectorTp}
  #scalar
  a::ScalarTp
  b::ScalarTp
  #vectors
  x::Vector{VectorTp}
  p::Vector{VectorTp}
  u::Vector{VectorTp}

  axpyZpbx{ScalarTp, VectorTp}(a_::ScalarTp, b_::ScalarTp, x_::Vector{VectorTp}, p_::Vector{VectorTp}, u_::Vector{VectorTp}) where {ScalarTp <: AbstractFloat} where {VectorTp <: Any}  = new(a_, b_, x_, p_, u_)

end #

@inline function (f::axpyZpbx)(i::Int64)
          f.x[i] = mm_mad(f.a, f.p[i], f.x[i])
          f.p[i] = mm_mad(f.b, f.p[i], f.u[i])
end


mutable struct norm2{VectorTp <: Any} <: GenericMultiblasFunctor{VectorTp, VectorTp}

  #vectors
  x::Vector{VectorTp}

  norm2{VectorTp}(x_::Vector{VectorTp}) where {VectorTp <: Any}  = new(x_)

end #

@inline function (f::norm2)(i::Int64)
          return mm_mul(f.x[i], f.x[i])
end

@inline function plus(x::AbstractFloat, y::AbstractFloat)
          return mm_add(x, y)
end

end #GenericBlas


