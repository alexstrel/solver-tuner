module BlasIntrinsics

using Base:llvmcall

# https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=FMA

# Plain baseline registers
const half2   = NTuple{2, VecElement{Float16}}
const half4   = NTuple{4, VecElement{Float16}}
#
const float2  = NTuple{2, VecElement{Float32}}
const float4  = NTuple{4, VecElement{Float32}}
const float8  = NTuple{8, VecElement{Float32}}
const float16 = NTuple{16,VecElement{Float32}}
#
const double2 = NTuple{2, VecElement{Float64}}
const double4 = NTuple{4, VecElement{Float64}}
const double8 = NTuple{8, VecElement{Float64}}


@inline m256dfma(a,b,c) = ccall("llvm.fma.v4f64", llvmcall, double4, (double4, double4, double4), a, b, c)
@inline m256fma(a,b,c ) = ccall("llvm.fma.v8f32", llvmcall, float8, (float8, float8, float8), a, b, c)

@inline m512dfma(a,b,c) = ccall("llvm.fma.v8f64", llvmcall, double8, (double8, double8, double8), a, b, c)
@inline m512fma(a,b,c ) = ccall("llvm.fma.v16f32", llvmcall, float16, (float16, float16, float16), a, b, c)

@inline prefetch(a, i1, i2, i3) = ccall("llvm.prefetch", llvmcall, Cvoid, (Ptr{Cchar}, Int32, Int32, Int32), a, i1, i2, i3)

@inline prefetchT0(a) = mmprefetch(a, Int32(0), Int32(3), Int(1))
@inline prefetchT1(a) = mmprefetch(a, Int32(0), Int32(2), Int(1))
@inline prefetchT2(a) = mmprefetch(a, Int32(0), Int32(1), Int(1))

@inline function mm_mul(x::T, y::T) where T <:AbstractFloat
   return x * y
end

@inline function mm_mul(x::float4, y::float4)
  (VecElement(x[1].value * y[1].value),
   VecElement(x[2].value * y[2].value),
   VecElement(x[3].value * y[3].value),
   VecElement(x[4].value * y[4].value))
end

@inline function mm_mul(x::double2, y::double2)
  (VecElement(x[1].value * y[1].value),
   VecElement(x[2].value * y[2].value))
end

@inline function mm_mul(x::float8, y::float8)
  (VecElement(x[1].value * y[1].value),
   VecElement(x[2].value * y[2].value),
   VecElement(x[3].value * y[3].value),
   VecElement(x[4].value * y[4].value),
   VecElement(x[5].value * y[5].value),
   VecElement(x[6].value * y[6].value),
   VecElement(x[7].value * y[7].value),
   VecElement(x[8].value * y[8].value))
end

@inline function mm_mul(x::double4, y::double4)
  (VecElement(x[1].value * y[1].value),
   VecElement(x[2].value * y[2].value),
   VecElement(x[3].value * y[3].value),
   VecElement(x[4].value * y[4].value))
end


@inline function mm_add(x::T, y::T) where T <:AbstractFloat
   return x + y
end

@inline function mm_add(x::float4, y::float4)
  (VecElement(x[1].value + y[1].value),
   VecElement(x[2].value + y[2].value),
   VecElement(x[3].value + y[3].value),
   VecElement(x[4].value + y[4].value))
end

@inline function mm_add(x::double2, y::double2)
  (VecElement(x[1].value + y[1].value),
   VecElement(x[2].value + y[2].value))
end

@inline function mm_add(x::float8, y::float8)
  (VecElement(x[1].value + y[1].value),
   VecElement(x[2].value + y[2].value),
   VecElement(x[3].value + y[3].value),
   VecElement(x[4].value + y[4].value),
   VecElement(x[5].value + y[5].value),
   VecElement(x[6].value + y[6].value),
   VecElement(x[7].value + y[7].value),
   VecElement(x[8].value + y[8].value))
end

@inline function mm_add(x::double4, y::double4)
  (VecElement(x[1].value + y[1].value),
   VecElement(x[2].value + y[2].value),
   VecElement(x[3].value + y[3].value),
   VecElement(x[4].value + y[4].value))
end

@inline function mm_sub(x::T, y::T) where T <:AbstractFloat
   return x - y
end

@inline function mm_sub(x::float4, y::float4)
  (VecElement(x[1].value - y[1].value),
   VecElement(x[2].value - y[2].value),
   VecElement(x[3].value - y[3].value),
   VecElement(x[4].value - y[4].value))
end

@inline function mm_sub(x::double2, y::double2)
  (VecElement(x[1].value - y[1].value),
   VecElement(x[2].value - y[2].value))
end

@inline function mm_sub(x::float8, y::float8)
  (VecElement(x[1].value - y[1].value),
   VecElement(x[2].value - y[2].value),
   VecElement(x[3].value - y[3].value),
   VecElement(x[4].value - y[4].value),
   VecElement(x[5].value - y[5].value),
   VecElement(x[6].value - y[6].value),
   VecElement(x[7].value - y[7].value),
   VecElement(x[8].value - y[8].value))
end

@inline function mm_sub(x::double4, y::double4)
  (VecElement(x[1].value - y[1].value),
   VecElement(x[2].value - y[2].value),
   VecElement(x[3].value - y[3].value),
   VecElement(x[4].value - y[4].value))
end

@inline function mm_mad(a::T, x::T, y::T) where T <:AbstractFloat
   return a * x + y
end

@inline function mm_mad(a::T, x::float4, y::float4) where T <: AbstractFloat
  (VecElement(a * x[1].value + y[1].value),
   VecElement(a * x[2].value + y[2].value),
   VecElement(a * x[3].value + y[3].value),
   VecElement(a * x[4].value + y[4].value))
end

@inline function mm_mad(a::T, x::double2, y::double2) where T <: AbstractFloat
  (VecElement(a * x[1].value + y[1].value),
   VecElement(a * x[2].value + y[2].value))
end

@inline function mm_mad(a::T, x::float8, y::float8) where T <: AbstractFloat
   z = float8((a, a, a, a, a, a, a, a))
   return m256fma(z,x,y)
end

@inline function mm_mad(a::T, x::double4, y::double4) where T <: AbstractFloat
   z = double4((a, a, a, a))
   return m256dfma(z,x,y)
end

end #QJuliaIntrinsics
