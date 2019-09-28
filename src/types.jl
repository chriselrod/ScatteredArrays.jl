

struct ScatteredArray{E,M,T,N,Np1} <: AbstractScatteredArray{E,M,T,N,Np1}
    data::Array{E,Np1}
end
struct VectorizedScatteredArray{E,M,T,N,Np1} <: AbstractScatteredArray{E,M,T,N,Np1}
    ptr::Ptr{E}
    size::NTuple{Np1,Int}
end
struct ScatteredArrayView{E,M,T,new_N,N,Np1,V} <: AbstractScatteredArray{E,M,T,new_N,Np1}
    ptr::Ptr{E}
    size::NTuple{new_N,Int}
    full_size::NTuple{Np1,Int}
    view::V
end

"""
Data axis for a chunked_array of mats are:
W x (chunked_array[2:N]) x mat x chunked_array[1] ÷ W
"""
struct ChunkedArray{E,M,T,N,Np2,U<:Unsigned} <: AbstractScatteredArray{E,M,T,N,Np2}
    data::Array{E,Np2}
    size::NTuple{N,Int}
    mask::U
end
struct VectorizedChunkedArray{E,M,T,N,Np2} <: AbstractScatteredArray{E,M,T,N,Np2}
    ptr::Ptr{E}
    fullsize::NTuple{Np2,Int}
end
struct ChunkedArrayView{E,M,T,new_N,N,Np2,V,U<:Unsigned} <: AbstractScatteredArray{E,M,T,new_N,Np2}
    ptr::Ptr{E}
    size::NTuple{new_N,Int}
    full_size::NTuple{Np2,Int}
    view::V
    mask::U
end

const AbstractVectorizedScatteredArray{E,M,T,N,Np1} = Union{VectorizedScatteredArray{E,M,T,N,Np1},VectorizedChunkedArray{E,M,T,N,Np1}}
const AbstractScatteredArrayView{E,M,T,new_N,N,Np1,V} = Union{ScatteredArrayView{E,M,T,new_N,N,Np1,V},ChunkedArrayView{E,M,T,new_N,N,Np1,V}}


@inline Base.pointer(ScA::ScatteredArray) = pointer(ScA.data)
@inline Base.pointer(ScA::Union{AbstractVectorizedScatteredArray,AbstractScatteredArrayView}) = ScA.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, ScA::AbstractScatteredArray{T}) where {T} = pointer(ScA)
@inline function Base.unsafe_convert(::Type{Ptr{T}}, ScA::AbstractScatteredArray) where {T}
    Base.unsafe_convert(Ptr{T}, pointer(ScA))
end
@inline Base.convert(::Type{Ptr{T}}, ScA::AbstractScatteredArray) where T = Base.unsafe_convert(Ptr{T}, ScA)


@generated function Base.length(vsa::AbstractVectorizedScatteredArray{E,M,T,N}) where {E,M,T,N}
    quote
        $(Expr(:meta,:inline))
        $(Expr(:call, :*, [:(vsa.size[$n]) for n ∈ 1:N]...))
    end
end
@inline Base.size(ScA::AbstractScatteredArrayView) = ScA.size

"""
Expression for constructing an instance of T from the arguments.
The default constructor is `Expr(:call, T, args...)`.

For example,

T = SVector{4,Float64};
args = [:a,:b,:c,:d];
Expr(:call, T, args...)
# :((SArray{Tuple{4},Float64,1,4})(a, b, c, d))

defines how to construct the type. When extending this to custom types, either ensure
that the default constuctor is supported, or define your own `construct_expr` method.
For example, when `T <: Tuple`, `construct_expr` dispatches to:
    Expr(:tuple, args...)
instead.

"""
function construct_expr(::Type{T}, args) where T
    Expr(:call, T, args...)
end
function construct_expr(::Type{T}, args) where {T <: Tuple}
    Expr(:tuple, args...)
end
function construct_expr(::Type{T}, args) where {S,T <: SArray{S}}
    Expr(:call, :(SArray{$S}), args...)
end
function construct_expr(::Type{T}, args) where {S,T <: ConstantFixedSizeArray{S}}
    Expr(:call, :(ConstantFixedSizeArray{$S}), Expr(:tuple, args...))
end
