module ScatteredArrays


using VectorizationBase, SIMDPirates, StaticArrays, Base.Cartesian

export ScatteredArray, ScatteredVector, ScatteredMatrix, LinearStorage

const MultiDimIndex{N} = Union{Tuple{Vararg{<:Integer, N}}, NTuple{N,<:Integer}, CartesianIndex{N}}

# fall back. An alternative definition should be provided
# if an array can be accessed via linear indexing in order (ie, 1,2,3..),
# and all elements are reached before length(A). Eg, if
# a triangular matrix is defined and does not store the off triangle.
# type_length(::NTuple{N}) where N= N
# type_length(::SArray{S,T,N,L}) where {S,T,N,L} = L
type_length(::Type{NTuple{N}}) where N= N
type_length(::Type{SArray{S,T,N,L}}) where {S,T,N,L} = L
type_length(::T) where T = type_length(T)
type_length(::Type{A}) where {T,A<:AbstractArray{T}} = sizeof(A) ÷ sizeof(T)

# type_length(::Type{T}) where T = throw("type_length(::Type{$T}) has not been defined.")
# function type_length(::T) where T
#     @show T
#     throw("Stop!")
# end

"""
LinearStorage allows one to define a linear storing behavior different from standard linear indexing,
in case the two may differ. For example, consider a `L = LowerTrianglar{MMatrix{3,3,Float64,9}}`.
One is likely to want the standard linear indexing behavior for operations like `L .* A`.
However, for the construction of ScatteredArrays, one would prefer to only access the lower triangular elements.
Thus, ScatteredArrays accesses elements via LinearStorage, which default to the standard LinearIndex,
so that one may optionally provide other methods if one wants different behavior.
"""
struct LinearStorage end
@inline Base.getindex(A::AbstractArray, ::LinearStorage, i::Vararg{Any,N} where N) = getindex(A, i...)
@inline Base.setindex!(A::AbstractArray, v, ::LinearStorage, i::Vararg{Any,N} where N) = setindex!(A, v, i...)

abstract type AbstractScatteredArray{E,M,T<:AbstractArray{E,M},N,Np1} <: AbstractArray{T,N} end
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

function VectorizationBase.vectorizable(ScA::ScatteredArray{E,M,T,N,Np1}) where {E,M,T,N,Np1}
    ptr_data = pointer(ScA.data)
    VectorizedScatteredArray{E,M,T,N,Np1}(ptr_data, size(ScA.data))
end
@generated function Base.length(vsa::VectorizedScatteredArray{E,M,T,N}) where {E,M,T,N}
    quote
        $(Expr(:meta,:inline))
        $(Expr(:call, :*, [:(vsa.size[$n]) for n ∈ 1:N]...))
    end
end

@inline Base.size(ScA::ScatteredArrayView) = ScA.size

const ScatteredVector{E,M,T} = ScatteredArray{E,M,T,1,2}
const ScatteredMatrix{E,M,T} = ScatteredArray{E,M,T,2,3}

function ScA_setindex_quote(N, L)
    q = quote $(Expr(:meta, :inline)) end
    inds = [:(i[$j]) for j ∈ 1:N]
    for j ∈ 1:L
        push!(q.args, Expr(:call, :setindex!, :(ScA.data), :(v[LinearStorage(), $j]), inds..., j))
    end
    q
end

@generated function Base.setindex!(ScA::ScatteredArray{E,M,T,N,Np1}, v::T,
                                   # i::Vararg{<:Integer,N}) where {T,E,M,N,Np1}
                                   i::Vararg{<:Integer,N}) where {E,M,T,N,Np1}
    ScA_setindex_quote(N, type_length(T))
end

@generated function Base.setindex!(ScA::ScatteredArray{E,M,T,N,Np1}, v::T,
                                    # i::CartesianIndex{N}) where {T,E,M,N,Np1}
                                    i::MultiDimIndex{N}) where {E,M,T,N,Np1}
    ScA_setindex_quote(N, type_length(T))
end



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
@generated function Base.getindex(ScA::ScatteredArray{E,M,T,N,Np1}, i::Vararg{<:Integer,N}) where {E,M,T,N,Np1}
    inds = [:(i[$n]) for n in 1:N]
    ind_expr = [Expr(:call, :getindex, :(ScA.data), inds..., j) for j in 1:type_length(T)]
    quote
        $(Expr(:meta, :inline))
        $(construct_expr(T, ind_expr))
    end
end

@generated function Base.getindex(ScA::ScatteredArray{E,M,T,N,Np1},
                        i::MultiDimIndex{N}) where {E,M,T,N,Np1}
    inds = [:(i[$n]) for n in 1:N]
    ind_expr = [Expr(:call, :getindex, :(ScA.data), inds..., j) for j in 1:type_length(T)]
    quote
        $(Expr(:meta, :inline))
        $(construct_expr(T, ind_expr))
    end
end


## Lifted from metaprogramming documentation
# @generated function sub2ind_quote(dims::NTuple{N}, I::NTuple{N}) where N
#    ex = :(I[$N] - 1)
#    for i = (N - 1):-1:1
#        ex = :(I[$i] - 1 + dims[$i] * $ex)
#    end
#    :($ex + 1)
# end
@generated function sub2ind(dims::NTuple{N}, I::NTuple{N}) where N
   ex = :(I[$N] - 1)
   for i = (N - 1):-1:1
       ex = :(I[$i] - 1 + dims[$i] * $ex)
   end
   quote
       $(Expr(:meta,:inline))
       $ex + 1
   end
end

function scattered_array_view_inds(V, N)
    inds = Expr[]
    i = 0
    for n ∈ 1:N
        if V.parameters[n] == Colon
            i += 1
            push!(inds, :(i[$i]))
        else
            push!(inds, :(ScA.view[$n]))
        end
    end
    inds
end

@generated function Base.getindex(ScA::ScatteredArrayView{E,M,T,new_N,N,Np1,V},
                        i::Vararg{<:Any,new_N}) where {E,M,T,new_N,N,Np1,V}
    inds = scattered_array_view_inds(V, N)
    ind_expr = Expr[]
    for j in 1:type_length(T)
        push!(ind_expr, quote
            I = $(Expr(:tuple,inds...,j))
            unsafe_load(ScA.ptr, sub2ind(ScA.full_size, I))
        end)
    end
    quote
        $(Expr(:meta, :inline))
        dims =
        $(construct_expr(T, ind_expr))
    end
end
@generated function Base.setindex!(ScA::ScatteredArrayView{E,M,T,new_N,N,Np1,V}, v,
                        i::Vararg{<:Any,new_N}) where {E,M,T,new_N,N,Np1,V}
    inds = scattered_array_view_inds(V, N)
    q = quote
        $(Expr(:meta, :inline))
    end
    for j ∈ 1:type_length(T)
        push!(q.args, quote
            unsafe_store!(ScA.ptr, v[LinearStorage(), $j], sub2ind(ScA.full_size, $(Expr(:tuple,inds...,j))))
        end)
    end
    push!(q.args, :(v))
    q
end
@inline function Base.view(ScA::ScatteredArray{E,M,T,N,Np1}, i::Vararg{<:Any,N}) where {E,M,T,N,Np1}
    ScatteredArrayView(ScA, i)
end
@generated function ScatteredArrayView(ScA::ScatteredArray{E,M,T,N,Np1}, i::V) where {E,M,T,N,Np1,V}
    new_N = 0
    new_size = Expr(:tuple)
    for n ∈ 1:N
        if V.parameters[n] == Colon
            new_N += 1
            push!(new_size.args, :(full_size[$n]))
        end
    end
    quote
        full_size = size(ScA.data)
        ScatteredArrayView{$E,$M,$T,$new_N,$N,$Np1,$V}(pointer(ScA.data), $new_size, full_size, i)
    end
end


# @inline function SArray{Tuple{S},T,N,L}(vs::Vararg{SIMDPirates.SVec{W,T},L}) where {S,T,N,L,W}
#     SArray{S}(vs...)
# end

function vload_quote(N, W, E, T)
    inds = [:(i[$n]) for n in 1:N]
    quote
        $(Expr(:meta,:inline))
        $(construct_expr(T, [Expr(:call, :vload, :(SIMDPirates.SVec{$W,$E}), :(vScA.ptr),
            Expr(:call, :sub2ind, :(vScA.size), Expr(:tuple, inds..., j))) for j in 1:type_length(T)]))
    end
end

@generated function SIMDPirates.vload(::Type{SIMDPirates.SVec{W,E}}, vScA::VectorizedScatteredArray{E,M,T,N,Np1},
                                                # i::CartesianIndex{N}) where {E,M,T,N,Np1,W}
                                                i::Vararg{<:Integer,N}) where {E,M,T,N,Np1,W}
    vload_quote(N, W, E, T)
end
@generated function SIMDPirates.vload(::Type{SIMDPirates.SVec{W,E}}, vScA::VectorizedScatteredArray{E,M,T,N,Np1},
                                                # i::Vararg{<:Integer,N}) where {E,M,T,N,Np1,W}
                                                i::MultiDimIndex{N}) where {E,M,T,N,Np1,W}
    vload_quote(N, W, E, T)
end

@generated function SIMDPirates.vload(::Type{SIMDPirates.SVec{W,E}}, vScA::VectorizedScatteredArray{E,M,T,N,Np1},
                                                # i::Vararg{<:Integer,N}) where {E,M,T,N,Np1,W}
                                                i::Integer) where {E,M,T,N,Np1,W}
    quote
        $(Expr(:meta,:inline))
        $(construct_expr(T, [Expr(:call, :vload, :(SIMDPirates.SVec{$W,$E}), :(vScA.ptr),
            :(i + length(vScA)*$(j-1)) ) for j in 1:type_length(T)]))
    end
end

@noinline thrownotisbitserror(T) = throw("The type is required to be isbits, but $T is not.")
@noinline throwdimensionmismatcherror(dimA, dimB) = throw("Dimensions $dimA != $dimB.")

function Base.copyto!(ScA::ScatteredArray{E,M,T,N,Np1}, A::AbstractArray{T}) where {E,M,T,N,Np1}
    @boundscheck size(ScA) == size(A) || throwdimensionmismatcherror(size(ScA), size(A))
    @inbounds for i in eachindex(A)
        ScA[i] = A[i]
    end
    ScA
end


@generated function ScatteredArray(A::AbstractArray{T,N}) where {E,M,T <: AbstractArray{E,M},N}
    isbitstype(T) || thrownotisbitserror(T)
    TL = type_length(T)
    quote
        ScA = ScatteredArray{$E,$M,$T,$N,$(N+1)}(
            Array{$E}(undef, $(Expr(:tuple, [:(size(A,$n)) for n ∈ 1:N]..., TL)) )
        )
        @inbounds copyto!(ScA, A)
    end
end
@generated function ScatteredArray{E,M,T,N,Np1}(::UndefInitializer, I::Vararg{<:Integer,N}) where {E,M,T <: AbstractArray{E,M},N,Np1}
    isbitstype(T) || thrownotisbitserror(T)
    N + 1 == Np1 || throw("N + 1 = $(N + 1) != Np1 = $Np1")
    # @show T
    TL = type_length(T)
    quote
        ScatteredArray{$E,$M,$T,$N,$Np1}(
            Array{$E}(undef, $(Expr(:tuple, [:(I[$n]) for n ∈ 1:N]..., TL)) )
        )
    end
end

@generated function Base.size(ScA::ScatteredArray{E,M,T,N,Np1}) where {E,M,T,N,Np1}
    quote
        $(Expr(:meta, :inline))
        $(Expr(:tuple, [:(size(ScA,$n)) for n ∈ 1:N]...))
    end
end
@inline Base.size(ScA::ScatteredArray, n::Integer) = size(ScA.data, n)



end # module
