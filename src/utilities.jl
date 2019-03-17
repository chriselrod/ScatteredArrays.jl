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

abstract type AbstractScatteredArray{E,M,A<:AbstractArray{E,M},N,Np1} <: AbstractArray{A,N} end


@generated function Base.getindex(A::PaddedMatrices.AbstractFixedSizePaddedArray{S,T,N,P,L}, ::LinearStorage, i::Vararg{<:Integer,NI}) where {S,T,N,P,L,NI}
    if N == 1 || ( (S.parameters[1] == P) && (NI == 1) )
        @assert NI == 1
        return quote
            $(Expr(:meta,:inline))
            ind = i[1]
            @boundscheck ind > $L && PaddedMatrices.ThrowBoundsError()
            @inbounds A[ind]
        end
    elseif NI == 1
        return quote
            $(Expr(:meta,:inline))
            nrep, nrem = divrem(i[1], $(S.parameters[1]))
            ind = nrep * $P + nrem
            @boundscheck ind > $L && PaddedMatrices.ThrowBoundsError()
            @inbounds A[ind]
        end
    else
        @assert N == NI
        return quote
            $(Expr(:meta,:inline))
            $(Expr(:meta,:propagate_inbounds))
            A[i...]
        end
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


@noinline thrownotisbitserror(T) = throw("The type is required to be isbits, but $T is not.")
@noinline throwdimensionmismatcherror(dimA, dimB) = throw("Dimensions $dimA != $dimB.")
