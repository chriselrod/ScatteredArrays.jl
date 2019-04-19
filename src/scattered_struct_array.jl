
function VectorizationBase.vectorizable(ScA::ScatteredArray{E,M,T,N,Np1}) where {E,M,T,N,Np1}
    ptr_data = pointer(ScA.data)
    VectorizedScatteredArray{E,M,T,N,Np1}(ptr_data, size(ScA.data))
end

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
    # The strategy is to calculate the first index, as well as the stride between indices.
    # After the first index is calculated, the remaining indices can be calcualted much more
    # efficiently via the stide.
    q = quote
        $(Expr(:meta, :inline))
        stride = $(Expr(:call, :*, [:(ScA.full_size[$n]) for n ∈ 1:N]...))
        ind_1 = sub2ind(ScA.full_size, $(Expr(:tuple,inds...,1)))
    end
    for j ∈ 2:type_length(T)
        push!(q.args, :($(Symbol(:ind_,j)) = ind_1 + stride * $(j-1)))
    end
    push!(q.args, construct_expr(T, [:(unsafe_load(ScA.ptr, $(Symbol(:ind_,j)))) for j ∈ 1:type_length(T)]))
    q
end
@generated function Base.setindex!(ScA::ScatteredArrayView{E,M,T,new_N,N,Np1,V}, v,
                        i::Vararg{<:Any,new_N}) where {E,M,T,new_N,N,Np1,V}
    inds = scattered_array_view_inds(V, N)
    q = quote
        $(Expr(:meta, :inline))
        stride = $(Expr(:call, :*, [:(ScA.full_size[$n]) for n ∈ 1:N]...))
        ind_1 = sub2ind(ScA.full_size, $(Expr(:tuple,inds...,1)))
        unsafe_store!(ScA.ptr, v[LinearStorage(), 1], ind_1)
    end
    for j ∈ 2:type_length(T)
        push!(q.args, quote
            unsafe_store!(ScA.ptr, v[LinearStorage(), $j], ind_1 + stride * $(j-1))
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

function scattered_vload_quote(N, W, E, T, V)
    inds = [:(i[$n]) for n in 1:N]
    quote
        $(Expr(:meta,:inline))
        $(construct_expr(T, [Expr(:call, :vload, V, :(vScA.ptr),
            Expr(:call, :sub2ind, :(vScA.size), Expr(:tuple, inds..., j))) for j in 1:type_length(T)]))
    end
end

@generated function SIMDPirates.vload(::Type{SIMDPirates.SVec{W,E}}, vScA::VectorizedScatteredArray{E,M,T,N,Np1},
                                                # i::CartesianIndex{N}) where {E,M,T,N,Np1,W}
                                                i::Vararg{<:Integer,N}) where {E,M,T,N,Np1,W}
    scattered_vload_quote(N, W, E, T, SIMDPirates.SVec{W,E})
end
@generated function SIMDPirates.vload(::Type{SIMDPirates.SVec{W,E}}, vScA::VectorizedScatteredArray{E,M,T,N,Np1},
                                                # i::Vararg{<:Integer,N}) where {E,M,T,N,Np1,W}
                                                i::MultiDimIndex{N}) where {E,M,T,N,Np1,W}
    scattered_vload_quote(N, W, E, T, SIMDPirates.SVec{W,E})
end

@generated function SIMDPirates.vload(::Type{SIMDPirates.SVec{W,E}}, vScA::VectorizedScatteredArray{E,M,T,N,Np1},
                                                # i::Vararg{<:Integer,N}) where {E,M,T,N,Np1,W}
                                                i::Integer) where {E,M,T,N,Np1,W}
    quote
        $(Expr(:meta,:inline))
        $(construct_expr(T, [Expr(:call, :vload, SIMDPirates.SVec{W,E}, :(vScA.ptr),
            :(i + length(vScA)*$(j-1)) ) for j in 1:type_length(T)]))
    end
end
@generated function SIMDPirates.vload(
                ::Type{SIMDPirates.SVec{W,E}}, vScA::VectorizedScatteredArray{E,M,T,N,Np1}
            ) where {E,M,T,N,Np1,W}
    quote
        $(Expr(:meta,:inline))
        $(construct_expr(T, [Expr(:call, :vload, SIMDPirates.SVec{W,E}, :(vScA.ptr + length(vScA)*$(sizeof(E)*(j-1))) ) for j in 1:type_length(T)]))
    end
end
@generated function SIMDPirates.vload(
                ::Type{SIMDPirates.SVec{W,Ef1}}, vScA::VectorizedScatteredArray{Ef2,M,T,N,Np1}
            ) where {Ef1,Ef2,M,T,N,Np1,W}
    quote
        $(Expr(:meta,:inline))
        $(construct_expr(T,
            [Expr(:call, :convert, SVec{W,Ef1},
                Expr(:call, :vload, SIMDPirates.SVec{W,Ef2}, :(vScA.ptr + length(vScA)*$(sizeof(Ef2)*(j-1))) )) for j in 1:type_length(T)
        ])
        )
    end
end

@inline function Base.:+(i::Integer, v::VectorizedScatteredArray{E,M,T,N,Np1}) where {E,M,T,N,Np1}
    VectorizedScatteredArray{E,M,T,N,Np1}(v.ptr + sizeof(E) * i, v.size)
end
@inline function Base.:+(v::VectorizedScatteredArray{E,M,T,N,Np1}, i::Integer) where {E,M,T,N,Np1}
    VectorizedScatteredArray{E,M,T,N,Np1}(v.ptr + sizeof(E) * i, v.size)
end

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
