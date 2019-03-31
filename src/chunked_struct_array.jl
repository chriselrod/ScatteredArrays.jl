
function VectorizationBase.vectorizable(ScA::ChunkedArray{E,M,T,N,Np2}) where {E,M,T,N,Np2}
    VectorizedChunkedArray{E,M,T,N,Np2}(pointer(ScA.data), size(ScA.data))
end

const ChunkedVector{E,M,T} = ChunkedArray{E,M,T,1,2}
const ChunkedMatrix{E,M,T} = ChunkedArray{E,M,T,2,3}

function chunked_ind_base_quote(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    quote
        $(Expr(:meta, :inline))
        i1 = 1+((i[1]-1) & $(W-1))
        i2 = 1+((i[1]-1) >> $Wshift)
    end
end

function CcA_setindex_quote(N, L, T)
    q = chunked_ind_base_quote(T)
    inds = [:(i[$j]) for j ∈ 2:N]
    for j ∈ 1:L
        push!(q.args, Expr(:call, :setindex!, :(ScA.data), :(v[LinearStorage(), $j]), :i1, inds..., j,  :i2))
    end
    q
end

@generated function Base.setindex!(ScA::ChunkedArray{E,M,T,N,Np2}, v::T,
                                   # i::Vararg{<:Integer,N}) where {T,E,M,N,Np2}
                                   i::Vararg{<:Integer,N}) where {E,M,T,N,Np2}
    CcA_setindex_quote(N, type_length(T), E)
end

@generated function Base.setindex!(ScA::ChunkedArray{E,M,T,N,Np2}, v::T,
                                    # i::CartesianIndex{N}) where {T,E,M,N,Np2}
                                    i::MultiDimIndex{N}) where {E,M,T,N,Np2}
    CcA_setindex_quote(N, type_length(T), E)
end



@generated function Base.getindex(ScA::ChunkedArray{E,M,T,1,3}, i::Integer, j::Integer) where {E,M,T}
    q = chunked_ind_base_quote(E)
    # inds = [:(i[$n]) for n in 2:N]
    ind_expr = [Expr(:call, :getindex, :(ScA.data), :i1, k, :i2) for k ∈ 1:type_length(T)]
    push!(q.args, construct_expr(T, ind_expr))
    q
end
@generated function Base.getindex(ScA::ChunkedArray{E,M,T,N,Np2}, i::Vararg{<:Integer,N}) where {E,M,T,N,Np2}
    q = chunked_ind_base_quote(E)
    inds = [:(i[$n]) for n in 2:N]
    ind_expr = [Expr(:call, :getindex, :(ScA.data), :i1, inds..., j, :i2) for j ∈ 1:type_length(T)]
    push!(q.args, construct_expr(T, ind_expr))
    q
end

@generated function Base.getindex(ScA::ChunkedArray{E,M,T,N,Np2},
                        i::MultiDimIndex{N}) where {E,M,T,N,Np2}
    q = chunked_ind_base_quote(E)
    inds = [:(i[$n]) for n in 2:N]
    ind_expr = [Expr(:call, :getindex, :(ScA.data), :i1, inds..., j, :i2) for j ∈ 1:type_length(T)]
    push!(q.args, construct_expr(T, ind_expr))
    q
end




function chunked_array_view_inds(V, N, T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    inds = Vector{Expr}(undef, N)
    if V.parameters[1] == Colon
        ind1 = :(i[1])
    else
        ind1 = :(ScA.view[1])
    end
    inds[1] = :( 1 + (($ind1-1) & $(W-1)) )
    i = 1
    for n ∈ 2:N
        if V.parameters[n] == Colon
            i += 1
            inds[n] = :(i[$i])
        else
            inds[n] = :(ScA.view[$n])
        end
    end
    inds, :(1 + (($ind1-1) >> $Wshift))
end

@generated function Base.getindex(ScA::ChunkedArrayView{E,M,T,new_N,N,Np2,V},
                        i::Vararg{<:Any,new_N}) where {E,M,T,new_N,N,Np2,V}
    inds, indslast = chunked_array_view_inds(V, N, E)
    # The strategy is to calculate the first index, as well as the stride between indices.
    # After the first index is calculated, the remaining indices can be calcualted much more
    # efficiently via the stide.
    q = quote
        $(Expr(:meta, :inline))
        stride = $(Expr(:call, :*, [:(ScA.full_size[$n]) for n ∈ 1:N]...))
        ind_1 = sub2ind(ScA.full_size, $(Expr(:tuple, inds..., 1, indslast)))
    end
    for j ∈ 2:type_length(T)
        push!(q.args, :($(Symbol(:ind_,j)) = ind_1 + stride * $(j-1)))
    end
    push!(q.args, construct_expr(T, [:(unsafe_load(ScA.ptr, $(Symbol(:ind_,j)))) for j ∈ 1:type_length(T)]))
    q
end
@generated function Base.setindex!(ScA::ChunkedArrayView{E,M,T,new_N,N,Np2,V}, v,
                        i::Vararg{<:Any,new_N}) where {E,M,T,new_N,N,Np2,V}
    inds, indslast = chunked_array_view_inds(V, N, E)
    q = quote
        $(Expr(:meta, :inline))
        stride = $(Expr(:call, :*, [:(ScA.full_size[$n]) for n ∈ 1:N]...))
        ind_1 = sub2ind(ScA.full_size, $(Expr(:tuple, inds..., 1, indslast)))
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
@inline function Base.view(ScA::ChunkedArray{E,M,T,N,Np2}, i::Vararg{<:Any,N}) where {E,M,T,N,Np2}
    ChunkedArrayView(ScA, i)
end
@generated function ChunkedArrayView(ScA::ChunkedArray{E,M,T,N,Np2}, i::V) where {E,M,T,N,Np2,V}
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
        ChunkedArrayView{$E,$M,$T,$new_N,$N,$Np2,$V}(pointer(ScA.data), $new_size, full_size, i)
    end
end


# @inline function SArray{Tuple{S},T,N,L}(vs::Vararg{SIMDPirates.SVec{W,T},L}) where {S,T,N,L,W}
#     SArray{S}(vs...)
# end
@inline function Base.:+(i::Integer, v::VectorizedChunkedArray{E,M,T,N,Np2}) where {E,M,T,N,Np2}
    VectorizedChunkedArray{E,M,T,N,Np2}(v.ptr + sizeof(T) * i, v.size)
end
@inline function Base.:+(v::VectorizedChunkedArray{E,M,T,N,Np2}, i::Integer) where {E,M,T,N,Np2}
    VectorizedChunkedArray{E,M,T,N,Np2}(v.ptr + sizeof(T) * i, v.size)
end
@generated function SIMDPirates.vload(::Type{V}, vScA::VectorizedChunkedArray{E,M,T,N,Np2}) where {E,M,T,N,Np2,W, V <: Union{SIMDPirates.SVec{W,E},SIMDPirates.Vec{W,E}}}
    W_full, Wshift_full = VectorizationBase.pick_vector_width_shift(E)
    inds = [:(i[$n]) for n in 2:N]
    quote
        $(Expr(:meta,:inline))
        $(construct_expr(T, [Expr(:call, :vload, V, :(vScA.ptr + $(W_full*j))) for j ∈ 0:type_length(T)-1]))
    end
end

function chunked_vload_quote(N, W, E, T, V)
    W_full, Wshift_full = VectorizationBase.pick_vector_width_shift(E)
    inds = [:(i[$n]) for n in 2:N]
    quote
        $(Expr(:meta,:inline))
        $(construct_expr(T, [Expr(:call, :vload, V, :(vScA.ptr),
            Expr(:call, :sub2ind, :(vScA.size), Expr(:tuple, 1, inds..., j, 1 + :( (i[1]-1) >> $Wshift_full ) ))) for j in 1:type_length(T)]))
    end
end

@generated function SIMDPirates.vload(::Type{V}, vScA::VectorizedChunkedArray{E,M,T,N,Np2},
                                                # i::CartesianIndex{N}) where {E,M,T,N,Np2,W}
                                                i::Vararg{<:Integer,N}) where {E,M,T,N,Np2,W, V <: Union{SIMDPirates.SVec{W,E},SIMDPirates.Vec{W,E}}}
    chunked_vload_quote(N, W, E, T, V)
end
@generated function SIMDPirates.vload(::Type{V}, vScA::VectorizedChunkedArray{E,M,T,N,Np2},
                                                # i::Vararg{<:Integer,N}) where {E,M,T,N,Np2,W}
                                                i::MultiDimIndex{N}) where {E,M,T,N,Np2,W, V <: Union{SIMDPirates.SVec{W,E},SIMDPirates.Vec{W,E}}}
    chunked_vload_quote(N, W, E, T, V)
end
"""
For vload(::Type{<:SVec}, ::VectorizedChunkedArray), we assume the loads are alligned with chunk boundaries.
"""
@generated function SIMDPirates.vload(::Type{V}, vScA::VectorizedChunkedArray{E,M,T,N,Np2},
                                                # i::Vararg{<:Integer,N}) where {E,M,T,N,Np2,W}
                                                i::Integer) where {E,M,T,N,Np2,W, V <: Union{SIMDPirates.SVec{W,E},SIMDPirates.Vec{W,E}}}
    W_full, Wshift_full = VectorizationBase.pick_vector_width_shift(E)
    WE = W_full * sizeof(E)
    TL = type_length(T)
    quote
        $(Expr(:meta,:inline))
        ind = $(TL*sizeof(E)) * (i-1)
        $(construct_expr(T, [Expr(:call, :vload, V, :(vScA.ptr + ind + $(WE*j))) for j in 0:TL-1]))
    end
end


function Base.copyto!(ScA::ChunkedArray{E,M,T,N,Np2}, A::AbstractArray{T}) where {E,M,T,N,Np2}
    @boundscheck size(ScA) == size(A) || throwdimensionmismatcherror(size(ScA), size(A))
    @inbounds for i in eachindex(A)
        ScA[i] = A[i]
    end
    ScA
end


@generated function ChunkedArray(A::AbstractArray{T,N}) where {E,M,T <: AbstractArray{E,M},N}
    isbitstype(T) || thrownotisbitserror(T)
    TL = type_length(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(E)
    quote
        ScA = ChunkedArray{$E,$M,$T,$N,$(N+2)}(
            Array{$E}(undef, $(Expr(:tuple, W, [:(size(A,$n)) for n ∈ 2:N]..., TL, :(size(A,1) >> $Wshift) )) )
        )
        @inbounds copyto!(ScA, A)
    end
end
@generated function ChunkedArray{E,M,T,N,Np2}(::UndefInitializer, I::Vararg{<:Integer,N}) where {E,M,T <: AbstractArray{E,M},N,Np2}
    isbitstype(T) || thrownotisbitserror(T)
    N + 2 == Np2 || throw("N + 1 = $(N + 1) != Np2 = $Np2")
    # @show T
    TL = type_length(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(E)
    quote
        ChunkedArray{$E,$M,$T,$N,$Np2}(
            Array{$E}(undef, $(Expr(:tuple, W, [:(I[$n]) for n ∈ 2:N]..., TL, :(I[1] >> $Wshift))) )
        )
    end
end

@generated function Base.size(ScA::ChunkedArray{E,M,T,N,Np2}) where {E,M,T,N,Np2}
    quote
        $(Expr(:meta, :inline))
        s = size(ScA.data)
        @inbounds $(Expr(:tuple, :(s[1]*s[end]), [:(s[$n]) for n ∈ 2:N]...))
    end
end
@inline Base.size(ScA::ChunkedArray{E,M,T,N,Np2}, n::Integer) where {E,M,T,N,Np2} = n == 1 ? size(ScA.data, 1)*size(ScA.data, Np2) : size(ScA.data, n)
@inline function Base.length(ScA::ChunkedArray{E,M,T,N,Np2}) where {E,M,T,N,Np2}
    s = size(ScA.data)
    @inbounds l = s[Np2]
    @inbounds for n ∈ 1:N
        l *= s[n]
    end
    l
end
