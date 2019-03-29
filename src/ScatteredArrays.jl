module ScatteredArrays


using VectorizationBase, SIMDPirates, StaticArrays, Base.Cartesian, PaddedMatrices
import PaddedMatrices: type_length

export ScatteredArray, ScatteredVector, ScatteredMatrix, LinearStorage, ChunkedArray

const MultiDimIndex{N} = Union{Tuple{Vararg{<:Integer, N}}, NTuple{N,<:Integer}, CartesianIndex{N}}



include("utilities.jl")
include("types.jl")
include("scattered_struct_array.jl")
include("chunked_struct_array.jl")


end # module
