module ScatteredArrays


using VectorizationBase, SIMDPirates, StaticArrays, Base.Cartesian, PaddedMatrices
import PaddedMatrices: type_length

export ScatteredArray, ScatteredVector, ScatteredMatrix, LinearStorage

const MultiDimIndex{N} = Union{Tuple{Vararg{<:Integer, N}}, NTuple{N,<:Integer}, CartesianIndex{N}}



include("utilities.jl")
include("types.jl")
include("scattered_struct_array.jl")
include("chunked_Struct_array.jl")


end # module
