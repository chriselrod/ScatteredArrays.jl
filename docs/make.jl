using ScatteredArrays
using Documenter

makedocs(;
    modules=[ScatteredArrays],
    authors="Chris Elrod",
    repo="https://github.com/chriselrod/ScatteredArrays.jl/blob/{commit}{path}#L{line}",
    sitename="ScatteredArrays.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chriselrod.github.io/ScatteredArrays.jl",
    ),
    pages=[
        "Home" => "index.md",
    ],
    strict=false,
)

deploydocs(;
    repo="github.com/chriselrod/ScatteredArrays.jl",
)
