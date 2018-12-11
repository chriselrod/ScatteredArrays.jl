using Documenter, ScatteredArrays

makedocs(;
    modules=[ScatteredArrays],
    format=:html,
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/chriselrod/ScatteredArrays.jl/blob/{commit}{path}#L{line}",
    sitename="ScatteredArrays.jl",
    authors="Chris Elrod",
    assets=[],
)

deploydocs(;
    repo="github.com/chriselrod/ScatteredArrays.jl",
    target="build",
    julia="1.0",
    deps=nothing,
    make=nothing,
)
