using WebDataset
using Documenter

DocMeta.setdocmeta!(WebDataset, :DocTestSetup, :(using WebDataset); recursive=true)

makedocs(;
    modules=[WebDataset],
    authors="Thomas Breuel",
    repo="https://github.com/ThomasBreuel/WebDataset.jl/blob/{commit}{path}#{line}",
    sitename="WebDataset.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ThomasBreuel.github.io/WebDataset.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ThomasBreuel/WebDataset.jl",
)
