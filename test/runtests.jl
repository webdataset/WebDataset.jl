using WebDataset
using Test

@testset "WebDataset.jl" begin
    sampletar = "sample.tar"

    # generic_open
    @test read(generic_open("pipe:echo hello"), String) == "hello\n"
    @test read(generic_open("/dev/null"), String) == ""

    # braceexpand
    @test braceexpand("") == [""]
    @test braceexpand("abc") == ["abc"]
    @test braceexpand("a{1..3}bc") == ["a1bc", "a2bc", "a3bc"]
    @test braceexpand("a{1..2}{1..2}bc") == ["a11bc", "a12bc", "a21bc", "a22bc"]

    # tar file iterator
    @test tariterator(open(sampletar)) |> collect |> length == 180

    # sample iteration
    @test sampleiterator(open(sampletar)) |> collect |> length == 90
    @test sampleiterator(open(sampletar)) |> sampleshuffle |> collect |> length == 90

    # transforms
    l = sampleiterator(open(sampletar)) |> it->sampletransforms(it, default_decoders) |> collect
    @test length(l) == 90
    first = l[1][".img"]
    @test size(first) == (28, 28)

    # batching
    l = sampleiterator(open(sampletar)) |> it->samplebatching(it, 10) |> collect
    @test length(l) == 9
    first = l[1][".png"]
    @test length(first) == 10

    # data loader
    ds = DatasetDescriptor(
        sources=[sampletar, sampletar],
        shuffle=10,
        batchsize=10,
        decoding=default_decoders,
        ntasks=2,
    )
    l = dataloader(ds) |> collect
    @test length(l) == 18
    @test length(l[1][".img"]) == 10
    @test size(l[1][".img"][1]) == (28, 28)
end