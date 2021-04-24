using WebDataset
using Test

@testset "WebDataset.jl" begin
    # generic_open
    @test read(generic_open("pipe:echo hello"), String) == "hello\n"
    @test read(generic_open("/dev/null"), String) == ""

    # braceexpand
    @test braceexpand("") == [""]
    @test braceexpand("abc") == ["abc"]
    @test braceexpand("a{1..3}bc") == ["a1bc", "a2bc", "a3bc"]
    @test braceexpand("a{1..2}{1..2}bc") == ["a11bc", "a12bc", "a21bc", "a22bc"]

    # tar file iterator
    @test tariterator(open("test/sample.tar")) |> collect |> length == 180

    # sample iteration
    @test sampleiterator(open("test/sample.tar")) |> collect |> length == 90
    @test sampleiterator(open("test/sample.tar")) |> sampleshuffle |> collect |> length == 90

    # transforms
    l = sampleiterator(open("test/sample.tar")) |> it->sampletransforms(it, default_decoders) |> collect
    @test length(l) == 90
    first = l[1][".img"]
    @test size(first) == (28, 28)

    # batching
    l = sampleiterator(open("test/sample.tar")) |> it->samplebatching(it, 10) |> collect
    @test length(l) == 9
    first = l[1][".png"]
    @test length(first) == 10

    # data loader
    ds = DatasetDescriptor(
        sources=["test/sample.tar", "test/sample.tar"],
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