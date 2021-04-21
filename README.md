
# Operation

WebDatasets consists of a collection of tar files ("shards"). Each tar file is read sequentially, and multiple shards may be read in parallel.

Within each tar file, files with a common basename but different extensions are grouped together into training samples. The "basename" is defined here as the directory name plus the file name up to the first "." in the file name.

Training samples for learning stereo models might contain a sequence of samples of the form:

```
098953.left.jpg
098953.right.jpg
098953.depth.png16
194432.left.jpg
194432.right.jpg
194432.depth.png16
```

Training samples are represented as `Dict{String,Any}` instances in the library.

Dataset loading takes place in multiple stages:

```
tar file reading -> grouping into samples -> decoding of samples -> augmentation of samples -> general mapping -> batching -> collating
```

The `dataloader` function will set up such a loading pipeline and run it in parallel in multiple threads. The entire process is described by a dataset descriptor:

```
@with_kw mutable struct DatasetDescriptor
    # list of shards
    sources::Array{Union{String,Cmd}} = []
    # size of the inline shuffle buffer
    shuffle::Int = 1000
    # batchsize (0 for no batching)
    batchsize::Int = 16
    # decoding functions
    decoding::Array{Pair{Union{String,Regex,Renamer},Function}} = [""=>x->x]
    # augmentation functions
    augmenting::Array{Pair{Union{String,Regex,Renamer},Function}} = [""=>x->x]
    # a general mapping function running after augmentation
    mapping::Function = x->x
    # collation functions running after batching
    collating::Array{Pair{Union{String,Regex,Renamer},Function}} = [""=>x->x]
    # channel size for output channel
    csize::Int = 100
    # number of concurrent tasks/shards for loading
    ntasks::Int = 4
    # more verbose output
    verbose::Bool = false
    # even more verbose output
    debug::Bool = false
end
```

Here, `decoding`, `augmenting`, and `collating` are lists of mapping rules that are searched sequentially for a matching rule.  The default decoders are very simple:


```
default_decoders = [
    ".cls" => clsdecode,
    Rename(r".(jpg|jpg|png|p?m)$", ".img") => imdecode,
    ".json" => jsondecode,
]
```

This says that every file ending in `.cls` should be decoded with the `clsdecode` function and every file ending in `.json` should be decoded using `jsondecode`. Files matching the regular expression should have their matching portion renamed to `.img` and should be decoded using the `imdecode` function.

Decoder functions should take `UInt8` arrays containing the binary file content and decode into whatever data structure is desired. Augmentation functions take the decoded outputs as inputs and produce new outputs. Collating functions take a list of decoded items (e.g., all files ending in `.img`) and can, for example, stack them together into arrays.

# Simple Example

In this example, we start by defining a simple `DatasetDescriptor`; we're using the default decoders.


```julia
using Pkg; Pkg.add("Flux")
```

    [32m[1m    Updating[22m[39m registry at `~/.julia/registries/General`
    [32m[1m    Updating[22m[39m git-repo `https://github.com/JuliaRegistries/General.git`
    [32m[1m   Resolving[22m[39m package versions...
    [32m[1m  No Changes[22m[39m to `~/proj/WebDataset.jl/Project.toml`
    [32m[1m  No Changes[22m[39m to `~/proj/WebDataset.jl/Manifest.toml`



```julia
using WebDataset
using Images
using Flux: batch

function image_augmentation(image)
    # generate a fixed size output image for batching here
    # here, we just return a mock image for demonstration
    result = zeros(Float32, 3, 256, 256)
    return result
end

augmenting = [
    # call whatever augmentation you like for individual images
    ".img" => image_augmentation,
    # this line ensures that any other sample components are carried along unchanged
    "" => x->x
]

collating = [
    # .cls is turned  into an Array{Int}
    ".cls" => data->Array{Int}(data),
    # RGB images are batched using Flux.batch
    ".img" => l->batch(l),
]

shards = braceexpand("pipe:curl -L -s http://storage.googleapis.com/nvdata-coco/coco-train2014-seg-{000000..000003}.tar")
shards = braceexpand("/work-2020/shards/imagenet/imagenet-train-{000000..000146}.tar")[1:4]

desc = DatasetDescriptor(
    sources=shards,
    shuffle=1000,
    batchsize=32,
    decoding=default_decoders,
    augmenting=augmenting,
    collating=collating,
)
```




    DatasetDescriptor
      sources: Array{Union{Cmd, String}}((4,))
      shuffle: Int64 1000
      batchsize: Int64 32
      decoding: Array{Pair{Union{Regex, String, WebDataset.Renamer}, Function}}((3,))
      augmenting: Array{Pair{Union{Regex, String, WebDataset.Renamer}, Function}}((2,))
      mapping: #23 (function of type WebDataset.var"#23#33")
      collating: Array{Pair{Union{Regex, String, WebDataset.Renamer}, Function}}((2,))
      csize: Int64 100
      ntasks: Int64 4
      verbose: Bool false
      debug: Bool false




Here is a non-threaded loader that's useful for debugging.


```julia
if false
    ch = debugloader(desc, desc.sources[1:2])
    sample = take!(ch)
    @show size(sample[".img"])
end;
```

This is the multi-threaded loader used for actual training. It loads samples in parallel but delivers them as a simple iterator.


```julia
desc.ntasks = 4
(count, total) = (0, 0)
global last_sample
@time for sample in dataloader(desc)
    last_sample = sample
    count += 1
    total += length(sample["__key__"])
    if count % 100 == 1; @show count, total; end
end
(count, total)
```

    (count, total) = (1, 32)
    (count, total) = (101, 3232)
    (count, total) = (201, 6432)
    (count, total) = (301, 9632)
    (count, total) = (401, 12832)
    (count, total) = (501, 16032)
    (count, total) = (601, 19232)
    (count, total) = (701, 22432)
    (count, total) = (801, 25632)
    (count, total) = (901, 28832)
    (count, total) = (1001, 32032)
     73.420912 seconds (19.44 M allocations: 121.839 GiB, 2.16% gc time, 0.14% compilation time)





    (1052, 33607)



We are getting a nicely batched sample with additional information about sample keys.


```julia
last_sample
```




    Dict{String, Any} with 3 entries:
      "__key__" => ["0010656", "0937756", "0729658", "0587807", "0307037", "0965382", "1158354", "0432406", "0…
      ".img"    => Float32[0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]…
      ".cls"    => [8, 732, 568, 457, 240, 753, 904, 337, 430, 53, 462, 818, 436, 554, 609, 577, 1, 540, 981]



Four Julia threads give us about 450 decoded images per second. The code scales completely linearly, though Julia multithreading seems to have some overhead when running many threads in parallel.


```julia
33607/73.4
```




    457.86103542234326



Exceptions in the multithreaded code are available here.


```julia
WebDataset.num_exceptions, WebDataset.last_exception
```




    (Base.Threads.Atomic{Int64}(0), nothing)



# Under the Hood

The `dataloader` function gives you one particular way of multithreaded data loading and augmentation. It is built from a number of simple iterators that you can reuse and recombine in other ways if you like. You can also use distributed computing for distributed loading and augmentation. The core functionality is expressed by this composition of iterators:

```Julia
# open either a file or a pipe: source
stream = generic_open(source)

# iterate over samples in the .tar archive
raw = sampleiterator(stream)

# shuffle samples with a shuffle buffer
shuffled = sampleshuffle(raw, desc.shuffle)

# decode the samples based on decoding rules
decoded = sampletransforms(shuffled, desc.decoding)

# augment the samples based on agumentation rules
augmented = sampletransforms(decoded, desc.augmenting)

# batch the samples to the given batch size
batched = samplebatching(augmented, desc.batchsize)

# collate individual fields based on collating rules
collated = sampletransforms(batched, desc.collating)
```



```julia

```
