module WebDataset

using TarIterators
using Images
using FileIO, ImageIO, ImageMagick
using Transducers
using Transducers: Transducer, R_, next, inner, xform
using HTTP
using JSON
using ResumableFunctions
using Base.Iterators
using Test
using Base.Threads
using Parameters

export tariterator
export sampleiterator
export sampleshuffle
export sampletransforms
export samplebatching
export transpose_batch
export default_decoders
export map_by_suffix
export braceexpand
export allsplitext
export loadproc
export dataloader
export debugloader
export DatasetDescriptor
export Rename
export Continue

global dataloader_tids

# path names

@noinline function allsplitext(s)
    if s == ""; return ("", ""); end
    slash = findlast("/", s)
    slash = slash == nothing ? (0,) : slash
    fname = s[slash[1]+1:length(s)]
    dot = findfirst(".", fname)
    dot = dot == nothing ? (0,) : dot
    split = slash[1]+dot[1]
    return s[1:split-1], s[split:length(s)]
end

# brace expansion for filenames

@noinline function braceexpand(s)
    expansion = r"[{][0-9]+[.][.][0-9]+[}]"
    m = match(expansion, s)
    if m == nothing
        return [s]
    end
    prefix = s[1:m.offset-1]
    rest = braceexpand(s[m.offset+length(m.match):length(s)])
    range = s[m.offset+1:m.offset+length(m.match)-2]
    lohi = split(range, "..")
    (lo, hi) = [parse(Int, x) for x in lohi]
    result = []
    for i in lo:hi
        for r in rest
            expanded = prefix * string(i, pad=length(lohi[1])) * r
            push!(result, expanded)
        end
    end
    return result
end

# .tar file reader

@noinline function substr(header, from, size)
    lo = from+1
    hi = from+size
    s = String(header[lo:hi])
    if length(s) == 0; return s; end
    return rstrip(s, ['\0'])
end

@noinline function decode_header(header)
    htype = substr(header, 156, 1)
    fname = substr(header, 0, 100)
    prefix = substr(header, 345, 155)
    raw_size = substr(header, 124, 12)
    size = 0
    @test !occursin("\0", raw_size)
    if raw_size != ""; size = parse(Int64, raw_size, base=8); end
    blocks = Int(floor((size + 511) / 512))
    return (htype, prefix*fname, size, blocks)
end

@noinline function next_file(stream)
    header = Array{UInt8}(undef, 512)
    read!(stream, header)
    htype, fname, n, blocks = decode_header(header)
    buffer = Array{UInt8}(undef, blocks*512)
    read!(stream, buffer)
    return (htype, fname, buffer[1:n])
end

@noinline @resumable function tariterator(stream)
    count = 0
    while(!eof(stream))
        htype, fname, buffer = next_file(stream)
        if htype != "0"; continue; end
        @yield (fname, buffer)
        count = count + 1
    end
end

# grouping tar entries into samples

valid_sample(d) = (d != nothing && length(d) > 1)

@noinline @resumable function sampleiterator(shard::String)
    stream = open(shard)
    current_prefix = nothing
    current_sample = nothing
    for (fname, value) in tariterator(stream)
        (prefix, suffix) = allsplitext(fname)
        if prefix == ""; continue; end
        if current_sample === nothing || prefix != current_prefix
            if valid_sample(current_sample)
                @yield current_sample
            end
            current_sample = Dict{String,Any}("__key__"=>prefix)
            current_prefix = prefix
        end
        current_sample[suffix] = value
    end
    if valid_sample(current_sample)
        @yield current_sample
    end
end

# shuffling

@noinline @resumable function sampleshuffle(source, bufsize=1000)
    buffer = []
    iter = iterate(source)
    while iter != nothing
        if length(buffer) < bufsize
            push!(buffer, iter[1])
            iter = iterate(source, iter[2])
        end
        iter == nothing && break
        index = rand(UInt64) % length(buffer) + 1
        result = buffer[index]
        buffer[index] = iter[1]
        @yield result
        iter = iterate(source, iter[2])
    end
    while length(buffer) > 0
        @yield pop!(buffer)
    end
end

# decoding and augmentation

struct EOF
    key
end

struct Rename
    key
    value
end

struct Continue
    key
    value
end


function kv_apply(f, key, value)
    try
        return f(value)
    catch e
        if e isa MethodError
            return f(key, value)
        end
        rethrow()
    end
end

@noinline function map_by_suffix(sample, transformers)
    result = Dict()
    for (key, value) in sample
        if key[1] == '_'
            result[key] = value
            continue
        end
        for (suffix, f) in transformers
            if endswith(lowercase(key), suffix)
                out = kv_apply(f, key, value)
                if out isa Rename
                    result[out.key] = out.value
                    break
                elseif out isa Continue
                    result[out.key] = out.value
                else
                    result[key] = out
                    break
                end
            end
        end
    end
    return result
end

@noinline @resumable function sampletransforms(source, transformers)
    for sample in source
        @yield map_by_suffix(sample, transformers)
    end
end

default_decoders = [
    ".cls" => data->parse(Int64, String(data)),
    ".jpg" => data->ImageMagick.load_(data),
    ".jpeg" => data->ImageMagick.load_(data),
    ".png" => data->ImageMagick.load_(data),
    ".ppm" => data->ImageMagick.load_(data),
    ".pgm" => data->ImageMagick.load_(data),
    ".pbm" => data->ImageMagick.load_(data),
    ".json" => data->JSON.parse(String(data)),
]

# batching

@noinline function transpose_batch(sample)
    keys = Set{String}(key for d in sample for (key, _) in d)
    result = Dict{String,Any}()
    for key in keys
        result[key] = [d[key] for d in sample]
    end
    if length(Set(length(l) for (_, l) in result)) != 1
        throw(error("not all keys are present in all samples"))
    end
    return result
end

@noinline @resumable function samplebatching(source, batchsize; minsize=1)
    current_batch = []
    for sample in source
        if batchsize == 0
            @yield sample
        elseif length(current_batch) >= batchsize
            @yield transpose_batch(current_batch)
            current_batch = []
        else
            push!(current_batch, sample)
        end
    end
    if length(current_batch) >= minsize
        @yield transpose_batch(current_batch)
    end
end

# multithreaded loader

@with_kw mutable struct DatasetDescriptor
    sources::Array{String} = []
    shuffle::Int = 1000
    batchsize::Int = 16
    decoding::Array{Pair{String,Function}} = [""=>x->x]
    augmenting::Array{Pair{String,Function}} = [""=>x->x]
    collating::Array{Pair{String,Function}} = [""=>x->x]
    csize::Int = 100
    ntasks::Int = 4
    verbose::Bool = false
    debug::Bool = false
end

@noinline function loadproc(desc::DatasetDescriptor, inch::Channel, outch::Channel, eof; maxcount=1e30)
    count = 0
    for source in inch
        desc.debug && @show source
        raw = sampleiterator(source)
        shuffled = sampleshuffle(raw, desc.shuffle)
        decoded = sampletransforms(shuffled, desc.decoding)
        augmented = sampletransforms(decoded, desc.augmenting)
        batched = samplebatching(augmented, desc.batchsize)
        collated = sampletransforms(batched, desc.collating)
        for batch in collated
            desc.debug && @show batch["__key__"]
            put!(outch, batch)
            sleep(0.0001)
            count += 1
            if count > maxcount; return; end
        end
    end
    put!(outch, eof)
end

@noinline @resumable function dataloader(desc::DatasetDescriptor, inch::Channel)
    global dataloader_tids
    outch = Channel(desc.csize)
    dataloader_tids = [Threads.@spawn loadproc(desc, inch, outch, EOF(i)) for i in 1:desc.ntasks]
    running = Set(1:desc.ntasks)
    while true
        sample = take!(outch)
        if isa(sample, EOF)
            delete!(running, sample.key)
            length(running) == 0 && break
            continue
        end
        @yield sample
    end
end

function dataloader(desc::DatasetDescriptor)
    fnames::Array{String} = []
    for d in desc.sources
        append!(fnames, braceexpand(d))
    end
    sources = Channel(length(fnames))
    for fname in fnames
        put!(sources, fname)
    end
    close(sources)
    return dataloader(desc, sources)
end

@noinline function debugloader(desc::DatasetDescriptor, sources; maxcount=10)
    inch = Channel(length(sources))
    for s in sources; put!(inch, s); end
    close(inch)
    outch = Channel(min(maxcount+1, 100))
    loadproc(desc, inch, outch, EOF(0); maxcount=maxcount)
    return outch
end

end
