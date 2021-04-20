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

export tariterator
export sampleiterator
export sampleshuffle
export sampletransforms
export samplebatching
export transpose_batch
export default_decoders
export map_by_suffix
export loaderproc
export dataloader

# .tar file reader

function substr(header, from, size)
    lo = from+1
    hi = from+size
    s = String(header[lo:hi])
    if length(s) == 0; return s; end
    return rstrip(s, ['\0'])
end

function decode_header(header)
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

function next_file(stream)
    header = Array{UInt8}(undef, 512)
    read!(stream, header)
    htype, fname, n, blocks = decode_header(header)
    buffer = Array{UInt8}(undef, blocks*512)
    read!(stream, buffer)
    return (htype, fname, buffer[1:n])
end

@resumable function tariterator(stream)
    count = 0
    while(!eof(stream))
        htype, fname, buffer = next_file(stream)
        if htype != "0"; continue; end
        @yield (fname, buffer)
        count = count + 1
    end
end

# grouping tar entries into samples

valid_sample(d) = (d != undef && length(d) > 1)

@resumable function sampleiterator(shards)
    for shard in shards
        stream = open(shard)
        current_prefix = undef
        current_sample = undef
        for (fname, value) in tariterator(stream)
            prefix, suffix = splitext(fname)
            if prefix == ""; continue; end
            if current_sample === undef || prefix != current_prefix
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
end

# shuffling
#
@resumable function sampleshuffle(source, bufsize=1000)
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

function map_by_suffix(sample, transformers)
    result = Dict()
    for (key, value) in sample
        if key[1] == '_'
            result[key] = value
            continue
        end
        for (suffix, f) in transformers
            if endswith(lowercase(key), suffix)
                new_value = f(value)
                if new_value != undef
                    result[key] = new_value
                end
                break
            end
        end
    end
    return result
end

@resumable function sampletransforms(source, transformers)
    for sample in source
        @yield map_by_suffix(sample, transformers)
    end
end

default_decoders = [
    (".cls", data->parse(Int64, String(data))),
    (".jpg", data->ImageMagick.load_(data)),
    (".jpeg", data->ImageMagick.load_(data)),
    (".png", data->ImageMagick.load_(data)),
    (".ppm", data->ImageMagick.load_(data)),
    (".pgm", data->ImageMagick.load_(data)),
    (".pbm", data->ImageMagick.load_(data)),
    (".json", (data->JSON.parse(String(data)))),
    ("", data->undef),
]

# batching

function transpose_batch(sample)
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

@resumable function samplebatching(source, batchsize; minsize=1)
    current_batch = []
    for sample in source
        if length(current_batch) >= batchsize
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

function loaderproc(item, batches::Channel;
        decoders=default_decoders,
        augmentations=[], ntasks=4,
        csize=100, shuffle=1000, batchsize=64)
    iterator = sampleiterator([item])
    decoded = sampletransforms(iterator, decoders)
    augmented = sampletransforms(iterator, augmentations)
    shuffled = sampleshuffle(decoded, shuffle)
    batched = samplebatching(shuffled, batchsize)
    for batch in batched
        put!(batches, batch)
    end
end

end
