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

export tariterator
export tar_stage
export counted
export default_decoders, default_preproc, default_collation
export collate, rename, transform
export itemkey, make_sample, dv_tranpose

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

function itemkey(item)
    return splitext(item[1])[1]
end

function make_sample(items)
    result = Dict{String,Any}()
    for (path, value) in items
        if value == undef; continue; end
        if path == ""; continue; end
        (n, e) = splitext(path)
        result[e] = value
        result["__key__"] = n
    end
    return result
end

function tar_stage(inch, outch; decoders=default_decoders, maxcount=1e30)
    count = 0
    while true
        shard = take!(inch)
        @show shard
        stream = open(shard)
        @show stream
        # foreach(tariterator(stream) |> PartitionBy(itemkey) |> Map(make_sample)) do sample
        foreach(tariterator(stream)) do sample
            if count < 5;
                msg = (string(sample)*" "^100)[1:100]
                @show count, msg
            end
            put!(outch, sample)
            count += 1
            if count > maxcount; error("stop"); end
        end
        close(stream)
        if count > maxcount; break; end
    end
end

function rename1(x, l)
    for a in l
        if a == x
            return l[1], true
        end
    end
    return x, false
end

function rename(sample, renames)
    result = Dict()
    for (key, value) in sample
        if key[1] == '_'; result[key] = value; continue; end
        for r in renames
            (key, found) = rename1(key, r)
            if found; break; end
        end
        result[key] = value
    end
    return result
end


function transform(sample, transformers)
    result = Dict()
    for (key, value) in sample
        if key[1] == '_'; result[key] = value; continue; end
        for (suffix, f) in transformers
            if endswith(lowercase(key), suffix)
                (new_key, new_value) = f(value)
                if new_value != undef
                    result[key] = new_value
                end
                break
            end
        end
    end
    return result
end

function dv_transpose(sample)
    keys = Set(key for d in sample for (key, _) in d)
    result = Dict()
    for key in keys
        result[key] = [d[key] for d in sample]
    end
    if length(Set(length(l) for (_, l) in result)) != 1
        throw(error("not all keys are present in all samples"))
    end
    return result
end


function collate_images_strict(l)
    l = [channelview(a) for a in l]
    shapes = [x for x in Set(map(size, l))]
    if length(shapes) != 1
        error("collate_images_strict: inconsistent shapes: "*string(shapes))
    end
    shape = shapes[1]
    result = zeros(eltype(l[1]), shape...)
    for a in l
        slice = map(hi->1:hi, size(a))
        result[slice...] = a
    end
    return result
end


function collate_images_expand(l)
    l = [channelview(a) for a in l]
    shape = reduce((x,y)->max.(x,y), map(size, l))
    result = zeros(eltype(l[1]), shape...)
    for a in l
        slice = map(hi->1:hi, size(a))
        result[slice...] = a
    end
    return result
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

default_renames = [
    [".img", ".jpg", ".jpg", ".png", ".ppm", ".pgm", ".pbm"],
]

default_collation = [
    (".jpg", collate_images_strict),
    (".cls", Vector),
]

default_preproc = [
    (".jpg", image->(RGB.(image))[1:30, 1:30])
    ("", x->x)
]

counted(x) = @time count(_->true, x)

end
