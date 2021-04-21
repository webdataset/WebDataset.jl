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
export map_by_rules
export braceexpand
export allsplitext
export loadproc
export dataloader
export debugloader
export DatasetDescriptor
export Rename
export Continue
export clsdecode
export imdecode
export jsondecode
export generic_open

global dataloader_tids = nothing
global last_exception = nothing
global num_exceptions = Threads.Atomic{Int}(0)
global max_exceptions = 999999999

# path names

function allsplitext(s)
    if s == ""; return ("", ""); end
    slash = findlast("/", s)
    slash = slash == nothing ? (0,) : slash
    fname = s[slash[1]+1:length(s)]
    dot = findfirst(".", fname)
    dot = dot == nothing ? (0,) : dot
    split = slash[1]+dot[1]
    return s[1:split-1], s[split:length(s)]
end

# generic I/O

function generic_open(shard; verbose=false)
    if startswith(shard, "pipe:")
        shard = Cmd(Base.shell_split(shard[6:length(shard)]))
    end
    verbose && @show shard
    return open(shard)
end

# brace expansion for filenames

function braceexpand(s::String)
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

function braceexpand(l::Vector{String})
    expanded = braceexpand(l[1])
    if length(l) < 2
        return expanded
    end

end

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

valid_sample(d) = (d != nothing && length(d) > 1)

@resumable function sampleiterator(stream)
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

struct EOF
    key
end

abstract type Renamer end

struct Rename <: Renamer
    key
    new_key
end

struct Continue <: Renamer
    key
    new_key
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

function maprule(key::String, value, rule::String, f)
    !endswith(lowercase(key), rule) && return nothing
    return (key, f(value))
end

function maprule(key::String, value, rule::Regex, f)
    match(rule, lowercase(key)) == nothing && return nothing
    return (key, f(value))
end

function maprule(key::String, value, rule::Renamer, f)
    out = maprule(key, value, rule.key, f)
    out == nothing && return nothing
    return (rule.new_key, out[2])
end

function maprule(key::String, value, rule::Type, f)
    !isa(value, rule) && return nothing
    return (key, f(value))
end


function map_by_rules(sample, transformers)
    result = Dict{String,Any}()
    for (key, value) in sample
        if key[1] == '_'
            result[key] = value
            continue
        end
        for (rule, f) in transformers
            output = maprule(key, value, rule, f)
            output == nothing && continue
            new_key, new_value = output
            result[new_key] = new_value
            !isa(rule, Continue) && break
        end
    end
    return result
end

@resumable function sampletransforms(source, transformers; handle="ignore")
    for sample in source
        result = missing
        try
            result = map_by_rules(sample, transformers)
        catch exn
            @show exn
            global last_exception
            global num_exceptions
            last_exception = exn
            Threads.atomic_add!(num_exceptions, 1)
            if num_exceptions[] > max_exceptions
                error("too many exceptions")
            end
            continue
        end
        @yield result
    end
end

function imdecode(data)
    return ImageMagick.load_(data)
end

function clsdecode(data; lo=-999999, hi=999999)
    value = parse(Int64, String(data))
    if value < lo || value > hi
        error("cls value out of range")
    end
    return value
end

function jsondecode(data; query="")
    query == "" || error("unimplemented")
    return JSON.parse(String(data))
end

default_decoders = [
    ".cls" => clsdecode,
    Rename(r".(jpg|jpg|png|p?m)$", ".img") => imdecode,
    ".json" => jsondecode,
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

@resumable function ldebug(it)
    for sample in it
        @show sample["__key__"], string(sample)[1:100]
        @yield sample
    end
end


function loadproc(desc::DatasetDescriptor, inch::Channel, outch::Channel, eof; maxcount=1e30)
    count = 0
    for source in inch
        desc.debug && @show source
        stream = generic_open(source)
        raw = sampleiterator(stream)
        shuffled = sampleshuffle(raw, desc.shuffle)
        decoded = sampletransforms(shuffled, desc.decoding)
        augmented = sampletransforms(decoded, desc.augmenting)
        mapped = (desc.mapping(x) for x in augmented)
        batched = samplebatching(mapped, desc.batchsize)
        collated = sampletransforms(batched, desc.collating)
        for batch in collated
            desc.debug && @show batch["__key__"]
            put!(outch, batch)
            sleep(0.0001)
            count += 1
            if count > maxcount; return; end
        end
        close(stream)
    end
    put!(outch, eof)
end

function checkfailed(tids)
    for t in tids
        if istaskfailed(t)
            wait(t)
        end
    end
end

@resumable function dataloader(desc::DatasetDescriptor, inch::Channel)
    global num_exceptions
    global dataloader_tids
    Threads.atomic_xchg!(num_exceptions, 0)
    outch = Channel(desc.csize)
    tids = [Threads.@spawn loadproc(desc, inch, outch, EOF(i)) for i in 1:desc.ntasks]
    dataloader_tids = tids
    running = Set(1:desc.ntasks)
    while true
        checkfailed(tids)
        tids = [t for t in tids if !istaskdone(t)]
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
    fnames = desc.sources
    sources = Channel(length(fnames))
    for fname in fnames
        put!(sources, fname)
    end
    close(sources)
    return dataloader(desc, sources)
end

function debugloader(desc::DatasetDescriptor, sources; maxcount=10)
    inch = Channel(length(sources))
    for s in sources; put!(inch, s); end
    close(inch)
    outch = Channel(min(maxcount+1, 100))
    loadproc(desc, inch, outch, EOF(0); maxcount=maxcount)
    return outch
end

end
