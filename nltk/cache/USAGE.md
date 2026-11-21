# Persistent pipeline cache

NLTK's standard entry points (`sent_tokenize`, `word_tokenize`, `pos_tag`,
`pos_tag_sents`) and any parser wrapped with
`nltk.cache.pipeline.CachedParser` can store their per-input results on disk
and reuse them across processes. An interrupted or repeated batch run over a
large corpus then reuses results instead of re-tokenizing and re-tagging every
document.

The feature is strictly opt-in. Without the `cache` keyword every function
behaves exactly as before, value for value and byte for byte.

## Enabling the cache

The `cache` keyword accepts a `PipelineCache`, a directory path, or `True`
for the default location (`~/.nltk_cache`):

```python
from nltk import word_tokenize, pos_tag
from nltk.cache import PipelineCache

cache = PipelineCache(cache_dir="/var/cache/nltk")

# First run: computes and stores every result.
tagged = pos_tag(word_tokenize(text, cache=cache), cache=cache)

# A later process pointing at the same directory gets a disk hit.
cache2 = PipelineCache(cache_dir="/var/cache/nltk")
assert pos_tag(word_tokenize(text, cache=cache2), cache=cache2) == tagged
```

Passing a plain string is equivalent to constructing a cache in that
directory:

```python
word_tokenize(text, cache="/var/cache/nltk")
```

The same keyword exists on `sent_tokenize`, `word_tokenize`, `pos_tag` and
`pos_tag_sents`. It is keyword-only, so existing positional calls are
unaffected. For parsers, wrap the parser object:

```python
from nltk.cache.pipeline import CachedParser

cached_parser = CachedParser(
    parser, cache, processor_name="chart-en", corpus_version=grammar_path
)
for tree in cached_parser.parse(tokens):
    ...
```

## What makes a cache key

A key is the SHA-256 digest of a JSON descriptor with **three** components; an
entry is reused only when all three match:

1. the input digest -- SHA-256 of the canonicalized text or token list,
2. the processor version -- the NLTK release plus the function name and every
   output-affecting parameter (`language`, `preserve_line`, `lang`, `tagset`),
3. the corpus/model version -- a content digest of the installed Punkt or
   perceptron model files, or an explicit version string.

Replacing a model file changes its content digest, upgrading NLTK changes the
embedded release, and changing a parameter changes the descriptor; in every
case the old entry is simply not matched. A text hash alone is never the key.

Entries live in version-segregated directories, so different versions never
share an entry and can be cleaned independently:

```
<root>/<stage>/<processor-slug>-<hash>/<corpus-slug>-<hash>/<key>.entry
```

## Statistics

`cache.stats_snapshot()` returns a dictionary with `hits`, `misses`,
`skipped` (results that could not be serialized), `corrupt` (bad entries
discarded and recomputed), `degraded` (unwritable cache), `evictions`,
`writes` and `hit_rate`:

```python
cache.stats_snapshot()
# {'hits': 9, 'misses': 6, 'skipped': 0, 'corrupt': 0, 'degraded': 0,
#  'evictions': 0, 'writes': 6, 'hit_rate': 0.6}
```

## Clearing

`PipelineCache.clear` supports three selection modes:

```python
cache.clear()                          # everything
cache.clear(older_than=cutoff)         # entries older than an epoch time
cache.clear(processor_version="...")   # one processor version
cache.clear(corpus_version="...")      # one model/corpus version
```

## Size bound and eviction

`PipelineCache(cache_dir=..., max_size_bytes=...)` bounds total on-disk size
(default 1 GiB; `None` disables the bound). When the bound is exceeded the
least-recently-used entries are evicted down to 90% of it. Reusing an entry
refreshes its timestamp, so frequently reused results survive. A cache
constructed without a bound can still be shrunk on demand:

```python
cache.trim_to_size(max_bytes=256 * 1024 * 1024)  # manual LRU trim
```

## Failure handling

- **Corrupt entry** -- any checksum, magic, format or descriptor mismatch is
  logged at WARNING level, counted in `stats['corrupt']` and recomputed; a
  wrong value is never returned. The embedded full descriptor means even a
  deliberate SHA-256 input collision cannot return another input's result.
- **Unwritable directory** -- one WARNING is logged and the cache degrades to
  pass-through (always recompute, never raise), counted as `degraded`.
- **Non-serializable output** (for example an unregistered `Tree` from a
  parser) -- the result is returned correctly but not stored, counted as
  `skipped`. Token lists and tagged-token tuples are always cacheable.
- **Atomic writes** -- each entry is written to a temporary file and published
  with `os.replace` (plus `fsync`), so a killed process leaves no partial
  entry. An advisory file lock plus an in-process lock guard directory
  maintenance across threads and processes.

## Concurrency, determinism and random seeds

The cache adds no randomness: keys are content digests and stored values are
exact serializations of the deterministic output the processor would have
returned anyway. If the wrapped processor itself uses a random seed (for
example a training run), its output is only as deterministic as that seed;
the cache faithfully stores whatever was computed. Two workers that race to
compute the same key may both compute it (last writer wins via the atomic
replace), but both computed the same function over the same input, so every
reader still observes one identical, valid entry. Parallelism changes *who
computes first*, never the value. There is no shared in-memory result table,
so worker count cannot change a result; only an explicitly seeded (or
seedless) processor matters.

## Costs

A cache miss pays for JSON/base64 encoding, one file write and an `fsync`; a
hit pays for one file read, checksum verification and decoding. Measured on a
20,000-document workload (60 tokens each, `tracemalloc`; numbers vary run to
run with disk scheduling, so a representative range is given):

| Scenario                    | Wall time                | Peak Python heap        |
|-----------------------------|--------------------------|-------------------------|
| No cache (recompute)        | ~37–42 s (baseline)      | 137.8 MiB               |
| Cold cache (compute+write)  | +8% to +30% vs baseline  | 138.1 MiB (+0.3 MiB)    |
| Warm cache (disk reuse)     | ~3.6–4.4x faster         | 188.0 MiB (+~50 MiB)    |

Reproduce with `artifacts/bench/bench_cache.py` (set
`BENCH_HEAVY_ITERATIONS=0` for a deliberately cheap, IO-bound workload where
caching is not a win, to see the crossover). The warm-path heap increase is
the per-entry decode buffer (the JSON and base64 intermediates for one
result, released as soon as the entry is consumed); nothing is retained in
memory between calls. The cache introduces one extra serialized *copy* of
each result on the I/O path and one lock synchronization point per
write/eviction, but no background threads and no in-memory result table. For
very cheap operations (plain regex splitting) the disk round trip can cost
more than recomputation; for tagging and parsing the warm cache is a clear
net win.

## Compatibility with old artifacts

The cache is a new, additive facility: it reads no old outputs and changes no
existing function's return value. Entries written by one NLTK release are not
read by another because the release version is part of every key, so old
cache directories are simply inert after an upgrade -- delete them with
`cache.clear()` or remove the directory. Models and corpora produced by older
NLTK versions remain usable exactly as before; only the *cache entries* are
version-gated, never the underlying data. Cached entries are JSON (not
pickle), so they cannot execute code on load.

## Relation to NLTK's existing caches

`functools.lru_cache` on the tokenizer/tagger constructors and
`nltk.data._resource_cache` memoize *model loading* inside one process. They
hold models in memory, cannot be shared with a new process, and have no notion
of input content or corpus version. The disk cache stores *stage results*,
keyed by input plus processor and corpus versions. The two layers coexist:
models stay memoized in memory, per-document results persist on disk.
