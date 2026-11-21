# Natural Language Toolkit: persistent disk cache for pipeline results
#
# Copyright (C) 2001-2026 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Persistent, version-aware disk cache for NLTK pipeline stages.

This module adds a *cross-process* cache for the intermediate results of the
text-processing pipeline (sentence/word tokenization, part-of-speech tagging,
and arbitrary parsing/chunking stages). It is strictly opt-in: no public NLTK
function changes behaviour unless a cache is supplied through the new
``cache=`` keyword.

How it differs from the caches NLTK already has
-----------------------------------------------
NLTK already memoizes *model construction* inside a process
(``functools.lru_cache`` on ``_get_punkt_tokenizer`` / ``_get_tagger``) and
keeps loaded resources in ``nltk.data._resource_cache``. Those caches hold
loaded models, live in process memory, and evaporate when the process exits.
Re-running a million-document batch therefore re-tokenizes and re-tags every
document from scratch. The :class:`PipelineCache` in this module instead stores
*per-input stage outputs* on disk, keyed by the input together with the
processor and corpus versions, so a later process reuses them. The two layers
compose: model loading stays in memory, stage results live on disk.

Cache keys
----------
A key never consists of the text hash alone. It is the SHA-256 digest of a
JSON descriptor containing three components, and a new entry is used whenever
any one of them changes:

1. the input (``text_hash`` -- SHA-256 of the canonicalized input),
2. the processor version (code/algorithm/parameters, e.g.
   ``"nltk-3.10.4|word_tokenize|preserve_line=False|lang=english"``),
3. the corpus/model version (a hash of the loaded model file, or a caller
   supplied version string).

Storage layout
--------------
::

    <root>/<stage>/<processor-slug>/<processor-version-hash>/
        <corpus-slug>/<corpus-version-hash>/<input-hash>.entry

Version components therefore get their own directories: different processor or
corpus versions never share an entry and can be cleaned independently. Each
entry is written through a temporary file in the same directory and published
with :func:`os.replace`, so a killed process never leaves a half-written
entry. Entries embed their full key descriptor and a payload checksum; any
mismatch (including a deliberately constructed SHA-256 collision on the input
hash) is detected and the entry is recomputed.

Degradation
-----------
If the cache root is not writable, the cache logs one warning and silently
degrades to pass-through (never caching, always recomputing). A corrupt or
unreadable entry is skipped, a warning is logged, and the value is recomputed.
"""

import base64
import contextlib
import datetime
import hashlib
import json
import logging
import os
import pathlib
import shutil
import threading
import uuid

from nltk import __version__ as NLTK_VERSION
from nltk import jsontags
from nltk.jsontags import safe_json_loads

logger = logging.getLogger(__name__)

#: Magic prefix marking a cache entry file; also carries the on-disk format.
ENTRY_MAGIC = "NLTKCACHE1"
#: Size cap (bytes) for one entry payload; large results bypass the cache.
MAX_ENTRY_BYTES = 64 * 1024 * 1024
#: Default upper bound on total cache size (1 GiB).
DEFAULT_MAX_SIZE_BYTES = 1024 * 1024 * 1024
#: Fraction of ``max_size_bytes`` that eviction shrinks the cache back to.
EVICTION_TARGET_RATIO = 0.9
#: Tolerance (bytes) before an asynchronous size check triggers eviction.
_SIZE_JITTER = 1024 * 1024

# Encode an explicit NLTK version once. Stage descriptors embed this so that
# installing a new NLTK release invalidates every previously written entry.
# (NLTK_VERSION is assigned above from ``nltk.__version__``.)


def _canonical_input(value):
    """
    Canonical JSON rendering of a stage input.

    Lists/tuples of strings (token lists, tagged-token lists) and plain strings
    are the expected inputs. Sorting is disabled (order is meaningful in a
    sentence); ``ensure_ascii=False`` keeps Unicode stable across platforms.
    """
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    )


def _sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes_hex(data):
    return hashlib.sha256(data).hexdigest()


# Tuple preservation ------------------------------------------------------
# Plain JSON (and jsontags) turns tuples into lists, but several NLTK stage
# outputs are contractually tuples -- ``pos_tag`` returns ``list(tuple)``.
# Rather than mutating the global jsontags registry (which models user-facing
# objects, not this storage format), the payload codec wraps tuples in a
# marked one-key dict so the round trip is exact and byte-stable.
# The marker deliberately does NOT start with "!": jsontags claims that
# namespace and its decoder raises on unknown "!"-prefixed one-key dicts.
_TUPLE_MARKER = "#nltk.cache.tuple#"


def _encode_tuples(obj):
    if isinstance(obj, tuple):
        return {_TUPLE_MARKER: [_encode_tuples(item) for item in obj]}
    if isinstance(obj, list):
        return [_encode_tuples(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _encode_tuples(value) for key, value in obj.items()}
    return obj


def _decode_tuples(obj):
    if isinstance(obj, dict) and len(obj) == 1 and _TUPLE_MARKER in obj:
        return tuple(_decode_tuples(item) for item in obj[_TUPLE_MARKER])
    if isinstance(obj, dict):
        return {key: _decode_tuples(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_decode_tuples(item) for item in obj]
    return obj


def dump_payload(result):
    """Serialize a stage result, preserving tuple structure."""
    return jsontags.JSONTaggedEncoder().encode(_encode_tuples(result))


def load_payload(text):
    """Inverse of :func:`dump_payload`.

    Tuple markers are applied *after* jsontags decoding: jsontags treats any
    ``"!"``-prefixed one-key dict as its own tag namespace, so letting its
    decoder see our marker first would raise "Unknown tag". The encoder writes
    plain JSON dicts for markers, which survive a jsontags decode untouched.
    """
    return _decode_tuples(jsontags.JSONTaggedDecoder().decode(text))


# Allowed leaf / container types for cacheable results. Anything outside this
# set (e.g. an unregistered list subclass such as ``nltk.tree.Tree``) would be
# silently flattened to a plain list by the JSON encoder and come back as the
# wrong type -- a correctness violation, not merely a cache miss. Such results
# are refused up front and simply not cached.
_SCALAR_TYPES = (str, int, float, bool, type(None))


def _verify_cacheable(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        raise TypeError("circular reference in cache result")
    if isinstance(obj, _SCALAR_TYPES):
        return
    if isinstance(obj, (tuple, list, dict)):
        # Reject subclasses unless jsontags knows how to round-trip them:
        # plain json.dumps would strip the subclass silently.
        if type(obj) not in (tuple, list, dict) and not hasattr(
            type(obj), "json_tag"
        ):
            raise TypeError(
                f"result contains non-serializable type {type(obj).__name__!r}; "
                "caching this stage output is not supported"
            )
        seen.add(obj_id)
        if isinstance(obj, dict):
            for key, value in obj.items():
                _verify_cacheable(key, seen)
                _verify_cacheable(value, seen)
        else:
            for item in obj:
                _verify_cacheable(item, seen)
        seen.discard(obj_id)
        return
    if not hasattr(type(obj), "json_tag"):
        raise TypeError(
            f"result contains non-serializable type {type(obj).__name__!r}; "
            "caching this stage output is not supported"
        )


def _slugify(text, fallback="x"):
    """Filesystem-safe rendering of a version string for directory names."""
    cleaned = "".join(
        ch if (ch.isalnum() or ch in ".-_") else "_" for ch in str(text)
    ).strip(".-_")
    return (cleaned or fallback)[:64]


def resource_version(resource_path):
    """
    Derive a stable corpus/model version string from a resource on disk.

    The version combines the absolute path, file size and SHA-256 content
    digest, so replacing the model file (same path, different bytes) changes
    the version even when the caller did not pass an explicit version. The
    digest of an already loaded file object is computed from its current
    position; callers normally pass a path.
    """
    if hasattr(resource_path, "read"):
        stream = resource_path
        stream.seek(0)
        digest = hashlib.sha256()
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            digest.update(chunk)
        return f"stream:sha256:{digest.hexdigest()}"
    path = os.path.abspath(os.fspath(resource_path))
    h = hashlib.sha256()
    h.update(path.encode("utf-8"))
    size = os.path.getsize(path)
    with open(path, "rb") as fh:  # sandboxed-open ok: caller-supplied model path (operator-chosen)
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return f"{path}:{size}:sha256:{h.hexdigest()}"


class CacheStats:
    """Mutable hit/miss bookkeeping for one :class:`PipelineCache`."""

    __slots__ = (
        "hits",
        "misses",
        "skipped",
        "corrupt",
        "degraded",
        "evictions",
        "writes",
    )

    def __init__(self):
        #: Number of requests served from a verified cache entry.
        self.hits = 0
        #: Number of requests recomputed and stored as fresh entries.
        self.misses = 0
        #: Number of recomputations whose result could not be cached
        #: (non-serializable output or over-sized payload).
        self.skipped = 0
        #: Number of corrupt entries skipped and recomputed.
        self.corrupt = 0
        #: Number of times an unwritable cache forced pass-through.
        self.degraded = 0
        #: Number of entries removed by size-limit eviction.
        self.evictions = 0
        #: Number of fresh entries written.
        self.writes = 0

    def as_dict(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "skipped": self.skipped,
            "corrupt": self.corrupt,
            "degraded": self.degraded,
            "evictions": self.evictions,
            "writes": self.writes,
            "hit_rate": (
                self.hits / (self.hits + self.misses)
                if (self.hits + self.misses)
                else 0.0
            ),
        }

    def reset(self):
        self.hits = self.misses = self.skipped = self.corrupt = 0
        self.degraded = self.evictions = self.writes = 0

    def __repr__(self):
        return f"CacheStats({self.as_dict()})"


class PipelineCache:
    """
    A persistent, version-aware, size-bounded disk cache for stage results.

    The cache is safe to share between threads *and* processes: every write is
    atomic (temp file + :func:`os.replace`) and directory operations are
    guarded by an advisory lock file. Concurrent first-time computations of
    the same key may duplicate work, but they never corrupt the cache and the
    result is always deterministic -- last writer wins, and every writer
    computed the same function over the same input.

    :param cache_dir: Root directory of the cache. Defaults to
        ``~/.nltk_cache``. Created on first use.
    :type cache_dir: str or os.PathLike
    :param max_size_bytes: Upper bound on total on-disk size. When exceeded,
        least-recently-used entries are evicted down to
        ``EVICTION_TARGET_RATIO`` of the bound. ``None`` disables the bound.
    :type max_size_bytes: int or None
    """

    def __init__(self, cache_dir=None, max_size_bytes=DEFAULT_MAX_SIZE_BYTES):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".nltk_cache")
        self.cache_dir = os.path.abspath(os.fspath(cache_dir))
        self.max_size_bytes = max_size_bytes
        self.stats = CacheStats()
        self._lock = threading.RLock()
        self._known_writable = None
        #: Approximate current size; reconciled against the filesystem on
        #: eviction, so it cannot drift far enough to defeat the bound.
        self._approx_size = 0
        self._size_dirty = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_or_compute(
        self,
        stage,
        processor_version,
        corpus_version,
        input_value,
        compute_fn,
    ):
        """
        Return the cached result of ``compute_fn(input_value)`` or compute it.

        :param stage: Pipeline stage name, e.g. ``"tokenize"``, ``"tag"``,
            ``"parse"``. Becomes the first path component of the entry.
        :type stage: str
        :param processor_version: Full processor descriptor (name, code
            version and every parameter that can change the output).
        :type processor_version: str
        :param corpus_version: Corpus/model descriptor. Use
            :func:`resource_version` for a model file, or any explicit string.
        :type corpus_version: str
        :param input_value: The stage input (str, or nested tuples/lists of
            strings). Must be JSON-serializable.
        :param compute_fn: Zero-argument callable producing the result.
        :return: The cached or freshly computed result, always equal to
            ``compute_fn()``.
        """
        descriptor = self._build_descriptor(
            stage, processor_version, corpus_version, input_value
        )
        entry_path = self._entry_path(descriptor)

        # Degradation is accounted exactly once per request, here at the entry
        # point, so the read and write helpers do not both count it.
        if not self._ensure_writable():
            self.stats.degraded += 1
            return compute_fn()

        cached = self._read_entry(entry_path, descriptor)
        if cached is not _MISS:
            self.stats.hits += 1
            return cached

        result = compute_fn()
        if self._write_entry(entry_path, descriptor, result):
            self.stats.misses += 1
        else:
            # Unserializable/oversized result, or a write failure detected
            # after the initial writability probe.
            self.stats.skipped += 1
        return result

    def clear(self, older_than=None, processor_version=None, corpus_version=None):
        """
        Remove cache entries. Three independent selection modes:

        - ``older_than``: an epoch timestamp (float) or a
          :class:`~datetime.datetime`; entries last modified before it are
          removed.
        - ``processor_version`` / ``corpus_version``: exact version strings;
          every entry belonging to that version directory is removed.

        With no arguments the whole cache root is removed. Returns the number
        of entry files deleted.
        """
        with self._locked():
            if not os.path.isdir(self.cache_dir):
                return 0
            if processor_version is None and corpus_version is None and older_than is None:
                removed = self._count_entries(self.cache_dir)
                shutil.rmtree(self.cache_dir, ignore_errors=True)
                self._size_dirty = True
                return removed

            cutoff = None
            if older_than is not None:
                if isinstance(older_than, datetime.datetime):
                    cutoff = older_than.timestamp()
                else:
                    cutoff = float(older_than)

            pv_dir = (
                f"{_slugify(processor_version)}-{_sha256_hex(processor_version)[:12]}"
                if processor_version is not None
                else None
            )
            cv_dir = (
                f"{_slugify(corpus_version)}-{_sha256_hex(corpus_version)[:12]}"
                if corpus_version is not None
                else None
            )

            removed = 0
            for dirpath, _dirnames, filenames in os.walk(self.cache_dir):
                # pathlib normalizes both "/" and os.sep, including Windows.
                parts = pathlib.Path(dirpath).parts
                if pv_dir is not None and pv_dir not in parts:
                    continue
                if cv_dir is not None and cv_dir not in parts:
                    continue
                for name in filenames:
                    if not name.endswith(".entry"):
                        continue
                    path = os.path.join(dirpath, name)
                    if cutoff is not None:
                        try:
                            if os.path.getmtime(path) >= cutoff:
                                continue
                        except OSError:
                            continue
                    with contextlib.suppress(OSError):
                        os.remove(path)
                        removed += 1
            self._prune_empty_dirs(self.cache_dir)
            self._size_dirty = True
            if removed:
                logger.info("Cache clear removed %d entries", removed)
            return removed

    def current_size_bytes(self):
        """Return the total size in bytes of all entry files under the root."""
        total = 0
        if not os.path.isdir(self.cache_dir):
            return 0
        for dirpath, _dirnames, filenames in os.walk(self.cache_dir):
            for name in filenames:
                if name.endswith(".entry"):
                    with contextlib.suppress(OSError):
                        total += os.path.getsize(os.path.join(dirpath, name))
        self._approx_size = total
        self._size_dirty = False
        return total

    def trim_to_size(self, max_bytes=None):
        """
        Explicitly shrink the cache to at most *max_bytes* using LRU eviction.

        This is the manual counterpart of the automatic size bound: the
        constructor evicts only when writes cross ``max_size_bytes``, whereas
        this method runs the same least-recently-used removal on demand (for
        example from a maintenance job after other data filled the disk).
        With no argument the constructor's bound is used. Returns the number
        of entries removed.
        """
        if max_bytes is None:
            max_bytes = self.max_size_bytes
        if max_bytes is None:
            return 0
        with self._locked():
            self._approx_size = self.current_size_bytes()
            if self._approx_size <= max_bytes:
                return 0
            before = self.stats.evictions
            self._evict_lru(max_bytes)
            return self.stats.evictions - before

    def stats_snapshot(self):
        """Return the current statistics as a plain dictionary."""
        return self.stats.as_dict()

    # ------------------------------------------------------------------
    # Key construction and paths
    # ------------------------------------------------------------------
    def _build_descriptor(self, stage, processor_version, corpus_version, input_value):
        canonical = _canonical_input(input_value)
        return {
            "stage": str(stage),
            "processor_version": str(processor_version),
            "corpus_version": str(corpus_version),
            "text_hash": _sha256_hex(canonical),
        }

    def _entry_path(self, descriptor):
        pv = descriptor["processor_version"]
        cv = descriptor["corpus_version"]
        key_material = json.dumps(descriptor, sort_keys=True)
        key_hash = _sha256_hex(key_material)
        parts = [
            self.cache_dir,
            _slugify(descriptor["stage"]),
            f"{_slugify(pv)}-{_sha256_hex(pv)[:12]}",
            f"{_slugify(cv)}-{_sha256_hex(cv)[:12]}",
            f"{key_hash}.entry",
        ]
        return os.path.join(*parts)

    # ------------------------------------------------------------------
    # Reads: verification, corruption handling, collision detection
    # ------------------------------------------------------------------
    def _read_entry(self, path, descriptor):
        # Writability is checked once by the caller; here a missing entry is
        # the ordinary miss path.
        try:
            with open(path, "rb") as fh:  # sandboxed-open ok: caller-named cache dir (operator path)
                raw = fh.read()
        except FileNotFoundError:
            return _MISS
        except OSError as exc:
            logger.warning("Cache entry unreadable (%r); recomputing", path)
            self.stats.corrupt += 1
            return _MISS

        try:
            text = raw.decode("utf-8")
            lines = text.split("\n")
            if len(lines) < 4 or lines[0] != ENTRY_MAGIC:
                raise ValueError("bad magic")
            stored_descriptor = safe_json_loads(
                base64.urlsafe_b64decode(lines[1]),
                context="PipelineCache entry descriptor",
            )
            stored_payload_hash = lines[2]
            payload_b64 = "\n".join(lines[3:])
            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            if len(payload_bytes) > MAX_ENTRY_BYTES:
                raise ValueError("payload over size cap")
            if _sha256_bytes_hex(payload_bytes) != stored_payload_hash:
                raise ValueError("payload checksum mismatch")
            # Full descriptor equality: a text-hash collision that somehow
            # produced the same file name still cannot pass, because the
            # processor/corpus versions and the true text hash are compared.
            if stored_descriptor != descriptor:
                raise ValueError("key descriptor mismatch (possible collision)")
            payload_text = payload_bytes.decode("utf-8")
            result = load_payload(payload_text)
        except (
            ValueError,
            KeyError,
            TypeError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning(
                "Corrupt cache entry discarded and recomputed: %r (%r)", path, exc
            )
            self.stats.corrupt += 1
            with contextlib.suppress(OSError):
                os.remove(path)
            return _MISS

        # Refresh atime/mtime so LRU eviction reflects actual reuse. A failed
        # touch is harmless: the entry itself is verified and usable.
        with contextlib.suppress(OSError):
            os.utime(path, None)
        return result

    # ------------------------------------------------------------------
    # Writes: atomic publication and size enforcement
    # ------------------------------------------------------------------
    def _write_entry(self, path, descriptor, result):
        """Publish *result* atomically. Return True on write, False on skip."""
        try:
            _verify_cacheable(result)
            payload_text = dump_payload(result)
        except TypeError as exc:
            # Non-JSON-serializable output (e.g. an unregistered Tree parse
            # result): do not poison the pipeline, simply skip caching.
            logger.warning(
                "Result is not cache-serializable; not caching (%r)", exc
            )
            return False
        payload_bytes = payload_text.encode("utf-8")
        if len(payload_bytes) > MAX_ENTRY_BYTES:
            logger.warning(
                "Result of %d bytes exceeds cache entry cap (%d); not caching",
                len(payload_bytes),
                MAX_ENTRY_BYTES,
            )
            return False

        payload_hash = _sha256_bytes_hex(payload_bytes)
        body = "\n".join(
            [
                ENTRY_MAGIC,
                base64.urlsafe_b64encode(
                    json.dumps(descriptor, sort_keys=True).encode("utf-8")
                ).decode("ascii"),
                payload_hash,
                base64.urlsafe_b64encode(payload_bytes).decode("ascii"),
            ]
        )

        directory = os.path.dirname(path)
        tmp_path = None
        with self._locked():
            try:
                os.makedirs(directory, exist_ok=True)
                # UUID keeps the temp name unique even for distinct keys
                # written back-to-back from the same thread/process.
                tmp_path = os.path.join(
                    directory,
                    f".tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}",
                )
                with open(tmp_path, "w", encoding="utf-8") as fh:  # sandboxed-open ok: atomic publish inside caller-named cache dir
                    fh.write(body)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, path)
            except OSError as exc:
                logger.warning("Cache write failed; continuing without cache (%r)", exc)
                if tmp_path is not None:
                    with contextlib.suppress(OSError):
                        os.remove(tmp_path)
                self._known_writable = False
                # Counted as a skipped store by get_or_compute; the warning
                # already records the degradation.
                return False
            self.stats.writes += 1
            self._approx_size += len(body.encode("utf-8"))
            self._enforce_size()
            return True

    def _enforce_size(self):
        if self.max_size_bytes is None:
            return
        if self._size_dirty or self._approx_size > self.max_size_bytes + _SIZE_JITTER:
            self._approx_size = self.current_size_bytes()
        if self._approx_size <= self.max_size_bytes:
            return
        target = int(self.max_size_bytes * EVICTION_TARGET_RATIO)
        removed_bytes = self._evict_lru(target)
        logger.warning(
            "Cache exceeded %d bytes; LRU eviction freed %d bytes "
            "(%d entries total)",
            self.max_size_bytes,
            removed_bytes,
            self.stats.evictions,
        )

    def _evict_lru(self, target_bytes):
        """Remove least-recently-used entries until size <= target. Bytes freed."""
        entries = []
        for dirpath, _dirnames, filenames in os.walk(self.cache_dir):
            for name in filenames:
                if not name.endswith(".entry"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    entries.append((os.path.getmtime(path), os.path.getsize(path), path))
                except OSError:
                    continue
        # Oldest access/modification time first = least recently used.
        entries.sort(key=lambda item: item[0])
        freed = 0
        for _mtime, size, path in entries:
            with contextlib.suppress(OSError):
                os.remove(path)
                freed += size
                self.stats.evictions += 1
            if self._approx_size - freed <= target_bytes:
                break
        self._approx_size -= freed
        self._prune_empty_dirs(self.cache_dir)
        self._size_dirty = True
        return freed

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------
    def _ensure_writable(self):
        """
        Verify (once, then on demand) that the cache root can be written.
        On failure the cache degrades to pass-through instead of raising.
        """
        if self._known_writable is True:
            return True
        with self._lock:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                probe = os.path.join(self.cache_dir, ".write_probe")
                with open(probe, "w") as fh:  # sandboxed-open ok: writability probe inside caller-named cache dir
                    fh.write("ok")
                os.remove(probe)
            except OSError as exc:
                if self._known_writable is not False:
                    logger.warning(
                        "Cache directory %r is not writable (%r); "
                        "caching disabled for this process",
                        self.cache_dir,
                        exc,
                    )
                self._known_writable = False
                return False
            self._known_writable = True
            return True

    @contextlib.contextmanager
    def _locked(self):
        """In-process lock plus an advisory cross-process file lock."""
        self._lock.acquire()
        lock_path = os.path.join(self.cache_dir, ".cache.lock")
        lock_fh = None
        try:
            if self._known_writable:
                try:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    lock_fh = open(lock_path, "a+")  # sandboxed-open ok: advisory lock file inside caller-named cache dir
                    _lock_file(lock_fh)
                except OSError:
                    # Locking unsupported/unavailable: the atomic replace
                    # guarantees entry integrity even without the advisory lock.
                    lock_fh = None
            yield
        finally:
            if lock_fh is not None:
                with contextlib.suppress(OSError):
                    _unlock_file(lock_fh)
                lock_fh.close()
            self._lock.release()

    @staticmethod
    def _count_entries(root):
        count = 0
        for _dirpath, _dirnames, filenames in os.walk(root):
            count += sum(1 for name in filenames if name.endswith(".entry"))
        return count

    @staticmethod
    def _prune_empty_dirs(root):
        for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
            if dirpath == root:
                continue
            with contextlib.suppress(OSError):
                if not os.listdir(dirpath):
                    os.rmdir(dirpath)


class _Miss:
    """Sentinel for a cache miss (``None`` is a legitimate cached value)."""

    __slots__ = ()

    def __repr__(self):
        return "<cache miss>"


_MISS = _Miss()


# ----------------------------------------------------------------------
# Advisory file locking (fcntl on POSIX, msvcrt on Windows)
# ----------------------------------------------------------------------
def _lock_file(fh):
    try:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    except ImportError:
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)


def _unlock_file(fh):
    try:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except ImportError:
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


__all__ = [
    "PipelineCache",
    "CacheStats",
    "resource_version",
    "DEFAULT_MAX_SIZE_BYTES",
    "MAX_ENTRY_BYTES",
    "NLTK_VERSION",
]
