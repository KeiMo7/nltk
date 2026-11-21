# Natural Language Toolkit: Corpus and processing provenance
#
# Copyright (C) 2001-2026 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""Content-addressed provenance for corpora, processors and processing runs.

NLTK corpora are distributed without a version field: a few readers expose an
ad hoc ``version`` attribute (VerbNet) or a file date (BCP47), but most do
not.  Two installations can therefore hold different text under the same
corpus name, and two tokenisation runs that look identical in their arguments
can produce different tokens.  This module gives a corpus, a processor and a
processing run a *content-derived* identity so two results can be compared
before they are trusted as equivalent.

Design rules (see ``web/provenance.rst`` for the user guide):

* A version is the SHA-256 of the corpus content, never a directory name, a
  download time or a file modification time.  Two copies of the same bytes get
  the same version whatever their paths are called.
* Everything is opt-in.  Nothing is recorded until a :func:`record` session (or
  :func:`enable`) is active, and no existing return value changes.
* Per-file digests are streamed and cached by a stat signature, so repeated
  fingerprints do not re-read a large corpus.
* Result files never contain an absolute path: ``source_id`` is relative to
  the NLTK data root, and path-shaped parameter values are collapsed to a
  basename.

Typical use::

    from nltk import provenance, word_tokenize

    with provenance.record() as run:
        for sample_id, text in samples:
            run.add_sample(sample_id, word_tokenize(text))
    result = run.result()
    result.save("run.json")              # corpus/processor/sample digests

    old = provenance.ProcessingResult.load("old.json")
    diff = provenance.compare(old, result)
    if not diff.comparable:
        print(diff.reasons)              # corpus or processor version differs
    print(diff.changed_samples)          # sample ids whose output changed
"""

import base64
import hashlib
import json
import os
import re
import threading
import warnings

from nltk.termsec import sanitize_terminal

__all__ = [
    "SCHEMA",
    "CorpusFingerprint",
    "CorpusUnavailableError",
    "IncomparableResultsError",
    "ProcessorFingerprint",
    "ProcessingResult",
    "Comparison",
    "ProvenanceWarning",
    "RecordingSession",
    "compare",
    "corpus_fingerprint",
    "enable",
    "disable",
    "export_manifest",
    "is_recording",
    "manifest",
    "processor_fingerprint",
    "record",
    "recording",
    "sample_digest",
]

#: Schema id written into every result, so a consumer can refuse a format it
#: does not understand rather than silently misread it.
SCHEMA = "nltk-provenance/1"
CORPUS_SCHEMA = "nltk-corpus/1"
PROCESSOR_SCHEMA = "nltk-processor/1"
MANIFEST_SCHEMA = "nltk-corpus-manifest/1"
CACHE_SCHEMA = "nltk-provenance-cache/1"

#: Streamed read size.  A hash is computed in O(1) memory; no corpus file is
#: ever held whole, so provenance adds no full-corpus copies.
_CHUNK = 1024 * 1024

_NO_VERSION = "unknown"


class CorpusUnavailableError(LookupError):
    """A corpus or model resource could not be found.

    The message names the download that installs it, so a missing corpus fails
    with an actionable instruction instead of only the bare lookup miss.
    """


class IncomparableResultsError(ValueError):
    """Two results used different corpus or processor versions."""


class ProvenanceWarning(UserWarning):
    """A content digest could not be computed; the version is recorded as
    ``unknown`` and processing continues."""


# ---------------------------------------------------------------------------
# Helpers: canonicalisation, absolute-path hygiene, digests
# ---------------------------------------------------------------------------


def _nltk_version():
    try:
        import nltk

        return str(nltk.__version__)
    except Exception:  # pragma: no cover - only during a broken import
        return _NO_VERSION


def _collapse_path(value):
    """Render an absolute path as ``<absolute>/<basename>``.

    Result files are compared and committed across machines, so an absolute
    path is both non-portable and a small information leak.  A path-shaped
    *parameter* is collapsed; a path-shaped field NLTK itself writes is made
    relative to the data root instead (see :func:`_source_id`).
    """
    stripped = value.rstrip("\\/")
    base = os.path.basename(stripped.replace("\\", "/"))
    return "<absolute>/" + base if base else "<absolute>"


#: An absolute filesystem token inside free text (a message, a parameter value).
#: The lookbehind keeps URL schemes ("https://…") and the "sha256:…" digests out.
_ABS_TOKEN = re.compile(r"(?<![\w:/>])(?:/[^\s'\"<>|]+|[A-Za-z]:[\\/][^\s'\"<>|]*)")


def _scrub_paths(text):
    """Collapse absolute filesystem paths embedded in free text.

    Result files are compared across machines, so a path that slipped into an
    error message or a parameter must not make the file machine-specific.
    """
    if "/" not in text and "\\" not in text:
        return text
    return _ABS_TOKEN.sub(lambda match: _collapse_path(match.group(0)), text)


def _canonical_obj(obj):
    """A JSON-ready value with deterministic ordering and no absolute paths.

    Used for parameter maps and arbitrary processor outputs.  Sets are sorted,
    tuples become lists, bytes are base64 text and an unknown object falls back
    to ``repr`` (a processor that returns such an object should supply its own
    digest instead; the repr may carry an address).
    """
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return _scrub_paths(obj)
    if isinstance(obj, bytes):
        return "base64:" + base64.b64encode(obj).decode("ascii")
    if isinstance(obj, dict):
        return {
            str(k): _canonical_obj(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(obj, (list, tuple)):
        return [_canonical_obj(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [_canonical_obj(v) for v in sorted(obj, key=_sort_key)]
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return _canonical_obj(to_dict())
        except Exception:
            pass
    return _scrub_paths(repr(obj))


def _sort_key(value):
    try:
        return (0, json.dumps(_canonical_obj(value), sort_keys=True))
    except Exception:  # pragma: no cover - defensive
        return (1, repr(value))


def canonical_json(obj):
    """Deterministic JSON: sorted keys, no insignificant whitespace."""
    return json.dumps(
        _canonical_obj(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=True,
    )


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample_digest(output):
    """A content digest of one processing output.

    The output is canonicalised (sorted keys, tuples as lists) so two runs that
    build the same value in a different order get the same digest.  Returns a
    ``"sha256:<hex>"`` string.
    """
    return "sha256:" + _sha256_text("nltk-sample/1\0" + canonical_json(output))


def _sanitize_error(exc, roots=()):
    """An exception message safe for a result file: no absolute paths."""
    text = f"{type(exc).__name__}: {exc}"
    for root in roots:
        if isinstance(root, str) and root:
            text = text.replace(root, "<root>")
    # Whatever an exception chose to embed, collapse any remaining absolute
    # path-shaped token.
    return _scrub_paths(text)


# ---------------------------------------------------------------------------
# Digest cache
# ---------------------------------------------------------------------------


class _DigestCache:
    """Per-file SHA-256 values keyed by a stat signature.

    A key is the SHA-256 of the file's canonical open target, so the cache file
    itself holds no absolute path.  The signature changes when the file does
    (size, mtime, ctime and, on POSIX, inode), so a changed corpus is re-hashed
    rather than served from a stale entry.  Both the in-memory map and the
    on-disk map are best-effort: a cache problem never fails a fingerprint.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._mem = {}
        self._disk_loaded = False
        self._disk = {}
        self._path = None
        self._path_resolved = False
        self._dirty = False

    # -- location ----------------------------------------------------------
    def _cache_path(self):
        if self._path_resolved:
            return self._path
        self._path_resolved = True
        if os.environ.get("NLTK_PROVENANCE_NO_CACHE"):
            self._path = None
            return None
        explicit = os.environ.get("NLTK_PROVENANCE_CACHE")
        if explicit:
            explicit = os.path.expanduser(explicit)
            self._path = (
                explicit
                if explicit.endswith(".json")
                else os.path.join(explicit, "digests.json")
            )
            return self._path
        try:
            import nltk.data

            for entry in nltk.data.path:
                base = os.path.expanduser(str(getattr(entry, "path", entry)))
                if os.path.isdir(base) and os.access(base, os.W_OK | os.X_OK):
                    self._path = os.path.join(base, "provenance_digests.json")
                    break
        except Exception:
            self._path = None
        return self._path

    def _load(self):
        if self._disk_loaded:
            return
        self._disk_loaded = True
        path = self._cache_path()
        if not path or not os.path.isfile(path):
            return
        try:
            from nltk.jsontags import safe_json_load
            from nltk.pathsec import open as _safe_open

            with _safe_open(path, "r", context="nltk.provenance cache") as f:
                data = safe_json_load(f, context="nltk.provenance cache")
            if data.get("schema") == CACHE_SCHEMA:
                entries = data.get("entries")
                if isinstance(entries, dict):
                    self._disk = entries
        except Exception:
            # Unreadable/foreign cache is treated as absent; a fingerprint must
            # not be refused because a cache file is corrupt.
            self._disk = {}

    def _flush(self):
        if not self._dirty:
            return
        path = self._cache_path()
        if not path:
            self._dirty = False
            return
        self._dirty = False
        payload = json.dumps(
            {"schema": CACHE_SCHEMA, "entries": self._disk},
            sort_keys=True,
            separators=(",", ":"),
        )
        tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            from nltk.pathsec import open as _safe_open

            with _safe_open(tmp, "w", context="nltk.provenance cache") as f:
                f.write(payload)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # -- get/set -----------------------------------------------------------
    def get(self, key, signature):
        entry = self._mem.get(key)
        if entry is not None and entry[0] == signature:
            return entry[1]
        with self._lock:
            self._load()
            entry = self._disk.get(key)
            if entry is not None and entry.get("sig") == signature:
                digest = entry.get("sha256")
                if isinstance(digest, str):
                    self._mem[key] = (signature, digest)
                    return digest
        return None

    def put(self, key, signature, digest):
        # Deliberately does not flush: a digest pass touches every file of a
        # corpus, and rewriting the cache once per file would be quadratic.
        # The caller flushes once when the pass ends (see _aggregate_digest).
        with self._lock:
            self._mem[key] = (signature, digest)
            self._load()
            self._disk[key] = {"sig": signature, "sha256": digest}
            self._dirty = True

    def flush(self):
        with self._lock:
            self._flush()

    def clear(self):
        with self._lock:
            self._mem.clear()
            self._disk.clear()
            self._dirty = True
            self._flush()


_CACHE = _DigestCache()


def _cache_key(pointer):
    """Hashed cache key for a pointer: the cache stores no absolute path."""
    import nltk.data

    if isinstance(pointer, nltk.data.ZipFilePathPointer):
        identity = f"zip\0{pointer.zipfile.filename}\0{pointer.entry}"
    else:
        identity = f"file\0{os.path.realpath(str(pointer))}"
    return _sha256_text(identity)


def _file_signature(pointer):
    import nltk.data

    try:
        if isinstance(pointer, nltk.data.ZipFilePathPointer):
            info = pointer.zipfile.getinfo(pointer.entry)
            stat = os.stat(pointer.zipfile.filename)
            return [
                "zip",
                info.CRC,
                info.file_size,
                stat.st_size,
                stat.st_mtime_ns,
            ]
        stat = os.stat(str(pointer))
        return [
            "fs",
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            getattr(stat, "st_ino", 0),
            getattr(stat, "st_dev", 0),
        ]
    except Exception:
        return None


def _stream_digest(pointer):
    """SHA-256 of a pointer's bytes, streamed in bounded chunks."""
    digest = hashlib.sha256()
    with pointer.open() as stream:
        while True:
            chunk = stream.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_digest(pointer, force=False):
    """Content digest of one file, from the cache when its stat matches."""
    key = None
    signature = None
    if not force:
        try:
            key = _cache_key(pointer)
            signature = _file_signature(pointer)
        except Exception:
            key = signature = None
    if key is not None and signature is not None:
        cached = _CACHE.get(key, signature)
        if cached is not None:
            return cached
    digest = _stream_digest(pointer)
    if key is not None and signature is not None:
        _CACHE.put(key, signature, digest)
    return digest


def _list_resource_files(pointer):
    """``(relative name, pointer)`` pairs for a directory-or-file resource.

    Handles both a real directory and a directory entry inside a zip, the two
    shapes :func:`nltk.data.find` returns.
    """
    import nltk.data

    if isinstance(pointer, nltk.data.ZipFilePathPointer):
        prefix = pointer.entry
        if prefix and not prefix.endswith("/"):
            info = pointer.zipfile.getinfo(prefix)
            if not info.is_dir():
                return [(os.path.basename(prefix), pointer)]
            prefix += "/"
        pairs = []
        for name in pointer.zipfile.namelist():
            if name.endswith("/") or not name.startswith(prefix):
                continue
            rel = name[len(prefix) :]
            if not rel:
                continue
            pairs.append(
                (
                    rel.replace("\\", "/"),
                    nltk.data.ZipFilePathPointer(pointer.zipfile, name),
                )
            )
        return sorted(pairs)

    path = str(pointer)
    if os.path.isfile(path):
        return [(os.path.basename(path), pointer)]
    pairs = []
    for dirname, subdirs, files in os.walk(path):
        subdirs.sort()
        for name in sorted(files):
            full = os.path.join(dirname, name)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            pairs.append((rel, nltk.data.FileSystemPathPointer(full)))
    return sorted(pairs)


def _aggregate_digest(schema, pairs, force=False):
    """Digest over sorted ``(name, pointer)`` pairs, streaming each file."""
    hasher = hashlib.sha256(schema.encode("utf-8") + b"\0")
    try:
        for name, pointer in sorted(pairs, key=lambda item: item[0]):
            hasher.update(name.replace("\\", "/").encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(_file_digest(pointer, force=force).encode("ascii"))
            hasher.update(b"\0")
    finally:
        # One write per pass, not per file; a cache failure stays best-effort.
        _CACHE.flush()
    return "sha256:" + hasher.hexdigest()


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def _corpus_name(root):
    import nltk.data

    if isinstance(root, nltk.data.ZipFilePathPointer):
        entry = root.entry.rstrip("/")
        if entry:
            return os.path.basename(entry)
        return os.path.basename(root.zipfile.filename).removesuffix(".zip")
    path = str(getattr(root, "path", root)).rstrip("\\/")
    return os.path.basename(path) or "corpus"


def _relative_to_data_root(path):
    """*path* relative to the data root that contains it, POSIX separators."""
    try:
        import nltk.data

        real = os.path.realpath(path)
        for entry in nltk.data.path:
            base = os.path.realpath(
                os.path.expanduser(str(getattr(entry, "path", entry)))
            )
            try:
                if real == base or real.startswith(base + os.sep):
                    rel = os.path.relpath(real, base).replace(os.sep, "/")
                    return rel if rel and rel != "." else None
            except ValueError:
                continue
    except Exception:
        pass
    return None


def _source_id(root, name):
    """A portable, non-absolute identifier for a corpus root."""
    import nltk.data

    try:
        if isinstance(root, nltk.data.ZipFilePathPointer):
            zip_rel = _relative_to_data_root(root.zipfile.filename)
            base = (
                zip_rel
                if zip_rel
                else os.path.basename(root.zipfile.filename).removesuffix(".zip")
            )
            entry = root.entry.rstrip("/")
            return f"{base}/{entry}" if entry else base
        path = str(getattr(root, "path", root))
        rel = _relative_to_data_root(path)
        if rel:
            return rel
    except Exception:
        pass
    return name


def _declared_version(reader):
    """A reader's own version attribute, when it has one (VerbNet, BCP47).

    Best-effort and never triggers a lookup: attributes live on the constructed
    reader.  This value is recorded for display only; comparison always uses
    the content digest.
    """
    try:
        value = getattr(reader, "version", None)
        if callable(value):
            return None
        if isinstance(value, str) and value:
            return value
    except Exception:
        pass
    return None


def _reader_pairs(reader, fileids):
    pairs = []
    for fileid in fileids:
        pointer = reader.abspath(fileid)
        pairs.append((str(fileid).replace("\\", "/"), pointer))
    return pairs


def _reader_size(pairs):
    total = 0
    for _, pointer in pairs:
        try:
            total += pointer.file_size()
        except Exception:
            pass
    return total


def _corpus_fingerprint_from_reader(reader, fileids=None, force=False):
    name = _corpus_name(reader.root)
    source_id = _source_id(reader.root, name)
    ids = list(reader.fileids() if fileids is None else fileids)
    declared = _declared_version(reader)
    try:
        pairs = _reader_pairs(reader, ids)
        digest = _aggregate_digest(CORPUS_SCHEMA, pairs, force=force)
        size = _reader_size(pairs)
    except Exception as exc:
        message = (
            f"could not compute the content digest of corpus {name!r}: "
            f"{_sanitize_error(exc)}; version recorded as {_NO_VERSION!r}"
        )
        warnings.warn(sanitize_terminal(message), ProvenanceWarning, stacklevel=3)
        return CorpusFingerprint(
            name=name,
            version=_NO_VERSION,
            digest=None,
            entries=len(ids),
            size=None,
            declared_version=declared,
            version_source="unknown",
            source_id=source_id,
            error=_sanitize_error(exc, getattr(reader.root, "path", ())),
        )
    return CorpusFingerprint(
        name=name,
        version=digest,
        digest=digest,
        entries=len(ids),
        size=size,
        declared_version=declared,
        version_source="content",
        source_id=source_id,
        error=None,
    )


def _download_hint(resource_name):
    parts = resource_name.replace("\\", "/").strip("/").split("/")
    if len(parts) >= 2 and parts[0] in {
        "corpora",
        "tokenizers",
        "taggers",
        "chunkers",
        "grammars",
        "misc",
        "sentiment",
        "stemmers",
        "models",
        "help",
    }:
        package = parts[1]
    elif parts:
        package = parts[0]
    else:
        package = resource_name
    if package.endswith(".zip"):
        package = package[: -len(".zip")]
    return (
        f"Install it with:\n"
        f"    >>> import nltk\n"
        f"    >>> nltk.download({package!r})\n"
        f"or point nltk.data.path at an existing NLTK data directory."
    )


def corpus_fingerprint(source, *, fileids=None, force=False):
    """Return a :class:`CorpusFingerprint` for a reader or resource name.

    :param source: a :class:`~nltk.corpus.reader.api.CorpusReader` instance, a
        corpus/model resource name such as ``"brown"`` or
        ``"tokenizers/punkt_tab/english/"``, or a ``PathPointer``.
    :param fileids: restrict a reader or directory to these relative files.
    :param force: re-hash even when the digest cache has a matching entry.
    :raises CorpusUnavailableError: if a named resource is not installed.  The
        message names the ``nltk.download()`` call that installs it.
    """
    import nltk.data

    if hasattr(source, "fileids") and hasattr(source, "abspath"):
        return _corpus_fingerprint_from_reader(source, fileids=fileids, force=force)

    if isinstance(source, nltk.data.PathPointer):
        root = source
        if fileids is not None:
            pairs = [(str(fid).replace("\\", "/"), root.join(fid)) for fid in fileids]
        else:
            pairs = _list_resource_files(root)
        name = _corpus_name(root)
        try:
            digest = _aggregate_digest(CORPUS_SCHEMA, pairs, force=force)
        except Exception as exc:
            warnings.warn(
                sanitize_terminal(
                    f"could not compute the content digest of {name!r}: "
                    f"{_sanitize_error(exc)}"
                ),
                ProvenanceWarning,
                stacklevel=2,
            )
            return CorpusFingerprint(
                name=name,
                version=_NO_VERSION,
                digest=None,
                entries=len(pairs),
                size=None,
                declared_version=None,
                version_source="unknown",
                source_id=_source_id(root, name),
                error=_sanitize_error(exc),
            )
        return CorpusFingerprint(
            name=name,
            version=digest,
            digest=digest,
            entries=len(pairs),
            size=_reader_size(pairs),
            declared_version=None,
            version_source="content",
            source_id=_source_id(root, name),
            error=None,
        )

    resource_name = str(source)
    candidate = resource_name
    if "/" not in candidate:
        candidate = f"corpora/{candidate}"
    # A filesystem path is used as given.  A resource name keeps NLTK's normal
    # resolution order (and only falls back to a local path that actually
    # exists), so no existing loading priority changes.
    if os.path.isabs(resource_name):
        return corpus_fingerprint(
            nltk.data.FileSystemPathPointer(resource_name),
            fileids=fileids,
            force=force,
        )
    try:
        root = nltk.data.find(candidate)
    except LookupError as exc:
        if os.path.exists(resource_name):
            return corpus_fingerprint(
                nltk.data.FileSystemPathPointer(resource_name),
                fileids=fileids,
                force=force,
            )
        raise CorpusUnavailableError(
            f"Corpus {resource_name!r} is not installed. " + _download_hint(candidate)
        ) from exc
    return corpus_fingerprint(root, fileids=fileids, force=force)


def _resolve_resource(resource_name):
    """A PathPointer for a resource name, or for an existing filesystem path."""
    import nltk.data

    name = str(resource_name)
    if os.path.isabs(name):
        return nltk.data.FileSystemPathPointer(name), name
    try:
        return nltk.data.find(name), name
    except LookupError:
        if os.path.exists(name):
            return nltk.data.FileSystemPathPointer(name), name
        raise


def _resource_label(resource_name):
    """A path-independent label for a processor resource.

    The label is part of the digest, so it must not be an absolute path: the
    same model installed under two data roots is the same processor and must
    get the same version (and a result file must not carry the path).
    """
    name = str(resource_name)
    if os.path.isabs(name):
        relative = _relative_to_data_root(name)
        if relative:
            return relative
        base = os.path.basename(name.rstrip("\\/"))
        return f"<absolute>/{base}" if base else "<absolute>"
    return name.replace(os.sep, "/").rstrip("/")


def _processor_digest(resource_names, force=False):
    pairs = []
    for resource_name in resource_names:
        pointer, _ = _resolve_resource(resource_name)
        prefix = _resource_label(resource_name)
        for rel, file_pointer in _list_resource_files(pointer):
            pairs.append((f"{prefix}/{rel}", file_pointer))
    return _aggregate_digest(PROCESSOR_SCHEMA, pairs, force=force), len(pairs)


def processor_fingerprint(
    name,
    *,
    version=None,
    resources=(),
    params=None,
    force=False,
):
    """Identify a processor by content and configuration.

    :param name: a stable processor name, e.g. ``"nltk.tag.PerceptronTagger"``.
    :param version: a declared version to fall back to when no *resources* are
        given.  The package version is used when this is omitted.
    :param resources: resource names whose content defines the processor (model
        directories or files).  When present, their digest is the version and
        the float ``version`` argument is recorded separately as
        ``declared_version`` only if supplied.
    :param params: configuration that changes output (language, tagset, random
        seed, ...); absolute paths are collapsed.
    """
    params = _canonical_obj(params or {})
    nltk_version = _nltk_version()
    declared = str(version) if version is not None else None
    if resources:
        try:
            digest, entries = _processor_digest(resources, force=force)
        except Exception as exc:
            message = (
                f"could not compute the content digest of processor {name!r}: "
                f"{_sanitize_error(exc)}; version recorded as {_NO_VERSION!r}"
            )
            warnings.warn(sanitize_terminal(message), ProvenanceWarning, stacklevel=2)
            return ProcessorFingerprint(
                name=name,
                version=_NO_VERSION,
                digest=None,
                nltk_version=nltk_version,
                declared_version=declared,
                version_source="unknown",
                resources=tuple(_resource_label(r) for r in resources),
                params=params,
                error=_sanitize_error(exc),
            )
        return ProcessorFingerprint(
            name=name,
            version=digest,
            digest=digest,
            nltk_version=nltk_version,
            declared_version=declared,
            version_source="content",
            resources=tuple(_resource_label(r) for r in resources),
            params=params,
            error=None,
        )
    return ProcessorFingerprint(
        name=name,
        version=declared or nltk_version,
        digest=None,
        nltk_version=nltk_version,
        declared_version=declared,
        version_source="declared" if declared else "package",
        resources=(),
        params=params,
        error=None,
    )


# ---------------------------------------------------------------------------
# Built-in processor detection
# ---------------------------------------------------------------------------

#: True while at least one recording session is active.  Processors and readers
#: test this module global directly on their hot path, so the off state costs a
#: single attribute lookup and no work.
_ACTIVE = False

_SESSIONS = []
_SESSIONS_LOCK = threading.RLock()


def _snapshot_sessions():
    with _SESSIONS_LOCK:
        return tuple(_SESSIONS)


def is_recording():
    """Whether a recording session is currently active."""
    return _ACTIVE


#: How the built-in tokenisers/taggers are named and where their model lives.
_BUILTIN_PROCESSORS = {
    "punkt": {
        "name": "nltk.tokenize.PunktTokenizer",
        "resource": "tokenizers/punkt_tab/{language}/",
        "param": "language",
    },
    "perceptron": {
        "name": "nltk.tag.PerceptronTagger",
        "resource": "taggers/averaged_perceptron_tagger_{lang}/",
        "param": "lang",
    },
    "universal_tagset": {
        "name": "nltk.tag.mapping.tagset_mapping",
        "resource": "taggers/universal_tagset/",
        "param": "tagset",
    },
}


def _builtin_spec(kind, params):
    spec = _BUILTIN_PROCESSORS[kind]
    return (kind, canonical_json(_canonical_obj(params))), spec


def _resolve_builtin(spec_key):
    kind, encoded = spec_key
    from nltk.jsontags import safe_json_loads

    spec = _BUILTIN_PROCESSORS[kind]
    params = safe_json_loads(encoded, context="nltk.provenance builtin")
    resource = spec["resource"].format(**params)
    return processor_fingerprint(spec["name"], resources=(resource,), params=params)


def _note_processor(kind, **params):
    """Record a built-in processor use, if any recording session is active."""
    if not _ACTIVE:
        return
    key = _builtin_spec(kind, params)[0]
    for session in _snapshot_sessions():
        session._observe_processor_key(key)


def _note_punkt(language="english", **extra):
    params = {"language": language, **extra}
    _note_processor("punkt", **params)


def _note_perceptron(lang="eng", tagset=None):
    _note_processor("perceptron", lang=lang, tagset=tagset)
    # A tagset mapping is a separate data resource whose content changes the
    # output as much as the model does, so record it too when one was used.
    if tagset:
        _note_processor("universal_tagset", tagset=tagset)


def _note_corpus_reader(reader):
    """Record a corpus reader constructed while a session is active.

    Called from ``CorpusReader.__init__``; the first check is a module-global
    boolean test, so the off state costs one function call and changes nothing.
    """
    if not _ACTIVE:
        return
    for session in _snapshot_sessions():
        session._observe_reader(reader)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class CorpusFingerprint:
    """The identity of one corpus: name, content version, entry count, digest.

    ``version`` is ``"sha256:<hex>"`` for a corpus whose files could be read,
    or ``"unknown"`` when digesting failed (a warning is emitted and processing
    continues).  ``source_id`` is relative to the NLTK data root and is never an
    absolute path.
    """

    __slots__ = (
        "name",
        "version",
        "digest",
        "entries",
        "size",
        "declared_version",
        "version_source",
        "source_id",
        "error",
    )

    def __init__(
        self,
        name,
        version,
        digest=None,
        entries=0,
        size=None,
        declared_version=None,
        version_source="content",
        source_id=None,
        error=None,
    ):
        self.name = str(name)
        self.version = str(version)
        self.digest = digest
        self.entries = int(entries)
        self.size = size
        self.declared_version = declared_version
        self.version_source = version_source
        self.source_id = str(source_id) if source_id is not None else self.name
        self.error = error

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            version=data["version"],
            digest=data.get("digest"),
            entries=data.get("entries", 0),
            size=data.get("size"),
            declared_version=data.get("declared_version"),
            version_source=data.get("version_source", "content"),
            source_id=data.get("source_id", data["name"]),
            error=data.get("error"),
        )

    def to_dict(self):
        return {
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "entries": self.entries,
            "size": self.size,
            "declared_version": self.declared_version,
            "version_source": self.version_source,
            "source_id": self.source_id,
            "error": self.error,
        }

    def __eq__(self, other):
        return (
            isinstance(other, CorpusFingerprint) and self.to_dict() == other.to_dict()
        )

    def __hash__(self):
        return hash(canonical_json(self.to_dict()))

    def __repr__(self):
        return (
            f"CorpusFingerprint(name={self.name!r}, version={self.version!r}, "
            f"entries={self.entries}, source_id={self.source_id!r})"
        )


class ProcessorFingerprint:
    """The identity of a processor: content digest, NLTK version, parameters."""

    __slots__ = (
        "name",
        "version",
        "digest",
        "nltk_version",
        "declared_version",
        "version_source",
        "resources",
        "params",
        "error",
    )

    def __init__(
        self,
        name,
        version,
        digest=None,
        nltk_version=None,
        declared_version=None,
        version_source="content",
        resources=(),
        params=None,
        error=None,
    ):
        self.name = str(name)
        self.version = str(version)
        self.digest = digest
        self.nltk_version = str(nltk_version or _nltk_version())
        self.declared_version = declared_version
        self.version_source = version_source
        self.resources = tuple(str(r) for r in resources)
        self.params = _canonical_obj(params or {})
        self.error = error

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            version=data["version"],
            digest=data.get("digest"),
            nltk_version=data.get("nltk_version"),
            declared_version=data.get("declared_version"),
            version_source=data.get("version_source", "content"),
            resources=data.get("resources", ()),
            params=data.get("params", {}),
            error=data.get("error"),
        )

    def to_dict(self):
        return {
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "nltk_version": self.nltk_version,
            "declared_version": self.declared_version,
            "version_source": self.version_source,
            "resources": list(self.resources),
            "params": self.params,
            "error": self.error,
        }

    def __eq__(self, other):
        return (
            isinstance(other, ProcessorFingerprint)
            and self.to_dict() == other.to_dict()
        )

    def __hash__(self):
        return hash(canonical_json(self.to_dict()))

    def __repr__(self):
        return (
            f"ProcessorFingerprint(name={self.name!r}, version={self.version!r}, "
            f"params={self.params!r})"
        )


class ProcessingResult:
    """The versions used by one processing run plus per-sample output digests.

    A result is JSON-serialisable and contains no absolute paths.  ``samples``
    maps a caller-chosen id to the output digest from :func:`sample_digest`.
    """

    __slots__ = ("schema", "nltk_version", "corpora", "processors", "samples", "meta")

    def __init__(
        self,
        corpora=(),
        processors=(),
        samples=None,
        meta=None,
        nltk_version=None,
        schema=SCHEMA,
    ):
        self.schema = schema
        self.nltk_version = str(nltk_version or _nltk_version())
        self.corpora = tuple(corpora)
        self.processors = tuple(processors)
        self.samples = dict(samples or {})
        self.meta = _canonical_obj(meta or {})

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, data, *, verify_schema=True):
        if not isinstance(data, dict):
            raise ValueError("a provenance result must be a JSON object")
        schema = data.get("schema")
        if verify_schema and schema != SCHEMA:
            raise ValueError(
                f"unsupported provenance schema {schema!r}; expected {SCHEMA!r}"
            )
        samples = data.get("samples") or {}
        if not isinstance(samples, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in samples.items()
        ):
            raise ValueError("'samples' must map sample ids to digest strings")
        return cls(
            corpora=[CorpusFingerprint.from_dict(c) for c in data.get("corpora", ())],
            processors=[
                ProcessorFingerprint.from_dict(p) for p in data.get("processors", ())
            ],
            samples=samples,
            meta=data.get("meta") or {},
            nltk_version=data.get("nltk_version"),
            schema=schema,
        )

    @classmethod
    def load(cls, path):
        """Read a result written by :meth:`save` (sandbox-checked)."""
        from nltk.pathsec import open as _safe_open

        from nltk.jsontags import safe_json_load

        with _safe_open(str(path), "r", context="nltk.provenance.load") as f:
            return cls.from_dict(safe_json_load(f, context="nltk.provenance.load"))

    def to_dict(self):
        return {
            "schema": self.schema,
            "nltk_version": self.nltk_version,
            "corpora": [c.to_dict() for c in self.corpora],
            "processors": [p.to_dict() for p in self.processors],
            "samples": dict(sorted(self.samples.items())),
            "meta": self.meta,
        }

    def to_json(self):
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )

    def save(self, path):
        """Write the result as JSON through the NLTK path sandbox."""
        from nltk.pathsec import open as _safe_open

        with _safe_open(str(path), "w", context="nltk.provenance.save") as f:
            f.write(self.to_json())
            f.write("\n")

    # -- combination -------------------------------------------------------
    def merge(self, other):
        """Combine two partial results from parallel workers.

        Both sides must agree on every corpus and processor version and on the
        parameter map; sample ids must be disjoint.  Because samples are keyed
        and results are sorted on output, the merged result does not depend on
        which worker finished first (provided ids are unique).
        """
        comparison = compare(self, other)
        if not comparison.comparable:
            raise IncomparableResultsError(
                "cannot merge results with different versions: "
                + "; ".join(comparison.reasons)
            )
        overlap = set(self.samples) & set(other.samples)
        if overlap:
            raise ValueError(
                "cannot merge results with overlapping sample ids: "
                + ", ".join(sorted(overlap))
            )
        if canonical_json(self.meta) != canonical_json(other.meta):
            raise ValueError("cannot merge results with different meta values")
        samples = dict(self.samples)
        samples.update(other.samples)
        return ProcessingResult(
            corpora=self.corpora,
            processors=self.processors,
            samples=samples,
            meta=self.meta,
            nltk_version=self.nltk_version,
        )

    def check_comparable(self, other):
        """Raise :class:`IncomparableResultsError` if versions differ."""
        comparison = compare(self, other)
        if not comparison.comparable:
            raise IncomparableResultsError(
                "results are not directly comparable: " + "; ".join(comparison.reasons)
            )
        return comparison

    def __eq__(self, other):
        return isinstance(other, ProcessingResult) and self.to_dict() == other.to_dict()

    def __repr__(self):
        return (
            f"ProcessingResult(corpora={len(self.corpora)}, "
            f"processors={len(self.processors)}, samples={len(self.samples)})"
        )


class Comparison:
    """The verdict of :func:`compare`.

    ``comparable`` is false when a corpus/processor version or configuration
    differs (or is unknown on either side) and says why in ``reasons``.  Sample
    changes are reported either way, so an upgrade can be inspected even though
    its results are not directly comparable.
    """

    __slots__ = (
        "comparable",
        "reasons",
        "changed_samples",
        "added_samples",
        "removed_samples",
        "unchanged_samples",
        "same_outputs",
    )

    def __init__(
        self,
        comparable,
        reasons,
        changed_samples=(),
        added_samples=(),
        removed_samples=(),
        unchanged_samples=(),
    ):
        self.comparable = bool(comparable)
        self.reasons = tuple(reasons)
        self.changed_samples = tuple(changed_samples)
        self.added_samples = tuple(added_samples)
        self.removed_samples = tuple(removed_samples)
        self.unchanged_samples = tuple(unchanged_samples)
        self.same_outputs = not (
            self.changed_samples or self.added_samples or self.removed_samples
        )

    def to_dict(self):
        return {
            "comparable": self.comparable,
            "reasons": list(self.reasons),
            "changed_samples": list(self.changed_samples),
            "added_samples": list(self.added_samples),
            "removed_samples": list(self.removed_samples),
            "unchanged_samples": list(self.unchanged_samples),
            "same_outputs": self.same_outputs,
        }

    def __repr__(self):
        return (
            f"Comparison(comparable={self.comparable}, "
            f"changed={len(self.changed_samples)}, reasons={self.reasons!r})"
        )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _corpus_versions(result):
    seen = {}
    for corpus in result.corpora:
        seen.setdefault(corpus.name, set()).add(corpus.version)
    return {name: sorted(versions) for name, versions in seen.items()}


def _processor_versions(result):
    seen = {}
    for processor in result.processors:
        signature = (processor.version, canonical_json(processor.params))
        seen.setdefault(processor.name, set()).add(signature)
    return {
        name: sorted(signatures, key=lambda item: (item[0], item[1]))
        for name, signatures in seen.items()
    }


def _describe_versions(values):
    return ", ".join(values)


def compare(a, b):
    """Compare two results (or two loaded JSON dicts).

    Corpus and processor versions are both checked.  A content-digest mismatch,
    an unknown digest, a changed processor parameter (language, tagset, seed)
    and a differing NLTK version all make the results not directly comparable.
    Sample output digests are diffed regardless, naming the ids that changed.

    :rtype: Comparison
    """
    if isinstance(a, dict):
        a = ProcessingResult.from_dict(a)
    if isinstance(b, dict):
        b = ProcessingResult.from_dict(b)
    if not isinstance(a, ProcessingResult) or not isinstance(b, ProcessingResult):
        raise TypeError("compare() expects ProcessingResult instances or dicts")

    reasons = []
    if a.nltk_version != b.nltk_version:
        reasons.append(f"NLTK version differs: {a.nltk_version} != {b.nltk_version}")

    corpus_a, corpus_b = _corpus_versions(a), _corpus_versions(b)
    for name in sorted(set(corpus_a) | set(corpus_b)):
        if name not in corpus_a:
            reasons.append(f"corpus {name!r} is present only in the second result")
        elif name not in corpus_b:
            reasons.append(f"corpus {name!r} is present only in the first result")
        else:
            versions_a, versions_b = corpus_a[name], corpus_b[name]
            if versions_a != versions_b:
                reasons.append(
                    f"corpus {name!r} version differs: "
                    f"{_describe_versions(versions_a)} != "
                    f"{_describe_versions(versions_b)}"
                )
            elif _NO_VERSION in versions_a:
                reasons.append(
                    f"corpus {name!r} version is unknown on both sides; "
                    "comparability cannot be established"
                )

    processor_a, processor_b = _processor_versions(a), _processor_versions(b)
    for name in sorted(set(processor_a) | set(processor_b)):
        if name not in processor_a:
            reasons.append(f"processor {name!r} is present only in the second result")
        elif name not in processor_b:
            reasons.append(f"processor {name!r} is present only in the first result")
        else:
            if processor_a[name] != processor_b[name]:
                reasons.append(
                    f"processor {name!r} version/configuration differs: "
                    f"{_describe_versions(v for v, _ in processor_a[name])} != "
                    f"{_describe_versions(v for v, _ in processor_b[name])}"
                )
            elif _NO_VERSION in {v for v, _ in processor_a[name]}:
                reasons.append(
                    f"processor {name!r} version is unknown; comparability "
                    "cannot be established"
                )

    ids_a, ids_b = set(a.samples), set(b.samples)
    changed = sorted(sid for sid in ids_a & ids_b if a.samples[sid] != b.samples[sid])
    added = sorted(ids_b - ids_a)
    removed = sorted(ids_a - ids_b)
    unchanged = sorted(sid for sid in ids_a & ids_b if a.samples[sid] == b.samples[sid])

    return Comparison(
        comparable=not reasons,
        reasons=reasons,
        changed_samples=changed,
        added_samples=added,
        removed_samples=removed,
        unchanged_samples=unchanged,
    )


# ---------------------------------------------------------------------------
# Manifest export
# ---------------------------------------------------------------------------


def _as_fingerprint(item, force=False):
    if isinstance(item, CorpusFingerprint):
        return item
    if isinstance(item, ProcessingResult):
        return None
    return corpus_fingerprint(item, force=force)


def manifest(*items, force=False):
    """Build a corpus manifest for everything a run used.

    Accepts :class:`CorpusFingerprint` objects, readers, resource names and/or a
    :class:`ProcessingResult` (whose corpora are included).  The returned dict
    is JSON-serialisable and path-free.
    """
    fingerprints = []
    for item in items:
        if isinstance(item, ProcessingResult):
            fingerprints.extend(item.corpora)
            continue
        fingerprints.append(_as_fingerprint(item, force=force))
    unique = {}
    for fingerprint in fingerprints:
        unique[(fingerprint.name, fingerprint.source_id, fingerprint.version)] = (
            fingerprint
        )
    ordered = [unique[key].to_dict() for key in sorted(unique)]
    return {
        "schema": MANIFEST_SCHEMA,
        "nltk_version": _nltk_version(),
        "corpora": ordered,
    }


def export_manifest(path, *items, force=False):
    """Write :func:`manifest` as JSON through the NLTK path sandbox."""
    from nltk.pathsec import open as _safe_open

    payload = manifest(*items, force=force)
    with _safe_open(str(path), "w", context="nltk.provenance.export_manifest") as f:
        json.dump(payload, f, sort_keys=True, ensure_ascii=False, indent=2)
        f.write("\n")
    return payload


# ---------------------------------------------------------------------------
# Recording sessions
# ---------------------------------------------------------------------------


class RecordingSession:
    """Collect corpora, processors and per-sample output digests for one run.

    Use it as a context manager (see :func:`record`).  Corpus readers built and
    built-in tokeniser/tagger models loaded while it is active are observed
    automatically; anything else can be added explicitly with
    :meth:`add_corpus`, :meth:`add_processor` and :meth:`add_sample`.

    A session is process-local and thread-safe.  Two threads may add samples
    concurrently; results stay deterministic because everything is keyed by
    sample id and sorted on output, and an id may be added only once (a second
    add with a different digest raises rather than picking a race winner).
    """

    def __init__(self, *, processor=None, params=None, capture_corpora=True):
        self._lock = threading.RLock()
        self._corpus_specs = {}
        self._processor_keys = set()
        self._samples = {}
        self._explicit_corpora = []
        self._explicit = []
        self._capture_corpora = bool(capture_corpora)
        self._entered = False
        self._stopped = False
        if processor is not None:
            if isinstance(processor, ProcessorFingerprint):
                self.add_processor(processor)
            else:
                self._explicit.append((str(processor), _canonical_obj(params or {})))
        elif params is not None:
            self._explicit.append(("nltk", _canonical_obj(params)))

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        global _ACTIVE
        with _SESSIONS_LOCK:
            if self._entered or self._stopped:
                raise RuntimeError("this recording session has already been used")
            _SESSIONS.append(self)
            _ACTIVE = True
            self._entered = True
        return self

    def stop(self):
        global _ACTIVE
        with _SESSIONS_LOCK:
            if self in _SESSIONS:
                _SESSIONS.remove(self)
            _ACTIVE = bool(_SESSIONS)
        self._stopped = True

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    # -- collection --------------------------------------------------------
    def _observe_reader(self, reader):
        if not self._capture_corpora:
            return
        try:
            root = reader.root
            # Key on the root's resolved location, not id(): an id() can be
            # reused once an earlier reader is collected.  This stays in
            # memory only; the result file records source_id, never a path.
            location = (
                f"{root.zipfile.filename}\0{root.entry}"
                if hasattr(root, "zipfile")
                else os.path.realpath(str(root))
            )
            key = (location, tuple(str(f) for f in reader.fileids()))
        except Exception:
            return
        with self._lock:
            self._corpus_specs.setdefault(key, reader)

    def _observe_processor_key(self, key):
        with self._lock:
            self._processor_keys.add(key)

    def add_corpus(self, source, *, fileids=None, force=False):
        """Explicitly include a corpus; returns its fingerprint."""
        fingerprint = corpus_fingerprint(source, fileids=fileids, force=force)
        with self._lock:
            self._explicit_corpora.append(fingerprint)
        return fingerprint

    def add_processor(self, processor):
        """Include an explicit :class:`ProcessorFingerprint`."""
        if isinstance(processor, ProcessorFingerprint):
            with self._lock:
                self._explicit.append(processor)
            return processor
        raise TypeError("add_processor() expects a ProcessorFingerprint")

    def add_sample(self, sample_id, output=None, *, digest=None):
        """Record one sample's output (or a precomputed digest).

        ``sample_id`` must be unique within the session; recording the same id
        twice with a different output is a caller error, because which value
        won would otherwise depend on scheduling.  Returns the digest.
        """
        if digest is None:
            if output is None:
                raise ValueError("add_sample() needs an output or a digest")
            digest = sample_digest(output)
        if not isinstance(digest, str) or not digest:
            raise ValueError("a sample digest must be a non-empty string")
        sample_id = str(sample_id)
        with self._lock:
            previous = self._samples.get(sample_id)
            if previous is not None and previous != digest:
                raise ValueError(
                    f"sample id {sample_id!r} was recorded twice with different "
                    "outputs; sample ids must be unique within a run"
                )
            self._samples[sample_id] = digest
        return digest

    # -- result ------------------------------------------------------------
    def result(self, *, force=False):
        """Freeze the collected run into a :class:`ProcessingResult`."""
        with self._lock:
            readers = list(self._corpus_specs.values())
            processor_keys = set(self._processor_keys)
            explicit_corpora = list(self._explicit_corpora)
            explicit = list(self._explicit)
            samples = dict(self._samples)

        corpora = []
        seen = set()
        for fingerprint in explicit_corpora:
            key = (fingerprint.name, fingerprint.source_id, fingerprint.version)
            if key not in seen:
                seen.add(key)
                corpora.append(fingerprint)
        for reader in readers:
            fingerprint = _corpus_fingerprint_from_reader(reader, force=force)
            key = (fingerprint.name, fingerprint.source_id, fingerprint.version)
            if key not in seen:
                seen.add(key)
                corpora.append(fingerprint)

        processors = []
        seen_processors = set()
        for key in sorted(processor_keys):
            fingerprint = _resolve_builtin(key)
            marker = (fingerprint.name, canonical_json(fingerprint.to_dict()))
            if marker not in seen_processors:
                seen_processors.add(marker)
                processors.append(fingerprint)
        for item in explicit:
            if isinstance(item, ProcessorFingerprint):
                fingerprint = item
            else:
                name, params = item
                fingerprint = processor_fingerprint(name, params=params)
            marker = (fingerprint.name, canonical_json(fingerprint.to_dict()))
            if marker not in seen_processors:
                seen_processors.add(marker)
                processors.append(fingerprint)

        # A result always identifies the toolkit that produced it, even when no
        # built-in processor was detected (custom processing still depends on
        # NLTK's code).
        if not processors:
            processors.append(processor_fingerprint("nltk", version=_nltk_version()))

        corpora.sort(key=lambda f: (f.name, f.source_id, f.version))
        processors.sort(key=lambda f: (f.name, f.version, canonical_json(f.params)))
        return ProcessingResult(
            corpora=corpora,
            processors=processors,
            samples=samples,
        )


def record(*, processor=None, params=None, capture_corpora=True):
    """Start recording provenance for the duration of a ``with`` block.

    :param processor: an optional processor name (or
        :class:`ProcessorFingerprint`) used by the run.  Built-in tokenisers and
        taggers are otherwise detected automatically.
    :param params: processor configuration that affects output, including a
        random seed when the processor is stochastic.
    :param capture_corpora: observe corpus readers constructed inside the
        block (default true).  Disable when opening unrelated corpora.
    :rtype: RecordingSession
    """
    return RecordingSession(
        processor=processor, params=params, capture_corpora=capture_corpora
    )


def recording(*, processor=None, params=None, capture_corpora=True):
    """Alias for :func:`record` (reads naturally as ``with recording():``)."""
    return record(processor=processor, params=params, capture_corpora=capture_corpora)


def enable(*, processor=None, params=None, capture_corpora=True):
    """Start a session that lasts until :func:`disable` is called."""
    return record(
        processor=processor, params=params, capture_corpora=capture_corpora
    ).start()


def disable():
    """Stop every active session started with :func:`enable`."""
    for session in _snapshot_sessions():
        session.stop()
