# Natural Language Toolkit: pipeline integration for the disk cache
#
# Copyright (C) 2001-2026 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Glue between :class:`~nltk.cache.PipelineCache` and the standard pipeline
entry points (``sent_tokenize`` / ``word_tokenize`` / ``pos_tag`` /
``pos_tag_sents``) and a small :class:`CachedParser` wrapper for any
:class:`~nltk.parse.api.ParserI`.

Everything here is inert until a caller passes ``cache=...``. The cached and
uncached code paths share the same underlying calls, so their outputs are
identical value-for-value and serialize to the same bytes.
"""

import functools
import hashlib
import os

from nltk.cache import NLTK_VERSION, PipelineCache

# Processor descriptors. These strings are part of the on-disk key format:
# change one only when the processor code/parameters change semantics, because
# every edit invalidates previously written entries for that stage.
SENT_TOKENIZE_PROCESSOR = "nltk-{ver}|sent_tokenize|preserve_line=False"
WORD_TOKENIZE_PROCESSOR = "nltk-{ver}|word_tokenize|preserve_line={pl}"
POS_TAG_PROCESSOR = "nltk-{ver}|pos_tag|lang={lang}|tagset={tagset}"
PARSE_PROCESSOR = "nltk-{ver}|CachedParser|{name}"

# Resource names resolved through nltk.data.find for the default processors.
_PUNKT_RESOURCE = "tokenizers/punkt_tab/{lang}/"
_TAGGER_RESOURCE = "taggers/averaged_perceptron_tagger_{lang}/"


def as_cache(cache):
    """
    Normalize the public ``cache=`` argument.

    Accepts ``None``/``False`` (caching off), a :class:`PipelineCache`,
    a directory path, or ``True`` (a cache in the default location).
    """
    if cache is None or cache is False:
        return None
    if isinstance(cache, PipelineCache):
        return cache
    if cache is True:
        return PipelineCache()
    return PipelineCache(cache_dir=os.fspath(cache))


@functools.lru_cache(maxsize=32)
def corpus_version_for_resource(resource_name):
    """
    Return a content-based version string for an installed NLTK resource.

    For a filesystem resource the version covers every file below the
    resource path (path, size and SHA-256 per file). For a zip-hosted
    resource the containing archive's path, size and digest are used. The
    result is memoized per process: hashing a multi-megabyte model on every
    document would otherwise erase the time savings the cache exists for.
    Replacing the model file changes the digest and therefore invalidates
    every entry derived from the old model.
    """
    from nltk.data import FileSystemPathPointer, ZipFilePathPointer, find
    from nltk.pathsec import open as pathsec_open

    pointer = find(resource_name)
    digest = hashlib.sha256()

    if isinstance(pointer, ZipFilePathPointer):
        archive = pointer.zipfile.filename
        digest.update(os.path.abspath(archive).encode("utf-8"))
        digest.update(str(os.path.getsize(archive)).encode("ascii"))
        # The archive is an installed resource located via nltk.data.find,
        # so it resolves inside a trusted data root: open it via the sandbox.
        with pathsec_open(archive, "rb", context="cache.resource_version") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(pointer.entry.encode("utf-8"))
        return f"zip:{resource_name}:sha256:{digest.hexdigest()}"

    if isinstance(pointer, FileSystemPathPointer):
        root = pointer.path
    else:  # pragma: no cover - defensive: unknown pointer flavour
        root = str(pointer)

    digest.update(os.path.abspath(root).encode("utf-8"))
    if os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                if os.path.islink(path) or not os.path.isfile(path):
                    continue
                rel = os.path.relpath(path, root)
                digest.update(rel.encode("utf-8"))
                digest.update(str(os.path.getsize(path)).encode("ascii"))
                with pathsec_open(
                    path, "rb", context="cache.resource_version"
                ) as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        digest.update(chunk)
    else:
        digest.update(str(os.path.getsize(root)).encode("ascii"))
        with pathsec_open(root, "rb", context="cache.resource_version") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"fs:{resource_name}:sha256:{digest.hexdigest()}"


# ----------------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------------
def cached_sent_tokenize(sent_tokenize_impl, text, language, cache):
    """Cache wrapper around the real ``sent_tokenize`` implementation."""
    cache = as_cache(cache)
    if cache is None:
        return sent_tokenize_impl(text, language)
    processor = SENT_TOKENIZE_PROCESSOR.format(ver=NLTK_VERSION)
    try:
        corpus_version = corpus_version_for_resource(
            _PUNKT_RESOURCE.format(lang=language)
        )
    except (LookupError, OSError):
        # Let the underlying implementation raise its normal error on miss;
        # only skip the version lookup when the resource cannot be resolved.
        corpus_version = f"unresolved:{_PUNKT_RESOURCE.format(lang=language)}"
    return cache.get_or_compute(
        stage="sent_tokenize",
        processor_version=f"{processor}|lang={language}",
        corpus_version=corpus_version,
        input_value=text,
        compute_fn=lambda: sent_tokenize_impl(text, language),
    )


def cached_word_tokenize(word_tokenize_impl, text, language, preserve_line, cache):
    """Cache wrapper around the real ``word_tokenize`` implementation."""
    cache = as_cache(cache)
    if cache is None:
        return word_tokenize_impl(text, language, preserve_line)
    processor = WORD_TOKENIZE_PROCESSOR.format(ver=NLTK_VERSION, pl=preserve_line)
    try:
        if preserve_line:
            # No Punkt model is consulted in this mode.
            corpus_version = "preserve_line:no-punkt"
        else:
            corpus_version = corpus_version_for_resource(
                _PUNKT_RESOURCE.format(lang=language)
            )
    except (LookupError, OSError):
        corpus_version = f"unresolved:{_PUNKT_RESOURCE.format(lang=language)}"
    return cache.get_or_compute(
        stage="word_tokenize",
        processor_version=f"{processor}|lang={language}",
        corpus_version=corpus_version,
        input_value=text,
        compute_fn=lambda: word_tokenize_impl(text, language, preserve_line),
    )


# ----------------------------------------------------------------------
# Tagging
# ----------------------------------------------------------------------
def cached_pos_tag(pos_tag_impl, tokens, tagset, lang, cache):
    """Cache wrapper around the real ``pos_tag`` implementation."""
    cache = as_cache(cache)
    if cache is None:
        return pos_tag_impl(tokens, tagset, lang)
    processor = POS_TAG_PROCESSOR.format(
        ver=NLTK_VERSION, lang=lang, tagset=tagset
    )
    try:
        corpus_version = corpus_version_for_resource(
            _TAGGER_RESOURCE.format(lang=lang)
        )
    except (LookupError, OSError):
        corpus_version = f"unresolved:{_TAGGER_RESOURCE.format(lang=lang)}"
    return cache.get_or_compute(
        stage="pos_tag",
        processor_version=processor,
        corpus_version=corpus_version,
        # A tuple and a list of the same tokens canonicalize to one key, and
        # the payload codec restores tuples, so the list(tuple(str, str))
        # return contract survives the disk round trip.
        input_value=list(tokens),
        compute_fn=lambda: pos_tag_impl(tokens, tagset, lang),
    )


# ----------------------------------------------------------------------
# Parsing (generic wrapper for any ParserI)
# ----------------------------------------------------------------------
class CachedParser:
    """
    Wrap any parser exposing ``parse(tokens)`` (and optionally
    ``parse_sents(sentences)``) with a :class:`PipelineCache`.

    The parser itself supplies the corpus version through a caller-provided
    model path or explicit version string, so grammar/model changes invalidate
    cached parses. Outputs must be JSON-tag serializable; if they are not, the
    wrapper transparently skips caching for those inputs.

    :param parser: A parser following :class:`~nltk.parse.api.ParserI`.
    :param cache: A :class:`PipelineCache`, directory path, or ``True`` for
        the default cache location.
    :param processor_name: Stable name of the parsing algorithm (defaults to
        the parser class name). Embedded in the processor version.
    :param corpus_version: Explicit corpus/grammar/model version string, or a
        path whose content digest is used via
        :func:`~nltk.cache.resource_version`.
    """

    def __init__(self, parser, cache, processor_name=None, corpus_version=None):
        from nltk.cache import resource_version

        self._parser = parser
        self._cache = as_cache(cache)
        self._name = processor_name or parser.__class__.__name__
        if corpus_version is None:
            corpus_version = "unspecified"
        elif os.path.exists(os.fspath(corpus_version)):
            corpus_version = resource_version(corpus_version)
        self._corpus_version = str(corpus_version)
        self._processor = PARSE_PROCESSOR.format(
            ver=NLTK_VERSION, name=self._name
        )

    def parse(self, tokens):
        if self._cache is None:
            return self._parser.parse(tokens)
        return self._cache.get_or_compute(
            stage="parse",
            processor_version=self._processor,
            corpus_version=self._corpus_version,
            input_value=list(tokens),
            compute_fn=lambda: list(self._parser.parse(tokens)),
        )

    def parse_sents(self, sentences):
        return [
            self.parse(sent) for sent in sentences
        ]

    def __repr__(self):
        return f"<CachedParser {self._name!r} cache={self._cache!r}>"


__all__ = [
    "as_cache",
    "corpus_version_for_resource",
    "cached_sent_tokenize",
    "cached_word_tokenize",
    "cached_pos_tag",
    "CachedParser",
]
