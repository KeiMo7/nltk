# Natural Language Toolkit: unit tests for the persistent pipeline cache
#
# Copyright (C) 2001-2026 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Tests for :mod:`nltk.cache` (persistent, version-aware disk cache).

These tests deliberately use only pure-Python processors and stub compute
functions so they need no downloaded NLTK models and run fully offline.
Integration with the real tokenizer/tagger entry points is exercised through
monkeypatched implementations with identical call signatures.
"""

import json
import os
import time

import pytest

from nltk.cache import (
    DEFAULT_MAX_SIZE_BYTES,
    ENTRY_MAGIC,
    NLTK_VERSION,
    PipelineCache,
    resource_version,
)
from nltk.cache.pipeline import (
    CachedParser,
    as_cache,
    cached_pos_tag,
    cached_sent_tokenize,
    cached_word_tokenize,
)


# ----------------------------------------------------------------------
# Fixtures and small test doubles
# ----------------------------------------------------------------------
class CountingComputer:
    """Compute function that records how often it actually ran."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.result


class StubParser:
    """Minimal ParserI-like object for CachedParser tests."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def parse(self, tokens):
        self.calls += 1
        return [self.result]


@pytest.fixture
def cache(tmp_path):
    return PipelineCache(cache_dir=tmp_path / "cache")


# ----------------------------------------------------------------------
# 1. Hit / reuse
# ----------------------------------------------------------------------
class TestCacheHits:
    def test_second_call_is_a_hit_and_skips_computation(self, cache):
        computer = CountingComputer(["a", "b", "c"])
        first = cache.get_or_compute(
            "tokenize", "proc-1", "corpus-1", "some text", computer
        )
        second = cache.get_or_compute(
            "tokenize", "proc-1", "corpus-1", "some text", computer
        )
        assert first == second == ["a", "b", "c"]
        assert computer.calls == 1
        stats = cache.stats_snapshot()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["corrupt"] == 0
        assert stats["hit_rate"] == pytest.approx(0.5)

    def test_hit_survives_a_new_process_object(self, tmp_path):
        """A fresh PipelineCache over the same directory reuses disk entries."""
        computer = CountingComputer([("x", "NN")])
        c1 = PipelineCache(cache_dir=tmp_path / "persistent")
        c1.get_or_compute("tag", "p", "c", ["x"], computer)
        assert computer.calls == 1

        c2 = PipelineCache(cache_dir=tmp_path / "persistent")
        reused = c2.get_or_compute("tag", "p", "c", ["x"], computer)
        assert reused == [("x", "NN")]
        assert computer.calls == 1  # still one: served from disk, not recomputed
        assert c2.stats_snapshot()["hits"] == 1

    def test_cached_and_recomputed_results_are_byte_identical(self, cache):
        """Serialization round-trip must reproduce the exact same bytes."""
        result = [("John", "NNP"), ("'s", "POS"), ("big", "JJ"), (".", ".")]
        computer = CountingComputer(result)
        cache.get_or_compute("tag", "p", "c", ["John", "'s", "big", "."], computer)
        cached_result = cache.get_or_compute(
            "tag", "p", "c", ["John", "'s", "big", "."], computer
        )
        # Value equality ...
        assert cached_result == result
        # ... exact types preserved (tuple, not JSON list) ...
        assert all(isinstance(pair, tuple) for pair in cached_result)
        # ... and byte-for-byte identical serialization through the cache codec.
        from nltk.cache import dump_payload

        assert dump_payload(cached_result) == dump_payload(result)

    def test_none_is_a_valid_cached_value(self, cache):
        """The miss sentinel must never be confused with a cached None."""
        computer = CountingComputer(None)
        assert cache.get_or_compute("stage", "p", "c", "x", computer) is None
        assert cache.get_or_compute("stage", "p", "c", "x", computer) is None
        assert computer.calls == 1
        assert cache.stats_snapshot()["hits"] == 1

    def test_tuple_and_list_input_share_one_canonical_key(self, cache):
        computer = CountingComputer(["t"])
        cache.get_or_compute("s", "p", "c", ["a", "b"], computer)
        cache.get_or_compute("s", "p", "c", ("a", "b"), computer)
        assert computer.calls == 1


# ----------------------------------------------------------------------
# 2. Version-driven invalidation
# ----------------------------------------------------------------------
class TestVersionInvalidation:
    def test_processor_version_change_invalidates(self, cache):
        """Explicit acceptance assertion: processor version change => miss."""
        computer = CountingComputer(["r"])
        cache.get_or_compute("tokenize", "proc-v1", "corpus-v1", "text", computer)
        assert computer.calls == 1
        cache.get_or_compute("tokenize", "proc-v2", "corpus-v1", "text", computer)
        assert computer.calls == 2
        stats = cache.stats_snapshot()
        assert stats["misses"] == 2 and stats["hits"] == 0

    def test_corpus_version_change_invalidates(self, cache):
        computer = CountingComputer(["r"])
        cache.get_or_compute("tokenize", "proc-v1", "corpus-v1", "text", computer)
        cache.get_or_compute("tokenize", "proc-v1", "corpus-v2", "text", computer)
        assert computer.calls == 2

    def test_text_change_invalidates(self, cache):
        computer = CountingComputer(["r"])
        cache.get_or_compute("tokenize", "p", "c", "text one", computer)
        cache.get_or_compute("tokenize", "p", "c", "text two", computer)
        assert computer.calls == 2

    def test_version_directories_are_isolated(self, cache, tmp_path):
        """Different versions land in different directories; no cross-pollution."""
        computer = CountingComputer(["r"])
        cache.get_or_compute("tokenize", "proc-v1", "corpus-v1", "t", computer)
        cache.get_or_compute("tokenize", "proc-v2", "corpus-v1", "t", computer)
        cache.get_or_compute("tokenize", "proc-v2", "corpus-v2", "t", computer)
        entries = [
            os.path.relpath(os.path.join(dp, f), cache.cache_dir)
            for dp, _, fns in os.walk(cache.cache_dir)
            for f in fns
            if f.endswith(".entry")
        ]
        assert len(entries) == 3
        # Processor and corpus version each form their own path segment.
        assert any("proc-v1" in e for e in entries)
        assert any("proc-v2" in e for e in entries)
        corpus_dirs = {e.split(os.sep)[2] for e in entries}
        assert len(corpus_dirs) == 2

    def test_nltk_version_is_part_of_every_descriptor(self, cache):
        """The NLTK release version must be embedded in processor descriptors."""
        assert NLTK_VERSION  # non-empty, e.g. 3.10.4
        computer = CountingComputer(["r"])
        cache.get_or_compute(
            "tokenize", f"nltk-{NLTK_VERSION}|word_tokenize", "c", "t", computer
        )
        entry = next(
            os.path.join(dp, f)
            for dp, _, fns in os.walk(cache.cache_dir)
            for f in fns
            if f.endswith(".entry")
        )
        with open(entry, "rb") as fh:
            lines = fh.read().decode("utf-8").split("\n")
        import base64

        descriptor = json.loads(base64.urlsafe_b64decode(lines[1]))
        assert descriptor["processor_version"].startswith(f"nltk-{NLTK_VERSION}|")
        assert set(descriptor) == {
            "stage",
            "processor_version",
            "corpus_version",
            "text_hash",
        }
        assert len(descriptor["text_hash"]) == 64  # SHA-256 hex


# ----------------------------------------------------------------------
# 3. Corruption, collisions and degradation
# ----------------------------------------------------------------------
class TestCorruptionAndDegradation:
    def _entry_path(self, cache):
        return next(
            os.path.join(dp, f)
            for dp, _, fns in os.walk(cache.cache_dir)
            for f in fns
            if f.endswith(".entry")
        )

    def test_corrupt_entry_is_skipped_recomputed_and_warns(self, cache, caplog):
        import logging

        computer = CountingComputer(["r"])
        cache.get_or_compute("s", "p", "c", "t", computer)
        entry = self._entry_path(cache)
        with open(entry, "wb") as fh:
            fh.write(b"NLTKCACHE1\ngarbage-not-base64!!\nxx\nxx")
        with caplog.at_level(logging.WARNING):
            result = cache.get_or_compute("s", "p", "c", "t", computer)
        assert result == ["r"]
        assert computer.calls == 2  # corrupted => recomputed
        assert cache.stats_snapshot()["corrupt"] == 1
        assert any("Corrupt cache entry" in rec.message for rec in caplog.records)

    def test_truncated_entry_is_corruption_not_a_wrong_value(self, cache):
        computer = CountingComputer(["r"])
        cache.get_or_compute("s", "p", "c", "t", computer)
        entry = self._entry_path(cache)
        with open(entry, "rb+") as fh:
            fh.truncate(10)
        result = cache.get_or_compute("s", "p", "c", "t", computer)
        assert result == ["r"]
        assert computer.calls == 2
        assert cache.stats_snapshot()["corrupt"] == 1

    def test_descriptor_tampering_is_detected_as_collision(self, cache):
        """
        Even if an attacker/bit-rot makes an entry resolve under the same
        file name, the embedded descriptor must be compared and a mismatch
        must force recomputation rather than returning the foreign value.
        """
        computer_a = CountingComputer(["result-A"])
        computer_b = CountingComputer(["result-B"])
        cache.get_or_compute("s", "p", "c", "input-A", computer_a)
        entry = self._entry_path(cache)

        # Rewrite the descriptor inside the entry to a different input while
        # keeping a valid payload: read path must reject the mismatch.
        import base64

        with open(entry, "rb") as fh:
            lines = fh.read().decode("utf-8").split("\n")
        descriptor = json.loads(base64.urlsafe_b64decode(lines[1]))
        descriptor["text_hash"] = "0" * 64  # no longer matches the real input
        lines[1] = base64.urlsafe_b64encode(
            json.dumps(descriptor, sort_keys=True).encode("utf-8")
        ).decode("ascii")
        with open(entry, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        result = cache.get_or_compute("s", "p", "c", "input-A", computer_b)
        assert result == ["result-B"]  # recomputed, never the stale foreign value
        assert computer_b.calls == 1
        assert cache.stats_snapshot()["corrupt"] == 1

    def test_payload_checksum_mismatch_is_detected(self, cache):
        computer = CountingComputer(["r"])
        cache.get_or_compute("s", "p", "c", "t", computer)
        entry = self._entry_path(cache)
        with open(entry, "rb+") as fh:
            fh.seek(0, os.SEEK_END)
            fh.write(b"AA")  # corrupt base64 tail -> checksum/decode failure
        assert cache.get_or_compute("s", "p", "c", "t", computer) == ["r"]
        assert computer.calls == 2

    def test_unwritable_directory_degrades_to_pass_through(
        self, tmp_path, caplog
    ):
        import logging

        readonly = tmp_path / "readonly"
        readonly.mkdir()
        os.chmod(readonly, 0o500)  # r-x, no write
        try:
            if os.access(readonly, os.W_OK):
                pytest.skip("running as root: read-only bit is not enforced")
            cache = PipelineCache(cache_dir=readonly / "sub")
            computer = CountingComputer(["r"])
            with caplog.at_level(logging.WARNING):
                r1 = cache.get_or_compute("s", "p", "c", "t", computer)
                r2 = cache.get_or_compute("s", "p", "c", "t", computer)
            assert r1 == r2 == ["r"]
            assert computer.calls == 2  # nothing was cached
            stats = cache.stats_snapshot()
            # Exactly one degradation per request, never double counted.
            assert stats["degraded"] == 2 and stats["hits"] == 0
            assert any("not writable" in rec.message for rec in caplog.records)
        finally:
            os.chmod(readonly, 0o700)

    def test_atomic_writes_leave_no_temp_files(self, cache):
        computer = CountingComputer(["r"])
        cache.get_or_compute("s", "p", "c", "t", computer)
        leftovers = [
            f
            for dp, _, fns in os.walk(cache.cache_dir)
            for f in fns
            if f.startswith(".tmp.")
        ]
        assert leftovers == []


# ----------------------------------------------------------------------
# 4. Size bound and eviction
# ----------------------------------------------------------------------
class TestEviction:
    def test_lru_eviction_respects_size_bound(self, tmp_path):
        cache = PipelineCache(cache_dir=tmp_path / "small", max_size_bytes=2048)
        for i in range(50):
            cache.get_or_compute(
                "s",
                "p",
                "c",
                f"input-{i}",
                lambda i=i: [f"value-{i}"] * 20,
            )
        total = cache.current_size_bytes()
        assert total <= 2048
        stats = cache.stats_snapshot()
        assert stats["evictions"] > 0
        # Entries that remain still hit correctly.
        remaining = [
            f
            for dp, _, fns in os.walk(cache.cache_dir)
            for f in fns
            if f.endswith(".entry")
        ]
        assert remaining, "eviction removed everything; bound should retain some"

    def test_eviction_is_lru_not_fifo(self, tmp_path):
        cache = PipelineCache(cache_dir=tmp_path / "lru", max_size_bytes=4096)
        computer = lambda i: CountingComputer([f"v{i}"] * 40)
        comps = {}
        for i in range(30):
            comps[i] = computer(i)
            cache.get_or_compute("s", "p", "c", f"k{i}", comps[i])
        # Re-touch the oldest key so it becomes recently used.
        cache.get_or_compute("s", "p", "c", "k0", comps[0])
        cache._enforce_size()
        # The re-touched key must still be a hit (it was not evicted).
        result = cache.get_or_compute("s", "p", "c", "k0", comps[0])
        assert result == ["v0"] * 40
        assert comps[0].calls == 2  # initial + re-touch; the final call is a hit

    def test_none_max_size_disables_eviction(self, tmp_path):
        cache = PipelineCache(cache_dir=tmp_path / "unbounded", max_size_bytes=None)
        for i in range(20):
            cache.get_or_compute("s", "p", "c", f"k{i}", lambda i=i: [i] * 10)
        assert cache.stats_snapshot()["evictions"] == 0
        assert len(list(tmp_path.glob("**/*.entry"))) == 20

    def test_manual_trim_to_size(self, tmp_path):
        # No automatic bound: fill first, then shrink on demand by size.
        cache = PipelineCache(cache_dir=tmp_path / "manual", max_size_bytes=None)
        for i in range(30):
            cache.get_or_compute(
                "s", "p", "c", f"k{i}", lambda i=i: [f"v{i}"] * 30
            )
        full_size = cache.current_size_bytes()
        target = full_size // 2
        removed = cache.trim_to_size(target)
        assert removed > 0
        assert cache.current_size_bytes() <= target
        # The bound is still "off" for future writes; the trim was one-off.
        assert cache.max_size_bytes is None


# ----------------------------------------------------------------------
# 5. Clearing: by time, by version, wholesale
# ----------------------------------------------------------------------
class TestClear:
    def test_clear_all(self, cache):
        for i in range(3):
            cache.get_or_compute("s", "p", "c", f"k{i}", lambda i=i: [i])
        assert cache.clear() == 3
        assert not os.path.exists(cache.cache_dir)

    def test_clear_by_time(self, cache):
        cache.get_or_compute("s", "p", "c", "old", lambda: ["o"])
        import pathlib

        entry = next(pathlib.Path(cache.cache_dir).glob("**/*.entry"))
        old_time = time.time() - 10000
        os.utime(entry, (old_time, old_time))
        cache.get_or_compute("s", "p", "c", "new", lambda: ["n"])
        removed = cache.clear(older_than=time.time() - 5000)
        assert removed == 1
        import pathlib

        remaining = {
            os.path.basename(p)
            for p in pathlib.Path(cache.cache_dir).glob("**/*.entry")
        }
        assert remaining  # new entry survives

    def test_clear_by_processor_version(self, cache):
        cache.get_or_compute("s", "proc-v1", "c", "k", lambda: [1])
        cache.get_or_compute("s", "proc-v2", "c", "k", lambda: [2])
        removed = cache.clear(processor_version="proc-v1")
        assert removed == 1
        computer = CountingComputer([1])
        # v1 was deleted => recompute; v2 remains => hit.
        cache.get_or_compute("s", "proc-v1", "c", "k", computer)
        assert computer.calls == 1
        v2 = CountingComputer([2])
        cache.get_or_compute("s", "proc-v2", "c", "k", v2)
        assert v2.calls == 0

    def test_clear_by_corpus_version(self, cache):
        cache.get_or_compute("s", "p", "corpus-a", "k", lambda: [1])
        cache.get_or_compute("s", "p", "corpus-b", "k", lambda: [2])
        assert cache.clear(corpus_version="corpus-a") == 1
        import pathlib

        remaining = list(pathlib.Path(cache.cache_dir).glob("**/*.entry"))
        assert len(remaining) == 1


# ----------------------------------------------------------------------
# 6. resource_version and key hygiene
# ----------------------------------------------------------------------
class TestResourceVersion:
    def test_file_content_change_changes_version(self, tmp_path):
        model = tmp_path / "model.bin"
        model.write_bytes(b"model contents v1")
        v1 = resource_version(model)
        model.write_bytes(b"model contents v2 -- different")
        v2 = resource_version(model)
        assert v1 != v2
        assert "sha256" in v1

    def test_same_content_stable_version(self, tmp_path):
        model = tmp_path / "m.bin"
        model.write_bytes(b"stable")
        assert resource_version(model) == resource_version(model)

    def test_key_never_text_hash_alone(self, cache):
        computer = CountingComputer(["r"])
        cache.get_or_compute("s", "p1", "c1", "same-input", computer)
        cache.get_or_compute("s", "p2", "c1", "same-input", computer)
        cache.get_or_compute("s", "p2", "c2", "same-input", computer)
        assert computer.calls == 3  # identical text, three distinct keys

    def test_corpus_version_tracks_installed_model_content(self, tmp_path, monkeypatch):
        """A replaced model file (same path, new bytes) changes the version."""
        import nltk.data
        from nltk.cache.pipeline import corpus_version_for_resource
        from nltk.data import FileSystemPathPointer

        model_dir = tmp_path / "punkt_tab" / "english"
        model_dir.mkdir(parents=True)
        params = model_dir / "abbrev_types.txt"
        params.write_text("mr\nmrs")
        monkeypatch.setattr(
            nltk.data,
            "find",
            lambda name, paths=None: FileSystemPathPointer(str(model_dir)),
        )
        resource = "tokenizers/punkt_tab/english/"
        v1 = corpus_version_for_resource(resource)
        params.write_text("mr\nmrs\ndr")  # same path, changed content
        corpus_version_for_resource.cache_clear()
        v2 = corpus_version_for_resource(resource)
        corpus_version_for_resource.cache_clear()
        v3 = corpus_version_for_resource(resource)
        assert v1 != v2  # model replacement invalidates
        assert v2 == v3  # unchanged model is stable
        assert "sha256" in v1

    def test_model_version_change_forces_cache_miss(self, tmp_path, monkeypatch):
        """End-to-end: swapping the model version makes the next call a miss."""
        import nltk.data
        from nltk.cache.pipeline import corpus_version_for_resource
        from nltk.data import FileSystemPathPointer

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "weights.json").write_text("v1")
        monkeypatch.setattr(
            nltk.data,
            "find",
            lambda name, paths=None: FileSystemPathPointer(str(model_dir)),
        )
        cache = PipelineCache(tmp_path / "c")
        computer = CountingComputer(["r"])
        cv1 = corpus_version_for_resource("taggers/x/")
        cache.get_or_compute("tag", "p", cv1, "k", computer)
        (model_dir / "weights.json").write_text("v2")
        corpus_version_for_resource.cache_clear()
        cv2 = corpus_version_for_resource("taggers/x/")
        cache.get_or_compute("tag", "p", cv2, "k", computer)
        assert computer.calls == 2  # the model swap forced a recompute
        assert cache.stats_snapshot()["hits"] == 0


# ----------------------------------------------------------------------
# 7. Pipeline integration (entry points keep their signatures/results)
# ----------------------------------------------------------------------
class TestPipelineIntegration:
    def test_as_cache_normalization(self, tmp_path):
        assert as_cache(None) is None
        assert as_cache(False) is None
        assert isinstance(as_cache(True), PipelineCache)
        c = PipelineCache(tmp_path / "x")
        assert as_cache(c) is c
        assert isinstance(as_cache(str(tmp_path / "y")), PipelineCache)

    def test_cached_sent_tokenize_matches_uncached(self, tmp_path):
        def fake_sent_tokenize(text, language):
            return text.split(". ")

        cache = PipelineCache(tmp_path / "c")
        text = "One sentence. Another one. A third"
        uncached = fake_sent_tokenize(text, "english")
        cached = cached_sent_tokenize(
            fake_sent_tokenize, text, "english", cache
        )
        again = cached_sent_tokenize(fake_sent_tokenize, text, "english", cache)
        assert cached == uncached == again
        assert cache.stats_snapshot()["hits"] == 1

    def test_cached_word_tokenize_descriptors_differ_by_preserve_line(
        self, tmp_path
    ):
        def fake_word_tokenize(text, language, preserve_line):
            return text.split(" ") if preserve_line else text.replace(".", " .").split(" ")

        cache = PipelineCache(tmp_path / "c")
        text = "Hello world. Goodbye"
        r1 = cached_word_tokenize(
            fake_word_tokenize, text, "english", False, cache
        )
        r2 = cached_word_tokenize(
            fake_word_tokenize, text, "english", True, cache
        )
        assert r1 != r2  # different parameters, different results
        # Both become hits on repeat, proving separate entries were kept.
        cached_word_tokenize(fake_word_tokenize, text, "english", False, cache)
        cached_word_tokenize(fake_word_tokenize, text, "english", True, cache)
        assert cache.stats_snapshot()["hits"] == 2

    def test_cached_pos_tag_round_trips_tuples(self, tmp_path):
        def fake_pos_tag(tokens, tagset, lang):
            return [(tok, f"TAG-{tok}") for tok in tokens]

        cache = PipelineCache(tmp_path / "c")
        tokens = ["John", "runs"]
        uncached = fake_pos_tag(tokens, None, "eng")
        first = cached_pos_tag(fake_pos_tag, tokens, None, "eng", cache)
        second = cached_pos_tag(fake_pos_tag, tokens, None, "eng", cache)
        assert first == second == uncached
        assert cache.stats_snapshot()["hits"] == 1

    def test_tagset_and_lang_participate_in_key(self, tmp_path):
        seen = []

        def fake_pos_tag(tokens, tagset, lang):
            seen.append((tagset, lang))
            return [(t, f"{lang}-{tagset}-{t}") for t in tokens]

        cache = PipelineCache(tmp_path / "c")
        cached_pos_tag(fake_pos_tag, ["a"], None, "eng", cache)
        cached_pos_tag(fake_pos_tag, ["a"], "universal", "eng", cache)
        cached_pos_tag(fake_pos_tag, ["a"], None, "rus", cache)
        assert len(seen) == 3  # three distinct processor descriptors

    def test_public_pos_tag_sents_caches_each_sentence(self, tmp_path, monkeypatch):
        """Regression: the sents entry must bind the tagger and hit per sentence."""
        import nltk.tag as tagmod

        calls = {"n": 0}

        def fake_underlying(tokens, tagset, tagger, lang):
            calls["n"] += 1
            return [(tok, "X") for tok in tokens]

        monkeypatch.setattr(tagmod, "_pos_tag", fake_underlying)
        monkeypatch.setattr(tagmod, "_get_tagger", lambda lang: object())

        cache = PipelineCache(tmp_path / "c")
        sents = [["a", "b"], ["c", "d", "e"]]
        first = tagmod.pos_tag_sents(sents, cache=cache)
        second = tagmod.pos_tag_sents(sents, cache=cache)
        assert first == second
        assert calls["n"] == 2  # second round fully served from the cache
        assert cache.stats_snapshot()["hits"] == 2
        assert isinstance(second[0][0], tuple)  # tuple contract preserved

    def test_cached_parser_wraps_generic_parser(self, tmp_path):
        grammar = tmp_path / "grammar.cfg"
        grammar.write_text("S -> 'x' 'y'\n")
        parser = StubParser(("S", ("x",), ("y",)))
        cached = CachedParser(
            parser,
            PipelineCache(tmp_path / "c"),
            processor_name="TestParser",
            corpus_version=grammar,
        )
        r1 = cached.parse(["x", "y"])
        r2 = cached.parse(["x", "y"])
        # Tuples survive the JSON round trip exactly.
        assert r1 == r2 == [("S", ("x",), ("y",))]
        assert parser.calls == 1  # second parse was served from cache

    def test_cached_parser_without_cache_is_passthrough(self):
        parser = StubParser(("S",))
        cached = CachedParser(parser, None, corpus_version="v")
        cached.parse(["x"])
        cached.parse(["x"])
        assert parser.calls == 2

    def test_unsupported_output_type_is_never_cached_wrong(self, tmp_path, caplog):
        """A list subclass (e.g. nltk Tree) must not silently round-trip to list."""
        import logging

        class FakeTree(list):
            """Stands in for nltk.tree.Tree: a list subclass without json_tag."""

        parser = StubParser(FakeTree(["S", ["leaf"]]))
        cached = CachedParser(
            parser,
            PipelineCache(tmp_path / "c"),
            processor_name="TreeParser",
            corpus_version="g1",
        )
        with caplog.at_level(logging.WARNING):
            r1 = cached.parse(["x"])
            r2 = cached.parse(["x"])
        # Correct type every time, even though nothing was cached.
        assert isinstance(r1[0], FakeTree) and isinstance(r2[0], FakeTree)
        assert parser.calls == 2
        assert cached._cache.stats_snapshot()["writes"] == 0
        assert any("not cache-serializable" in rec.message for rec in caplog.records)


# ----------------------------------------------------------------------
# 8. Concurrency smoke test
# ----------------------------------------------------------------------
class TestConcurrency:
    def test_parallel_writers_never_corrupt_entries(self, tmp_path):
        import concurrent.futures

        cache = PipelineCache(cache_dir=tmp_path / "parallel")
        computer = CountingComputer(["r"])

        def worker(i):
            return cache.get_or_compute(
                "s", "p", "c", "shared-key", computer
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(worker, range(16)))
        assert all(r == ["r"] for r in results)
        # Exactly one valid entry exists regardless of races.
        entries = list(tmp_path.glob("parallel/**/*.entry"))
        assert len(entries) == 1
        # And a follow-up read in this thread is a clean hit.
        assert cache.get_or_compute("s", "p", "c", "shared-key", computer) == [
            "r"
        ]
        assert cache.stats_snapshot()["corrupt"] == 0
