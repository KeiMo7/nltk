# Natural Language Toolkit: provenance tests
#
# Copyright (C) 2001-2026 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""Corpus/processor provenance: versioning by content, comparison, degradation.

The four acceptance cases live here:

1. a normal run records a content-derived version for corpora and processor;
2. two results with different corpus versions are reported as not comparable;
3. a missing corpus fails with a download instruction;
4. re-running after an upgrade names the samples whose output changed.

Everything is opt-in: with no recording session, existing calls and their
values are untouched.
"""

import json
import os
import threading

import pytest

import nltk
import nltk.data
import nltk.provenance as provenance
from nltk.corpus.reader.plaintext import PlaintextCorpusReader


@pytest.fixture(autouse=True)
def _isolate():
    """Every test starts with no session, no in-memory cache and the cache off
    on disk, so one test's digests cannot mask another's file change."""
    provenance.disable()
    provenance._CACHE.clear()
    provenance._CACHE._mem.clear()
    provenance._CACHE._disk.clear()
    yield
    provenance.disable()


def make_corpus(root, files):
    """Create a plaintext corpus under *root* and return its reader."""
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return PlaintextCorpusReader(str(root), r".*\.txt")


def upgrade(corpus_root, name, text):
    """Replace one file in a corpus, as a data upgrade would."""
    path = corpus_root / name
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Normal processing records content-based versions
# ---------------------------------------------------------------------------


def test_normal_run_records_content_versions(tmp_path):
    root = tmp_path / "demo_corpus"
    reader = make_corpus(root, {"a.txt": "one two three\n", "b.txt": "four five\n"})
    processor = provenance.processor_fingerprint(
        "test.whitespace", resources=(str(root),), params={"language": "english"}
    )

    with provenance.record(capture_corpora=False) as run:
        run.add_corpus(reader)
        run.add_processor(processor)
        for i, word in enumerate(reader.words()):
            run.add_sample(f"s{i}", word.lower())
        result = run.result()

    assert result.schema == provenance.SCHEMA
    assert len(result.corpora) == 1
    corpus = result.corpora[0]
    assert corpus.name == "demo_corpus"
    assert corpus.entries == 2
    assert corpus.version.startswith("sha256:")
    assert corpus.version_source == "content"
    assert len(result.processors) == 1
    assert result.processors[0].version.startswith("sha256:")
    assert result.processors[0].params == {"language": "english"}
    assert sorted(result.samples) == ["s0", "s1", "s2", "s3", "s4"]


def test_version_is_content_not_directory_name(tmp_path):
    """Assertion required by the acceptance criteria: two identically-named
    corpora in different directories with the same bytes get the same version;
    a changed byte changes it."""
    a = make_corpus(tmp_path / "copy_one" / "same_name", {"x.txt": "alpha beta\n"})
    b = make_corpus(tmp_path / "copy_two" / "same_name", {"x.txt": "alpha beta\n"})
    assert a.root != b.root
    assert a.root.path != b.root.path

    va = provenance.corpus_fingerprint(a).version
    vb = provenance.corpus_fingerprint(b).version
    assert va == vb

    upgrade(tmp_path / "copy_two" / "same_name", "x.txt", "alpha beta gamma\n")
    changed = provenance.corpus_fingerprint(b, force=True).version
    assert changed != va


def test_version_ignores_download_time_and_directory(tmp_path):
    root = tmp_path / "timed"
    make_corpus(root, {"t.txt": "same\n"})
    first = provenance.corpus_fingerprint(str(root)).version
    # Touch the file's mtime only; the bytes are unchanged.
    stat = os.stat(root / "t.txt")
    os.utime(root / "t.txt", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    second = provenance.corpus_fingerprint(str(root), force=True).version
    assert first == second


def test_relative_entry_count_and_no_absolute_paths(tmp_path):
    root = tmp_path / "portable"
    make_corpus(root, {"a.txt": "x\n", "sub/b.txt": "y\n"})
    result = provenance.ProcessingResult(
        corpora=[provenance.corpus_fingerprint(root)],
        samples={"s": provenance.sample_digest(["x"])},
    )
    text = result.to_json()
    assert str(tmp_path) not in text
    assert str(root) not in text
    assert result.corpora[0].entries == 2


def test_sample_digest_is_order_insensitive_for_mappings(tmp_path):
    one = provenance.sample_digest({"a": [("x", "NN")], "b": 1})
    two = provenance.sample_digest({"b": 1, "a": [["x", "NN"]]})
    assert one == two
    assert one != provenance.sample_digest({"a": [["x", "VB"]], "b": 1})


def test_run_round_trips_through_json(tmp_path):
    root = tmp_path / "roundtrip"
    make_corpus(root, {"a.txt": "hello world\n"})
    result = provenance.ProcessingResult(
        corpora=[provenance.corpus_fingerprint(root)],
        samples={"s0": provenance.sample_digest(["hello", "world"])},
    )
    path = tmp_path / "run.json"
    result.save(str(path))
    loaded = provenance.ProcessingResult.load(str(path))
    assert loaded == result


# ---------------------------------------------------------------------------
# 2. Version mismatch is reported as not directly comparable
# ---------------------------------------------------------------------------


def test_version_mismatch_is_not_comparable(tmp_path):
    corpus = tmp_path / "evolving"
    make_corpus(corpus, {"a.txt": "old text\n"})
    before = provenance.ProcessingResult(
        corpora=[provenance.corpus_fingerprint(corpus)],
        samples={"s0": provenance.sample_digest(["old", "text"])},
    )

    upgrade(corpus, "a.txt", "new text\n")
    after = provenance.ProcessingResult(
        corpora=[provenance.corpus_fingerprint(corpus, force=True)],
        samples={"s0": provenance.sample_digest(["new", "text"])},
    )

    comparison = provenance.compare(before, after)
    assert not comparison.comparable
    assert any("evolving" in reason for reason in comparison.reasons)
    assert comparison.changed_samples == ("s0",)

    with pytest.raises(provenance.IncomparableResultsError):
        after.check_comparable(before)


def test_processor_version_mismatch_is_not_comparable():
    first = provenance.ProcessingResult(
        processors=[
            provenance.processor_fingerprint("p", version="1", params={"seed": 1})
        ],
        samples={"s": "sha256:aa"},
    )
    second = provenance.ProcessingResult(
        processors=[
            provenance.processor_fingerprint("p", version="2", params={"seed": 1})
        ],
        samples={"s": "sha256:aa"},
    )
    comparison = provenance.compare(first, second)
    assert not comparison.comparable
    assert any("processor 'p'" in reason for reason in comparison.reasons)


def test_unknown_version_is_not_comparable(tmp_path, monkeypatch):
    missing = tmp_path / "gone"
    # A reader whose files exist now; digesting is then forced to fail.
    reader = make_corpus(tmp_path / "flaky", {"a.txt": "x\n"})
    good = provenance.corpus_fingerprint(reader)

    def explode(*_args, **_kwargs):
        raise OSError("simulated read failure")

    monkeypatch.setattr(provenance, "_stream_digest", explode)
    with pytest.warns(provenance.ProvenanceWarning):
        unknown = provenance.corpus_fingerprint(reader, force=True)
    assert unknown.version == "unknown"
    assert unknown.version_source == "unknown"

    one = provenance.ProcessingResult(corpora=[good], samples={})
    two = provenance.ProcessingResult(corpora=[unknown], samples={})
    comparison = provenance.compare(one, two)
    assert not comparison.comparable
    assert any("unknown" in reason for reason in comparison.reasons)
    assert missing.exists() is False


# ---------------------------------------------------------------------------
# 3. A missing corpus fails with a download instruction
# ---------------------------------------------------------------------------


def test_missing_corpus_names_the_download(monkeypatch):
    def not_found(_name):
        raise LookupError("no such resource")

    monkeypatch.setattr(nltk.data, "find", not_found)
    with pytest.raises(provenance.CorpusUnavailableError) as excinfo:
        provenance.corpus_fingerprint("definitely_not_installed_155")
    message = str(excinfo.value)
    assert "not installed" in message
    assert "nltk.download('definitely_not_installed_155')" in message


def test_missing_resource_name_uses_its_package(monkeypatch):
    def not_found(_name):
        raise LookupError("no such resource")

    monkeypatch.setattr(nltk.data, "find", not_found)
    with pytest.raises(provenance.CorpusUnavailableError) as excinfo:
        provenance.corpus_fingerprint("tokenizers/punkt_tab/english/")
    assert "nltk.download('punkt_tab')" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. After an upgrade, re-running names changed samples
# ---------------------------------------------------------------------------


def test_upgrade_run_reports_changed_samples(tmp_path):
    corpus = tmp_path / "upgraded"
    make_corpus(corpus, {"a.txt": "alpha\n", "b.txt": "beta\n"})

    def run_once():
        session = provenance.record(capture_corpora=False)
        with session:
            session.add_corpus(corpus)
            for pid, name in (("p1", "a.txt"), ("p2", "b.txt")):
                text = (corpus / name).read_text(encoding="utf-8")
                session.add_sample(pid, text.split())
            return session.result()

    baseline = run_once()

    # Upgrade: p2 now tokenises differently, p1 does not.
    upgrade(corpus, "b.txt", "beta gamma\n")
    upgraded = run_once()

    comparison = provenance.compare(baseline, upgraded)
    assert not comparison.comparable  # the corpus content changed
    assert comparison.changed_samples == ("p2",)
    assert comparison.unchanged_samples == ("p1",)


# ---------------------------------------------------------------------------
# Opt-in / backwards compatibility
# ---------------------------------------------------------------------------


def test_no_session_means_no_recorded_versions(tmp_path):
    assert provenance.is_recording() is False
    reader = make_corpus(tmp_path / "quiet", {"a.txt": "x\n"})
    # Existing reader API is unchanged whether or not provenance exists.
    assert list(reader.words()) == ["x"]
    assert reader.fileids() == ["a.txt"]


def test_existing_corpus_api_is_unchanged(tmp_path):
    root = tmp_path / "plain"
    reader = make_corpus(root, {"a.txt": "Hello there.\nSecond line.\n"})
    assert list(reader.words()) == ["Hello", "there", ".", "Second", "line", "."]
    # sentence splitting needs the punkt model, which is optional; use a
    # regexp sentence tokenizer so the test does not depend on it.
    reader = PlaintextCorpusReader(
        str(root),
        r".*\.txt",
        sent_tokenizer=nltk.RegexpTokenizer(r"(?<=[.!?])\s+", gaps=True),
    )
    assert list(reader.sents()) == [
        ["Hello", "there", "."],
        ["Second", "line", "."],
    ]


def test_reader_observed_only_inside_session(tmp_path):
    outside = make_corpus(tmp_path / "outside", {"a.txt": "x\n"})
    assert outside is not None

    inside_root = tmp_path / "inside"
    with provenance.record() as session:
        make_corpus(inside_root, {"b.txt": "y\n"})
        result = session.result()

    names = {corpus.name for corpus in result.corpora}
    assert "inside" in names
    assert "outside" not in names


# ---------------------------------------------------------------------------
# Caching and cost
# ---------------------------------------------------------------------------


def test_repeated_fingerprint_reuses_cached_digest(tmp_path, monkeypatch):
    root = tmp_path / "cached"
    make_corpus(root, {"a.txt": "one\n", "b.txt": "two\n"})
    first = provenance.corpus_fingerprint(root)

    reads = []

    def counting(pointer):
        reads.append(str(pointer))
        return "deadbeef"

    monkeypatch.setattr(provenance, "_stream_digest", counting)
    second = provenance.corpus_fingerprint(root)
    assert second.version == first.version
    assert reads == []  # served from the stat-keyed cache, no second read


def test_changed_size_invalidates_cache(tmp_path):
    root = tmp_path / "invalidated"
    make_corpus(root, {"a.txt": "one\n"})
    first = provenance.corpus_fingerprint(root).version
    # Same name and mtime second, different bytes/length: the stat signature
    # differs, so the cache must not be reused.
    upgrade(root, "a.txt", "one two three four\n")
    second = provenance.corpus_fingerprint(root).version
    assert second != first


# ---------------------------------------------------------------------------
# Path hygiene and determinism
# ---------------------------------------------------------------------------


def test_absolute_path_parameters_are_collapsed(tmp_path):
    fingerprint = provenance.processor_fingerprint(
        "custom",
        params={"model": str(tmp_path / "weights.json"), "flag": True},
    )
    assert fingerprint.params["model"] == "<absolute>/weights.json"
    assert str(tmp_path) not in json.dumps(fingerprint.to_dict())


def test_manifest_export_has_no_absolute_paths(tmp_path):
    root = tmp_path / "manifested"
    make_corpus(root, {"a.txt": "x\n"})
    payload = provenance.export_manifest(str(tmp_path / "manifest.json"), root)
    text = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert payload["schema"] == provenance.MANIFEST_SCHEMA
    assert payload["corpora"][0]["entries"] == 1


def test_processor_version_is_the_same_in_two_data_roots(tmp_path, monkeypatch):
    """Two installs of the same model under different roots are the same
    processor: the version depends on the bytes, not on the root's path."""
    import nltk.data

    versions = []
    for name in ("root_a", "root_b"):
        root = tmp_path / name / "nltk_data"
        model = root / "corpora" / "demo_model"
        model.mkdir(parents=True)
        (model / "params.json").write_text('{"w": 1}\n', encoding="utf-8")
        monkeypatch.setattr(nltk.data, "path", [str(root)])
        fingerprint = provenance.processor_fingerprint("demo", resources=(str(model),))
        versions.append(fingerprint.version)
        assert str(root) not in json.dumps(fingerprint.to_dict())
        assert fingerprint.resources == ("corpora/demo_model",)
    assert versions[0] == versions[1]


def test_parallel_merge_is_order_independent(tmp_path):
    root = tmp_path / "parallel"
    make_corpus(root, {"a.txt": "x\n"})
    corpus = provenance.corpus_fingerprint(root)
    processor = provenance.processor_fingerprint("p", version="1")

    def worker(ids):
        samples = {sid: provenance.sample_digest([sid]) for sid in ids}
        return provenance.ProcessingResult(
            corpora=[corpus], processors=[processor], samples=samples
        )

    first = worker(["a", "b"]).merge(worker(["c", "d"]))
    second = worker(["c", "d"]).merge(worker(["a", "b"]))
    assert first == second
    assert sorted(first.samples) == ["a", "b", "c", "d"]

    (root / "a.txt").write_text("changed\n", encoding="utf-8")
    changed = provenance.corpus_fingerprint(root, force=True)
    assert changed.version != corpus.version
    with pytest.raises(provenance.IncomparableResultsError):
        provenance.ProcessingResult(
            corpora=[corpus], processors=[processor], samples={"z": "sha256:00"}
        ).merge(
            provenance.ProcessingResult(
                corpora=[changed],
                processors=[processor],
                samples={"y": "sha256:01"},
            )
        )


def test_duplicate_sample_id_with_different_output_is_refused():
    with provenance.record() as session:
        session.add_sample("s", ["a"])
        with pytest.raises(ValueError, match="recorded twice"):
            session.add_sample("s", ["b"])


def test_concurrent_sample_recording_is_deterministic(tmp_path):
    with provenance.record() as session:

        def add(n):
            for i in range(20):
                session.add_sample(f"{n}-{i}", [n, i])

        threads = [threading.Thread(target=add, args=(n,)) for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        result = session.result()
    assert len(result.samples) == 80
    assert list(result.to_dict()["samples"]) == sorted(result.samples)


# ---------------------------------------------------------------------------
# Comparison of an old (pre-provenance) result and JSON inputs
# ---------------------------------------------------------------------------


def test_compare_accepts_plain_dicts():
    payload_a = provenance.ProcessingResult(
        processors=[provenance.processor_fingerprint("p", version="1")],
        samples={"s": "sha256:aa"},
    ).to_dict()
    payload_b = json.loads(json.dumps(payload_a))
    comparison = provenance.compare(payload_a, payload_b)
    assert comparison.comparable
    assert comparison.changed_samples == ()
    assert comparison.same_outputs is True


def test_unsupported_schema_is_refused():
    with pytest.raises(ValueError, match="unsupported provenance schema"):
        provenance.ProcessingResult.from_dict({"schema": "something/9", "samples": {}})
