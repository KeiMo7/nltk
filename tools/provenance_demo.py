#!/usr/bin/env python3
# Natural Language Toolkit: provenance demo
#
# Copyright (C) 2001-2026 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""Reproduce the provenance acceptance artefacts:

* a corpus manifest for a corpus a run used;
* an upgrade-difference sample: the same texts processed before and after one
  corpus file changes, showing which samples changed and why the two results
  are not directly comparable.

Run it with an output directory::

    python tools/provenance_demo.py /tmp/provenance-demo

The demo builds its own data root under ``$HOME`` (NLTK's normal data locations
are read-only or absent here), writes ``manifest.json``, ``baseline.json``,
``upgraded.json`` and ``comparison.json``, and prints a short report.  The
corpus it creates is removed at the end; the output directory is not.

This is a demonstration script, not library code: it uses the builtin ``open``
to write the artefacts the user asked for.
"""

import json
import os
import shutil
import tempfile

import nltk
import nltk.data
import nltk.pathsec
from nltk import provenance
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.tokenize import TreebankWordTokenizer

SAMPLES = {
    "doc1.txt": "The Fulton County Grand Jury said Friday an investigation.\n",
    "doc2.txt": "Rudolph Agnew, 55 years old, was named a director.\n",
    "doc3.txt": "The jury further said in term-end presentments.\n",
}


def build_data_root():
    """A private NLTK data root under $HOME holding the demo corpus."""
    root = os.path.realpath(
        tempfile.mkdtemp(prefix=".nltk_provenance_demo_", dir=os.path.expanduser("~"))
    )
    corpus_dir = os.path.join(root, "corpora", "demo_news")
    os.makedirs(corpus_dir)
    for name, text in SAMPLES.items():
        with open(os.path.join(corpus_dir, name), "w", encoding="utf-8") as f:
            f.write(text)
    nltk.data.path.insert(0, root)
    nltk.pathsec._ALLOWED_ROOTS_CACHE = None
    nltk.pathsec._LAST_DATA_PATHS = None
    return root, corpus_dir


def run(tokenizer, corpus_dir):
    """Tokenise every sample in a fresh recording session."""
    processor = provenance.processor_fingerprint(
        "nltk.tokenize.TreebankWordTokenizer",
        params={"language": "english"},
    )
    with provenance.record(capture_corpora=False) as session:
        session.add_corpus(corpus_dir)
        session.add_processor(processor)
        for name in sorted(os.listdir(corpus_dir)):
            with open(os.path.join(corpus_dir, name), encoding="utf-8") as f:
                text = f.read()
            session.add_sample(name, tokenizer.tokenize(text))
        return session.result()


def main(out_dir):
    out_dir = os.path.realpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # ``ProcessingResult.save`` writes through NLTK's path sandbox, so the
    # output directory has to be an allowed data root.  A program that keeps
    # its artefacts elsewhere does the same (or writes the JSON itself with
    # ``result.to_json()``).
    nltk.data.path.insert(0, out_dir)
    nltk.pathsec._ALLOWED_ROOTS_CACHE = None
    nltk.pathsec._LAST_DATA_PATHS = None
    data_root, corpus_dir = build_data_root()
    try:
        tokenizer = TreebankWordTokenizer()

        baseline = run(tokenizer, corpus_dir)
        baseline.save(os.path.join(out_dir, "baseline.json"))
        manifest = provenance.export_manifest(
            os.path.join(out_dir, "manifest.json"), baseline
        )

        # "Upgrade": replace one document's contents, as a data release would.
        with open(os.path.join(corpus_dir, "doc2.txt"), "w", encoding="utf-8") as f:
            f.write("Rudolph Agnew, 55, was named director of the new board.\n")

        upgraded = run(tokenizer, corpus_dir)
        upgraded.save(os.path.join(out_dir, "upgraded.json"))

        comparison = provenance.compare(baseline, upgraded)
        with open(os.path.join(out_dir, "comparison.json"), "w", encoding="utf-8") as f:
            json.dump(comparison.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")

        print("corpus manifest")
        for corpus in manifest["corpora"]:
            print(
                f"  {corpus['name']}: version={corpus['version']} "
                f"entries={corpus['entries']} source_id={corpus['source_id']}"
            )
        print("\nbaseline processor")
        for proc in baseline.processors:
            print(
                f"  {proc.name}: version={proc.version} "
                f"source={proc.version_source} params={proc.params}"
            )
        print("\nupgrade comparison")
        print(f"  comparable: {comparison.comparable}")
        for reason in comparison.reasons:
            print(f"  reason: {reason}")
        print(f"  changed samples: {list(comparison.changed_samples)}")
        print(f"  unchanged samples: {list(comparison.unchanged_samples)}")
        print(f"\nartefacts written to {os.path.abspath(out_dir)}")
    finally:
        shutil.rmtree(data_root, ignore_errors=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python tools/provenance_demo.py <output-dir>")
    main(sys.argv[1])
