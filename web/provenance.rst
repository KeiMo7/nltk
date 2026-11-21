Corpus and Processing Provenance
================================

NLTK corpora are distributed without a version field.  A few readers expose an
ad hoc ``version`` attribute (VerbNet) or a file date (BCP47), but most do not,
so a corpus name alone does not tell you **which bytes** were read.  Two
machines can hold different text under the same corpus name, and two
tokenisation runs can therefore produce different tokens from the same input.

:mod:`nltk.provenance` gives a corpus, a processor and a processing run a
*content-derived* identity, so two results can be compared before they are
trusted as equivalent: which differed, which samples changed, and why they are
not directly comparable.

The module is **opt-in**.  Nothing is recorded and no existing return value
changes unless you open a recording session with ``provenance.record()`` (or
``provenance.enable()``).

Quick start
-----------

Record a run, and save its result:

.. code-block:: python

    import nltk
    from nltk import provenance
    from nltk.corpus import brown

    with provenance.record() as run:
        for i, sent in enumerate(brown.sents()[:2]):
            run.add_sample(f"brown-{i}", nltk.word_tokenize(" ".join(sent)))
        result = run.result()

    result.save("run.json")

The saved JSON records, per corpus, its name, content version (``sha256:…``),
entry count and a content digest; per processor, the model digest and its
configuration (language, tagset, …); and per sample, the digest of its output.
It contains **no absolute paths**.

Compare two runs:

.. code-block:: python

    old = provenance.ProcessingResult.load("run-old.json")
    new = provenance.ProcessingResult.load("run.json")
    diff = provenance.compare(old, new)
    if not diff.comparable:
        print(diff.reasons)          # e.g. "corpus 'brown' version differs: …"
    print(diff.changed_samples)      # sample ids whose output changed
    print(diff.unchanged_samples)    # sample ids whose output stayed the same

Export the corpus manifest for reproduction:

.. code-block:: python

    provenance.export_manifest("manifest.json", brown)
    # or, from a recorded run:
    provenance.export_manifest("manifest.json", old)

``ProcessingResult.save`` and ``export_manifest`` write through NLTK's path
sandbox, like every other NLTK file write: the destination must be inside a
directory on ``nltk.data.path``.  To write elsewhere, serialize with
``result.to_json()`` (or ``provenance.manifest(...)``) and write the text
yourself.

What is versioned
-----------------

Corpus
    ``version`` is the SHA-256 of the sorted ``(relative file name, file
    digest)`` pairs of the corpus files.  It is derived from bytes only, so
    two copies of the same content in differently-named directories get the
    same version, and a download timestamp or file mtime never affects it.
    ``entries`` is the file count; ``source_id`` is the root relative to the
    NLTK data root (never absolute).

Processor
    ``processor_fingerprint(name, resources=…, params=…)`` digests the model
    files that define a processor and records the configuration that changes
    its output.  When there are no model files, the declared version (or the
    installed NLTK ``__version__``) is used instead, and
    ``version_source`` says which of the three applies (``content``,
    ``declared`` or ``package``).  Stochastic processors should pass their
    seed through ``params`` so two runs with different seeds are not reported
    as comparable.

Built-in processors are detected automatically: ``sent_tokenize``/
``word_tokenize`` record the ``punkt_tab`` model digest, and ``pos_tag``/
``pos_tag_sents`` record the ``averaged_perceptron_tagger_*/`` model digest,
together with the language and tagset they were called with.

Comparison
----------

``compare(a, b)`` checks corpus versions, processor versions and
configurations, and the NLTK version.  A mismatch, or an ``unknown`` version
on either side, makes ``comparable`` false and fills ``reasons``.  Sample output
digests are diffed either way, so an upgrade that makes two results
incomparable still reports exactly which samples changed.

``ProcessingResult.check_comparable(other)`` raises
``IncomparableResultsError`` rather than returning a verdict, for callers that
want the comparison to be a precondition.

Merging parallel work
---------------------

``ProcessingResult.merge`` combines partial results from workers.  It refuses
results whose corpus/processor versions differ and refuses overlapping sample
ids.  Because samples are keyed and sorted on output, the merged result does
not depend on the order in which workers finished, provided sample ids are
unique.  A session records each id once; adding the same id with a different
output raises instead of letting scheduling decide the winner.

Cost
----

* Digests are streamed in bounded chunks; no corpus file is held whole, so
  provenance adds no full-corpus copy.
* Per-file digests are cached by a stat signature (size, mtime, ctime, inode):
  a repeat fingerprint re-stats but does not re-read a large corpus.  The
  cache is best-effort; a corrupt or unwritable cache is treated as absent.
  Set ``NLTK_PROVENANCE_NO_CACHE=1`` to disable it, or
  ``NLTK_PROVENANCE_CACHE=<dir>`` to choose where it lives (the default is the
  first writable ``nltk.data.path`` entry).  The cache is written through the
  usual path sandbox, so an explicit location must also be inside a data root;
  if it is not, the cache is simply not written.
* The first fingerprint of a corpus reads every file once, at I/O speed.

Known limitations
-----------------

* Cache signatures use the change time (``st_ctime``) to notice an in-place
  rewrite.  On filesystems where that field is a creation time rather than a
  change time (some Windows configurations), a file rewritten in place to the
  same size with its modification time restored could be served from a stale
  entry.  Pass ``force=True`` (or disable the cache) when a corpus may have
  been replaced that way; a fresh install or a normal upgrade always changes
  the signature.
* The version describes the files a reader was constructed to read
  (``fileids()``), not every file under the corpus directory.  A file in the
  directory that no reader selects does not affect the version.
* Built-in detection covers ``sent_tokenize``/``word_tokenize`` (Punkt) and
  ``pos_tag``/``pos_tag_sents`` (averaged perceptron, plus the universal
  tagset map).  A tokeniser or tagger constructed directly records only what
  you pass to ``add_processor``/``processor_fingerprint``.

Degradation
-----------

* A missing corpus raises :class:`nltk.provenance.CorpusUnavailableError`, a
  ``LookupError``, whose message names the ``nltk.download(…)`` call that
  installs it.
* A digest that cannot be computed (unreadable file, transient I/O error)
  emits :class:`nltk.provenance.ProvenanceWarning`, records the version as
  ``unknown`` and lets processing continue.  Two results with an unknown
  version are reported as not directly comparable.

What does *not* change
----------------------

* Existing corpus-reader, tokeniser and tagger APIs and their return values are
  untouched, whether or not a session is active.
* With no session active, the hooks in ``CorpusReader.__init__`` and the
  built-in tokeniser/tagger entry points do one boolean test and return.
* NLTK's corpus loading priority (``nltk.data.path`` order, ``find``
  resolution) is unchanged: a resource name is still resolved by ``find``
  first; a local path is used only if ``find`` misses and the path exists.

Determinism under parallelism
-----------------------------

A recording session collects into maps keyed by sample id.  Adding the same id
twice with different output raises, so scheduling can never decide which value
wins; adding distinct ids from several threads is safe.  ``merge`` of partial
worker results is independent of worker completion order, and both ``merge``
and output serialisation sort keys, so the JSON is byte-identical for the same
set of samples.  Fingerprinting performs no random sampling; a stochastic
processor should put its seed in ``params`` so two different seeds are
reported as not comparable rather than silently compared.

Compatibility
-------------

Results written by this schema (``nltk-provenance/1``) are self-describing and
rejected if the schema is not recognised.  Data produced by an NLTK older than
this feature has no provenance record; it remains fully usable, but it cannot
be compared against a versioned result — treat it as ``unknown`` rather than
assuming equivalence.  Nothing in existing corpora or saved models needs to be
migrated: versions are computed from the bytes already on disk, on demand.

Why not reuse what already exists
---------------------------------

Two existing mechanisms look related but answer a different question:

* a few readers expose a declared ``version`` (VerbNet's ``"2.1"``, BCP47's
  file date).  That is a label chosen by the corpus packager, not the bytes
  on disk; it is recorded as ``declared_version`` for display, and never used
  as the comparison key.
* the downloader verifies an md5/sha256 over the **download archive**.  That
  checks that a download arrived intact; it says nothing about the unpacked
  files a reader actually sees (or about a corpus installed by hand or from
  another archive).  It is also unavailable for corpora that were never
  downloaded through NLTK.

``nltk.provenance`` therefore digests the unpacked files a reader reads.  The
older mechanisms keep working unchanged and are not replaced: provenance is an
additional, opt-in layer, and its result records its own schema id so the two
never get confused.

Switches
--------

``provenance.enable()`` / ``provenance.disable()``
    Start/stop a session that spans a whole process rather than a ``with``
    block.

``NLTK_PROVENANCE_NO_CACHE``
    Disable the on-disk digest cache.

``NLTK_PROVENANCE_CACHE``
    Directory (or ``.json`` file) for the digest cache.
