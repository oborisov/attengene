# Changelog

## v0.9.0 - 2026-06-14

Live-streaming retrieval trace and a ClinVar deletion-matching fix.

- New: the retrieval trace now streams live. Each database's line appears the
  moment that search finishes (building the trace block incrementally) rather
  than all at once after retrieval completes, so there's no blank wait while
  the databases are queried. `route_and_retrieve` gained an optional `on_step`
  callback (no-op by default, so non-streaming callers are unchanged); the
  per-line text is shared with the all-at-once trace so the two forms can't
  drift. Still gated by `RAG_TRACE`.
- Fix: ClinVar deletions/duplications stored without trailing bases were
  missed. ClinVar is inconsistent about trailing bases (it stores both
  `c.730_731delAG` and the base-less `c.1521_1523del` - the CFTR F508del form).
  A clinician typing the base-bearing form (`c.1521_1523delCTT`) found nothing,
  because the exact lookup matched only that literal. The lookup now tries both
  the base-bearing and base-less form for a simple trailing del/dup event, so it
  matches whichever ClinVar stored (compound delins keeps its single form). The
  gene filter still constrains breadth.

## v0.8.0 - 2026-06-14

Streaming retrieval trace and lowercase gene-symbol routing.

- New: retrieval trace. When streaming, the answer is now preceded by a
  collapsible "Retrieval trace" block showing the query classification and,
  per database (ClinVar, GeneReviews, NephroGenetics, PubMed), whether it was
  searched, how many results it returned, and the top hit - or "skipped" with
  the routing reason. It surfaces what the RAG router actually did, which makes
  routing gaps and off-target retrieval visible. Emitted as ordinary OpenAI
  content (a Markdown `<details>` block), so it works in any OpenAI-compatible
  client without provider-specific status events. Gated by the new `RAG_TRACE`
  env var (default on; set `0`/`false`/`off` to disable). See `.env.example`.
- Fix: lowercase gene symbols in queries. Gene-symbol extraction only matched
  uppercase tokens, so a query like `muc1 gene` or `kmt2d gene` silently routed
  as a phenotype query - skipping the variant-level path (ClinVar, PubMed) and
  leaving the GeneReviews search unanchored, which could drift to a sibling
  gene's chapter. Lowercase/mixed-case tokens are now accepted when a gene cue
  is adjacent (`X gene`, `gene X`, `variant/mutation in X`) and normalized to
  the canonical uppercase symbol. Phenotype prose with no gene cue still yields
  no gene, so general questions keep routing as phenotype queries.

## v0.7.0 - 2026-06-14

Retired-chapter filtering and dead-link guards for GeneReviews citations.

- New: retired GeneReviews chapters are excluded from retrieval by default.
  Retired/historical chapters carry a "RETIRED CHAPTER, FOR HISTORICAL
  REFERENCE ONLY" marker in their title; these are detected at index time
  (handling the en-dash and box-drawing-dash variants seen in the corpus),
  persisted as a `retired` boolean on `genereviews_chunks`, and default-excluded
  in retrieval so they never appear as a cited clinical source. An
  `include_retired` opt-in is available for historical lookups.
- Fix: dead Bookshelf citation links. GeneReviews source URLs are now built as
  `/books/<id>/` only for valid numeric NCBI Bookshelf IDs. Chapters carrying a
  shortname-derived fallback id (e.g. `NBKwagner`) - which never resolved to a
  real Bookshelf page - now link to the GeneReviews landing page instead of
  presenting a broken link to a clinician.
- Indexer hardening: `index_genereviews.py` accepts a directory argument
  (expanded to its `*.nxml` files) and aborts before any `--clear` if 0 sections
  were parsed, so a bad path can no longer wipe the live index.
- Schema: `genereviews_chunks` gains a `retired BOOLEAN NOT NULL DEFAULT false`
  column, with an in-place backfill migration documented for existing indexes.

## v0.6.0 - 2026-06-09

Role-boundary guardrails and citation grounding, surfaced by a manual walk
through a gold-standard query catalog against a live instance.

- New: patient-specific role boundary. The assistant now declines requests to
  diagnose, prognosticate for, manage, or advise on patient communication for a
  **specific individual** ("diagnose this 5-year-old...", "should I tell my
  patient they will get cancer?") and directs the user to a qualified clinical
  geneticist. General and educational questions ("how is cystic fibrosis
  diagnosed?", "what is the surveillance interval in Lynch syndrome?") are
  explicitly unaffected and answered normally. Enforced across three layers: the
  system prompt, a pre-generation keyword check, and a post-generation pattern
  check that catches third-person differential delivery for a named patient.
- New: pathogenicity-classification boundary. Requests to generate, score,
  reclassify, or give an opinion on a variant's pathogenicity (e.g. "apply the
  ACMG criteria and give me a final score", "is this VUS pathogenic in your
  opinion?") are declined - the system reports existing ClinVar classifications
  but never generates its own.
- Fix: fabricated provenance on no-evidence answers. When nothing relevant was
  retrieved, the model could fill its inline "Source:" line from its own
  training, citing databases the pipeline does not use (e.g. NCBI Gene, UniProt)
  for a gene with no corpus hit. The Source line is now sanitized against an
  allowlist (ClinVar, GeneReviews, NephroGenetics, PubMed); fabricated tuples
  are dropped.
- Fix: no-evidence answers leaked stray references. The phrase matcher that
  suppresses the References block on a "no relevant evidence" answer missed
  several common phrasings ("No X is described in the evidence", "does not
  document ... in the evidence", "... based on the retrieved evidence", "... in
  the provided references"); those off-target citations no longer appear.

## v0.5.3 - 2026-06-07

Three clinical-correctness fixes for variant citations, surfaced by real chat
logs and verified against the live ClinVar corpus:

- Fix: wrong-gene citation bleed. HGVS coordinates are not unique across the
  genome - a coordinate like `c.203C>T (p.Thr68Met)` exists in more than one
  gene (e.g. ALPL and GP1BB), and `c.526G>A (p.Ala176Thr)` in ALPL and PNPT1.
  The exact ClinVar lookup used the parsed gene only as a sort tie-breaker, so
  the same-coordinate variant from the *wrong* gene could still ride in under
  the result limit and be presented as an authoritative citation. The gene is
  now a hard filter when confidently parsed: a gene-filtered-empty result is the
  correct true negative (the answer falls back to prose) rather than a leaked
  wrong-gene hit. When no gene is parsed, the previous ungated behavior is kept
  so a correct hit is never zeroed out.
- Fix: duplicate "References" block. Some models append their own References
  list in their answer, and the citation post-processor then appended a second,
  canonical one. Post-processing now strips any model-written References block
  before rebuilding the canonical one, and the streaming path stops forwarding
  the model's own block mid-stream (since shipped tokens can't be recalled).
- Fix: truncated PubMed citation titles. Titles and abstracts carry inline
  markup (e.g. `<i>GENE</i>`); the XML parser stopped at the first child element
  and silently truncated the title (e.g. dropping a trailing gene name). It now
  walks the whole subtree so the full title is captured.

## v0.5.2 - 2026-06-06

- Fix: when the model reported in German that a queried variant or condition was
  not in the retrieved evidence (e.g. "keine relevanten Belege gefunden",
  "nicht in den Belegen enthalten"), the References-suppression failed to fire
  and stray off-target citations were still appended. The suppression now
  recognizes German not-found phrasings in addition to English, so a German "no
  evidence" answer correctly drops the misleading reference list. (The model
  already answered German prompts in German; only the citation post-processing
  was English-only.)

## v0.5.1 - 2026-06-06

- Fix: variant queries with no gene symbol (e.g. a follow-up "what about the
  variant c.526G>A, p.(Ala176Thr)") crashed the exact ClinVar lookup with an
  invalid SQL ordering clause. The query then fell back to semantic prose
  search, which drifted to unrelated articles and could attach a wrong citation
  to the variant. The exact lookup now handles gene-less queries correctly and
  returns the right ClinVar entries.

## v0.5.0 - 2026-06-06

- Feature: structured HGVS lookup for variant queries. Variant-level questions
  are now answered by parsing the gene symbol and HGVS tokens out of the query
  (e.g. `c.526G>A`, `p.(Ala176Thr)`, gene `ALPL`) and doing an exact match
  against ClinVar, instead of fuzzy-matching the whole sentence. This is more
  precise and gives clean true negatives: a variant that genuinely isn't in
  ClinVar returns no evidence rather than off-target near-matches.
- Fix: resolves a regression where conversational phrasing around a variant
  ("what about the variant c.526G>A ... in ALPL") diluted the fuzzy-match score
  below the v0.4.1 similarity floor, causing a correct, indexed pathogenic
  variant to be wrongly reported as not found. The exact-lookup tier is immune
  to this; the fuzzy hybrid search remains as a fallback for variant queries
  with no parseable HGVS token (e.g. "what variants are known in BRCA1").
- Parsing normalizes common input variations: HTML-escaped `&gt;`, letter case,
  parentheses around the protein change, and `del`/`dup`/`ins` events.

## v0.4.1 - 2026-06-06

- Fix: ClinVar hybrid search no longer surfaces off-target variants when the
  queried variant is absent. The hybrid score is min-max normalized, which
  stretched a signal-free pool of trigram-coincidence matches (variants from
  unrelated genes sharing an HGVS-like substring) across the full score range
  and listed them as authoritative-but-wrong citations. A raw lexical-
  similarity floor now drops sub-threshold candidates before normalization:
  a real HGVS/name match scores well above the floor, so a query for a
  variant that isn't in ClinVar returns no ClinVar evidence instead of noise.

## v0.4.0 - 2026-06-06

- Feature: ClinVar variants are now searched in the chat pipeline. Previously
  variant-level questions ("what about c.526G>A in ALPL") were answered only
  from narrative sources and missed the variant even when it was indexed.
  The router now runs hybrid lexical+semantic search over ClinVar when the
  query carries a gene symbol or HGVS notation, and cites the matching
  ClinVar variation IDs. Phenotype-only queries still skip ClinVar so the
  variants table does not add noise.
- Fix: the References block is suppressed when the answer states the queried
  item is absent from the retrieved evidence, so off-target neighbour sources
  are no longer listed as if they supported a "not found" answer.
- Fix: the transient "Searching databases..." status line is no longer
  persisted into the stored response text.
- Audit: each query is now classified (variant / phenotype / non-clinical)
  and the classification plus retrieved variant IDs are recorded.

## v0.3.1 - 2026-06-02

- Fix: GeneReviews retrieval no longer cites an unrelated gene's chapter.
  When a query names a gene that isn't in the indexed corpus, pure semantic
  search returned the densest "genetics prose" chunk (an off-target gene
  scoring ~0.55), which surfaced as a citation even when the answer said
  "no evidence". Retrieval is now anchored on gene symbols extracted from
  the query (passed as a gene filter), so off-target chapters can't leak
  in; absent genes return an empty GeneReviews result cleanly.
- Fix: raised the GeneReviews similarity floor from 0.3 to 0.6. On-target
  chunks score ~0.7-0.8 while noise tops out around 0.55, so the old 0.3
  threshold filtered nothing.

## v0.3.0 - 2026-06-02

- Fix: the API image now builds on a fresh clone. The `Dockerfile` copies
  `data/genereviews/`, but those lookup tables were gitignored, so
  `docker/podman compose up` failed at the `COPY data/genereviews/` step.
  The two factual NBK-id/gene/disease lookup tables (the demo data the API
  loads at runtime) are now committed; copyrighted GeneReviews prose stays
  out of the repo.
- Podman: all image references are fully qualified (`docker.io/...`) in the
  compose file and Dockerfiles, so the stack runs under stock Podman without
  an `unqualified-search-registries` entry in `registries.conf`. Quickstart
  documents the `podman compose` path.

## v0.2.0 - 2026-06-02

- RAG prompt: responses now open with a 2-3 sentence ANSWER synthesis
  (condition + gene(s), evidence-only, ending with a source list) before
  the FINDINGS / GAPS audit trail
- RAG prompt: tightened FINDINGS to at most 6 bullets, one fact each, with
  non-matching conditions collapsed into a single rule-out line instead of
  one bullet per retrieved entry
- Local backend: optional `BACKEND_LOCAL_KEY` sends `Authorization: Bearer`
  on outbound calls (chat and the startup `/models` probe) for a
  llama-server running with `--api-key`; empty/unset keeps the no-auth path

## v0.1.0 - 2026-05-30

Initial public release.

- FastAPI backend exposing an OpenAI-compatible `/v1/chat/completions`
  and `/v1/models` surface
- Hybrid retrieval (pg_trgm lexical + pgvector semantic) over ClinVar
- Semantic retrieval over GeneReviews and NephroGenetics
- Multi-backend LLM dispatch: a local OpenAI-compatible endpoint
  (`attengene-local`) and an optional Mistral cloud backend
  (`attengene-mistral`, opt-in via `BACKEND_MISTRAL_KEY`)
- Bundled embeddings server (BAAI/bge-large-en-v1.5, 1024-dim) with
  CPU and CUDA images
- Audit logging, citation post-processing, pre/post guardrails
- Indexing scripts for ClinVar (with a 25-gene starter dataset),
  GeneReviews (NXML), and NephroGenetics
- Docker Compose stack covering database, embeddings, and API; LLM is
  intentionally external
