# Illustrated textbook revision — 2026-10-24

This revision deepens the already published Weeks 1–6 rather than publishing another week. The canonical source and public address remain the personal homepage repository and `/deepinfer/`.

## Scope

- L1–L12 now connect motivations, prerequisites, units, derivations, complete small worked cases, runnable demonstrations, boundary cases, counterexamples, and self-checks. Hardware extensions are explicitly separated from the standard-library Python core.
- Twenty-one local SVG diagrams cover the learning map, execution, memory, numerics, tiling, Attention, online normalization, KV capacity/mapping, quantization, ownership, prefix reuse, and scheduling/metrics. Each has accessible title/description text and no active or external content.
- Foundations/prep, Labs 0–5, HW1/2/5/6, both quizzes, Project1 and both published Project2 phases have aligned reading/practice paths. Full HW1/2 follow L7/L8; Project1's implementation dependency is Q7 → Q9 → Q6.
- Existing starter/reference implementations, assignment APIs, scoring denominators and browser progress identifiers remain compatible. Quiz2 retains five questions worth two points each, with an explicit quantization transfer question.
- A new self-study route connects the entire published curriculum. Course-root instructions distinguish the downloadable ZIP from the personal homepage source tree.

## Reader and packaging

- Standalone Markdown images become captioned figures with original-image links. Small screens scroll each diagram internally, preserving readable labels without widening the document.
- The mobile table of contents starts collapsed and remains learner-controlled. Safe disclosure parsing accepts separate or adjacent `details`/`summary` tags while escaping other HTML and preserving code literals.
- Reader CSS/JS URLs use content hashes. Both course entry points remain functional, with the root entry rendering the course directly rather than a redirect or iframe.
- All diagrams and the new guide are bundled in the offline ZIP. The archive still excludes private reports, working CSV files, runtime caches and unexpected file types.

## Verification

- All 15 registered reference assignment suites pass, both in the source tree and an independently extracted offline package.
- Sixty-nine Python regression tests pass. These include executable textbook examples, independent numerical/state checks, safe rendering/archive tests, the existing 300 seeded scheduling replays, all 18 experiment rows, and six mutation checks.
- Site checks cover both entry points, all 11 routes, 46 generated documents, 1,361 local references and unchanged progress keys. Storage and document-reader interaction tests pass.
- Browser checks cover all 12 chapters at a 390-pixel mobile viewport: the document client width and scroll width are both 375 pixels. Diagram bounds and label intersections were checked for all 21 figures; one lifecycle label overlap was corrected and visually rechecked. Representative desktop/light/dark layouts and disclosure interactions were inspected.
- The local Docker daemon is unavailable, so the complete Jekyll production build is delegated to the existing deployment workflow. That workflow runs the course build, all reference assignment suites and all course regressions before copying the course to the published site.
- Whole-repository Prettier warnings predate this revision; only changed maintainer files, tests and workflow configuration are formatted. The course subtree retains its existing formatting exclusion.

## Reproduce

Run from the personal homepage repository root:

```bash
python -B _courses/build_deepinfer.py
node _courses/tests/test_site.cjs
node _courses/tests/test_storage.cjs
node _courses/tests/test_doc_reader.cjs
python -B _courses/deepinfer/course/tools/grade.py --solution
python -B -m unittest discover -s _courses/tests -p 'test_*.py'
```

Passing these checks establishes consistency of the published examples and software contracts, not a guarantee that every learner has mastered the material. The projects remain teaching models, not a measured GPU benchmark or an integrated production inference engine. Weeks 7–15 remain unpublished.
