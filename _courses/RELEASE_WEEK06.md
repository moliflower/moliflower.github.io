# DeepInfer Week 6 — 2026-10-24

This batch adds L11/L12, Lab5, HW6, and Project2 Phase2 (the Week 6 checkpoint). The maintained source remains this personal homepage repository, published at `/deepinfer/`; the retired Sites project is not used.

## Teaching scope

- Decode-first scheduling, per-request prefill chunks, iteration-boundary admission and completion.
- The first output is emitted at the end of the last prefill chunk; an output length of one needs no decode iteration.
- TTFT, request-weighted TPOT, singleton handling, and request SLO goodput with an explicit observation window.
- A fixed-tick simulation, not a real model, calibrated GPU benchmark, or integrated KV allocator. Under the simulator's budget constraint all defined TPOT values are one tick; the materials explain why this does not establish a real GPU performance claim.
- Phase1 files and grading remain separate. Each of Weeks 5 and 6 has its own 100-point self-assessment; Unit1 remains unchanged.

## Verification

- All 15 registered reference assignment suites pass, including 23 new public tests across the three new assignments.
- Three additional integration tests compare 300 seeded workloads with an independently structured replay oracle, cross-check all 18 experiment rows against HW6, and verify that six deliberately incorrect semantic variants fail public tests.
- Empty starters fail as expected rather than reporting a false pass. The new prediction and hint paths work.
- Both entry points render all 11 routes; 45 generated notes have valid local targets and progress keys. Storage regression checks pass.
- Desktop and 390-pixel mobile viewport previews were checked; the mobile document client width was 375 pixels with no document overflow. Phase2 returns to the short calendar URL and new tasks appear in the mobile practice center; no browser console errors were observed.
- The full local Jekyll build is unavailable because the Docker daemon is not running. The deployment workflow performs the production build and now also runs all reference solutions, storage tests, and Week 6 integration checks before publication.
- The first deployment attempt encountered an upstream runner change: `ubuntu-latest` selected Ubuntu 26.04, which has no Ruby 3.3.5 binary. The deploy job is now pinned to `ubuntu-24.04`, matching the previous successful production build without changing the site's Ruby or gem versions.

## Reproduce from the repository root

```bash
python -B _courses/build_deepinfer.py
node _courses/tests/test_site.cjs
node _courses/tests/test_storage.cjs
python -B _courses/deepinfer/course/tools/grade.py --solution
python -B _courses/tests/test_week06.py
python -B _courses/deepinfer/course/projects/project2_scheduler/experiment.py --solution
```

Learners run `experiment.py` without `--solution` to use their own implementation. Keep personal reports, CSV results, and QA screenshots outside the public course tree. The ZIP bundles the unimplemented starters and separate, explicitly named reference solutions.
