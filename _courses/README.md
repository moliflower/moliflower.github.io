# Standalone courses

`deepinfer/` is the canonical source of DeepInfer 101, served at `/deepinfer/`. Maintain the course in this personal homepage repository; the previous Sites publication is retired.

The leading underscore keeps these files out of Jekyll processing. The deployment workflow copies them into `_site/deepinfer/` after Jekyll and PurgeCSS finish, preserving Markdown downloads, relative links, and the course's independent styles.

Edit Markdown, Python assignments, and the course site's HTML/CSS/JS directly under `deepinfer/course/`. Run these commands from this repository's root:

```powershell
python _courses/build_deepinfer.py
node _courses/tests/test_site.cjs _courses/deepinfer/course/site
node _courses/tests/test_storage.cjs _courses/deepinfer/course/site
python _courses/deepinfer/course/tools/grade.py --solution
```

The builder regenerates lecture HTML and the runnable ZIP; deployment also runs it automatically. Never add personal reports, QA attempts, credentials, or working files to the public `deepinfer/` directory. The ZIP includes public reference solutions; student progress remains browser-local. Upstream CUDA implementation links stay pinned to the original DeepInfer repository and are separate from this course's source.

Commit the refreshed course files on `main` to use the existing GitHub Actions deployment. Do not edit `gh-pages` directly: the next site build replaces it.
