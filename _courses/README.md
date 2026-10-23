# Standalone courses

`deepinfer/` is the canonical source of DeepInfer 101, served at `/deepinfer/`. Maintain the course in this personal homepage repository; the previous Sites publication is retired.

The leading underscore keeps these files out of Jekyll processing. The deployment workflow copies them into `_site/deepinfer/` after Jekyll and PurgeCSS finish, preserving Markdown downloads, relative links, and the course's independent styles.

The builder generates a complete homepage at `deepinfer/index.html`, not a redirect or iframe. It reuses `course/site/index.html` as its template and keeps a relative asset prefix for both online and offline use. Hash navigation stays at `/deepinfer/#/...`, and lecture return links point back to this entry. The old `course/site/index.html` remains functional for existing bookmarks; edit the template rather than the generated root entry. The ZIP also includes the root entry.

Edit Markdown, Python assignments, and the course site's HTML/CSS/JS directly under `deepinfer/course/`. Run these commands from this repository's root:

```powershell
python _courses/build_deepinfer.py
node _courses/tests/test_site.cjs _courses/deepinfer/course/site
node _courses/tests/test_storage.cjs _courses/deepinfer/course/site
node _courses/tests/test_doc_reader.cjs
python -B _courses/deepinfer/course/tools/grade.py --solution
python -B -m unittest discover -s _courses/tests -p 'test_*.py'
```

The builder regenerates lecture HTML and the runnable ZIP; deployment also runs it automatically. Never add personal reports, QA attempts, credentials, or working files to the public `deepinfer/` directory. The ZIP includes public reference solutions; student progress remains browser-local. Upstream CUDA implementation links stay pinned to the original DeepInfer repository and are separate from this course's source.

The illustrated textbook covers the published Weeks 1–6 only. Edit diagrams as local SVG sources in `deepinfer/course/figures/`, with a `viewBox`, accessible title/description, and no scripts or external resources. Standalone Markdown images become captioned, zoomable figures in the reader; on narrow screens diagrams scroll inside their own frame rather than shrinking labels into illegibility. Reader styles and scripts receive content-hash URLs, while existing assignment IDs and progress keys remain unchanged. New numerical examples should have regression checks under `_courses/tests/`, outside the learner assignment tree.

Commit the refreshed course files on `main` to use the existing GitHub Actions deployment. Do not edit `gh-pages` directly: the next site build replaces it.
