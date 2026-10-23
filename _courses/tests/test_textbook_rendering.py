"""Validate the self-contained textbook renderer and accessible vector figures."""

import importlib.util
from pathlib import Path
import re
import sys
import unittest
import xml.etree.ElementTree as ElementTree
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "deepinfer/course"
spec = importlib.util.spec_from_file_location("textbook_notes_builder", COURSE / "tools/build_notes.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class TestTextbookRendering(unittest.TestCase):
    def setUp(self):
        builder.CURRENT_SRC_DIR = "course/lectures"

    def test_standalone_figure_caption_zoom_and_escaping(self):
        title, body, headings = builder.parse_markdown(
            '# Test\n\n![A "quoted" & B](../figures/learning-map.svg)\n\n## Read the arrows\n\nText.', "fallback")
        self.assertEqual(title, "Test")
        self.assertIn('<figure class="textbook-figure">', body)
        self.assertIn('src="../../figures/learning-map.svg"', body)
        self.assertIn('alt="A &quot;quoted&quot; &amp; B"', body)
        self.assertIn('rel="noopener"', body)
        self.assertEqual(len(headings), 1)
        self.assertIn("点击图解可查看原图", body)

    def test_image_after_paragraph_and_code_are_not_confused(self):
        body, headings = builder.convert_blocks([
            "Intro", "![Diagram](../figures/learning-map.svg)", "", "```python",
            "print('<script>')", "```", "", "<details>", "<summary>先预测，再展开</summary>",
            "", "Expected: **3**", "</details>",
        ])
        self.assertIn("<p>Intro</p>\n<figure", body)
        self.assertNotIn("<script>", body)
        self.assertIn("&lt;script&gt;", body)
        self.assertIn('<details class="study-hint">', body)
        self.assertIn("<strong>3</strong>", body)

    def test_all_figures_are_local_accessible_and_passive(self):
        figures = list((COURSE / "figures").glob("*.svg"))
        self.assertTrue(figures)
        for file in figures:
            with self.subTest(file=file.name):
                root = ElementTree.parse(file).getroot()
                self.assertEqual(root.tag.split("}")[-1], "svg")
                self.assertTrue(root.get("viewBox"))
                tags = {element.tag.split("}")[-1] for element in root.iter()}
                self.assertTrue({"title", "desc"}.issubset(tags))
                self.assertFalse({"script", "foreignObject", "image"}.intersection(tags))
                for element in root.iter():
                    for name, value in element.attrib.items():
                        local_name = name.split("}")[-1].lower()
                        self.assertFalse(local_name.startswith("on"))
                        if local_name in {"href", "src"}:
                            self.assertTrue(value.startswith("#"), f"external resource: {value}")
                self.assertNotRegex(file.read_text(encoding="utf-8"), r"url\(\s*['\"]?(?:https?:|data:)")

    def test_inline_disclosures_nested_blocks_and_code_literals(self):
        body, headings = builder.convert_blocks([
            "Predict first.", "<details><summary>核对</summary>", "", "Answer.",
            "<details>", "<summary>Further reasoning</summary>", "", "**Detail**", "</details>",
            "```html", "<details><summary>literal</summary>", "</details>", "```", "</details>",
        ])
        self.assertEqual(body.count('<details class="study-hint">'), 2)
        self.assertIn("<p>Predict first.</p>\n<details", body)
        self.assertIn("<summary>核对</summary>", body)
        self.assertIn("&lt;details&gt;&lt;summary&gt;literal&lt;/summary&gt;", body)
        with self.assertRaises(ValueError):
            builder.convert_blocks(["<details><summary>Missing close</summary>", "Answer."])

    def test_book_registry_and_mobile_contents(self):
        sources = {doc["src"] for doc in builder.DOCS}
        self.assertIn("course/LEARNING_PATH.md", sources)
        doc = next(item for item in builder.DOCS if item["out"] == "learning-path.html")
        doc = {**doc, "parsed_title": "学习路线"}
        page = builder.build_page(doc, "<p>Example</p>", [(2, "先手算", "h-1")])
        self.assertIn('<details class="doc-toc" id="doc-toc" open>', page)
        self.assertIn('<nav aria-label="本页章节">', page)
        self.assertIn("手算与练习请按正文分段完成", page)
        self.assertNotIn("data-key=", page)

    def test_published_chapter_images_resolve(self):
        for doc in builder.DOCS:
            source = doc["src_path"].read_text(encoding="utf-8")
            for description, target in re.findall(r"!\[([^\]]*)\]\(([^)\s]+)\)", source):
                if target.startswith(("http:", "https:")):
                    continue
                self.assertTrue(description.strip(), f"missing alt: {doc['src']}")
                self.assertTrue((doc["src_path"].parent / target).is_file(), f"missing figure: {doc['src']} -> {target}")

    def test_all_authored_disclosures_render_as_interactive_hints(self):
        for doc in builder.DOCS:
            source = doc["src_path"].read_text(encoding="utf-8")
            fenced = False
            disclosures = 0
            for line in source.splitlines():
                if line.strip().startswith("```"):
                    fenced = not fenced
                if not fenced and line.strip().startswith("<details>"):
                    disclosures += 1
            builder.CURRENT_SRC_DIR = str(Path(doc["src"]).parent).replace("\\", "/")
            title, body, headings = builder.parse_markdown(source, doc["title_zh"])
            self.assertEqual(body.count('<details class="study-hint">'), disclosures, doc["src"])

    def test_offline_archive_contains_current_figures_and_guide(self):
        with zipfile.ZipFile(COURSE.parent / "deepinfer-101-course.zip") as archive:
            self.assertIsNone(archive.testzip())
            sources = [COURSE / "LEARNING_PATH.md", *sorted((COURSE / "figures").glob("*.svg"))]
            for source in sources:
                relative = source.relative_to(COURSE.parent).as_posix()
                with self.subTest(file=relative):
                    self.assertTrue(archive.read(relative) == source.read_bytes(), f"Rebuild stale archive member: {relative}")
            reader = archive.read("course/site/notes/learning-path.html").decode("utf-8")
            self.assertIn('href="../../../index.html#/calendar"', reader)
            self.assertIn('src="../../figures/learning-map.svg"', reader)


if __name__ == "__main__":
    unittest.main(verbosity=2)
