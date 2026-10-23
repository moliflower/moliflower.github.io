"""Build the self-contained DeepInfer course inside the personal homepage repo."""

from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parent / "deepinfer"
DIRECTORIES = {"course", "doc", "notes"}
FILES = {"LICENSE", "START_HERE.md"}
EXTENSIONS = {".html", ".css", ".js", ".md", ".py", ".ps1", ".bat"}


def main():
    subprocess.run([sys.executable, "-B", str(ROOT / "course/tools/build_notes.py")], check=True)
    sources = []
    for source in sorted(ROOT.rglob("*")):
        relative = source.relative_to(ROOT)
        if source.is_symlink():
            raise ValueError(f"Symlinks cannot be published: {relative}")
        if not source.is_file() or source.name in {"index.html", "deepinfer-101-course.zip"} and source.parent == ROOT:
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            raise ValueError(f"Private or generated runtime data in public tree: {relative}")
        if source.name.lower() in {"report.md", "experiment.csv"}:
            raise ValueError(f"Personal work in public tree: {relative}")
        if relative.as_posix() not in FILES and (relative.parts[0] not in DIRECTORIES or source.suffix not in EXTENSIONS):
            raise ValueError(f"Unexpected public file: {relative}")
        sources.append((source, relative))
    archive = ROOT / "deepinfer-101-course.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for source, relative in sources:
            content = source.read_bytes()
            if relative.as_posix() in {"course/site/index.html", "course/site/app.js"}:
                content = content.replace(
                    'href="../../deepinfer-101-course.zip" download>下载课程代码（含参考答案）'.encode(),
                    'href="../README.md">打开课程指南'.encode(),
                ).replace(
                    'href="../../deepinfer-101-course.zip" download>下载课程代码'.encode(),
                    'href="../README.md">课程文件'.encode(),
                )
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, content)
    print(f"Built {len(sources)} course files; archive {archive.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
