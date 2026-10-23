#!/usr/bin/env python3
"""DeepInfer 101 local autograder.

Usage:
    python course/tools/grade.py            # run every assignment
    python course/tools/grade.py lab00      # run one assignment
    python course/tools/grade.py --list     # list assignment names
    python course/tools/grade.py --verbose  # show unittest output
    python course/tools/grade.py prep --question q1 --failfast
    python course/tools/grade.py prep --question q1 --hint 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from assignment_specs import SPECS

COURSE_ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENTS = {
    "foundations": COURSE_ROOT / "foundations",
    "prep": COURSE_ROOT / "prep",
    "lab00": COURSE_ROOT / "labs" / "lab00",
    "lab01": COURSE_ROOT / "labs" / "lab01",
    "lab02": COURSE_ROOT / "labs" / "lab02",
    "lab03": COURSE_ROOT / "labs" / "lab03",
    "hw01": COURSE_ROOT / "hw" / "hw01",
    "hw02": COURSE_ROOT / "hw" / "hw02",
    "project1": COURSE_ROOT / "projects" / "project1",
    "lab04": COURSE_ROOT / "labs" / "lab04",
    "hw05": COURSE_ROOT / "hw" / "hw05",
    "project2": COURSE_ROOT / "projects" / "project2",
    "lab05": COURSE_ROOT / "labs" / "lab05",
    "hw06": COURSE_ROOT / "hw" / "hw06",
    "project2_scheduler": COURSE_ROOT / "projects" / "project2_scheduler",
}

# Selectors are explicit: a misspelled question must never silently run 0 tests.
QUESTIONS = {
    "prep": {
        "q0": ("TestRun", "00-run.md", ["先运行已完成的 demo.py，不用修改它。", "命令应从 DeepInfer 根目录运行；检查当前路径。", "执行 python course/prep/demo.py，随后运行 q0 测试。"]),
        "q1": ("TestMaximum", "01-debug.md", ["用全负数输入观察 best 的初值。", "分别考虑空列表、单个元素、全负数。", "非空列表用第一个元素初始化 best；空列表按规格抛 ValueError。"]),
        "q2": ("TestMatrix", "02-matrix.md", ["分别检查行是否共享，以及 total 在哪一层清零。", "修改副本的一格，观察原矩阵；手算输出的每一格。", "每行都要新建列表；每开始计算一个 C[i][j] 时令 total=0。"]),
        "q3": ("TestTesting", "03-tests.md", ["测试工具本身也要能发现错误结果。", "浮点数用绝对误差；先检查两个结果长度是否一致。", "误差大于 atol 时抛 AssertionError；向量加法先拒绝长度不匹配，再逐项相加。"]),
    },
    "lab00": {
        "q0": ("TestWalkthrough", "README.md", ["这一题只运行现成示例。", "运行 python course/labs/lab00/walkthrough.py。", "输出应有 c=[11, 22, 33, 44, 55] 和最后一组的跳过标记。"]),
        "q1": ("TestIndex TestVectorAdd", "README.md", ["先在纸上列 bid=0/1、tid=0..3 的全局索引。", "每组从 bid*block_dim 开始，组内再加 tid。", "实现 coalesced_index；vector_add 的访存骨架和尾部判断已经提供。"]),
        "q2": ("TestStrided", "README.md", ["结果不变，只改变线程分配元素的顺序。", "对照连续索引，交换 bid 和 tid 的职责。", "strided_index 返回 tid*grid_dim+bid；不要修改输出顺序。"]),
        "q3": ("TestCompare", "README.md", ["复用前两题，不要再写 kernel。", "分别取得两个 stats 字典的 global_transactions。", "返回两个整数构成的 tuple，而不是整个 stats。"]),
        "q4": ("TestRoofline", "README.md", ["分别求搬数据的时间和计算时间，注意单位。", "GB/s 乘 1e9，TFLOP/s 乘 1e12。", "取两个时间的最大值；相等归 memory；带宽和算力必须为正。"]),
        "q5": ("TestScale", "README.md", ["本题独立复用 tracer，做 c[i]=a[i]*factor。", "make_vector_gpu(a, [0]*len(a)) 可提供输入、输出地址。", "一线程处理一元素，guard i<n；每元素只有一次 load 和一次 store。"]),
        "q6": ("TestDebug", "README.md", ["先读 traceback 最下面的出错地址，再看 n。", "检查 i==n 时是否仍然执行了读写。", "最后合法索引为 n-1；修正 vector_add_debug 的边界判断。"]),
    },
}


for assignment, entries in SPECS.items():
    QUESTIONS[assignment] = {
        q: (spec[0], "README.md", list(spec[3:])) for q, spec in entries.items()
    }


def prediction_matches(answer, expected):
    """Accept equivalent decimal numerals, never evaluate student input."""
    if answer == expected:
        return True
    try:
        actual_number, expected_number = Decimal(answer), Decimal(expected)
        return actual_number.is_finite() and actual_number == expected_number
    except InvalidOperation:
        return False


def unlock(name, question, read_input=input):
    spec = SPECS[name][question]
    print(f"预测题 {name}/{question}：{spec[1]}")
    print("输入数值或指定单词；输入 exit 退出。预测不修改代码，也不锁住测试。")
    while True:
        try:
            answer = read_input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if answer == "exit":
            return False
        if prediction_matches(answer, spec[2]):
            print("预测正确。现在实现，再运行同题测试。")
            return True
        print("再推演一次。提示：" + spec[3])


def question_spec(name: str, question: str):
    return QUESTIONS[name][question]


# Keep unittest's normal tracebacks, and record results separately. Parsing its
# human-readable output loses subtest failures and breaks on test docstrings.
_WORKER = r'''
import json, sys, unittest
from pathlib import Path

report_path, verbose, failfast, *selectors = sys.argv[1:]
loader = unittest.TestLoader()
suite = loader.loadTestsFromNames(selectors) if selectors else loader.discover('.')

def ids(tests):
    for test in tests:
        if isinstance(test, unittest.TestSuite):
            yield from ids(test)
        else:
            yield test.id()

planned = list(ids(suite))
outcomes = {}

class Result(unittest.TextTestResult):
    def startTest(self, test):
        outcomes[test.id()] = 'running'
        super().startTest(test)
    def addSuccess(self, test):
        if outcomes[test.id()] == 'running':
            outcomes[test.id()] = 'pass'
        super().addSuccess(test)
    def addFailure(self, test, err):
        outcomes[test.id()] = 'fail'
        super().addFailure(test, err)
    def addError(self, test, err):
        outcomes[test.id()] = 'error'
        super().addError(test, err)
    def addSkip(self, test, reason):
        # A skipped subtest also leaves its parent question incomplete.
        parent = getattr(test, 'test_case', test)
        if outcomes.get(parent.id()) not in ('fail', 'error'):
            outcomes[parent.id()] = 'skip'
        super().addSkip(test, reason)
    def addExpectedFailure(self, test, err):
        outcomes[test.id()] = 'expected failure'
        super().addExpectedFailure(test, err)
    def addUnexpectedSuccess(self, test):
        outcomes[test.id()] = 'fail'
        super().addUnexpectedSuccess(test)
    def addSubTest(self, test, subtest, err):
        if err is not None:
            outcomes[test.id()] = 'fail' if issubclass(err[0], test.failureException) else 'error'
        super().addSubTest(test, subtest, err)

result = unittest.TextTestRunner(verbosity=int(verbose), failfast=failfast == '1',
                                 resultclass=Result).run(suite)
Path(report_path).write_text(json.dumps({'planned': planned, 'outcomes': outcomes,
                                       'tests_run': result.testsRun}), encoding='utf-8')
if not result.testsRun:
    print('ERROR: no tests collected; check test filenames and selectors.', file=sys.stderr)
complete = bool(planned) and all(outcomes.get(test) == 'pass' for test in planned)
sys.exit(0 if result.wasSuccessful() and complete else 1)
'''


def _run_unittest(cwd: Path, verbose: bool, selectors=(), failfast: bool = False,
                  timeout: float = 30, report=None) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "results.json"
        command = [sys.executable, "-B", "-c", _WORKER, str(report_path),
                   "2" if verbose else "1", "1" if failfast else "0", *selectors]
        try:
            proc = subprocess.run(
                command, cwd=cwd, text=True, encoding="utf-8",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                capture_output=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            def decoded(value):
                return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
            output = decoded(exc.stdout) + decoded(exc.stderr)
            output += f"\nTIMEOUT: 超过 {timeout:g}s。检查循环是否推进、递归是否到达终止条件；可用 --timeout 调整秒数。\n"
            if verbose:
                print(output)
            return False, output
        if report is not None and report_path.exists():
            report.update(json.loads(report_path.read_text(encoding="utf-8")))
    output = (proc.stdout or "") + (proc.stderr or "")
    if verbose:
        print(output, end="" if output.endswith("\n") else "\n")
    return proc.returncode == 0, output


def run_one(name: str, verbose: bool, solution: bool, question=None, failfast=False,
            timeout=30, report=None) -> tuple[bool, str]:
    path = ASSIGNMENTS[name]
    if not path.is_dir():
        return False, f"missing directory: {path}"
    selectors = []
    if question:
        classes = question_spec(name, question)[0].split()
        module = f"test_{name}"
        selectors = [f"{module}.{cls}" for cls in classes]
    if not solution:
        return _run_unittest(path, verbose, selectors, failfast, timeout, report)

    # Copy the assignment to a temp directory and replace each student file
    # with its reference solution. This validates the tests without touching
    # the student's work.
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / name
        shutil.copytree(path, work, ignore=shutil.ignore_patterns("__pycache__"))
        for sol in work.glob("*_solution.py"):
            target = sol.with_name(sol.name[: -len("_solution.py")] + ".py")
            shutil.copyfile(sol, target)
        return _run_unittest(work, verbose, selectors, failfast, timeout, report)


def question_feedback(name, report, question=None):
    """Return question status from actual test IDs, including unrun tests."""
    feedback = []
    planned = report.get("planned", [])
    outcomes = report.get("outcomes", {})
    for q, spec in QUESTIONS.get(name, {}).items():
        if question and q != question:
            continue
        selectors = [f"test_{name}.{part}" for part in spec[0].split()]
        selected = {test for test in planned if any(
            test == selector or test.startswith(selector + ".") for selector in selectors)}
        if not selected:
            continue
        passed = sum(outcomes.get(test) == "pass" for test in selected)
        failed = any(outcomes.get(test) in ("fail", "error") for test in selected)
        status = "FAIL" if failed else "PASS" if passed == len(selected) else "未完成"
        feedback.append((q, status, passed, len(selected)))
    return feedback


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DeepInfer 101 tests.")
    parser.add_argument("assignments", nargs="*", help="prep, lab00, lab01, hw01, project1")
    parser.add_argument("--list", action="store_true", help="list assignment names")
    parser.add_argument("--verbose", action="store_true", help="show unittest output")
    parser.add_argument("--solution", action="store_true", help="run the reference solutions")
    parser.add_argument("--question", help="只运行一道题，例如 prep --question q1")
    parser.add_argument("--hint", type=int, choices=(1, 2, 3), help="显示该题的一层提示，不运行测试")
    parser.add_argument("--failfast", action="store_true", help="遇到第一个失败就停止该项测试")
    parser.add_argument("--timeout", type=float, default=30, help="每份作业的测试超时秒数（默认 30）")
    parser.add_argument("--unlock", action="store_true", help="编码前预测输入输出；支持题号见 --list")
    args = parser.parse_args()

    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout 必须是有限正数")

    if args.hint and not args.question:
        parser.error("--hint 需要 --question")
    if args.unlock and (not args.question or args.hint or args.solution or args.list):
        parser.error("--unlock 需要 --question，且不与 hint/solution/list 一起使用")
    if args.question and (len(args.assignments) != 1):
        parser.error("--question 需要且只能指定一个实验，例如 prep --question q1")
    for name in args.assignments:
        if name not in ASSIGNMENTS:
            parser.error(f"未知实验 {name}；可选：{', '.join(ASSIGNMENTS)}")
    if args.question and args.question not in QUESTIONS.get(args.assignments[0], {}):
        available = ", ".join(QUESTIONS.get(args.assignments[0], {})) or "该实验暂不支持按题运行"
        parser.error(f"未知题号 {args.question}；{available}")
    if args.list:
        for name in args.assignments or ASSIGNMENTS:
            path = ASSIGNMENTS[name]
            print(f"{name:10s} {path.relative_to(COURSE_ROOT)}")
            for q, spec in QUESTIONS.get(name, {}).items():
                print(f"  {q}: {spec[0]}")
        return 0
    if args.unlock:
        name = args.assignments[0]
        if args.question not in SPECS.get(name, {}):
            parser.error("此题的预测练习在 README 中；交互预测支持 assignment_specs.py 中登记的题目")
        return 0 if unlock(name, args.question) else 1
    if args.hint:
        name = args.assignments[0]
        spec = question_spec(name, args.question)
        print(f"{name} / {args.question} / 提示 {args.hint}：{spec[2][args.hint - 1]}")
        print(f"回看：{(ASSIGNMENTS[name] / spec[1]).relative_to(COURSE_ROOT.parent)}")
        return 0

    names = args.assignments or list(ASSIGNMENTS)
    results: list[tuple[str, bool, str]] = []
    for name in names:
        print(f"== {name} ==")
        report = {}
        ok, output = run_one(name, args.verbose, args.solution, args.question, args.failfast,
                             args.timeout, report)
        results.append((name, ok, output))
        print("PASS" if ok else "FAIL")
        feedback = question_feedback(name, report, args.question)
        for q, status, passed, total in feedback:
            print(f"  {q:4s} {status:6s} {passed}/{total} tests passed")
        if not ok and not args.verbose:
            # Keep the first failure and its traceback, not just the last task.
            for line in output.strip().splitlines()[:45]:
                print("  " + line)
        if not ok:
            if args.question:
                print(f"下一步：python course/tools/grade.py {name} --question {args.question} --hint 1")
            elif name in QUESTIONS:
                first = next((q for q, status, _, _ in feedback if status == "FAIL"),
                             next(iter(QUESTIONS[name])))
                print(f"建议逐题开始：python course/tools/grade.py {name} --question {first} --failfast")
            print("NotImplementedError = 还没实现；AssertionError = 结果与期望不同；IndexError = 检查索引边界。")

    print()
    print("summary")
    print("-------")
    for name, ok, _ in results:
        print(f"{name:10s} {'PASS' if ok else 'FAIL'}")

    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
