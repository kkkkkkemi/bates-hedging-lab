"""Core is independent of product models; rates does not import equities."""

import ast
from pathlib import Path


def test_rates_and_core_import_boundaries():
    source = Path(__file__).resolve().parents[2] / "src" / "quantproj"
    for name, allowed in (("core", {"core"}), ("rates", {"rates", "core"})):
        for path in (source / name).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = (
                    [node.module]
                    if isinstance(node, ast.ImportFrom)
                    else ([part.name for part in node.names] if isinstance(node, ast.Import) else [])
                )
                for mod in modules:
                    if mod and mod.startswith("quantproj."):
                        assert mod.split(".")[1] in allowed, (path, mod)
