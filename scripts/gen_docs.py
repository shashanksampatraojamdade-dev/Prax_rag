"""Documentation generator for Prax RAG.

Regenerates the code-derived sections of docs/ARCHITECTURE.md and
docs/DEVELOPMENT.md by introspecting the actual source (settings, API routes,
module tree). Run via `make docs` (or directly) after changing code so the
guides never drift from the implementation.

Each generated block lives between HTML markers:
    <!-- AUTOGEN:section -->
    ...generated content...
    <!-- /AUTOGEN:section -->
Prose outside the markers is hand-written and preserved.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
DOCS = ROOT / "docs"


# --------------------------------------------------------------------------- #
# Section builders
# --------------------------------------------------------------------------- #
def build_settings_table() -> str:
    from app.core.config import Settings

    secret = {"postgres_password"}
    rows = ["| Env var | Type | Default | Notes |", "|---|---|---|---|"]
    for name, field in Settings.model_fields.items():
        env = name.upper()
        type_name = getattr(field.annotation, "__name__", str(field.annotation))
        default = "•••" if name in secret else field.default
        note = field.description or ""
        rows.append(f"| `{env}` | {type_name} | `{default}` | {note} |")
    return "\n".join(rows)


def build_routes_table() -> str:
    from app.main import create_app

    app = create_app()
    schema = app.openapi()
    api_prefix = "/api/v1"
    rows = ["| Method | Path | Summary |", "|---|---|---|"]
    entries = []
    for path, operations in schema.get("paths", {}).items():
        if not path.startswith(api_prefix):
            continue
        for method, op in operations.items():
            summary = op.get("summary", "")
            entries.append((path, method.upper(), summary))
    for path, method, summary in sorted(entries):
        rows.append(f"| {method} | `{path}` | {summary} |")
    return "\n".join(rows)


def build_module_map() -> str:
    lines = ["```"]
    for path in sorted(APP.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = path.relative_to(ROOT)
        doc = _module_docstring(path)
        suffix = f"  # {doc}" if doc else ""
        lines.append(f"{rel}{suffix}")
    lines.append("```")
    return "\n".join(lines)


def build_dependencies() -> str:
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = data.get("project", {}).get("dependencies", [])
    return "\n".join(f"- `{d}`" for d in deps)


def build_kb_settings() -> str:
    """KB-tuning knobs, auto-synced into docs/KB_IMPROVEMENTS.md."""
    from app.core.config import Settings

    kb_fields = [
        "chunk_strategy",
        "semantic_breakpoint_percentile",
        "enable_ocr",
        "enable_parent_child",
        "parent_chunk_size",
        "enable_deduplication",
        "enable_versioning",
        "enable_incremental",
        "enable_adaptive_chunking",
        "chunk_size",
        "chunk_overlap",
        "retrieval_top_k",
    ]
    rows = ["| Env var | Default | Purpose |", "|---|---|---|"]
    for name in kb_fields:
        field = Settings.model_fields.get(name)
        if field is None:
            continue
        env = name.upper()
        note = field.description or ""
        rows.append(f"| `{env}` | `{field.default}` | {note} |")
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _first_doc_line(func) -> str:
    doc = (func.__doc__ or "").strip()
    return doc.splitlines()[0] if doc else ""


def _module_docstring(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text())
        doc = ast.get_docstring(tree)
        return doc.splitlines()[0].strip() if doc else ""
    except SyntaxError:
        return ""


# --------------------------------------------------------------------------- #
# Marker replacement engine
# --------------------------------------------------------------------------- #
def replace_section(text: str, section: str, content: str) -> str:
    begin = f"<!-- AUTOGEN:{section} -->"
    end = f"<!-- /AUTOGEN:{section} -->"
    pattern = re.compile(
        re.escape(begin) + r".*?" + re.escape(end), flags=re.DOTALL
    )
    replacement = f"{begin}\n{content}\n{end}"
    if not pattern.search(text):
        raise SystemExit(f"Marker '{section}' not found in document")
    return pattern.sub(lambda _: replacement, text)


SECTION_BUILDERS = {
    "settings": build_settings_table,
    "routes": build_routes_table,
    "modules": build_module_map,
    "dependencies": build_dependencies,
    "kb_settings": build_kb_settings,
}


def update_doc(doc_path: Path, write: bool = True) -> bool:
    original = doc_path.read_text()
    updated = original
    for section, builder in SECTION_BUILDERS.items():
        if f"<!-- AUTOGEN:{section} -->" in updated:
            updated = replace_section(updated, section, builder())
    if updated != original:
        if write:
            doc_path.write_text(updated)
        return True
    return False


def main() -> int:
    import sys

    check_only = "--check" in sys.argv
    changed = []
    for name in ("ARCHITECTURE.md", "DEVELOPMENT.md", "KB_IMPROVEMENTS.md"):
        path = DOCS / name
        if not path.exists():
            continue
        if update_doc(path, write=not check_only):
            changed.append(name)

    if check_only:
        if changed:
            print("ERROR: docs are out of date:", ", ".join(changed))
            print("Run 'make docs' and commit the result.")
            return 1
        print("Docs are up to date.")
        return 0

    if changed:
        print("Updated:", ", ".join(changed))
    else:
        print("Docs already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
