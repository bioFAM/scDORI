"""Check relative help links reachable from the three main notebooks (offline)."""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


def document_text(path):
    """Include notebook source and rendered Markdown from executed cells."""
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text())
        parts = []
        for cell in notebook["cells"]:
            parts.append("".join(cell["source"]))
            for output in cell.get("outputs", []):
                parts.append("".join(output.get("data", {}).get("text/markdown", [])))
        return "\n".join(parts)
    return path.read_text()


def check_links(root):
    """Follow Markdown guides, check paths and heading anchors, list external URLs."""
    root = Path(root).resolve()
    queue = [root / "docs/notebooks" / f"{name}.ipynb" for name in ["preprocessing", "training", "downstream"]]
    seen, links, problems, external = set(), [], [], set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            problems.append({"source": str(path), "reason": "missing document"})
            continue
        for url in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", document_text(path)):
            split = urlsplit(url)
            if split.scheme:
                if split.scheme in {"http", "https"}:
                    external.add(url)
                continue
            target = (path.parent / unquote(split.path)).resolve() if split.path else path
            record = {"source": str(path.relative_to(root)), "url": url}
            links.append(record)
            if not target.exists():
                problems.append({**record, "reason": "missing target"})
                continue
            if split.fragment and target.suffix in {".md", ".ipynb"}:
                headings = re.findall(r"^#{1,6}\s+(.+)", document_text(target), re.M)
                anchors = {re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-") for h in headings}
                if unquote(split.fragment) not in anchors:
                    problems.append({**record, "reason": "missing heading anchor"})
            if target.suffix == ".md":
                queue.append(target)
    return {"root": str(root), "documents": len(seen), "relative_links": len(links),
            "problems": problems, "external_urls_not_checked_offline": sorted(external)}


def main():
    """Check a checkout or a portable executed-notebook project with the same layout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check_links(args.root)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
    raise SystemExit(bool(result["problems"]))


if __name__ == "__main__":
    main()
