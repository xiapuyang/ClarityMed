#!/usr/bin/env python3
"""Download MedRAG / MedCorp RAG corpora into data/download/medrag/.

WHY THIS SCRIPT EXISTS
----------------------
The MedRAG corpora live on Hugging Face (chunks) and NCBI (StatPearls raw
text); the precomputed MedCPT embeddings live on a UVA SharePoint. None of
those hosts is reachable from a restricted/CI sandbox, so this is meant to be
run *on your own machine* with normal network access.

It is deliberately disk-aware: the full MedCorp (PubMed + Wikipedia chunks,
let alone embeddings) is tens to >100 GB and will not fit on a typical laptop.
The default pulls only the clean, high-value, small corpus (StatPearls).

SOURCES (all verified against the MedRAG repo, commit-current as of 2024-02):
  - chunks:      https://huggingface.co/datasets/MedRAG/{pubmed,textbooks,wikipedia,statpearls}
  - statpearls:  https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz
                 chunked by Teddy-XiongGZ/MedRAG  src/data/statpearls.py
  - embeddings:  UVA SharePoint links baked into MedRAG src/utils.py (MedCPT only)

LICENSING (read before redistributing anything):
  - PubMed / StatPearls / Wikipedia(CC BY-SA): clean, redistributable per source terms.
  - Textbooks: 18 copyrighted USMLE textbooks. Research/eval only. Do NOT ship.
  - StatPearls text is NOT redistributed by MedRAG; you build it locally from NCBI.

USAGE
  python scripts/fetch_corpora.py --list
  python scripts/fetch_corpora.py                      # statpearls only (default)
  python scripts/fetch_corpora.py --corpus textbooks
  python scripts/fetch_corpora.py --corpus pubmed wikipedia
  python scripts/fetch_corpora.py --corpus textbooks --embeddings   # + precomputed MedCPT
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# --- repo root + default destination -------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT / "data" / "download" / "medrag"

NCBI_STATPEARLS = (
    "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"
)
MEDRAG_GIT = "https://github.com/Teddy-XiongGZ/MedRAG.git"

# Approximate *download* sizes (chunks). Real size is queried live when possible.
CORPORA = {
    "statpearls": dict(hf=None, approx_gb=0.3, note="built locally from NCBI; clean"),
    "textbooks": dict(
        hf="MedRAG/textbooks", approx_gb=0.1, note="COPYRIGHTED, research/eval only"
    ),
    "pubmed": dict(hf="MedRAG/pubmed", approx_gb=42.0, note="23.9M snippets; clean"),
    "wikipedia": dict(
        hf="MedRAG/wikipedia", approx_gb=12.0, note="CC BY-SA; general knowledge"
    ),
}

# Precomputed MedCPT embeddings (textbooks/pubmed/wikipedia only — statpearls
# is embedded locally by MedRAG). URLs verified from MedRAG src/utils.py.
# These are fragile SharePoint links; if they 404, re-embed locally (bge-m3).
EMB_MEDCPT = {
    "textbooks": "https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EQ8uXe4RiqJJm0Tmnx7fUUkBKKvTwhu9AqecPA3ULUxUqQ?download=1",
    "pubmed": "https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EVCuryzOqy5Am5xzRu6KJz4B6dho7Tv7OuTeHSh3zyrOAw?download=1",
    "wikipedia": "https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EXoxEANb_xBFm6fa2VLRmAcBIfCuTL-5VH6vl4GxJ06oCQ?download=1",
}
# Rough embedding sizes (MedCPT, 768-d). pubmed is the killer.
EMB_APPROX_GB = {"textbooks": 0.5, "pubmed": 75.0, "wikipedia": 22.0}


def human(gb: float) -> str:
    return f"{gb * 1024:.0f} MB" if gb < 1 else f"{gb:.1f} GB"


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1024**3


def live_size_gb(repo_id: str) -> float | None:
    """Sum LFS file sizes of a HF dataset's chunk/ dir. None if HF unreachable."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(repo_id, repo_type="dataset", files_metadata=True)
        total = 0
        for f in info.siblings:
            if not (f.rfilename or "").startswith("chunk/"):
                continue
            total += f.lfs.size if getattr(f, "lfs", None) else (f.size or 0)
        return total / 1024**3
    except Exception:
        return None


def cmd_list(dest: Path) -> None:
    print(f"Destination:   {dest}")
    print(f"Free on disk:  {free_gb(dest):.1f} GB\n")
    print(f"{'corpus':<11} {'chunks (approx)':>16}  {'+MedCPT emb':>12}  note")
    print("-" * 78)
    for name, c in CORPORA.items():
        emb = human(EMB_APPROX_GB[name]) if name in EMB_MEDCPT else "—"
        print(f"{name:<11} {human(c['approx_gb']):>16}  {emb:>12}  {c['note']}")
    print(
        "\nNote: sizes are approximate. Pass --corpus to download; StatPearls is the "
        "\ndefault. PubMed/Wikipedia embeddings are very large — prefer re-embedding "
        "\nlocally with bge-m3 for a bilingual (zh/en) retriever."
    )


def preflight(corpora: list[str], want_emb: bool, dest: Path) -> float:
    need = 0.0
    for name in corpora:
        live = live_size_gb(CORPORA[name]["hf"]) if CORPORA[name]["hf"] else None
        need += live if live is not None else CORPORA[name]["approx_gb"]
        if want_emb and name in EMB_MEDCPT:
            need += EMB_APPROX_GB[name]
    free = free_gb(dest)
    print(f"Estimated download: ~{need:.1f} GB   |   free on disk: {free:.1f} GB")
    if free < need * 1.15:
        print(
            f"\nABORT: not enough headroom (need ~{need * 1.15:.0f} GB incl. unzip/temp). "
            "Free space or pick a smaller --corpus set.",
            file=sys.stderr,
        )
        sys.exit(2)
    return need


def fetch_hf_chunks(repo_id: str, out: Path) -> None:
    from huggingface_hub import snapshot_download

    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(out),
        allow_patterns=["chunk/*", "README.md"],
        max_workers=8,
    )
    print(f"  done -> {out}")


def build_statpearls(dest: Path) -> None:
    """Clone MedRAG, pull NCBI tarball, run the official chunker."""
    build = dest / "_statpearls_build"
    repo = build / "MedRAG"
    if not repo.exists():
        build.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", MEDRAG_GIT, str(repo)], check=True
        )
    sp = repo / "corpus" / "statpearls"
    sp.mkdir(parents=True, exist_ok=True)
    tar = sp / "statpearls_NBK430685.tar.gz"
    if not tar.exists():
        print("  downloading StatPearls from NCBI bookshelf...")
        subprocess.run(["wget", "-c", NCBI_STATPEARLS, "-O", str(tar)], check=True)
    if not (sp / "statpearls_NBK430685").exists():
        subprocess.run(["tar", "-xzf", str(tar), "-C", str(sp)], check=True)
    print("  chunking StatPearls (needs `pip install tqdm`)...")
    subprocess.run(
        [sys.executable, "src/data/statpearls.py"], cwd=str(repo), check=True
    )
    final = dest / "statpearls" / "chunk"
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(sp / "chunk", final)
    print(f"  done -> {final}  ({len(list(final.glob('*.jsonl')))} files)")
    normalize_statpearls(dest)


def normalize_statpearls(dest: Path) -> None:
    """Rewrite MedRAG StatPearls chunks into our ingest schema.

    MedRAG ships ``{id, title, content, contents}`` with one chunk per line;
    our ``StatPearlsSource`` reads ``{doc_id, title, text}`` with one
    RawDocument per line. We aggregate all chunk-lines in a file into one
    article-level doc so our parent_child chunker (Unit 3) controls chunk
    sizing instead of MedRAG's pre-split. Output goes to a sibling
    ``normalized/`` dir; the original ``chunk/`` is left untouched for
    debugging or re-runs.
    """
    src = dest / "statpearls" / "chunk"
    out = dest / "statpearls" / "normalized"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    n_articles = 0
    for path in sorted(src.glob("*.jsonl")):
        title = ""
        parts: list[str] = []
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                if not title:
                    title = obj.get("title", "")
                content = obj.get("content") or obj.get("contents") or ""
                if content:
                    parts.append(content)
        if not parts:
            continue
        doc_id = path.stem
        (out / f"{doc_id}.jsonl").write_text(
            json.dumps(
                {"doc_id": doc_id, "title": title, "text": "\n\n".join(parts)},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        n_articles += 1
    print(f"  normalized {n_articles} articles -> {out}")


def fetch_embeddings(name: str, dest: Path) -> None:
    url = EMB_MEDCPT.get(name)
    if not url:
        print(f"  [skip] no precomputed MedCPT embeddings for {name}")
        return
    out = dest / name / "index" / "ncbi_MedCPT-Article-Encoder"
    out.mkdir(parents=True, exist_ok=True)
    zf = out / "embedding.zip"
    print(f"  downloading {name} MedCPT embeddings (~{human(EMB_APPROX_GB[name])})...")
    print("  [warning] SharePoint links are fragile; if this fails, re-embed locally.")
    try:
        subprocess.run(["wget", "-c", "-O", str(zf), url], check=True)
        subprocess.run(["unzip", "-o", str(zf), "-d", str(out)], check=True)
        zf.unlink(missing_ok=True)
        print(f"  done -> {out}")
    except subprocess.CalledProcessError:
        print(
            f"  [FAILED] {name} embeddings unavailable from SharePoint. "
            "Re-embed locally instead (see CORPORA.md).",
            file=sys.stderr,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--corpus",
        nargs="+",
        choices=list(CORPORA),
        default=["statpearls"],
        help="which corpora to download (default: statpearls)",
    )
    p.add_argument(
        "--embeddings",
        action="store_true",
        help="also fetch precomputed MedCPT embeddings (large; textbooks/pubmed/wikipedia only)",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"output dir (default: {DEFAULT_DEST})",
    )
    p.add_argument("--list", action="store_true", help="show corpora + sizes and exit")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = p.parse_args()

    if args.list:
        cmd_list(args.dest)
        return

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print(
            "Missing dependency. Run:  uv pip install huggingface_hub  (or pip install huggingface_hub)",
            file=sys.stderr,
        )
        sys.exit(1)

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Corpora: {', '.join(args.corpus)}   embeddings={args.embeddings}")
    preflight(args.corpus, args.embeddings, args.dest)
    if "textbooks" in args.corpus:
        print(
            "\n[notice] 'textbooks' is 18 COPYRIGHTED USMLE books — research/eval only, do not redistribute."
        )
    if not args.yes:
        if input("\nProceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("aborted.")
            return

    for name in args.corpus:
        print(f"\n=== {name} ===")
        if name == "statpearls":
            build_statpearls(args.dest)
        else:
            fetch_hf_chunks(CORPORA[name]["hf"], args.dest / name)
        if args.embeddings:
            fetch_embeddings(name, args.dest)

    print(f"\nAll done. Corpora under: {args.dest}")


if __name__ == "__main__":
    main()
