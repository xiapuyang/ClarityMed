#!/usr/bin/env python3
"""Download a UMLS Metathesaurus release via the NLM Download API.

WHY A SEPARATE SCRIPT
---------------------
``init_terminology.py`` is intentionally offline (seed + merge + validate
only). Network calls live here so that bootstrap path stays trivial to
reason about: no transitive auth failures, no proxy concerns, no
"why is my seed-only invocation hitting the internet."

This script does *not* normalize. It downloads the chosen release file
into ``$CLARITYMED_SHARED_DIR/terminology/raw/`` (or wherever
``--target-dir`` points) and stops. The next step is yours: turn
``MRCONSO.RRF`` into our JSONL schema with whatever tooling you prefer
(awk / pandas / DuckDB / a one-off Python script) and then::

    uv run python scripts/init_terminology.py --merge my_normalized.jsonl

WHY NOT ``umls_downloader``
---------------------------
``umls_downloader`` (last release 0.1.3, 2024-04) uses the legacy
TGT/CAS auth flow scraped through BeautifulSoup. NLM's current
documented automation API (``uts-ws.nlm.nih.gov/download``) takes a
single GET with an ``apiKey`` query parameter — strictly simpler and
the one NLM is steering integrations toward. Pinning a stale 3rd-party
dep with a 3-step HTML-form-scraping auth dance is more risk than a
30-line ``httpx`` call.

AUTH
----
You need a UTS account + API key. Sign up free:

    https://uts.nlm.nih.gov/uts/signup-login

Get / rotate the key:

    https://uts.nlm.nih.gov/uts/profile

Pass via ``--api-key`` or set ``UMLS_API_KEY`` in the environment.

USAGE
-----
::

    # See available versions
    uv run python scripts/download_umls.py --list

    # Download the current MRCONSO file (smallest, ~600 MB) — most common case
    uv run python scripts/download_umls.py --api-key $UMLS_API_KEY

    # Pin a specific release
    uv run python scripts/download_umls.py --version 2025AB

    # Full Metathesaurus subset (much larger, ~5 GB)
    uv run python scripts/download_umls.py --release-type umls-metathesaurus-full-subset

    # Custom destination
    uv run python scripts/download_umls.py --target-dir /mnt/nas/umls/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

# Allow ``python scripts/...`` invocation without ``uv run`` when src is on
# PYTHONPATH already; otherwise fall back to repo layout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from claritymed.stores.paths import shared_terminology_dir  # noqa: E402

RELEASES_URL = "https://uts-ws.nlm.nih.gov/releases"
DOWNLOAD_URL = "https://uts-ws.nlm.nih.gov/download"

# Release types worth exposing. ``mrconso-file`` covers 99% of the term
# expansion use case (concept names + aliases + languages) at a fraction
# of the size of the full Metathesaurus.
DEFAULT_RELEASE_TYPE = "umls-metathesaurus-mrconso-file"
SUPPORTED_RELEASE_TYPES = {
    "umls-metathesaurus-mrconso-file",  # ~600 MB — names only
    "umls-metathesaurus-full-subset",  # ~5 GB — names + semantic types + relations
    "umls-full-release",  # ~30 GB — everything
}

# Streaming download chunk. 8 MiB is a sweet spot for fewer syscalls
# without bloating memory on small VMs.
CHUNK_BYTES = 8 * 1024 * 1024

# UMLS files are zip archives; the first four bytes must be the PKZip
# magic number. NLM occasionally returns an HTML error page (200 OK with
# the auth failure inline) — checking the magic catches that.
ZIP_MAGIC = b"PK\x03\x04"


# --- release discovery --------------------------------------------------


def list_releases(release_type: str, *, only_current: bool = False) -> list[dict]:
    """Return release metadata from the NLM Release API (no auth needed)."""
    params: dict[str, str] = {"releaseType": release_type}
    if only_current:
        params["current"] = "true"
    with httpx.Client(timeout=30) as client:
        r = client.get(RELEASES_URL, params=params)
        r.raise_for_status()
    return r.json()


def resolve_release(release_type: str, version: str | None) -> dict:
    """Pick a specific release by version, or the current one when omitted.

    Raises ``SystemExit`` with a helpful message when ``version`` doesn't
    match any release — UMLS version strings (``2026AA`` / ``2025AB``)
    are easy to typo.
    """
    if version is None:
        rels = list_releases(release_type, only_current=True)
        if not rels:
            sys.exit(
                f"No current release found for releaseType={release_type!r}. "
                f"Try --list to see what's available."
            )
        return rels[0]
    rels = list_releases(release_type)
    for rel in rels:
        if rel.get("releaseVersion") == version:
            return rel
    available = ", ".join(r.get("releaseVersion", "?") for r in rels[:8])
    sys.exit(
        f"version={version!r} not found for releaseType={release_type!r}. "
        f"Recent versions: {available}. Use --list for the full list."
    )


# --- download -----------------------------------------------------------


def download(
    api_key: str,
    file_url: str,
    target: Path,
    *,
    force: bool,
) -> Path:
    """Stream the release into ``target``; verify zip magic before declaring success.

    The verification is the real-world payoff: NLM's failure mode is to
    serve an HTML error page with HTTP 200, which without the magic
    check would leave a "looks downloaded but corrupt" file in place
    and the operator chasing ghosts at normalize time.
    """
    if target.exists() and not force:
        size_mb = target.stat().st_size / (1024 * 1024)
        print(
            f"refusing to overwrite existing {target} ({size_mb:.1f} MB) — "
            f"pass --force to redownload.",
            file=sys.stderr,
        )
        sys.exit(1)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Stream into a sidecar then rename, so an interrupted download
    # cannot be mistaken for a complete one on a later run.
    tmp = target.with_suffix(target.suffix + ".partial")
    bytes_written = 0
    with httpx.Client(timeout=httpx.Timeout(60.0, read=None)) as client:
        with client.stream(
            "GET",
            DOWNLOAD_URL,
            params={"url": file_url, "apiKey": api_key},
            follow_redirects=True,
        ) as response:
            if response.status_code in (401, 403):
                sys.exit(
                    f"NLM rejected the API key (HTTP {response.status_code}). "
                    f"Double-check $UMLS_API_KEY against "
                    f"https://uts.nlm.nih.gov/uts/profile — every key rotation "
                    f"invalidates the previous one."
                )
            response.raise_for_status()
            total = response.headers.get("content-length")
            total_mb = f"{int(total) / (1024 * 1024):.0f} MB" if total else "?"
            print(f"downloading {file_url}  ({total_mb})")
            with tmp.open("wb") as fh:
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    fh.write(chunk)
                    bytes_written += len(chunk)
                    if total:
                        pct = 100.0 * bytes_written / int(total)
                        print(
                            f"  {bytes_written / (1024 * 1024):>7.1f} MB "
                            f"({pct:5.1f} %)",
                            end="\r",
                            file=sys.stderr,
                        )
    print(file=sys.stderr)  # newline after the progress overwrite

    with tmp.open("rb") as fh:
        head = fh.read(4)
    if head != ZIP_MAGIC:
        # Salvage the body for debugging — 99% of the time it's an HTML
        # auth-failure page disguised as a 200.
        broken = target.with_suffix(target.suffix + ".broken")
        tmp.rename(broken)
        sys.exit(
            f"Downloaded payload is not a zip archive (first 4 bytes: {head!r}). "
            f"Body saved to {broken} for inspection — usually an NLM HTML "
            f"error page returned with HTTP 200. Check that the API key is "
            f"current and your UTS license covers this release."
        )
    tmp.rename(target)
    return target


# --- CLI ----------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a UMLS release via the NLM Download API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print available versions for --release-type and exit.",
    )
    p.add_argument(
        "--release-type",
        default=DEFAULT_RELEASE_TYPE,
        choices=sorted(SUPPORTED_RELEASE_TYPES),
        help=(
            f"Which release to pull. Default {DEFAULT_RELEASE_TYPE!r} "
            f"(MRCONSO only, ~600 MB) — enough for term expansion."
        ),
    )
    p.add_argument(
        "--version",
        default=None,
        help="Pin a specific release version (e.g. 2026AA). Omit for current.",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("UMLS_API_KEY"),
        help="UTS API key. Defaults to $UMLS_API_KEY.",
    )
    p.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory. Defaults to "
            "$CLARITYMED_SHARED_DIR/terminology/raw/."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target file already exists.",
    )
    return p.parse_args()


def _print_releases(release_type: str) -> int:
    rels = list_releases(release_type)
    if not rels:
        print(f"no releases for releaseType={release_type!r}", file=sys.stderr)
        return 1
    print(f"available releases for {release_type}:")
    for r in rels:
        marker = " (current)" if r.get("current") else ""
        print(f"  {r.get('releaseVersion', '?'):<10} {r.get('fileName', '?')}{marker}")
    return 0


def main() -> int:
    args = _parse_args()
    if args.list:
        return _print_releases(args.release_type)

    if not args.api_key:
        sys.exit(
            "missing UTS API key. Get one at https://uts.nlm.nih.gov/uts/profile "
            "and pass --api-key or set $UMLS_API_KEY."
        )

    release = resolve_release(args.release_type, args.version)
    file_url = release["downloadUrl"]
    filename = release["fileName"]

    target_dir = args.target_dir or (shared_terminology_dir() / "raw")
    target = target_dir / filename

    download(args.api_key, file_url, target, force=args.force)

    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"\nwrote {target} ({size_mb:.1f} MB)")
    print()
    print("Next steps:")
    print(f"  1. unzip {target}")
    print("  2. normalize MRCONSO.RRF into the ConceptRecord JSONL schema")
    print("     (see scripts/init_terminology.py docstring for schema)")
    print("  3. uv run python scripts/init_terminology.py --merge <your.jsonl>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
