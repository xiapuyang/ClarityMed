"""Query expansion using a ``TermService``.

The current rule is intentionally simple:

1. Tokenize the query on whitespace + CJK character boundaries.
2. For each token (and a few short n-grams), call ``term_service.lookup``.
3. Collect every alias from every hit, in every language.
4. Append the unique aliases (minus the originals already in the query) to
   the original query string, joined by spaces.

Why so simple? In v1 the embedder is BGE-M3 (multilingual). Mixing the
surface form with synonyms / cross-lingual translations broadens both the
dense and sparse signals for retrieval. We do not need a full IR-style
query rewriter; a longer "concept bag" suffices for recall.
"""

from __future__ import annotations

import re

from claritymed.core.rag.terms.base import ConceptLanguage, TermService

# Token boundary: either a run of CJK ideographs (each character is a
# meaningful unit) or a run of ASCII word chars.
_TOKEN_RE = re.compile(r"[一-鿿]+|[A-Za-z][A-Za-z0-9'\-]*")

# Tokens up to this many words long are tried as multi-word terms
# ("acetylsalicylic acid"). Above this, lookup is per-word only.
MAX_NGRAM = 3


def expand_query(
    query: str,
    language: ConceptLanguage,
    term_service: TermService,
) -> str:
    """Return ``query`` extended with synonyms + cross-lingual aliases.

    When ``term_service`` is ``NoOpTermService`` (or no concepts match),
    returns ``query`` unchanged.
    """
    if not query.strip():
        return query

    tokens = _TOKEN_RE.findall(query)
    if not tokens:
        return query

    # Surfaces to look up: each token + sliding n-grams up to MAX_NGRAM.
    surfaces: list[str] = []
    surfaces.extend(tokens)
    for n in range(2, MAX_NGRAM + 1):
        for i in range(0, len(tokens) - n + 1):
            surfaces.append(" ".join(tokens[i : i + n]))

    # Seed seen_lower with the original query and every n-gram already in
    # it, so an alias matching one of those n-grams is not re-appended.
    seen_lower: set[str] = {_lower(query)}
    seen_lower.update(_lower(s) for s in surfaces)
    additions: list[str] = []

    for surface in surfaces:
        hits = term_service.lookup(surface, language)
        for hit in hits:
            for alias in hit.aliases:
                key = _lower(alias.text)
                if key in seen_lower:
                    continue
                seen_lower.add(key)
                additions.append(alias.text)

    if not additions:
        return query
    return query + " " + " ".join(additions)


def _lower(s: str) -> str:
    return s.strip().lower()
