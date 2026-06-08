"""Per-query system-collection selector.

Routing inputs:

* ``catalog`` — every system collection's ``CollectionMetadata``
* ``user_whitelist`` — ``Account.active_system_rag_collections``; when
  ``None`` falls back to ``retrieval.yaml:system_rag.default_active``;
  when an explicit empty list, the user has *opted out* of system RAG and
  the router returns ``[]`` unconditionally.

Filtering rules (in order):

1. **Whitelist gate** — only collections in the user's whitelist survive.
2. **Language gate** — collection must match the query language *or*
   advertise ``cross_lingual=True``. Cross-lingual is the BGE-M3 path:
   the embedder can search English text from a Chinese query and vice-
   versa, but only when the collection metadata says so.
3. **Topic-overlap gate** — score = matches / total topics, with
   ``authority_bias[tier]`` as the minimum score. Tier-1 (default 0.0)
   always survives once language is matched; lower tiers must prove
   relevance.

Output is sorted by score descending, capped to ``max_active``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from claritymed.core.rag.schemas import CollectionMetadata, RouterEntry

QueryLanguage = Literal["en", "zh"]


@dataclass(frozen=True)
class RoutingDecision:
    """Per-collection routing trace entry."""

    name: str
    selected: bool
    reason: str
    topic_score: float = 0.0


@dataclass(frozen=True)
class RouterTrace:
    """Per-request routing audit payload."""

    selected: list[str]
    considered: list[RoutingDecision] = field(default_factory=list)


@runtime_checkable
class Router(Protocol):
    """Per-query active-collection selector."""

    def select(
        self,
        query: str,
        language: QueryLanguage,
        user_whitelist: list[str] | None,
    ) -> list[str]: ...

    def select_with_trace(
        self,
        query: str,
        language: QueryLanguage,
        user_whitelist: list[str] | None,
    ) -> RouterTrace: ...


# English tokenization for topic overlap; CJK queries fall through to the
# substring check (each topic phrase is searched directly in the lowered
# query).
_EN_TOKEN_RE = re.compile(r"[a-z][a-z0-9'\-]*")


def _english_tokens(text: str) -> set[str]:
    return set(_EN_TOKEN_RE.findall(text.lower()))


class CollectionRouter(Router):
    """Rule-based router. v1 implementation."""

    def __init__(
        self,
        catalog: list[CollectionMetadata],
        config: RouterEntry,
        default_whitelist: list[str] | None = None,
    ) -> None:
        self._catalog = {c.name: c for c in catalog}
        self._max_active = config.max_active
        self._authority_bias = dict(config.authority_bias)
        # When the user passes ``None`` for user_whitelist, this list
        # (from ``retrieval.yaml:system_rag.default_active``) takes over.
        self._default_whitelist = (
            list(default_whitelist) if default_whitelist is not None else None
        )

    # --- Router protocol -----------------------------------------------

    def select(
        self,
        query: str,
        language: QueryLanguage,
        user_whitelist: list[str] | None,
    ) -> list[str]:
        return self.select_with_trace(query, language, user_whitelist).selected

    def select_with_trace(
        self,
        query: str,
        language: QueryLanguage,
        user_whitelist: list[str] | None,
    ) -> RouterTrace:
        # An explicit empty list is "user opted out of system RAG".
        if user_whitelist == []:
            return RouterTrace(
                selected=[],
                considered=[
                    RoutingDecision(
                        name=name,
                        selected=False,
                        reason="user_whitelist=[] (opted out)",
                    )
                    for name in self._catalog
                ],
            )

        whitelist = self._resolve_whitelist(user_whitelist)
        candidates = self._whitelist_gate(whitelist)
        considered: list[RoutingDecision] = []
        survivors: list[tuple[CollectionMetadata, float]] = []

        for c in self._catalog.values():
            if c.name not in candidates:
                considered.append(
                    RoutingDecision(
                        name=c.name,
                        selected=False,
                        reason="not in whitelist",
                    )
                )
                continue
            if not self._language_matches(c, language):
                considered.append(
                    RoutingDecision(
                        name=c.name,
                        selected=False,
                        reason=(
                            f"language {language} does not match "
                            f"{c.language} (cross_lingual={c.cross_lingual})"
                        ),
                    )
                )
                continue
            score = self._topic_score(query, c)
            threshold = self._authority_bias.get(c.authority_tier, 0.0)
            if score < threshold:
                considered.append(
                    RoutingDecision(
                        name=c.name,
                        selected=False,
                        reason=(
                            f"topic_score {score:.2f} < authority_bias"
                            f"[{c.authority_tier}]={threshold:.2f}"
                        ),
                        topic_score=score,
                    )
                )
                continue
            survivors.append((c, score))

        # Sort by topic score desc, then tier asc (prefer authoritative),
        # then name asc for stable order under ties.
        survivors.sort(key=lambda t: (-t[1], t[0].authority_tier, t[0].name))
        selected = [c.name for c, _ in survivors[: self._max_active]]

        for c, score in survivors:
            kept = c.name in selected
            considered.append(
                RoutingDecision(
                    name=c.name,
                    selected=kept,
                    reason="kept"
                    if kept
                    else f"capped by max_active={self._max_active}",
                    topic_score=score,
                )
            )

        return RouterTrace(selected=selected, considered=considered)

    # --- internals ------------------------------------------------------

    def _resolve_whitelist(self, user_whitelist: list[str] | None) -> set[str]:
        if user_whitelist is not None:
            return set(user_whitelist)
        if self._default_whitelist:
            return set(self._default_whitelist)
        # No whitelist at all → every catalog entry is a candidate.
        return set(self._catalog.keys())

    def _whitelist_gate(self, whitelist: set[str]) -> set[str]:
        return whitelist & set(self._catalog.keys())

    @staticmethod
    def _language_matches(c: CollectionMetadata, language: QueryLanguage) -> bool:
        return c.language == language or c.cross_lingual

    @staticmethod
    def _topic_score(query: str, c: CollectionMetadata) -> float:
        phrases = [*c.topics, *c.disease_codes]
        if not phrases:
            return 1.0  # no topics declared → not punished
        lowered = query.lower()
        en_tokens = _english_tokens(query)
        matches = 0
        for phrase in phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in lowered:
                matches += 1
                continue
            phrase_tokens = _english_tokens(phrase_lower)
            if phrase_tokens and phrase_tokens.issubset(en_tokens):
                matches += 1
        return matches / len(phrases)
