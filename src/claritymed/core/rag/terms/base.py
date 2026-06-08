"""TermService protocol + value types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

ConceptType = Literal["drug", "disease", "symptom", "procedure", "other"]
ConceptLanguage = Literal["en", "zh"]
AliasSource = Literal["umls", "cmekg", "rxnorm", "snomed", "user"]


@dataclass(frozen=True)
class Alias:
    """A surface form pointing at a concept."""

    text: str
    language: ConceptLanguage
    source: AliasSource


@dataclass(frozen=True)
class ConceptRecord:
    """Full record stored in the local terminology index."""

    concept_id: str
    type: ConceptType
    aliases: tuple[Alias, ...]


@dataclass(frozen=True)
class ConceptHit:
    """A surface-form lookup hit. ``surface`` is the form the caller typed;
    ``aliases`` is everything the concept knows about (including ``surface``).
    """

    concept_id: str
    surface: str
    language: ConceptLanguage
    score: float
    type: ConceptType
    aliases: tuple[Alias, ...]


@runtime_checkable
class TermService(Protocol):
    """Surface-form → concept lookup + cross-lingual alias retrieval."""

    def lookup(self, surface: str, language: ConceptLanguage) -> list[ConceptHit]:
        """Return concepts whose aliases match ``surface``. Empty on miss."""
        ...

    def cross_lingual_aliases(self, concept_id: str) -> list[Alias]:
        """All aliases for a concept across all languages."""
        ...
