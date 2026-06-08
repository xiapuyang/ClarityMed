"""Terminology subsystem (query expansion only in v1).

Plan §决策 8 chose **query expansion + post-retrieval entity linking**
over ingest-time NER: a term lookup at query time pulls synonyms /
cross-lingual aliases / brand-name↔generic mappings and joins them onto
the query string before the embedder sees it. No ingest-time NER pass
means no batched concept tagging cost; the trade-off is the term service
must cover the surface forms the user will type.

Two implementations:

* ``UmlsCmekgLocalService`` — loads a pre-normalized JSONL of concepts
  + aliases from ``data/terminology/`` (UMLS + CMeKG export the operator
  prepared offline). License-friendly: we never redistribute UMLS data.
* ``NoOpTermService`` — identity passthrough; used when the operator
  has not provisioned a terminology export, so retrieval still runs (with
  weaker recall on synonyms / cross-lingual).

Production data layout::

    data/terminology/
        concepts.jsonl   # one ConceptRecord per line
        README.md        # how to regenerate from UMLS + CMeKG

The ``ConceptRecord`` JSONL schema is::

    {
      "concept_id": "C0004057",                    # UMLS CUI or CMeKG id
      "type": "drug" | "disease" | "symptom" | "procedure",
      "aliases": [
        {"text": "aspirin", "language": "en", "source": "umls"},
        {"text": "acetylsalicylic acid", "language": "en", "source": "umls"},
        {"text": "ASA", "language": "en", "source": "umls"},
        {"text": "阿司匹林", "language": "zh", "source": "cmekg"}
      ]
    }
"""

from claritymed.core.rag.terms.base import (
    Alias,
    ConceptHit,
    ConceptRecord,
    ConceptType,
    TermService,
)
from claritymed.core.rag.terms.expansion import expand_query
from claritymed.core.rag.terms.factory import build_term_service
from claritymed.core.rag.terms.umls_cmekg import (
    NoOpTermService,
    UmlsCmekgLocalService,
)

__all__ = [
    "Alias",
    "ConceptHit",
    "ConceptRecord",
    "ConceptType",
    "NoOpTermService",
    "TermService",
    "UmlsCmekgLocalService",
    "build_term_service",
    "expand_query",
]
