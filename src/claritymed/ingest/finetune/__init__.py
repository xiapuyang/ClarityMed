"""Fine-tune material preprocessing (NOT indexed in RAG).

These corpora are conversational / synthetic (MedDialog-CN, Huatuo-26M)
— good signal for supervised fine-tuning, bad signal for retrieval
(unverified quality, mixed authority). Adapters here emit Alpaca-style
JSONL under ``data/finetune/`` and never write to Qdrant.
"""

from claritymed.ingest.finetune.meddialog_cn import (
    MedDialogStats,
    preprocess_meddialog_cn,
)

__all__ = ["MedDialogStats", "preprocess_meddialog_cn"]
