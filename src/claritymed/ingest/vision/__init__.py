"""Vision-side training and tuning entry points.

Subpackages per dataset (currently ``busi``) mirror the
``ingest/symptoms/<dataset_id>`` layout so the same five-step pipeline
(prepare / hparam / train / tune / promote) applies. The shared MLflow
helpers live at ``claritymed.ingest.mlflow_utils`` so symptoms and
vision share experiment-naming + tracking conventions.

See ``docs/vision-model-workflow.md`` for the operator runbook.
"""
