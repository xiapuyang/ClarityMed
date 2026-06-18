# Convenience targets for local-only workflows that the main `claritymed`
# CLI doesn't cover. Most project workflows live in `[project.scripts]`
# in pyproject.toml or under `claritymed <subcommand>` — this file exists
# for things that are explicitly out of CI scope (heavy bench runs that
# need real datasets + trained models on disk).

.PHONY: bench-drift

# Run the cross-dataset drift bench for one pair (e.g. PAIR=breast_us).
# Prereqs: trained model artifacts under `~/.claritymed/models/vision/`
# and source datasets downloaded under `~/.claritymed/data/vision/`.
# Script fails loud if either is missing — Make does not pre-check.
#
# Outputs land under `docs/benchmarks/cross_dataset_drift/` as a dated
# JSON + Markdown pair. See
# `docs/plans/2026-06-17-001-feat-cross-dataset-drift-bench-plan.md`
# for the full design.
bench-drift:
	@if [ -z "$(PAIR)" ]; then \
		echo "usage: make bench-drift PAIR=<pair_id>"; \
		echo "  known pair ids: breast_us"; \
		exit 1; \
	fi
	uv run python scripts/bench_cross_dataset_drift.py --pair $(PAIR)
