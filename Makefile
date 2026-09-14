# Convenience targets for local-only workflows that the main `claritymed`
# CLI doesn't cover. Most project workflows live in `[project.scripts]`
# in pyproject.toml or under `claritymed <subcommand>` — this file exists
# for things that are explicitly out of CI scope (heavy bench runs that
# need real datasets + trained models on disk).

.PHONY: bench-drift admin-ui admin-ui-dev admin-ui-test

# Build the admin SPA into src/claritymed/web/admin_ui/dist/. FastAPI's
# StaticFiles mount picks it up at /admin/* on the next server start.
# Run once after `uv sync` (and any time admin_ui/ source changes).
admin-ui:
	cd src/claritymed/web/admin_ui && npm install && npm run build

# Dev server for the admin SPA. Proxies /api and /auth to 127.0.0.1:8120
# so the chat backend on the default port serves the data calls. Use
# alongside `claritymed-web`.
admin-ui-dev:
	cd src/claritymed/web/admin_ui && npm install && npm run dev

# Vitest run. Useful for pre-commit / CI to verify admin_ui without a
# full repo-root npm setup.
admin-ui-test:
	cd src/claritymed/web/admin_ui && npm install && npm test


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
		echo "  known pair ids: breast_us, chest_xray"; \
		exit 1; \
	fi
	uv run python -m tests.benchmarks.cross_dataset_drift.run --pair $(PAIR)
