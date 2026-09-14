"""Record-import ingest module.

The skill `import-medical-record` builds a template directory; this
module loads it, applies cases via ``ManifestStore``, applies facts via
``ProfileStore``, and keeps an audit trail under ``data/_imports/<id>/``.

Public surface (consumed by the CLI in ``claritymed.cli.commands.record``):

* ``template_schema`` — pydantic contracts for the template directory.
* ``template_loader`` — walk + validate + ``import_id`` canonicalization.
* ``state`` — ``rows.jsonl`` / ``session.yaml`` helpers.
* ``case_writer.apply_case`` — apply one case.
* ``fact_writer.apply_facts`` — apply one user's facts bundle.
"""
