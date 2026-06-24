"""Job runner callables for the admin JobRegistry.

Each runner takes ``(JobSpec, JobRegistry)`` and is registered against a
:data:`claritymed.web.admin.jobs.JobKind` at lifespan startup. The runner
mutates spec state via ``registry.update`` / ``registry.append_stdout``
so every transition reaches the on-disk JSON mirror.
"""
