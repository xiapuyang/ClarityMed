"""LC25000 — shared Kaggle download source for the two per-organ histopath modules.

The upstream archive
(``andrewmvd/lung-and-colon-cancer-histopathological-images``) bundles
both lung and colon biopsies in a single 25k-image set. Lung and colon
are clinically independent — a patient's lung biopsy and a patient's
colon biopsy go through different specialists and different care
pathways — so this project ships them as TWO disease modules:

* :mod:`claritymed.ingest.vision.lung_histopath` (3-class:
  ``adenocarcinoma``, ``normal``, ``squamous_cell_carcinoma``).
* :mod:`claritymed.ingest.vision.colon_histopath` (2-class:
  ``adenocarcinoma``, ``normal``).

Both modules pull from the same Kaggle archive but train independent
checkpoints against organ-specific subsets of the data. To avoid
downloading the 5GB archive twice, both reuse this module's
:mod:`download` — the on-disk path is shared; the discovery in each
per-organ ``dataset.py`` walks only its organ subdir.

Same credentials story as the other vision modules: ``KAGGLE_USERNAME``
+ ``KAGGLE_KEY`` env vars or ``~/.kaggle/kaggle.json``.

Shared constants also live in :mod:`.source` — folder aliases that map
the upstream short names (``lung_aca`` → ``adenocarcinoma``,
``colon_n`` → ``normal``, etc.) plus the image extensions both
modules accept.
"""
