"""XGBoost symptom-prediction algorithm module.

Sibling to :mod:`claritymed.ingest.symptoms.typed_basd`. Provides an
``algorithm_module: xgb`` implementation of the Agent protocol the
DDXPlus adapter dispatches on. Ships as three concerns:

* :mod:`.encoding` — patient dict ↔ dense one-hot feature vector.
* :mod:`.ig_policy` — information-gain question selection over an
  XGBoost classifier's posterior.
* :mod:`.algorithm` — :class:`XgbAgent` bundling classifier +
  ev-marginals regressor + IG policy, plus joblib save/load.

Unlike :mod:`.ddxplus.__init__` this package does NOT self-register a
dataset adapter — XGBoost is an algorithm, not a dataset. It plugs into
the existing DDXPlus adapter via the ``algorithm_module`` field on
:class:`ModelSpec`.
"""
