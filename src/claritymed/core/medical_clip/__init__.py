"""BiomedCLIP-backed medical-image classification primitives.

The medical-clip module owns the ``Modality`` Literal — every other
module that needs to talk about image modality imports it from here so
the wire format and the disease-registry's ``accepted_modality`` field
can never drift apart.
"""
