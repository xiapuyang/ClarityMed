"""RSNA Pneumonia Detection Challenge — bbox-preserving detection adapter.

The classification path lives under :mod:`claritymed.ingest.vision.rsna_pneumonia`
and collapses the upstream CSV's bbox rows into image-level binary
labels via ``max(Target)``. This adapter takes the inverse stance:
keep the bboxes, emit YOLO-format labels (one .txt per image, one
line per bbox), and surface the result through the yolo_forge pipeline.

Both adapters share the upstream raw archive and the DICOM→PNG cache —
``prepare_fn`` here invokes the classification ``discover()`` purely
to populate the cache, then re-parses the labels CSV directly to keep
the bbox coordinates.
"""
