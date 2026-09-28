from __future__ import annotations

__all__ = ["GroupDRO", "RISEBatchTransform", "StartShiftInferencePreprocessor"]


def __getattr__(name: str):
    """Avoid importing torch-heavy modules for lightweight CLI/config tooling."""
    if name == "GroupDRO":
        from .robust import GroupDRO

        return GroupDRO
    if name in {"RISEBatchTransform", "StartShiftInferencePreprocessor"}:
        from .augment import RISEBatchTransform, StartShiftInferencePreprocessor

        return {
            "RISEBatchTransform": RISEBatchTransform,
            "StartShiftInferencePreprocessor": StartShiftInferencePreprocessor,
        }[name]
    raise AttributeError(name)
