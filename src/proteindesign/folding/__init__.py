from __future__ import annotations

from proteindesign.config import AlphaFoldConfig, ESMFoldConfig
from proteindesign.folding.base import FoldingPredictor, FoldingResult


def get_folding_predictor(model_name: str, **kwargs) -> FoldingPredictor:
    """Create a folding predictor by name.

    Args:
        model_name: ``"esmfold"`` or ``"alphafold2"``.
        **kwargs: Passed to the corresponding config dataclass.

    Returns:
        A FoldingPredictor instance.
    """
    if model_name == "esmfold":
        from proteindesign.folding.esmfold import ESMFoldPredictor

        config_fields = {k: v for k, v in kwargs.items() if hasattr(ESMFoldConfig, k)}
        return ESMFoldPredictor(ESMFoldConfig(**config_fields))
    elif model_name == "alphafold2":
        from proteindesign.folding.alphafold import AlphaFoldPredictor

        config_fields = {
            k: v for k, v in kwargs.items() if hasattr(AlphaFoldConfig, k)
        }
        return AlphaFoldPredictor(AlphaFoldConfig(**config_fields))
    else:
        raise ValueError(
            f"Unknown folding model: {model_name!r}. Use 'esmfold' or 'alphafold2'."
        )


__all__ = ["FoldingPredictor", "FoldingResult", "get_folding_predictor"]
