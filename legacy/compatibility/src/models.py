"""Compatibility exports for the historical ``src.models`` import path."""

from src.worm_species.models.factory import _load_model, build_model

__all__ = ["_load_model", "build_model"]
