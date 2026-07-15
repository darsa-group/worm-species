from __future__ import annotations

import re

import pandas as pd

from .labels import is_missing_label


def strip_final_number(value: str) -> str:
    """Remove the final barcode replicate number."""
    return re.sub(r"_\d+$", "", str(value))


def parse_taxonomy_from_barcode(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create genus, strict species, life-stage, and practical taxon columns."""
    data_cfg = cfg["data"]
    barcode_col = data_cfg.get("barcode_col", "barcode")
    if barcode_col not in df.columns:
        raise ValueError(f"barcode_col='{barcode_col}' not found in metadata.")

    barcode_base = df[barcode_col].astype(str).map(strip_final_number)
    extracted = barcode_base.str.extract(r"^(.+)_(Adult|Juvenile)$")
    parsed_taxon = extracted[0].where(extracted[0].notna(), barcode_base)
    parsed_life_stage = extracted[1]

    if "taxon_label" not in df.columns:
        df["taxon_label"] = parsed_taxon
    if "species_label" not in df.columns:
        df["species_label"] = parsed_taxon
    else:
        df["species_label"] = df["species_label"].where(
            df["species_label"].notna(), parsed_taxon
        )
    if "life_stage" not in df.columns:
        df["life_stage"] = parsed_life_stage
    else:
        df["life_stage"] = df["life_stage"].where(
            df["life_stage"].notna(), parsed_life_stage
        )
    if "genus" not in df.columns:
        df["genus"] = df["taxon_label"].astype(str).str.split("_").str[0]
    return df


def apply_taxonomic_uncertainty_rules(
    df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Mask uncertain strict-species labels while retaining other tasks."""
    data_cfg = cfg["data"]
    uncertainty_cfg = data_cfg.get("taxonomic_uncertainty", {})
    missing_values = data_cfg.get("missing_label_values", [])
    uncertain_labels = {
        str(value).strip()
        for value in uncertainty_cfg.get("uncertain_species_labels", [])
    }
    uncertain_patterns = [
        re.compile(pattern)
        for pattern in uncertainty_cfg.get("uncertain_species_patterns", [])
    ]
    resolved_overrides = {
        str(key).strip(): str(value).strip()
        for key, value in uncertainty_cfg.get(
            "resolved_species_label_overrides", {}
        ).items()
    }
    life_stage_overrides = {
        str(key).strip(): str(value).strip()
        for key, value in uncertainty_cfg.get("life_stage_overrides", {}).items()
    }

    raw_taxon = df["taxon_label"].astype(str).str.strip()
    for raw_label, resolved_label in resolved_overrides.items():
        df.loc[raw_taxon == raw_label, "species_label"] = resolved_label
    for raw_label, stage in life_stage_overrides.items():
        df.loc[raw_taxon == raw_label, "life_stage"] = stage

    species_text = df["species_label"].astype(str).str.strip()
    raw_taxon = df["taxon_label"].astype(str).str.strip()
    has_resolved_override = raw_taxon.isin(resolved_overrides.keys())
    uncertain_mask = raw_taxon.isin(uncertain_labels)
    for pattern in uncertain_patterns:
        uncertain_mask = uncertain_mask | species_text.apply(
            lambda value: bool(pattern.search(value))
        )
    uncertain_mask = uncertain_mask & ~has_resolved_override

    if uncertain_mask.any():
        print("\nStrict species labels masked because of taxonomic uncertainty:")
        print(df.loc[uncertain_mask, "taxon_label"].value_counts())
    df.loc[uncertain_mask, "species_label"] = pd.NA

    for column in ["genus", "species_label", "life_stage"]:
        if column in df.columns:
            missing_mask = df[column].apply(
                lambda value: is_missing_label(value, missing_values)
            )
            df.loc[missing_mask, column] = pd.NA

    df["__taxon_for_split__"] = df["species_label"].where(
        df["species_label"].notna(), df["genus"]
    )
    return df


def derive_taxonomy_and_stage(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Retain the legacy alternate taxonomy derivation helper."""
    df = df.copy()
    barcode_col = cfg.get("data", {}).get("barcode_col", "barcode")
    if barcode_col not in df.columns:
        return df
    barcode_base = df[barcode_col].astype("string").str.replace(
        r"_\d+$", "", regex=True
    )
    extracted = barcode_base.str.extract(r"^(.+)_(Adult|Juvenile)$", expand=True)
    taxon_from_barcode = extracted[0]
    stage_from_barcode = extracted[1]
    if "life_stage" not in df.columns:
        df["life_stage"] = stage_from_barcode
    else:
        df["life_stage"] = df["life_stage"].where(
            df["life_stage"].notna(), stage_from_barcode
        )
    if "species_label" not in df.columns:
        df["species_label"] = taxon_from_barcode
    else:
        df["species_label"] = df["species_label"].where(
            df["species_label"].notna(), taxon_from_barcode
        )
    if "species" not in df.columns:
        df["species"] = barcode_base
    taxon_for_genus = df["species_label"].where(
        df["species_label"].notna(), taxon_from_barcode
    )
    inferred_genus = taxon_for_genus.astype("string").str.split("_").str[0]
    if "genus" not in df.columns:
        df["genus"] = inferred_genus
    else:
        df["genus"] = df["genus"].where(df["genus"].notna(), inferred_genus)
    if cfg.get("data", {}).get("species_requires_binomial", True):
        species_as_str = df["species_label"].astype("string")
        genus_only = df["species_label"].notna() & ~species_as_str.str.contains(
            "_", regex=False, na=False
        )
        df.loc[genus_only, "species_label"] = pd.NA
    if "age" not in df.columns and "life_stage" in df.columns:
        df["age"] = df["life_stage"]
    return df
