"""Training-only samplers for biological combination balancing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import pandas as pd
import torch
from torch.utils.data import Sampler
from torch.utils.data import BatchSampler


class JointSpeciesStageSampler(Sampler[int]):
    """Balance observed species-stage combinations via individual-first draws.

    A draw selects a combination uniformly, an individual uniformly within
    that combination, and finally one image uniformly for that individual.
    Repeated images from the same worm therefore do not create extra sampling
    mass at either the combination or individual selection stage.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        species_col: str,
        stage_col: str,
        group_col: str,
        replacement: bool = True,
        samples_per_epoch: int | None = None,
        seed: int = 0,
    ) -> None:
        if any(
            column not in frame.columns
            for column in (species_col, stage_col, group_col)
        ):
            raise ValueError(
                "Joint species-stage sampling requires species, stage, and "
                "individual columns in the training dataframe."
            )
        self.frame = frame.reset_index(drop=True)
        self.replacement = bool(replacement)
        self.num_samples = (
            len(self.frame)
            if samples_per_epoch is None
            else int(samples_per_epoch)
        )
        if self.num_samples <= 0:
            raise ValueError("data.sampler.samples_per_epoch must be positive")
        if not self.replacement and self.num_samples > len(self.frame):
            raise ValueError(
                "Sampling without replacement cannot request more images than "
                "the training dataset contains."
            )
        self.seed = int(seed)
        self.epoch = 0

        grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, row in self.frame.iterrows():
            if pd.isna(row[species_col]) or pd.isna(row[stage_col]):
                continue
            combination = (str(row[species_col]), str(row[stage_col]))
            grouped[combination][str(row[group_col])].append(int(index))
        if not grouped:
            raise ValueError(
                "No observed, fully labelled species-stage combinations are "
                "available in the training split."
            )
        self._groups = {
            combination: {
                individual: tuple(indices)
                for individual, indices in individuals.items()
            }
            for combination, individuals in grouped.items()
        }
        self._combinations = tuple(sorted(self._groups))
        probability = 1.0 / len(self._combinations)
        self.summary = pd.DataFrame([
            {
                "species": species,
                "developmental_stage": stage,
                "individuals": len(self._groups[(species, stage)]),
                "images": sum(
                    len(indices)
                    for indices in self._groups[(species, stage)].values()
                ),
                "effective_combination_probability": probability,
            }
            for species, stage in self._combinations
        ])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.replacement:
            for _ in range(self.num_samples):
                combination = self._combinations[
                    int(torch.randint(
                        len(self._combinations), (1,), generator=generator
                    ).item())
                ]
                individuals = tuple(sorted(self._groups[combination]))
                individual = individuals[
                    int(torch.randint(
                        len(individuals), (1,), generator=generator
                    ).item())
                ]
                images = self._groups[combination][individual]
                yield images[
                    int(torch.randint(
                        len(images), (1,), generator=generator
                    ).item())
                ]
            return

        # Without replacement, preserve individual-level weighting by assigning
        # each image the inverse of its combination and individual image count.
        weights = torch.zeros(len(self.frame), dtype=torch.float64)
        for combination in self._combinations:
            individuals = self._groups[combination]
            for indices in individuals.values():
                image_probability = (
                    1.0
                    / len(self._combinations)
                    / len(individuals)
                    / len(indices)
                )
                weights[list(indices)] = image_probability
        selected = torch.multinomial(
            weights,
            self.num_samples,
            replacement=False,
            generator=generator,
        )
        yield from (int(index) for index in selected.tolist())


class CrossSpeciesStageContrastiveBatchSampler(BatchSampler):
    """Construct stage-balanced batches with cross-species positive pairs."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        species_col: str,
        stage_col: str,
        group_col: str,
        species_per_stage: int = 3,
        individuals_per_species_stage: int = 2,
        images_per_individual: int = 1,
        replacement: bool = True,
        samples_per_epoch: int | None = None,
        seed: int = 0,
        individual_dataset: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.species_per_stage = int(species_per_stage)
        self.individuals_per_species_stage = int(
            individuals_per_species_stage
        )
        self.images_per_individual = int(images_per_individual)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        self.individual_dataset = bool(individual_dataset)
        if min(
            self.species_per_stage,
            self.individuals_per_species_stage,
            self.images_per_individual,
        ) <= 0:
            raise ValueError("Cross-species sampler counts must be positive")

        nested: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for image_index, row in self.frame.iterrows():
            if any(pd.isna(row[column]) for column in (stage_col, species_col, group_col)):
                continue
            nested[str(row[stage_col])][str(row[species_col])][
                str(row[group_col])
            ].append(int(image_index))
        self.groups = {
            stage: {
                species: dict(individuals)
                for species, individuals in species_groups.items()
            }
            for stage, species_groups in nested.items()
            if len(species_groups) >= 2
        }
        if not self.groups:
            raise ValueError(
                "Cross-species stage batches require at least one stage with "
                "two observed species"
            )
        self._stages = tuple(sorted(self.groups))
        self.batch_size = sum(
            min(self.species_per_stage, len(self.groups[stage]))
            * self.individuals_per_species_stage
            * (1 if self.individual_dataset else self.images_per_individual)
            for stage in self._stages
        )
        requested = len(self.frame) if samples_per_epoch is None else int(samples_per_epoch)
        self.num_batches = max(1, (requested + self.batch_size - 1) // self.batch_size)
        all_barcodes = sorted(
            {str(value) for value in self.frame[group_col].dropna().unique()}
        )
        self._individual_index = {
            barcode: index for index, barcode in enumerate(all_barcodes)
        }
        self.summary = pd.DataFrame([
            {
                "developmental_stage": stage,
                "species": species,
                "individuals": len(individuals),
                "images": sum(len(indices) for indices in individuals.values()),
            }
            for stage, species_groups in self.groups.items()
            for species, individuals in species_groups.items()
        ])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    @staticmethod
    def _draw(values: list, count: int, generator: torch.Generator, replacement: bool) -> list:
        if not values:
            return []
        if replacement and len(values) < count:
            indices = torch.randint(len(values), (count,), generator=generator)
        else:
            indices = torch.randperm(len(values), generator=generator)[:min(count, len(values))]
        return [values[int(index)] for index in indices.tolist()]

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        for _ in range(self.num_batches):
            batch: list[int] = []
            for stage in self._stages:
                species_values = sorted(self.groups[stage])
                selected_species = self._draw(
                    species_values,
                    self.species_per_stage,
                    generator,
                    self.replacement,
                )
                for species in selected_species:
                    individuals = sorted(self.groups[stage][species])
                    selected_individuals = self._draw(
                        individuals,
                        self.individuals_per_species_stage,
                        generator,
                        self.replacement,
                    )
                    for individual in selected_individuals:
                        if self.individual_dataset:
                            batch.append(self._individual_index[individual])
                        else:
                            batch.extend(self._draw(
                                self.groups[stage][species][individual],
                                self.images_per_individual,
                                generator,
                                self.replacement,
                            ))
            if batch:
                yield batch


__all__ = [
    "CrossSpeciesStageContrastiveBatchSampler",
    "JointSpeciesStageSampler",
]
