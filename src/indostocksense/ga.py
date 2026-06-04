"""Genetic Algorithm utilities for hyperparameter optimization."""

import random
from dataclasses import dataclass


@dataclass
class SearchSpace:
    learning_rates: tuple = (1e-5, 2e-5, 3e-5, 5e-5)
    batch_sizes: tuple = (8, 16, 32)
    epochs: tuple = (1, 2, 3)
    lora_ranks: tuple = (4, 8, 16)
    lora_alphas: tuple = (8, 16, 32)
    lora_dropouts: tuple = (0.05, 0.1, 0.2)
    weight_decays: tuple = (0.0, 0.01, 0.05)


def create_individual(space: SearchSpace | None = None) -> dict:
    """Create one GA chromosome containing IndoBERT-LoRA hyperparameters."""
    space = space or SearchSpace()
    return {
        "learning_rate": random.choice(space.learning_rates),
        "batch_size": random.choice(space.batch_sizes),
        "epochs": random.choice(space.epochs),
        "lora_rank": random.choice(space.lora_ranks),
        "lora_alpha": random.choice(space.lora_alphas),
        "lora_dropout": random.choice(space.lora_dropouts),
        "weight_decay": random.choice(space.weight_decays),
    }


def create_population(size: int, space: SearchSpace | None = None) -> list[dict]:
    """Create the initial GA population."""
    return [create_individual(space) for _ in range(size)]

