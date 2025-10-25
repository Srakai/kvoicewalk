"""Genetic Algorithm for voice embedding exploration."""

import random
from typing import Any, Callable, List, Tuple
import torch
import numpy as np


class GeneticAlgorithm:
    """
    Genetic Algorithm optimizer for voice embeddings.
    Uses tournament selection, crossover, and adaptive mutation.
    """

    def __init__(
        self,
        population_size: int = 20,
        elite_size: int = 4,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.12,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        adaptive_mutation: bool = True,
    ):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.adaptive_mutation = adaptive_mutation

        self.population: List[torch.Tensor] = []
        self.fitness_scores: List[float] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.diversity_history: List[float] = []

    def initialize_population(
        self, initial_voices: List[torch.Tensor], voice_generator: Any
    ):
        """Initialize population with existing voices and variations."""
        self.population = []

        # Add initial voices (ensure CPU for GA operations)
        for voice in initial_voices[
            : min(len(initial_voices), self.population_size // 2)
        ]:
            self.population.append(voice.clone().cpu())

        # Generate variations
        while len(self.population) < self.population_size:
            base = random.choice(initial_voices)
            diversity = random.uniform(0.05, 0.2)
            new_voice = voice_generator.generate_voice(base, diversity)
            self.population.append(new_voice.cpu())

        self.fitness_scores = [0.0] * len(self.population)

    def evaluate_population(
        self, fitness_fn: Callable[[torch.Tensor], Tuple[float, dict]]
    ) -> List[Tuple[float, dict]]:
        """Evaluate fitness for all individuals in population."""
        results = []
        for i, individual in enumerate(self.population):
            fitness, details = fitness_fn(individual)
            self.fitness_scores[i] = fitness
            results.append((fitness, details))
        return results

    def selection_tournament(self) -> torch.Tensor:
        """Tournament selection - pick best from random subset."""
        tournament = random.sample(
            list(zip(self.population, self.fitness_scores)), self.tournament_size
        )
        winner = max(tournament, key=lambda x: x[1])
        return winner[0].clone().cpu()

    def crossover_blend(
        self, parent1: torch.Tensor, parent2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Blend crossover - weighted average of parents."""
        # Ensure both on CPU
        parent1 = parent1.cpu()
        parent2 = parent2.cpu()
        alpha = random.uniform(-1.0, 2.0)  # More aggressive extrapolation
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def crossover_uniform(
        self, parent1: torch.Tensor, parent2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniform crossover - randomly select from each parent."""
        # Ensure both on CPU
        parent1 = parent1.cpu()
        parent2 = parent2.cpu()
        mask = torch.rand_like(parent1) > 0.5
        child1 = torch.where(mask, parent1, parent2)
        child2 = torch.where(mask, parent2, parent1)
        return child1, child2

    def mutate(
        self, individual: torch.Tensor, std_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Apply Gaussian mutation to individual."""
        # Ensure on CPU
        individual = individual.cpu()
        std_tensor = std_tensor.cpu()

        # Adaptive mutation strength based on stagnation
        if self.adaptive_mutation and len(self.best_fitness_history) > 10:
            recent_improvement = (
                self.best_fitness_history[-1] - self.best_fitness_history[-10]
            )
            if recent_improvement < 0.01:  # Stagnation detected
                mutation_strength = self.mutation_strength * 2.0
            else:
                mutation_strength = self.mutation_strength
        else:
            mutation_strength = self.mutation_strength

        # Apply mutation to random subset of genes
        mask = torch.rand_like(individual) < self.mutation_rate
        noise = torch.randn_like(individual) * std_tensor * mutation_strength
        mutated = individual.clone()
        mutated[mask] += noise[mask]
        return mutated

    def evolve(self, voice_generator: Any) -> List[torch.Tensor]:
        """Perform one generation of evolution."""
        # Sort population by fitness
        sorted_pop = sorted(
            zip(self.population, self.fitness_scores), key=lambda x: x[1], reverse=True
        )

        # Elitism - keep best individuals (ensure CPU)
        new_population = [ind.clone().cpu() for ind, _ in sorted_pop[: self.elite_size]]

        # Track diversity
        if len(self.population) > 1:
            stacked = torch.stack([p.cpu().flatten() for p in self.population])
            diversity = torch.std(stacked, dim=0).mean().item()
            self.diversity_history.append(diversity)

        # Track best fitness
        best_fitness = sorted_pop[0][1]
        self.best_fitness_history.append(best_fitness)

        # Generate offspring
        std_tensor = voice_generator.std.cpu()  # Ensure CPU for GA operations
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.selection_tournament()
            parent2 = self.selection_tournament()

            # Crossover
            if random.random() < self.crossover_rate:
                if random.random() < 0.5:
                    child1, child2 = self.crossover_blend(parent1, parent2)
                else:
                    child1, child2 = self.crossover_uniform(parent1, parent2)
            else:
                child1, child2 = parent1.clone(), parent2.clone()

            # Mutation
            child1 = self.mutate(child1, std_tensor)
            child2 = self.mutate(child2, std_tensor)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population[: self.population_size]
        self.generation += 1

        return self.population

    def get_best(self) -> Tuple[torch.Tensor, float]:
        """Return best individual and its fitness."""
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx].clone().cpu(), self.fitness_scores[best_idx]

    def get_top_k(self, k: int) -> List[Tuple[torch.Tensor, float]]:
        """Return top k individuals and their fitness scores."""
        sorted_pop = sorted(
            zip(self.population, self.fitness_scores), key=lambda x: x[1], reverse=True
        )
        return [(ind.clone().cpu(), fit) for ind, fit in sorted_pop[:k]]

    def get_stats(self) -> dict:
        """Return statistics about current population."""
        return {
            "generation": self.generation,
            "best_fitness": max(self.fitness_scores),
            "mean_fitness": np.mean(self.fitness_scores),
            "diversity": self.diversity_history[-1] if self.diversity_history else 0.0,
            "population_size": len(self.population),
        }
