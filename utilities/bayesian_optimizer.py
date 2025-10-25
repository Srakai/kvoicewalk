"""Bayesian Optimization for voice embedding refinement."""

import torch
import numpy as np
from typing import List, Tuple, Callable, Optional
from collections import deque


class SimpleBayesianOptimizer:
    """
    Lightweight Bayesian Optimization using a simple surrogate model.
    Uses k-NN + local regression for fast surrogate modeling without heavy dependencies.
    """

    def __init__(
        self,
        embedding_dim: int,
        exploration_weight: float = 2.0,
        history_size: int = 100,
        k_neighbors: int = 5,
        local_search_radius: float = 0.1,
    ):
        self.embedding_dim = embedding_dim
        self.exploration_weight = exploration_weight
        self.history_size = history_size
        self.k_neighbors = k_neighbors
        self.local_search_radius = local_search_radius

        # Store evaluated points
        self.X_observed: deque = deque(maxlen=history_size)  # Embeddings
        self.y_observed: deque = deque(maxlen=history_size)  # Fitness scores

        self.iteration = 0
        self.best_value = -float("inf")
        self.best_point = None

    def add_observation(self, embedding: torch.Tensor, fitness: float):
        """Add an evaluated point to the history."""
        self.X_observed.append(embedding.cpu().detach().flatten())
        self.y_observed.append(fitness)

        if fitness > self.best_value:
            self.best_value = fitness
            self.best_point = embedding.clone()

        self.iteration += 1

    def predict_fitness(self, candidate: torch.Tensor) -> Tuple[float, float]:
        """
        Predict fitness using k-NN with distance weighting.
        Returns: (mean prediction, uncertainty estimate)
        """
        if len(self.X_observed) == 0:
            return 0.0, 1.0

        candidate_flat = candidate.cpu().detach().flatten()

        # Calculate distances to all observed points
        X_array = torch.stack(list(self.X_observed))
        distances = torch.norm(X_array - candidate_flat, dim=1)

        # Get k nearest neighbors
        k = min(self.k_neighbors, len(self.X_observed))
        nearest_indices = torch.argsort(distances)[:k]
        nearest_distances = distances[nearest_indices]
        nearest_values = torch.tensor([self.y_observed[i] for i in nearest_indices])

        # Distance-weighted average
        weights = 1.0 / (nearest_distances + 1e-6)
        weights = weights / weights.sum()

        mean_prediction = (weights * nearest_values).sum().item()

        # Uncertainty: average distance to neighbors (normalized)
        uncertainty = nearest_distances.mean().item()

        return mean_prediction, uncertainty

    def acquisition_ucb(self, candidate: torch.Tensor) -> float:
        """Upper Confidence Bound acquisition function."""
        mean, uncertainty = self.predict_fitness(candidate)
        return mean + self.exploration_weight * uncertainty

    def propose_candidate(
        self, base_embedding: torch.Tensor, voice_generator: any, n_candidates: int = 50
    ) -> torch.Tensor:
        """
        Propose next point to evaluate using acquisition function.
        Generates candidates around base_embedding and selects best by UCB.
        """
        candidates = []
        acquisition_scores = []

        # Generate candidates with varying diversity
        for _ in range(n_candidates):
            diversity = np.random.uniform(0.01, self.local_search_radius)
            candidate = voice_generator.generate_voice(base_embedding, diversity)
            candidates.append(candidate)

            # Evaluate acquisition function
            acq_score = self.acquisition_ucb(candidate)
            acquisition_scores.append(acq_score)

        # Return candidate with highest acquisition score
        best_idx = np.argmax(acquisition_scores)
        return candidates[best_idx]

    def optimize_local(
        self,
        starting_point: torch.Tensor,
        voice_generator: any,
        fitness_fn: Callable[[torch.Tensor], Tuple[float, dict]],
        n_iterations: int = 20,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, float, dict]:
        """
        Perform local Bayesian optimization around a starting point.
        Returns: (best_embedding, best_fitness, best_details)
        """
        current_best = starting_point.clone()
        current_fitness, current_details = fitness_fn(current_best)
        self.add_observation(current_best, current_fitness)

        for i in range(n_iterations):
            # Propose next candidate
            candidate = self.propose_candidate(current_best, voice_generator)

            # Evaluate
            fitness, details = fitness_fn(candidate)
            self.add_observation(candidate, fitness)

            # Update best
            if fitness > current_fitness:
                current_best = candidate.clone()
                current_fitness = fitness
                current_details = details

                if verbose:
                    print(f"BO Iter {i}: New best fitness = {fitness:.4f}")

        return current_best, current_fitness, current_details

    def refine_population(
        self,
        population: List[Tuple[torch.Tensor, float]],
        voice_generator: any,
        fitness_fn: Callable[[torch.Tensor], Tuple[float, dict]],
        n_iterations_per_individual: int = 10,
        verbose: bool = False,
    ) -> List[Tuple[torch.Tensor, float, dict]]:
        """
        Refine each individual in a population using local BO.
        Returns: List of (refined_embedding, fitness, details)
        """
        refined = []

        for idx, (embedding, fitness) in enumerate(population):
            if verbose:
                print(
                    f"\nRefining individual {idx + 1}/{len(population)} (fitness={fitness:.4f})"
                )

            # Add to history
            self.add_observation(embedding, fitness)

            # Local optimization
            refined_emb, refined_fit, refined_details = self.optimize_local(
                embedding,
                voice_generator,
                fitness_fn,
                n_iterations=n_iterations_per_individual,
                verbose=verbose,
            )

            refined.append((refined_emb, refined_fit, refined_details))

        return refined

    def get_stats(self) -> dict:
        """Return statistics about the optimization."""
        return {
            "iteration": self.iteration,
            "n_observations": len(self.X_observed),
            "best_value": self.best_value,
            "mean_value": np.mean(list(self.y_observed)) if self.y_observed else 0.0,
        }
