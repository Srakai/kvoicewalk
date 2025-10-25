"""Hybrid meta-learner combining Genetic Algorithm and Bayesian Optimization."""

import torch
from typing import Any, Callable, Tuple, List, Optional
import datetime
from pathlib import Path
import os
import json
import soundfile as sf
from tqdm import tqdm

from utilities.genetic_algorithm import GeneticAlgorithm
from utilities.bayesian_optimizer import SimpleBayesianOptimizer


class HybridMetaLearner:
    """
    Hybrid approach combining GA for exploration and BO for exploitation.

    Strategy:
    1. Use GA to explore the embedding space and find promising regions
    2. Every N generations, use BO to refine the best individuals
    3. Inject refined individuals back into GA population
    """

    def __init__(
        self,
        # GA parameters
        ga_population_size: int = 20,
        ga_elite_size: int = 4,
        ga_mutation_rate: float = 0.1,
        ga_crossover_rate: float = 0.7,
        # BO parameters
        bo_exploration_weight: float = 2.0,
        bo_local_search_radius: float = 0.1,
        # Hybrid parameters
        bo_refinement_interval: int = 5,  # Refine every N generations
        bo_refinement_candidates: int = 3,  # Top K to refine
        bo_iterations_per_candidate: int = 10,
        # Diversity injection
        diversity_injection_interval: int = 10,
        diversity_injection_count: int = 2,
    ):
        self.ga = GeneticAlgorithm(
            population_size=ga_population_size,
            elite_size=ga_elite_size,
            mutation_rate=ga_mutation_rate,
            crossover_rate=ga_crossover_rate,
            adaptive_mutation=True,
        )

        self.bo = None  # Will be initialized when we know embedding dimension
        self.bo_exploration_weight = bo_exploration_weight
        self.bo_local_search_radius = bo_local_search_radius

        self.bo_refinement_interval = bo_refinement_interval
        self.bo_refinement_candidates = bo_refinement_candidates
        self.bo_iterations_per_candidate = bo_iterations_per_candidate

        self.diversity_injection_interval = diversity_injection_interval
        self.diversity_injection_count = diversity_injection_count

        self.generation = 0
        self.total_evaluations = 0
        self.best_ever_fitness = -float("inf")
        self.best_ever_embedding = None
        self.best_ever_details = None

        # Statistics tracking
        self.fitness_history = []
        self.evaluation_history = []

        # Checkpoint management
        self.checkpoint_dir = None
        self.checkpoint_interval = 5  # Save every N generations

    def initialize(self, initial_voices: List[torch.Tensor], voice_generator: Any):
        """Initialize the meta-learner with starting population."""
        # Initialize GA population
        self.ga.initialize_population(initial_voices, voice_generator)

        # Initialize BO with embedding dimension
        embedding_dim = initial_voices[0].numel()
        self.bo = SimpleBayesianOptimizer(
            embedding_dim=embedding_dim,
            exploration_weight=self.bo_exploration_weight,
            local_search_radius=self.bo_local_search_radius,
        )

        print(f"Initialized Hybrid Meta-Learner:")
        print(f"  GA Population: {self.ga.population_size}")
        print(f"  Embedding Dim: {embedding_dim}")
        print(f"  BO Refinement every {self.bo_refinement_interval} generations")

    def save_checkpoint(self, checkpoint_path: Path, verbose: bool = True):
        """Save complete state for resuming optimization."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Save main checkpoint file
        checkpoint_data = {
            "generation": self.generation,
            "total_evaluations": self.total_evaluations,
            "best_ever_fitness": self.best_ever_fitness,
            "best_ever_details": self.best_ever_details,
            "fitness_history": self.fitness_history,
            "evaluation_history": self.evaluation_history,
            # GA state
            "ga_fitness_scores": self.ga.fitness_scores,
            "ga_generation": self.ga.generation,
            "ga_best_fitness_history": self.ga.best_fitness_history,
            "ga_diversity_history": self.ga.diversity_history,
            # BO state
            "bo_iteration": self.bo.iteration if self.bo else 0,
            "bo_best_value": self.bo.best_value if self.bo else -float("inf"),
            "bo_y_observed": list(self.bo.y_observed) if self.bo else [],
        }

        torch.save(checkpoint_data, checkpoint_path)

        # Save population tensors separately (can be large)
        population_path = checkpoint_path.with_suffix(".population.pt")
        torch.save(self.ga.population, population_path)

        # Save best embedding
        if self.best_ever_embedding is not None:
            best_path = checkpoint_path.with_suffix(".best.pt")
            torch.save(self.best_ever_embedding, best_path)

        # Save BO observed points
        if self.bo and len(self.bo.X_observed) > 0:
            bo_path = checkpoint_path.with_suffix(".bo_observed.pt")
            torch.save(list(self.bo.X_observed), bo_path)

        if verbose:
            print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path, verbose: bool = True):
        """Load complete state from checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load main checkpoint
        checkpoint_data = torch.load(checkpoint_path)

        self.generation = checkpoint_data["generation"]
        self.total_evaluations = checkpoint_data["total_evaluations"]
        self.best_ever_fitness = checkpoint_data["best_ever_fitness"]
        self.best_ever_details = checkpoint_data["best_ever_details"]
        self.fitness_history = checkpoint_data["fitness_history"]
        self.evaluation_history = checkpoint_data["evaluation_history"]

        # Restore GA state
        self.ga.fitness_scores = checkpoint_data["ga_fitness_scores"]
        self.ga.generation = checkpoint_data["ga_generation"]
        self.ga.best_fitness_history = checkpoint_data["ga_best_fitness_history"]
        self.ga.diversity_history = checkpoint_data["ga_diversity_history"]

        # Load population
        population_path = checkpoint_path.with_suffix(".population.pt")
        if population_path.exists():
            self.ga.population = torch.load(population_path)

        # Load best embedding
        best_path = checkpoint_path.with_suffix(".best.pt")
        if best_path.exists():
            self.best_ever_embedding = torch.load(best_path)

        # Restore BO state
        if self.bo:
            self.bo.iteration = checkpoint_data["bo_iteration"]
            self.bo.best_value = checkpoint_data["bo_best_value"]
            self.bo.y_observed = checkpoint_data["bo_y_observed"]

            # Load BO observed points
            bo_path = checkpoint_path.with_suffix(".bo_observed.pt")
            if bo_path.exists():
                self.bo.X_observed = torch.load(bo_path)

        if verbose:
            print(f"Checkpoint loaded: {checkpoint_path}")
            print(f"  Resuming from generation {self.generation}")
            print(f"  Total evaluations: {self.total_evaluations}")
            print(f"  Best fitness: {self.best_ever_fitness:.4f}")

    def evaluate_fitness(self, embedding: torch.Tensor, scorer) -> Tuple[float, dict]:
        """Wrapper for fitness evaluation that tracks statistics."""
        self.total_evaluations += 1

        # Use existing scoring logic
        best_results = {
            "score": 0.0,
            "target_similarity": 0.0,
            "self_similarity": 0.0,
            "feature_similarity": 0.0,
        }
        results, _ = scorer(best_results, embedding, min_similarity=-100.0)

        fitness = results.get("score", 0.0)

        # Update best ever
        if fitness > self.best_ever_fitness:
            self.best_ever_fitness = fitness
            self.best_ever_embedding = embedding.clone().cpu()
            self.best_ever_details = results

        self.fitness_history.append(fitness)
        self.evaluation_history.append(self.total_evaluations)

        return fitness, results

    def inject_diversity(self, voice_generator: Any):
        """Inject diverse individuals to prevent premature convergence."""
        # Get statistics of current population (ensure CPU)
        stacked = torch.stack([p.cpu() for p in self.ga.population])
        mean = stacked.mean(dim=0)

        # Generate diverse individuals far from current mean
        for _ in range(self.diversity_injection_count):
            diversity = 0.3  # High diversity
            new_individual = voice_generator.generate_voice(mean, diversity)

            # Replace random non-elite individual
            replace_idx = (
                self.ga.elite_size
                + torch.randint(
                    0, self.ga.population_size - self.ga.elite_size, (1,)
                ).item()
            )
            self.ga.population[replace_idx] = new_individual.cpu()
            self.ga.fitness_scores[replace_idx] = (
                0.0  # Will be evaluated next generation
            )

    def run_generation(
        self,
        voice_generator: Any,
        scorer: Callable,
        progress_bar: tqdm,
        verbose: bool = False,
    ) -> dict:
        """Run one generation of the hybrid algorithm."""
        # Evaluate current population
        eval_results = []
        for individual in self.ga.population:
            fitness, details = self.evaluate_fitness(individual, scorer)
            eval_results.append((fitness, details))

        # Update GA fitness scores
        for i, (fitness, _) in enumerate(eval_results):
            self.ga.fitness_scores[i] = fitness
            # Add to BO history
            self.bo.add_observation(self.ga.population[i], fitness)

        # Get current best
        best_individual, best_fitness = self.ga.get_best()
        ga_stats = self.ga.get_stats()

        if verbose:
            progress_bar.write(
                f"Gen {self.generation}: "
                f"Best={best_fitness:.4f}, "
                f"Mean={ga_stats['mean_fitness']:.4f}, "
                f"Diversity={ga_stats['diversity']:.4f}"
            )

        # Bayesian Optimization refinement phase
        if self.generation > 0 and self.generation % self.bo_refinement_interval == 0:
            if verbose:
                progress_bar.write(f"  -> Starting BO refinement phase...")

            # Get top K individuals
            top_k = self.ga.get_top_k(self.bo_refinement_candidates)

            # Refine each with BO
            refined_population = []
            for idx, (embedding, fitness) in enumerate(top_k):
                refined_emb, refined_fit, refined_details = self.bo.optimize_local(
                    starting_point=embedding,
                    voice_generator=voice_generator,
                    fitness_fn=lambda emb: self.evaluate_fitness(emb, scorer),
                    n_iterations=self.bo_iterations_per_candidate,
                    verbose=False,
                )

                if verbose:
                    progress_bar.write(
                        f"  -> Refined {idx+1}/{len(top_k)}: "
                        f"{fitness:.4f} -> {refined_fit:.4f}"
                    )

                # Inject refined individual back into population (replace worst)
                worst_idx = self.ga.fitness_scores.index(min(self.ga.fitness_scores))
                self.ga.population[worst_idx] = refined_emb.cpu()
                self.ga.fitness_scores[worst_idx] = refined_fit

        # Diversity injection
        if (
            self.generation > 0
            and self.generation % self.diversity_injection_interval == 0
        ):
            if verbose:
                progress_bar.write(f"  -> Injecting diversity...")
            self.inject_diversity(voice_generator)

        # Evolve GA population
        self.ga.evolve(voice_generator)
        self.generation += 1

        # Auto-save checkpoint
        if self.checkpoint_dir and self.generation % self.checkpoint_interval == 0:
            checkpoint_path = (
                self.checkpoint_dir / f"checkpoint_gen{self.generation}.pt"
            )
            self.save_checkpoint(checkpoint_path, verbose=False)
            if verbose:
                progress_bar.write(f"  -> Checkpoint saved: {checkpoint_path.name}")

        # Return statistics
        return {
            "generation": self.generation,
            "best_fitness": best_fitness,
            "best_ever_fitness": self.best_ever_fitness,
            "mean_fitness": ga_stats["mean_fitness"],
            "diversity": ga_stats["diversity"],
            "total_evaluations": self.total_evaluations,
            "best_details": eval_results[self.ga.fitness_scores.index(best_fitness)][1],
        }

    def optimize(
        self,
        voice_generator: Any,
        scorer: Callable,
        n_generations: int,
        results_dir: Path,
        output_name: str,
        target_audio_stem: str,
        speech_generator: Any,
        target_text: str,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: int = 5,
        resume_from: Optional[Path] = None,
        verbose: bool = True,
    ):
        """
        Main optimization loop.

        Args:
            voice_generator: Voice generation utility
            scorer: Fitness scoring function
            n_generations: Number of generations to run
            results_dir: Directory to save results
            output_name: Base name for output files
            target_audio_stem: Target audio filename stem
            speech_generator: Speech generation utility
            target_text: Text for speech generation
            checkpoint_dir: Directory for checkpoint files (default: results_dir/checkpoints)
            checkpoint_interval: Save checkpoint every N generations
            resume_from: Path to checkpoint file to resume from
            verbose: Print detailed progress
        """
        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = results_dir / "checkpoints"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval

        # Resume from checkpoint if provided
        start_generation = 0
        if resume_from:
            self.load_checkpoint(resume_from, verbose=verbose)
            start_generation = self.generation
            if verbose:
                print(f"Resuming from generation {start_generation}")

        progress_bar = tqdm(
            range(start_generation, n_generations),
            desc="Hybrid Meta-Learner Progress",
            initial=start_generation,
            total=n_generations,
        )

        for gen in progress_bar:
            stats = self.run_generation(
                voice_generator=voice_generator,
                scorer=scorer,
                progress_bar=progress_bar,
                verbose=verbose,
            )

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "Best": f"{stats['best_fitness']:.3f}",
                    "BestEver": f"{stats['best_ever_fitness']:.3f}",
                    "Mean": f"{stats['mean_fitness']:.3f}",
                    "Evals": stats["total_evaluations"],
                }
            )

            # Save best individual from this generation
            if stats["best_fitness"] == stats["best_ever_fitness"]:
                best_details = stats["best_details"]

                # Save voice tensor
                best_voice_path = (
                    f"{results_dir}/{output_name}_gen{gen}_"
                    f'{stats["best_fitness"]:.2f}_'
                    f'{best_details["target_similarity"]:.2f}_'
                    f"{target_audio_stem}.pt"
                )
                torch.save(self.best_ever_embedding, best_voice_path)

                # Generate and save audio
                audio = speech_generator.generate_audio(
                    target_text, self.best_ever_embedding
                )
                audio_path = best_voice_path.replace(".pt", ".wav")
                sf.write(audio_path, audio.cpu().numpy(), 24000)

                progress_bar.write(
                    f"\nGen {gen}: NEW BEST! "
                    f'Target Sim={best_details["target_similarity"]:.3f}, '
                    f'Self Sim={best_details["self_similarity"]:.3f}, '
                    f'Feature Sim={best_details["feature_similarity"]:.3f}, '
                    f'Score={best_details["score"]:.2f}'
                )

        # Save final checkpoint
        final_checkpoint = (
            self.checkpoint_dir / f"checkpoint_final_gen{self.generation}.pt"
        )
        self.save_checkpoint(final_checkpoint, verbose=True)

        # Final summary
        print(f"\n{'='*80}")
        print(f"Hybrid Meta-Learner Final Results")
        print(f"{'='*80}")
        print(f"Total Generations: {self.generation}")
        print(f"Total Evaluations: {self.total_evaluations}")
        print(f"Best Fitness: {self.best_ever_fitness:.4f}")
        if self.best_ever_details:
            print(
                f"  Target Similarity: {self.best_ever_details['target_similarity']:.4f}"
            )
            print(f"  Self Similarity: {self.best_ever_details['self_similarity']:.4f}")
            print(
                f"  Feature Similarity: {self.best_ever_details['feature_similarity']:.4f}"
            )
        print(f"Results saved to: {results_dir}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"{'='*80}\n")

        return self.best_ever_embedding, self.best_ever_fitness, self.best_ever_details
