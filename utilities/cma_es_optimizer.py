"""CMA-ES Optimizer for voice embedding optimization."""

import datetime
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import cma
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm


class CMAESOptimizer:
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimizer.
    State-of-the-art algorithm for continuous black-box optimization.
    """

    def __init__(
        self,
        population_size: int = None,  # Auto-determined by CMA-ES if None
        sigma0: float = 0.3,  # Initial step size
        checkpoint_interval: int = 5,
    ):
        self.population_size = population_size
        self.sigma0 = sigma0
        self.checkpoint_interval = checkpoint_interval

        self.es = None  # CMA-ES instance
        self.generation = 0
        self.total_evaluations = 0
        self.best_ever_fitness = -float("inf")
        self.best_ever_embedding = None
        self.best_ever_details = None

        # Statistics
        self.fitness_history = []
        self.evaluation_history = []
        self.checkpoint_dir = None

    def initialize(
        self, initial_voice: torch.Tensor, voice_generator: Any, verbose: bool = True
    ):
        """
        Initialize CMA-ES optimizer.

        Args:
            initial_voice: Starting voice embedding (best from initial population)
            voice_generator: Voice generator with std for normalization
            verbose: Print initialization info
        """
        # Convert initial voice to numpy array (flattened)
        x0 = initial_voice.cpu().detach().flatten().numpy()
        dim = len(x0)

        if verbose:
            print(
                f"\nInitializing sep-CMA-ES Optimizer (diagonal-only, memory-efficient)..."
            )
            print(f"  Embedding Dimension: {dim}")
            print(
                f"  This will use ~{dim * 8 / 1e6:.1f} MB instead of {dim * dim * 8 / 1e9:.1f} GB"
            )

        # Initialize CMA-ES with diagonal-only covariance (sep-CMA-ES)
        opts = {
            "verb_disp": 0,  # Silence default output
            "verbose": -1,
            "CMA_diagonal": True,  # Use separable CMA-ES (diagonal covariance only)
        }

        if self.population_size is not None:
            opts["popsize"] = self.population_size

        self.es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)

        if verbose:
            print(f"  ✓ Initialized successfully")
            print(f"  Population Size: {self.es.popsize}")
            print(f"  Initial Sigma: {self.sigma0}")

    def evaluate_fitness(
        self, embedding: torch.Tensor, scorer: Callable
    ) -> Tuple[float, dict]:
        """Wrapper for fitness evaluation with tracking."""
        self.total_evaluations += 1

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

    def save_checkpoint(self, checkpoint_path: Path, verbose: bool = True):
        """Save CMA-ES state for resuming."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            "generation": self.generation,
            "total_evaluations": self.total_evaluations,
            "best_ever_fitness": self.best_ever_fitness,
            "best_ever_details": self.best_ever_details,
            "fitness_history": self.fitness_history,
            "evaluation_history": self.evaluation_history,
            "cma_state": self.es.pickle_dumps(),  # Serialize CMA-ES state
        }

        torch.save(checkpoint_data, checkpoint_path)

        # Save best embedding
        if self.best_ever_embedding is not None:
            best_path = checkpoint_path.with_suffix(".best.pt")
            torch.save(self.best_ever_embedding, best_path)

        if verbose:
            print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path, verbose: bool = True):
        """Load CMA-ES state from checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_data = torch.load(checkpoint_path, weights_only=False)

        self.generation = checkpoint_data["generation"]
        self.total_evaluations = checkpoint_data["total_evaluations"]
        self.best_ever_fitness = checkpoint_data["best_ever_fitness"]
        self.best_ever_details = checkpoint_data["best_ever_details"]
        self.fitness_history = checkpoint_data["fitness_history"]
        self.evaluation_history = checkpoint_data["evaluation_history"]

        # Restore CMA-ES state
        self.es = cma.CMAEvolutionStrategy.pickle_loads(checkpoint_data["cma_state"])

        # Load best embedding
        best_path = checkpoint_path.with_suffix(".best.pt")
        if best_path.exists():
            self.best_ever_embedding = torch.load(best_path, weights_only=False)

        if verbose:
            print(f"Checkpoint loaded: {checkpoint_path}")
            print(f"  Resuming from generation {self.generation}")
            print(f"  Total evaluations: {self.total_evaluations}")
            print(f"  Best fitness: {self.best_ever_fitness:.4f}")

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
        resume_from: Optional[Path] = None,
        verbose: bool = True,
    ):
        """
        Main CMA-ES optimization loop.

        Args:
            voice_generator: Voice generation utility
            scorer: Fitness scoring function
            n_generations: Number of generations to run
            results_dir: Directory to save results
            output_name: Base name for output files
            target_audio_stem: Target audio filename stem
            speech_generator: Speech generation utility
            target_text: Text for speech generation
            checkpoint_dir: Directory for checkpoint files
            resume_from: Path to checkpoint file to resume from
            verbose: Print detailed progress
        """
        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = results_dir / "checkpoints"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Resume from checkpoint if provided
        start_generation = 0
        if resume_from:
            self.load_checkpoint(resume_from, verbose=verbose)
            start_generation = self.generation
            if verbose:
                print(f"Resuming from generation {start_generation}")

        progress_bar = tqdm(
            range(start_generation, n_generations),
            desc="CMA-ES Progress",
            initial=start_generation,
            total=n_generations,
        )

        for gen in progress_bar:
            # Ask CMA-ES for new candidate solutions
            solutions = self.es.ask()

            # Evaluate each candidate
            fitness_list = []
            for solution in solutions:
                # Convert numpy array to torch tensor
                embedding = torch.tensor(solution, dtype=torch.float32).reshape(
                    voice_generator.voices[0].shape
                )

                # Evaluate fitness
                fitness, details = self.evaluate_fitness(embedding, scorer)
                fitness_list.append(-fitness)  # CMA-ES minimizes, we want to maximize

            # Tell CMA-ES the fitness values
            self.es.tell(solutions, fitness_list)

            # Update generation counter
            self.generation += 1

            # Get statistics
            mean_fitness = -np.mean(fitness_list)
            best_fitness = -np.min(fitness_list)

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "Best": f"{best_fitness:.3f}",
                    "BestEver": f"{self.best_ever_fitness:.3f}",
                    "Mean": f"{mean_fitness:.3f}",
                    "Sigma": f"{self.es.sigma:.4f}",
                    "Evals": self.total_evaluations,
                }
            )

            # Log verbose output
            if verbose and gen % 5 == 0:
                progress_bar.write(
                    f"Gen {gen}: "
                    f"Best={best_fitness:.4f}, "
                    f"Mean={mean_fitness:.4f}, "
                    f"Sigma={self.es.sigma:.4f}"
                )

            # Save best individual if new best ever
            if best_fitness == self.best_ever_fitness:
                best_details = self.best_ever_details

                # Save voice tensor
                best_voice_path = (
                    f"{results_dir}/{output_name}_gen{gen}_"
                    f'{best_details["score"]:.2f}_'
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

            # Save checkpoint
            if gen % self.checkpoint_interval == 0 and gen > 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_gen{gen}.pt"
                self.save_checkpoint(checkpoint_path, verbose=False)
                if verbose:
                    progress_bar.write(f"  -> Checkpoint saved: {checkpoint_path.name}")

            # Check for convergence (optional early stopping)
            if self.es.stop():
                progress_bar.write("\nCMA-ES convergence criteria met. Stopping early.")
                break

        # Save final checkpoint
        final_checkpoint = (
            self.checkpoint_dir / f"checkpoint_final_gen{self.generation}.pt"
        )
        self.save_checkpoint(final_checkpoint, verbose=True)

        # Final summary
        print(f"\n{'='*80}")
        print(f"CMA-ES Final Results")
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
