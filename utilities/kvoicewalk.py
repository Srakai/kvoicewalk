import datetime
import gc
import os
import random
import traceback
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from tqdm import tqdm

from utilities.fitness_scorer import FitnessScorer
from utilities.initial_selector import InitialSelector
from utilities.speech_generator import SpeechGenerator
from utilities.voice_generator import VoiceGenerator
from utilities.hybrid_meta_learner import HybridMetaLearner

ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "out"


class KVoiceWalk:
    def __init__(
        self,
        target_audio: Path,
        target_text: str,
        other_text: str,
        voice_folder: str,
        interpolate_start: bool,
        population_limit: int,
        starting_voice: str,
        output_name: str,
    ) -> None:
        try:
            self.memcache_clear_freq = 100
            self.target_audio = target_audio
            self.target_text = target_text
            self.other_text = other_text

            # Determine device once
            from utilities.util import get_device

            self.device = get_device()

            self.speech_generator = SpeechGenerator(
                target_text=target_text,
                other_text=other_text,
                device=self.device,
            )
            self.fitness_scorer = FitnessScorer(
                str(target_audio),
                speech_generator=self.speech_generator,
            )
            # If starting_voice is provided, skip the expensive voice evaluation
            if starting_voice:
                from utilities.pytorch_sanitizer import load_voice_safely

                starting_voice_tensor = load_voice_safely(starting_voice)
                # Use only the starting voice, no need to evaluate all voices
                voices = [starting_voice_tensor]
                self.voice_generator = VoiceGenerator(voices, starting_voice)
                self.starting_voice = self.voice_generator.starting_voice
            else:
                # No starting voice provided, evaluate all voices to find best ones
                try:
                    self.initial_selector = InitialSelector(
                        self.fitness_scorer,
                        self.speech_generator,
                        str(target_audio),
                        target_text,
                        other_text,
                        voice_folder=voice_folder,
                    )
                except Exception as e:
                    raise Exception(f"Error: {e}")

                if interpolate_start:
                    voices = self.initial_selector.interpolate_search(population_limit)
                else:
                    voices = self.initial_selector.top_performer_start(population_limit)
                self.voice_generator = VoiceGenerator(voices, starting_voice)
                self.starting_voice = self.voice_generator.starting_voice
            self.best_voice = None  # Initialize best_voice as instance attribute
            self.output_name = output_name

        except Exception as e:
            print("FULL TRACEBACK:")
            traceback.print_exc()
            print(f"\nERROR: {e}")
            print(f"ERROR TYPE: {type(e)}")
            print(f"Error initializing KVoicewalk: {e}")
            raise SystemExit

    def random_walk(self, step_limit: int):
        # Score Initial Voice
        t = tqdm()
        self.best_voice = self.starting_voice
        best_results = {
            "score": 0.0,
            "target_similarity": 0.0,
            "self_similarity": 0.0,
            "feature_similarity": 0.0,
        }
        best_results, tps_report = self.fitness_scorer.score_voice(
            best_results=best_results,
            voice_tensor=self.starting_voice,
            min_similarity=-100.0,
        )
        t.write(
            f'Target Sim:{best_results["target_similarity"]:.3f}, Self Sim:{best_results["self_similarity"]:.3f}, Feature Sim:{best_results["feature_similarity"]:.2f}, Score:{best_results["score"]:.2f}'
        )

        # Create Results Directory
        now = datetime.datetime.now()
        results_dir = Path(
            OUT_DIR
            / f'{self.output_name}_{self.target_audio.stem}_{now.strftime("%Y%m%d_%H%M%S")}'
        )
        os.makedirs(results_dir, exist_ok=True)

        # Random Walk Loop
        progress_bar = tqdm(range(step_limit), desc="KVoiceWalk Progress")
        for i in progress_bar:
            # TODO: Expose to CLI
            diversity = random.uniform(0.01, 0.15)
            voice = self.voice_generator.generate_voice(self.best_voice, diversity)

            # Early function return saves audio generation compute
            min_similarity = best_results["target_similarity"] * 0.98
            voice_results, tps_report = self.fitness_scorer.score_voice(
                best_results, voice, min_similarity
            )

            # Set new winner if score is better
            if voice_results["score"] > best_results["score"]:
                best_results = voice_results
                # Move to CPU and clean up old best_voice
                old_voice = self.best_voice
                self.best_voice = voice.cpu()
                # Delete the old voice tensor to free memory
                if old_voice is not None:
                    del old_voice
            else:
                # Clean up voice tensor if it's not the new best
                del voice

            # Per config every # of steps, clear memory
            if i % self.memcache_clear_freq == 0 and i > 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                try:
                    progress_bar.write(
                        f'Step:{i:<4} Target Sim:{best_results["target_similarity"]:.3f} Self Sim:{best_results["self_similarity"]:.3f} Feature Sim:{best_results["feature_similarity"]:.3f} Score:{best_results["score"]:.2f} Diversity:{diversity:.2f}'
                    )
                except Exception:
                    print("")
                # Save results so folks can listen
                best_voice_name = f'{results_dir}/{self.output_name}_{i}_{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{self.target_audio.stem}.pt'
                torch.save(self.best_voice, best_voice_name)
                sf.write(
                    f'{results_dir}/{self.output_name}_{i}_{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{self.target_audio.stem}.wav',
                    best_results["audio"],
                    24000,
                )
                # TODO: Add config file for easy restarting runs from last save point

        # Print Final Results for Random Walk
        print(f"\n\nRandom Walk Final Results for {self.output_name}")
        print(f"Duration: {(t.format_dict['elapsed'] / 60):.2f} minutes")
        print(f"Best Voice: {best_voice_name}")
        print(f"Best Score: {best_results['score']:.2f}")
        print(f"Best Similarity: {best_results['target_similarity']:.2f}")
        print(f"Random Walk pt and wav files ---> {results_dir}")

        # clear memory at completion
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return

    def hybrid_meta_learn(
        self,
        n_generations: int = 50,
        ga_population_size: int = 20,
        ga_elite_size: int = 4,
        bo_refinement_interval: int = 5,
        bo_iterations_per_candidate: int = 10,
        checkpoint_interval: int = 5,
        resume_checkpoint: str = None,
        verbose: bool = True,
    ):
        """
        Run hybrid meta-learning optimization combining GA and BO.

        Args:
            n_generations: Number of generations to evolve
            ga_population_size: Size of GA population
            ga_elite_size: Number of elite individuals to preserve
            bo_refinement_interval: Apply BO refinement every N generations
            bo_iterations_per_candidate: BO iterations per refined individual
            verbose: Print detailed progress
        """
        # Create results directory
        now = datetime.datetime.now()
        results_dir = Path(
            OUT_DIR
            / f'{self.output_name}_{self.target_audio.stem}_hybrid_{now.strftime("%Y%m%d_%H%M%S")}'
        )
        os.makedirs(results_dir, exist_ok=True)

        # Initialize hybrid meta-learner
        meta_learner = HybridMetaLearner(
            ga_population_size=ga_population_size,
            ga_elite_size=ga_elite_size,
            ga_mutation_rate=0.1,
            ga_crossover_rate=0.7,
            bo_exploration_weight=2.0,
            bo_local_search_radius=0.15,
            bo_refinement_interval=bo_refinement_interval,
            bo_refinement_candidates=min(3, ga_elite_size),
            bo_iterations_per_candidate=bo_iterations_per_candidate,
            diversity_injection_interval=10,
            diversity_injection_count=2,
        )

        # Initialize population from voice generator's voices
        initial_voices = self.voice_generator.voices
        if len(initial_voices) < ga_population_size:
            # Generate additional voices if needed
            print(
                f"Expanding initial population from {len(initial_voices)} to {ga_population_size}"
            )
            while len(initial_voices) < ga_population_size:
                base = initial_voices[
                    len(initial_voices) % len(self.voice_generator.voices)
                ]
                diversity = random.uniform(0.1, 0.3)
                new_voice = self.voice_generator.generate_voice(base, diversity)
                initial_voices.append(new_voice)

        # Load checkpoint if resuming
        if resume_checkpoint:
            checkpoint_path = Path(resume_checkpoint)
            if not checkpoint_path.exists():
                print(f"Warning: Checkpoint not found: {resume_checkpoint}")
                print("Starting fresh run instead...")
                resume_checkpoint = None

        if not resume_checkpoint:
            meta_learner.initialize(initial_voices, self.voice_generator)
        else:
            # Initialize with dummy data first (required for BO setup)
            meta_learner.initialize(initial_voices, self.voice_generator)

        # Run optimization
        best_embedding, best_fitness, best_details = meta_learner.optimize(
            voice_generator=self.voice_generator,
            scorer=self.fitness_scorer.score_voice,
            n_generations=n_generations,
            results_dir=results_dir,
            output_name=self.output_name,
            target_audio_stem=self.target_audio.stem,
            speech_generator=self.speech_generator,
            target_text=self.target_text,
            checkpoint_dir=results_dir / "checkpoints",
            checkpoint_interval=checkpoint_interval,
            resume_from=Path(resume_checkpoint) if resume_checkpoint else None,
            verbose=verbose,
        )

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return best_embedding, best_fitness, best_details
