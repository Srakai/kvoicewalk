import datetime
import os
import random
import time
import traceback
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from utilities.fitness_scorer import FitnessScorer
from utilities.initial_selector import InitialSelector
from utilities.path_router import OUT_DIR
from utilities.speech_generator import SpeechGenerator
from utilities.voice_generator import VoiceGenerator


class KVoiceWalk:
    def __init__(self, target_audio: Path, target_text: str, other_text: str, voice_folder: str,
                 interpolate_start: bool, population_limit: int, starting_voice: str, output_name: str) -> None:
        try:
            self.target_audio = target_audio
            self.target_text = target_text
            self.other_text = other_text

            try:
                self.initial_selector = InitialSelector(str(target_audio), target_text, other_text,
                                                    voice_folder=voice_folder)
                print(f"self.initial_selector: passed")
            except Exception as e:
                raise Exception(f"Error: {e}")
            self.fitness_scorer = FitnessScorer(str(target_audio))
            print(f"fitness scorer passed")
            voices: list[torch.Tensor] = []
            if interpolate_start:
                voices = self.initial_selector.interpolate_search(population_limit)
                print(f"interpolate_start passed")
            else:
                voices = self.initial_selector.top_performer_start(population_limit)
                print(f"regular start passed")
            self.speech_generator = SpeechGenerator()
            print(f"speech gen passed")

            self.voice_generator = VoiceGenerator(voices, starting_voice)
            print(f"interpolate_start passed")
            # Either the mean or the supplied voice tensor
            self.starting_voice = self.voice_generator.starting_voice
            self.output_name = output_name
        except Exception as e:
            print("FULL TRACEBACK:")
            traceback.print_exc()
            print(f"\nERROR: {e}")
            print(f"ERROR TYPE: {type(e)}")
            print(f"Error initializing KVoicewalk: {e}")
            raise SystemExit

    def random_walk(self,step_limit: int):

        # Score Initial Voice
        best_voice = self.starting_voice
        best_results = self.score_voice(self.starting_voice)
        t = tqdm()
        t.write(f'Target Sim:{best_results["target_similarity"]:.3f}, Self Sim:{best_results["self_similarity"]:.3f}, Feature Sim:{best_results["feature_similarity"]:.2f}, Score:{best_results["score"]:.2f}')

        # Create Results Directory
        now = datetime.datetime.now()
        results_dir = Path(OUT_DIR / f'{self.output_name}_{self.target_audio.stem}_{now.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(results_dir, exist_ok=True)

        # Random Walk Loop

        for i in tqdm(range(step_limit)):
            # TODO: Expose to CLI
            diversity = random.uniform(0.01,0.15)
            voice = self.voice_generator.generate_voice(best_voice,diversity)

            # Early function return saves audio generation compute
            min_similarity = best_results["target_similarity"] * 0.98
            voice_results = self.score_voice(voice,min_similarity)

            # Set new winner if score is better
            if voice_results["score"] > best_results["score"]:
                best_results = voice_results
                best_voice = voice
                t.write(f'Step:{i:<4} Target Sim:{best_results["target_similarity"]:.3f} Self Sim:{best_results["self_similarity"]:.3f} Feature Sim:{best_results["feature_similarity"]:.3f} Score:{best_results["score"]:.2f} Diversity:{diversity:.2f}')
                # Save results so folks can listen
                torch.save(best_voice,
                           f'{results_dir}/{self.output_name}_{i}_{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{self.target_audio.stem}.pt')
                sf.write(
                    f'{results_dir}/{self.output_name}_{i}_{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{self.target_audio.stem}.wav',
                    best_results["audio"], 24000)
                # TODO: Add config file for easy restarting runs from last save point

        # Print Final Results for Random Walk
        print(f"Random Walk Final Results for {self.output_name}")
        print(f"Duration: {t.format_dict['elapsed']}")
        # print(f"Best Voice: {best_voice}") #TODO: add best voice model name
        print(f"Best Score: {best_results['score']:.2f}_")
        print(f"Best Similarity: {best_results['target_similarity']:.2f}_")
        print(f"Random Walk pt and wav files ---> {results_dir}")

        return

    # def score_voice(self,voice: torch.Tensor,min_similarity: float = 0.0) -> dict[str,Any]:
    #     """Using a harmonic mean calculation to provide a score for the voice in similarity"""
    #     audio = self.speech_generator.generate_audio(self.target_text, voice)
    #     target_similarity = self.fitness_scorer.target_similarity(audio)
    #     results: dict[str,Any] = {
    #         'audio': audio
    #     }
    #     # Bail early and save the compute if the similarity sucks
    #     if target_similarity > min_similarity:
    #         audio2 = self.speech_generator.generate_audio(self.other_text, voice)
    #         results.update(self.fitness_scorer.hybrid_similarity(audio,audio2,target_similarity))
    #     else:
    #         results["score"] = 0.0
    #         results["target_similarity"] = target_similarity
    #
    #     return results

    # def score_voice(self, voice_tensor: torch.Tensor, min_similarity: float):
    #     """Entire scoring pipeline on GPU"""
    #     device = voice_tensor.device
    #
    #     # Generate audio on GPU - returns tensor
    #     audio_tensor = self.speech_generator.generate_audio(self.target_text, voice_tensor)
    #
    #     # All similarity calculations on GPU - no file I/O!
    #     target_sim = self.fitness_scorer.target_similarity(audio_tensor)
    #
    #     # Generate second sample for self-similarity
    #     audio2_tensor = self.speech_generator.generate_audio(self.target_text, voice_tensor)
    #     self_sim = self.fitness_scorer.self_similarity(audio_tensor, audio2_tensor)
    #
    #     # Only move to CPU for feature extraction (if needed)
    #     features = self.fitness_scorer.extract_features(audio_tensor.cpu().numpy())
    #
    #     # Hybrid scoring
    #     results = self.fitness_scorer.hybrid_similarity(
    #         audio_tensor.cpu().numpy(),  # Only CPU when absolutely needed
    #         audio2_tensor.cpu().numpy(),
    #         target_sim
    #     )
    #
    #     return results

    def score_voice(self, voice_tensor: torch.Tensor, min_similarity: float):
        start_time = time.time()

        # Time audio generation
        audio_start = time.time()
        audio = self.speech_generator.generate_audio(self.target_text, voice_tensor)
        audio_time = time.time() - audio_start

        # Time similarity calculation
        sim_start = time.time()
        target_sim = self.fitness_scorer.target_similarity(audio)
        audio2 = self.speech_generator.generate_audio(self.target_text, voice_tensor)
        self_sim = self.fitness_scorer.self_similarity(audio, audio2)
        sim_time = time.time() - sim_start

        # Time feature extraction
        feature_start = time.time()
        features = self.fitness_scorer.extract_features(audio)
        feature_time = time.time() - feature_start

        # Time scoring
        score_start = time.time()
        results = self.fitness_scorer.hybrid_similarity(audio, audio2, target_sim)
        score_time = time.time() - score_start

        total_time = time.time() - start_time

        print(
            f"Audio gen: {audio_time:.3f}s, Similarity: {sim_time:.3f}s, Features: {feature_time:.3f}s, Scoring: {score_time:.3f}s, Total: {total_time:.3f}s")

        return results