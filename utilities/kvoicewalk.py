import datetime
import os
import random
import traceback
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from torch import Tensor
from tqdm import tqdm

from utilities.fitness_scorer import FitnessScorer
from utilities.initial_selector import InitialSelector
from utilities.kvw_informer import log_gpu_memory
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
            log_gpu_memory("Initializing Speech Generator")
            self.speech_generator = SpeechGenerator()
            try:
                log_gpu_memory("Selecting voices")
                self.initial_selector = InitialSelector(self.speech_generator, str(target_audio), target_text,
                                                        other_text,
                                                    voice_folder=voice_folder)
            except Exception as e:
                raise Exception(f"Error: {e}")
            log_gpu_memory("Scoring target audio")
            self.fitness_scorer = FitnessScorer(str(target_audio))

            log_gpu_memory("Selecting starting voices")
            if interpolate_start:
                voices = self.initial_selector.interpolate_search(population_limit)
            else:
                voices = self.initial_selector.top_performer_start(population_limit)
            log_gpu_memory("Initializing Voice Generator")
            self.voice_generator = VoiceGenerator(voices, starting_voice)
            self.clear_losers_from_memory()
            self.output_name = output_name
        except Exception as e:
            print("FULL TRACEBACK:")
            traceback.print_exc()
            print(f"\nERROR: {e}")
            print(f"ERROR TYPE: {type(e)}")
            print(f"Error initializing KVoicewalk: {e}")
            raise SystemExit

    def random_walk(self,step_limit: int):
        log_gpu_memory("Scoring initial voice")
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
        log_gpu_memory("Starting random walk run")
        for i in tqdm(range(step_limit)):
            # TODO: Expose to CLI
            diversity = random.uniform(0.01,0.15)
            log_gpu_memory("Generating best voice comparison")
            voice = self.voice_generator.generate_voice(best_voice,diversity)

            # Early function return saves audio generation compute
            min_similarity = best_results["target_similarity"] * 0.98
            log_gpu_memory("Scoring Voice results")
            voice_results = self.score_voice(voice, min_similarity)

            # Set new winner if score is better
            # Check GPU memory periodically
            if voice_results["score"] > best_results["score"]:
                self.clear_losers_from_memory()
                log_gpu_memory("Printing/Saving Results KVoicewalk:", view=True)
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

    def score_voice(self, voice_tensor: torch.Tensor, min_similarity: float = 0.0) -> dict[str, Any]:
        start_time = datetime.datetime.now()

        log_gpu_memory("Generating iterated voice audio")
        # Time audio generation
        audio_start = datetime.datetime.now()
        audio = self.speech_generator.generate_audio(self.target_text, voice_tensor)
        audio = Tensor.cpu(audio).numpy()
        audio_time = datetime.datetime.now() - audio_start
        results: dict[str, Any] = {
            'audio': audio
        }
        log_gpu_memory("Evaluating Target Sim")
        # Time target similarity calculation
        target_sim_start = datetime.datetime.now()
        target_sim = self.fitness_scorer.target_similarity(audio)
        target_sim_time = datetime.datetime.now() - target_sim_start

        log_gpu_memory("Checking target vs min sim")
        if target_sim > min_similarity:
            # Time self similarity calculation
            log_gpu_memory("Generating Audio2")
            audio2_start = datetime.datetime.now()
            audio2 = self.speech_generator.generate_audio(self.target_text, voice_tensor)
            # self_sim = self.fitness_scorer.self_similarity(audio, audio2)
            audio2_time = datetime.datetime.now() - audio2_start

            # Time hybrid sim scoring
            log_gpu_memory("Evaluating Hybrid Sim")
            score_start = datetime.datetime.now()
            # results.update(self.fitness_scorer.hybrid_similarity(audio, audio2, target_sim))
            results, feature_sim_time, self_sim_time, target_penalty_time = self.fitness_scorer.hybrid_similarity(audio,
                                                                                                                  audio2,
                                                                                                                  target_sim,
                                                                                                                  results)
            score_time = datetime.datetime.now() - score_start
            total_time = datetime.datetime.now() - start_time
            del audio2
            log_gpu_memory("Returning success results (target sim > min sim)")
            # print(
            #     f"Audio1 gen: {audio_time.total_seconds()}s, Audio1 gen: {audio2_time.total_seconds()}s, Target Similarity: {target_sim_time.total_seconds()}s,  Self Similarity: {self_sim_time.total_seconds()}s, Features: {feature_sim_time.total_seconds()}s, Penalty: {target_penalty_time.total_seconds()}s, Scoring: {score_time.total_seconds()}s, Total: {total_time.total_seconds()}s")

        else:
            results["score"] = 0.0
            results["target_similarity"] = target_sim
            # Clean up GPU memory
            del audio
            log_gpu_memory("Returning fail results (target sim < min sim)")

        return results

    def clear_losers_from_memory(self):
        # Get reference to the voices cache
        voices_cache = self.speech_generator.pipeline.voices

        # Create list of voices to keep (winners)
        voices_to_keep = {self.starting_voice}  # Always keep starting voice

        # Add current best voice to exempt list
        if hasattr(self, 'best_voice') and self.best_voice:
            voices_to_keep.add(self.best_voice)

        # Clean up all other voices
        voices_to_delete = []
        for voice_name in voices_cache:
            if voice_name not in voices_to_keep:
                voices_to_delete.append(voice_name)

        # Delete the non-winners
        for voice_name in voices_to_delete:
            del voices_cache[voice_name]
