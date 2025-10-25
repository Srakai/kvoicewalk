#!/usr/bin/env python3
"""
Standalone script to find optimal weights for the cost function.

This script generates audio samples from different voices using different text inputs,
calculates the three similarity metrics (target_similarity, self_similarity, feature_similarity),
and uses optimization to find the weights that best separate different voices while keeping
variations of the same voice close together.

Usage:
    python tune_weights.py
"""

import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.optimize import minimize
import torch
from tqdm import tqdm

from utilities.fitness_scorer import FitnessScorer
from utilities.speech_generator import SpeechGenerator
from utilities.pytorch_sanitizer import load_voice_safely


# Configuration
NUM_VOICES = 10  # Number of different voices to test
NUM_TEXTS_PER_VOICE = 5  # Number of different text inputs per voice
VOICES_DIR = Path("voices")
ROOT_DIR = Path(__file__).resolve().parent

# Sample texts for generating different audio variations
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming the world of artificial intelligence.",
    "Python is a versatile programming language used by millions of developers.",
    "Climate change poses significant challenges for future generations.",
    "Music has the power to evoke strong emotions and memories.",
    "Space exploration continues to reveal the mysteries of our universe.",
    "Healthy eating and regular exercise contribute to overall wellbeing.",
    "Technology advances at an exponential rate in the modern era.",
    "Education is the foundation of personal and societal development.",
    "The ocean covers more than seventy percent of Earth's surface.",
    "Art allows people to express creativity and cultural identity.",
    "Economic policies shape the prosperity of nations and communities.",
    "Communication skills are essential in both personal and professional life.",
    "Innovation drives progress and solves complex global problems.",
    "Nature provides countless resources that sustain life on our planet.",
]

# Fixed target for baseline comparison
TARGET_TEXT = "This is the target voice we are trying to match."
OTHER_TEXT = "This is a different sentence for self-similarity testing."


def select_voices(voices_dir: Path, n_voices: int) -> List[Path]:
    """Select N random voices from the voices directory."""
    all_voices = list(voices_dir.glob("*.pt"))
    if len(all_voices) < n_voices:
        print(f"Warning: Only {len(all_voices)} voices available, using all of them.")
        return all_voices
    return random.sample(all_voices, n_voices)


def generate_dataset(
    voices: List[Path], texts: List[str], n_texts_per_voice: int
) -> List[Dict[str, Any]]:
    """
    Generate dataset of metric scores for different voice-text combinations.

    Returns:
        List of dicts containing:
            - voice_name: str
            - text: str
            - target_similarity: float
            - self_similarity: float
            - feature_similarity: float
    """
    dataset = []

    # Use the first voice and text as the target reference
    target_voice_path = voices[0]
    target_voice = load_voice_safely(str(target_voice_path))

    print(f"\n=== Stage 1: Generating Dataset ===")
    print(f"Using target voice: {target_voice_path.stem}")
    print(
        f"Generating samples for {len(voices)} voices with {n_texts_per_voice} texts each...\n"
    )

    # Initialize speech generator and fitness scorer with target
    speech_generator = SpeechGenerator(
        target_text=TARGET_TEXT, other_text=OTHER_TEXT, device="cpu"
    )

    # Generate target audio for reference
    target_audio = speech_generator.generate_audio(TARGET_TEXT, target_voice)
    target_audio_numpy = target_audio.detach().cpu().numpy()

    # Save temporary target audio file
    import soundfile as sf

    target_wav_path = ROOT_DIR / "temp_target.wav"
    sf.write(str(target_wav_path), target_audio_numpy, 24000)

    # Initialize fitness scorer
    fitness_scorer = FitnessScorer(
        str(target_wav_path),
        speech_generator=speech_generator,
    )

    # Generate samples for each voice
    for voice_path in tqdm(voices, desc="Processing voices"):
        voice_name = voice_path.stem
        voice_tensor = load_voice_safely(str(voice_path))

        # Select random texts for this voice
        selected_texts = random.sample(texts, min(n_texts_per_voice, len(texts)))

        for text in selected_texts:
            # Update speech generator with new text
            speech_generator.target_text = text

            # Generate audio with this voice and text
            audio_tensor = speech_generator.generate_audio(text, voice_tensor)

            # Calculate metrics
            best_results = {
                "score": 0.0,
                "target_similarity": 0.0,
                "self_similarity": 0.0,
                "feature_similarity": 0.0,
            }

            results, _ = fitness_scorer.score_voice(
                best_results=best_results,
                voice_tensor=voice_tensor,
                min_similarity=-100.0,  # Don't skip any evaluation
            )

            # Store results
            dataset.append(
                {
                    "voice_name": voice_name,
                    "text": text,
                    "target_similarity": results["target_similarity"],
                    "self_similarity": results["self_similarity"],
                    "feature_similarity": results["feature_similarity"],
                }
            )

    # Clean up temporary file
    target_wav_path.unlink()

    print(f"\nGenerated {len(dataset)} samples total.")
    return dataset


def calculate_separation_score(
    weights: np.ndarray, dataset: List[Dict[str, Any]]
) -> float:
    """
    Calculate how well the given weights separate different voices.

    The separation score is:
        (average score for same-voice pairs) - (average score for different-voice pairs)

    We want to MAXIMIZE this, so we return the negative for minimization.
    """
    w_target, w_self, w_feature = weights

    # Ensure weights are positive and sum to 1 (normalization)
    if w_target < 0 or w_self < 0 or w_feature < 0:
        return 1e10  # Penalty for invalid weights

    weight_sum = w_target + w_self + w_feature
    if weight_sum == 0:
        return 1e10

    w_target /= weight_sum
    w_self /= weight_sum
    w_feature /= weight_sum

    # Calculate weighted scores for all samples
    for sample in dataset:
        sample["weighted_score"] = (
            w_target * sample["target_similarity"]
            + w_self * sample["self_similarity"]
            + w_feature * sample["feature_similarity"]
        )

    # Group samples by voice
    voices = {}
    for sample in dataset:
        voice_name = sample["voice_name"]
        if voice_name not in voices:
            voices[voice_name] = []
        voices[voice_name].append(sample["weighted_score"])

    # Calculate average intra-voice similarity (same voice, different texts)
    intra_voice_scores = []
    for voice_name, scores in voices.items():
        if len(scores) > 1:
            # All pairs within this voice
            for i in range(len(scores)):
                for j in range(i + 1, len(scores)):
                    intra_voice_scores.append(abs(scores[i] - scores[j]))

    # Calculate average inter-voice similarity (different voices)
    inter_voice_scores = []
    voice_names = list(voices.keys())
    for i in range(len(voice_names)):
        for j in range(i + 1, len(voice_names)):
            # Compare samples from different voices
            for score_i in voices[voice_names[i]]:
                for score_j in voices[voice_names[j]]:
                    inter_voice_scores.append(abs(score_i - score_j))

    # We want SMALL differences within voices and LARGE differences between voices
    if not intra_voice_scores or not inter_voice_scores:
        return 1e10

    avg_intra = np.mean(intra_voice_scores)
    avg_inter = np.mean(inter_voice_scores)

    # Separation score: inter-voice distance minus intra-voice distance
    # Higher is better, so we return negative for minimization
    separation = avg_inter - avg_intra

    return -separation


def optimize_weights(dataset: List[Dict[str, Any]]) -> Tuple[np.ndarray, float]:
    """
    Find optimal weights using scipy optimization.

    Returns:
        Tuple of (optimal_weights, best_separation_score)
    """
    print("\n=== Stage 2: Optimizing Weights ===")
    print("Finding weights that maximize voice separation...\n")

    # Initial guess: equal weights
    initial_weights = np.array([1.0, 1.0, 1.0])

    # Optimization bounds: all weights must be non-negative
    bounds = [(0, None), (0, None), (0, None)]

    # Run optimization
    result = minimize(
        lambda w: calculate_separation_score(w, dataset),
        initial_weights,
        method="L-BFGS-B",
        bounds=bounds,
        options={"disp": True},
    )

    # Normalize final weights
    optimal_weights = result.x
    optimal_weights /= np.sum(optimal_weights)

    best_score = -result.fun  # Convert back to positive separation score

    return optimal_weights, best_score


def main():
    """Main execution function."""
    print("=" * 60)
    print("Cost Function Weight Tuning Experiment")
    print("=" * 60)

    # Select random voices
    voices = select_voices(VOICES_DIR, NUM_VOICES)

    if len(voices) == 0:
        print("Error: No voices found in voices/ directory")
        return

    # Generate dataset
    dataset = generate_dataset(voices, SAMPLE_TEXTS, NUM_TEXTS_PER_VOICE)

    # Optimize weights
    optimal_weights, separation_score = optimize_weights(dataset)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOptimal Weights:")
    print(f"  target_similarity:  {optimal_weights[0]:.4f}")
    print(f"  self_similarity:    {optimal_weights[1]:.4f}")
    print(f"  feature_similarity: {optimal_weights[2]:.4f}")
    print(f"\nSeparation Score: {separation_score:.4f}")
    print("\nTo use these weights, update the hybrid_similarity method in")
    print("utilities/fitness_scorer.py to use a weighted sum:")
    print(f"\n    score = {optimal_weights[0]:.4f} * target_similarity + \\")
    print(f"            {optimal_weights[1]:.4f} * self_similarity + \\")
    print(f"            {optimal_weights[2]:.4f} * feature_similarity")
    print("=" * 60)


if __name__ == "__main__":
    main()
