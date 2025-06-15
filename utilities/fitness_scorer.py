import datetime
import warnings
from typing import Any

import librosa
import numpy as np
import torch
import torchaudio
from numpy._typing import NDArray
from speechbrain.inference.speaker import SpeakerRecognition
from torchaudio.prototype.transforms import ChromaSpectrogram

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class FitnessScorer:
    def __init__(self, target_wav: str, device: str = 'cuda'):
        """
        Initialize FitnessScorer with GPU optimization.

        Args:
            target_wav: Path to target audio file
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.target_wav = target_wav

        # Initialize SpeechBrain on GPU
        self.verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )

        # Pre-load target audio tensor to GPU (do this once!)
        self._target_tensor = self.verification.load_audio(target_wav).to(self.device)
        print(f"Target audio loaded to {self._target_tensor.device}")

        # Pre-compute target features for feature penalty calculation
        target_audio_numpy = self._target_tensor.cpu().numpy()
        self.target_features = self.extract_features(target_audio_numpy)
        print(f"Extracted {len(self.target_features)} target features")

    # Additional utility method for validation
    # def _validate_audio_input(self, audio_input) -> torch.Tensor:
    #     """
    #     Validate and convert audio input to proper tensor format.
    #
    #     Args:
    #         audio_input: Audio as tensor or numpy array
    #
    #     Returns:
    #         Validated tensor on correct device
    #     """
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #
    #     if isinstance(audio_input, np.ndarray):
    #         if audio_input.dtype not in [np.float32, np.float64]:
    #             raise ValueError(f"Unsupported numpy dtype: {audio_input.dtype}")
    #         audio_tensor = torch.from_numpy(audio_input.astype(np.float32)).to(device)
    #     elif isinstance(audio_input, torch.Tensor):
    #         audio_tensor = audio_input.to(device).float()
    #     else:
    #         raise TypeError(f"Unsupported audio input type: {type(audio_input)}")
    #
    #     # Ensure mono
    #     if len(audio_tensor.shape) > 1:
    #         audio_tensor = torch.mean(audio_tensor, dim=-1)
    #
    #     # Basic validation
    #     if audio_tensor.numel() == 0:
    #         raise ValueError("Empty audio tensor")
    #
    #     return audio_tensor

    def hybrid_similarity(self, audio_array: NDArray[np.float32] | torch.Tensor,
                          audio2_array: NDArray[np.float32] | torch.Tensor,
                          target_similarity: float, results: dict[str, Any]) -> tuple[
        dict[str, Any], datetime.timedelta, datetime.timedelta, datetime.timedelta | Any]:
        """
        Calculate hybrid similarity score combining target similarity, self similarity, and feature similarity.
        GPU-compatible version that accepts both numpy arrays and tensors.

        Args:
            audio_array: First audio signal (numpy array or torch tensor)
            audio2_array: Second audio signal (numpy array or torch tensor)
            target_similarity: Pre-calculated target similarity score

        Returns:
            Dictionary containing all similarity scores and final score
        """
        # Extract features using GPU-optimized method
        # Time feature extraction
        # log_gpu_memory("Evaluating Feature Sim")
        feature_start = datetime.datetime.now()
        features = self.extract_features(audio_array)
        feature_time = datetime.datetime.now() - feature_start

        # Ensure tensors for self_similarity calculation
        if isinstance(audio_array, np.ndarray):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            audio_tensor1 = torch.from_numpy(audio_array.astype(np.float32)).to(device)
        else:
            audio_tensor1 = audio_array.float()
            device = audio_tensor1.device

        if isinstance(audio2_array, np.ndarray):
            audio_tensor2 = torch.from_numpy(audio2_array.astype(np.float32)).to(device)
        else:
            audio_tensor2 = audio2_array.to(device).float()

        # Calculate self similarity with tensors
        # log_gpu_memory("Evaluating Self Sim")
        self_sim_start = datetime.datetime.now()
        self_similarity = self.self_similarity(audio_tensor1, audio_tensor2)
        self_sim_time = datetime.datetime.now() - self_sim_start

        # Calculate target feature penalty
        # log_gpu_memory("Evaluating Target Feature Penalty")
        target_penalty_start = datetime.datetime.now()
        target_features_penalty = self.target_feature_penalty(features)
        target_penalty_time = datetime.datetime.now() - target_penalty_start

        # Normalize and make higher = better
        feature_similarity = (100.0 - target_features_penalty) / 100.0
        if feature_similarity < 0.0:
            feature_similarity = 0.01

        # Prepare values for scoring
        values = np.array([target_similarity, self_similarity, feature_similarity])

        # Weights for potential future use (currently using unweighted harmonic mean)
        weights = np.array([0.48, 0.5, 0.02])

        # Harmonic mean calculation (unweighted as per current implementation)
        # Harmonic mean heavily penalizes low scores, encouraging balanced improvement
        score = len(values) / np.sum(1.0 / values)
        results.update({
            "score": float(score),
            "target_similarity": float(target_similarity),
            "self_similarity": float(self_similarity),
            "feature_similarity": float(feature_similarity),
            "weights": weights.tolist()  # Include weights for potential future use
        })
        # Clean up GPU memory
        cleanup_tensors = [audio_tensor1, audio_tensor2]
        for tensor_name in cleanup_tensors:
            try:
                del tensor_name
            except Exception as e:
                # Log which tensor is problematic but continue cleanup
                print(f"Warning: Could not delete {tensor_name}: {e}")
                continue
        return results, feature_time, self_sim_time, target_penalty_time

    def target_similarity(self, audio_tensor: torch.Tensor | NDArray[np.float32]) -> float:
        """
        Calculate similarity between generated audio and target audio.voice = voice.to(device)
        GPU-optimized version using direct tensor operations.

        Args:
            audio_tensor: Generated audio (tensor or numpy array)

        Returns:
            Similarity score (float)
        """
        # Ensure input is a tensor on the correct device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if isinstance(audio_tensor, np.ndarray):
            audio_tensor = torch.from_numpy(audio_tensor.astype(np.float32)).to(device)
        else:
            audio_tensor = audio_tensor.to(device).float()

        # Ensure mono audio
        if len(audio_tensor.shape) > 1:
            audio_tensor = torch.mean(audio_tensor, dim=-1)

        # Use pre-loaded target tensor (set in __init__)
        if not hasattr(self, '_target_tensor'):
            self._target_tensor = self.verification.load_audio(self.target_wav).to(device)

        target_tensor = self._target_tensor.to(device)

        # Create batches for SpeechBrain
        batch_target = target_tensor.unsqueeze(0)
        batch_audio = audio_tensor.unsqueeze(0)

        # Direct tensor verification
        score, _ = self.verification.verify_batch(batch_target, batch_audio)
        target_sim_score = float(score[0])

        # Clean up GPU memory
        cleanup_tensors = [audio_tensor, target_tensor, batch_target, batch_audio, score]
        for tensor_name in cleanup_tensors:
            try:
                del tensor_name
            except Exception as e:
                # Log which tensor is problematic but continue cleanup
                print(f"Warning: Could not delete {tensor_name}: {e}")
                continue
        return target_sim_score

    def target_feature_penalty(self, features: dict[str, Any]) -> float:
        """
        Calculate penalty for differences in audio features compared to target features.
        Optimized version with improved stability and error handling.

        Args:
            features: Dictionary of extracted audio features

        Returns:
            Penalty score (lower is better, 0 = perfect match)
        """
        if not hasattr(self, 'target_features') or not self.target_features:
            # If no target features are available, return neutral penalty
            return 50.0

        penalty = 0.0
        feature_count = 0

        for key, value in features.items():
            if key not in self.target_features:
                # Skip features that don't exist in target (for forward compatibility)
                continue

            target_val = self.target_features[key]

            # Handle different cases for robust calculation
            try:
                # Convert to float if needed (handles numpy types)
                current_val = float(value)
                target_val = float(target_val)

                # Handle near-zero or zero targets
                if abs(target_val) < 1e-8:
                    # For near-zero targets, use absolute difference
                    diff = abs(current_val)
                    # Scale by a reasonable factor to keep penalties in reasonable range
                    diff = min(diff * 100, 5.0)  # Cap at 500% penalty equivalent
                else:
                    # For non-zero targets, use relative difference
                    diff = abs((current_val - target_val) / target_val)
                    # Cap maximum penalty per feature to prevent extreme outliers
                    diff = min(diff, 5.0)  # Max 500% difference penalty

                penalty += diff
                feature_count += 1

            except (ValueError, TypeError, ZeroDivisionError):
                # Skip problematic features rather than crashing
                continue

        if feature_count == 0:
            # No valid features to compare
            return 50.0

        # Average penalty across features, convert to percentage scale
        average_penalty = penalty / feature_count
        return float(average_penalty * 100.0)

    def self_similarity(self, audio_tensor1: torch.Tensor | NDArray[np.float32],
                        audio_tensor2: torch.Tensor | NDArray[np.float32]) -> float:
        """
        Calculate self-similarity between two audio samples from the same voice.
        GPU-optimized version using direct tensor operations.

        Args:
            audio_tensor1: First audio sample (tensor or numpy array)
            audio_tensor2: Second audio sample (tensor or numpy array)

        Returns:
            Self-similarity score (float)
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Convert both inputs to tensors on GPU
        if isinstance(audio_tensor1, np.ndarray):
            audio_tensor1 = torch.from_numpy(audio_tensor1.astype(np.float32)).to(device)
        else:
            audio_tensor1 = audio_tensor1.to(device).float()

        if isinstance(audio_tensor2, np.ndarray):
            audio_tensor2 = torch.from_numpy(audio_tensor2.astype(np.float32)).to(device)
        else:
            audio_tensor2 = audio_tensor2.to(device).float()

        # Ensure mono audio for both
        if len(audio_tensor1.shape) > 1:
            audio_tensor1 = torch.mean(audio_tensor1, dim=-1)
        if len(audio_tensor2.shape) > 1:
            audio_tensor2 = torch.mean(audio_tensor2, dim=-1)

        # Create batches
        batch1 = audio_tensor1.unsqueeze(0)
        batch2 = audio_tensor2.unsqueeze(0)

        # Verify on GPU
        score, _ = self.verification.verify_batch(batch1, batch2)
        self_sim_score = float(score[0])

        # Clean up GPU memory
        cleanup_tensors = [audio_tensor1, audio_tensor2, batch1, batch2]
        for tensor_name in cleanup_tensors:
            try:
                del tensor_name
            except Exception as e:
                # Log which tensor is problematic but continue cleanup
                print(f"Warning: Could not delete {tensor_name}: {e}")
                continue
        return self_sim_score

    def extract_features(self, audio_array: NDArray[np.float32] | NDArray[np.float64] | torch.Tensor,
                         sr: int = 24000) -> dict[str, Any]:
        """
        Extract a comprehensive set of audio features for fingerprinting speech segments.
        GPU-optimized version using torchaudio where possible.

        Args:
            audio_array: Audio signal as numpy array or torch tensor
            sr: Sample rate (fixed at 24000 Hz)

        Returns:
            Dictionary containing extracted features
        """
        # Determine device - use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Convert input to tensor and ensure it's on GPU
        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.from_numpy(audio_array.astype(np.float32)).to(device)
        else:
            audio_tensor = audio_array.to(device).float()

        # Ensure mono (flatten stereo to mono if needed)
        if len(audio_tensor.shape) > 1 and audio_tensor.shape[-1] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=-1)
        elif len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.squeeze()

        # For CPU-only operations, keep a numpy version
        audio_numpy = audio_tensor.cpu().numpy()
        with torch.no_grad():
            # Initialize features dictionary
            features = {}

            # STFT parameters
            n_fft = 2048
            hop_length = 512

            # ===== CPU-ONLY FEATURES (keeping original librosa implementations) =====

            # Tonnetz - librosa
            tonnetz = librosa.feature.tonnetz(y=audio_numpy, sr=sr)
            features["tonnetz_mean"] = float(np.mean(tonnetz))
            features["tonnetz_std"] = float(np.std(tonnetz))

            # Rhythm features - librosa
            tempo, beat_frames = librosa.beat.beat_track(y=audio_numpy, sr=sr)
            features["tempo"] = float(tempo)

            if len(beat_frames) > 0:
                # Calculate beat_stats only if beats are detected
                beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                if len(beat_times) > 1:
                    beat_diffs = np.diff(beat_times)
                    features["beat_mean"] = float(np.mean(beat_diffs))
                    features["beat_std"] = float(np.std(beat_diffs))
                else:
                    features["beat_mean"] = 0.0
                    features["beat_std"] = 0.0
            else:
                features["beat_mean"] = 0.0
                features["beat_std"] = 0.0

            # Pitch and harmonics - librosa
            pitches, magnitudes = librosa.core.piptrack(y=audio_numpy, sr=sr, n_fft=n_fft, hop_length=hop_length)

            # For each frame, find the highest magnitude pitch
            pitch_values = []
            for i in range(magnitudes.shape[1]):
                index = magnitudes[:, i].argmax()
                pitch = pitches[index, i]
                if pitch > 0:  # Exclude zero pitch
                    pitch_values.append(pitch)

            if pitch_values:
                features["pitch_mean"] = float(np.mean(pitch_values))
                features["pitch_std"] = float(np.std(pitch_values))
            else:
                features["pitch_mean"] = 0.0
                features["pitch_std"] = 0.0

            # ===== GPU-ACCELERATED FEATURES =====

            # Basic features - GPU
            features["rms_energy"] = float(torch.sqrt(torch.mean(audio_tensor ** 2)).item())

            # Zero crossing rate - GPU implementation
            zero_crossings = torch.diff(torch.sign(audio_tensor), dim=0)
            features["zero_crossing_rate"] = float(torch.mean(torch.abs(zero_crossings)).item() / 2.0)

            # STFT for spectral features - GPU
            stft = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                              win_length=n_fft, return_complex=True, center=True)
            magnitude = torch.abs(stft)
            power = magnitude ** 2

            # Frequency bins
            freqs = torch.fft.fftfreq(n_fft, 1 / sr)[:n_fft // 2 + 1].to(device)

            # Spectral centroid - GPU
            weighted_freqs = freqs.unsqueeze(1) * magnitude
            spectral_centroids = torch.sum(weighted_freqs, dim=0) / (torch.sum(magnitude, dim=0) + 1e-8)
            features["spectral_centroid_mean"] = float(torch.mean(spectral_centroids).item())
            features["spectral_centroid_std"] = float(torch.std(spectral_centroids).item())

            # Spectral bandwidth - GPU
            freq_diff = (freqs.unsqueeze(1) - spectral_centroids.unsqueeze(0)) ** 2
            spectral_bandwidth = torch.sqrt(
                torch.sum(freq_diff * magnitude, dim=0) / (torch.sum(magnitude, dim=0) + 1e-8))
            features["spectral_bandwidth_mean"] = float(torch.mean(spectral_bandwidth).item())
            features["spectral_bandwidth_std"] = float(torch.std(spectral_bandwidth).item())

            # Spectral rolloff - GPU (85% of spectral energy)
            cumsum_power = torch.cumsum(power, dim=0)
            total_power = torch.sum(power, dim=0)
            rolloff_thresh = 0.85 * total_power
            rolloff_indices = torch.argmax((cumsum_power >= rolloff_thresh.unsqueeze(0)).float(), dim=0)
            rolloff_freqs = freqs[rolloff_indices]
            features["spectral_rolloff_mean"] = float(torch.mean(rolloff_freqs).item())
            features["spectral_rolloff_std"] = float(torch.std(rolloff_freqs).item())

            # Mel spectrogram - GPU
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128
            ).to(device)
            mel_spec = mel_transform(audio_tensor.unsqueeze(0)).squeeze(0)
            features["mel_spec_mean"] = float(torch.mean(mel_spec).item())
            features["mel_spec_std"] = float(torch.std(mel_spec).item())

            # MFCCs - GPU
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sr, n_mfcc=13, melkwargs={
                    'n_fft': n_fft, 'hop_length': hop_length, 'n_mels': 128
                }
            ).to(device)
            mfccs = mfcc_transform(audio_tensor.unsqueeze(0)).squeeze(0)

            # Store each MFCC coefficient mean and std
            for i in range(mfccs.shape[0]):
                features[f"mfcc{i + 1}_mean"] = float(torch.mean(mfccs[i]).item())
                features[f"mfcc{i + 1}_std"] = float(torch.std(mfccs[i]).item())

            # MFCC delta features (first derivative) - GPU
            mfcc_delta = torch.diff(mfccs, dim=1, prepend=mfccs[:, :1])
            for i in range(mfcc_delta.shape[0]):
                features[f"mfcc{i + 1}_delta_mean"] = float(torch.mean(mfcc_delta[i]).item())
                features[f"mfcc{i + 1}_delta_std"] = float(torch.std(mfcc_delta[i]).item())

            # Spectral flatness - GPU
            geometric_mean = torch.exp(torch.mean(torch.log(magnitude + 1e-8), dim=0))
            arithmetic_mean = torch.mean(magnitude, dim=0)
            flatness = geometric_mean / (arithmetic_mean + 1e-8)
            features["spectral_flatness_mean"] = float(torch.mean(flatness).item())
            features["spectral_flatness_std"] = float(torch.std(flatness).item())

            # Spectral contrast - GPU implementation
            # Create frequency bands
            n_bands = 6
            fmin = 200.0
            bands = torch.logspace(torch.log10(torch.tensor(fmin)),
                                   torch.log10(torch.tensor(sr / 2)),
                                   n_bands + 1).to(device)

            contrast_values = []
            for i in range(n_bands):
                # Find frequency bins in this band
                band_mask = (freqs >= bands[i]) & (freqs < bands[i + 1])
                if torch.sum(band_mask) == 0:
                    contrast_values.append(torch.tensor(0.0, device=device))
                    continue

                # Get magnitude in this band
                band_mag = magnitude[band_mask, :]

                # Calculate contrast (peak vs valley)
                if band_mag.numel() > 0:
                    # Peak: mean of top 20% values
                    top_k = max(1, int(0.2 * band_mag.shape[0]))
                    peaks, _ = torch.topk(band_mag, top_k, dim=0)
                    peak_val = torch.mean(peaks)

                    # Valley: mean of bottom 20% values
                    bottom_k = max(1, int(0.2 * band_mag.shape[0]))
                    valleys, _ = torch.topk(band_mag, bottom_k, dim=0, largest=False)
                    valley_val = torch.mean(valleys)

                    # Contrast ratio
                    contrast = peak_val / (valley_val + 1e-8)
                    contrast_values.append(contrast)
                else:
                    contrast_values.append(torch.tensor(0.0, device=device))

            contrast_gpu = torch.stack(contrast_values)
            features["spectral_contrast_mean"] = float(torch.mean(contrast_gpu).item())
            features["spectral_contrast_std"] = float(torch.std(contrast_gpu).item())

            # Energy features - GPU
            frame_length = hop_length
            frames = audio_tensor.unfold(0, frame_length, hop_length)
            energy = torch.sum(torch.abs(frames), dim=1)
            features["energy_mean"] = float(torch.mean(energy).item())
            features["energy_std"] = float(torch.std(energy).item())

            # Harmonics-to-noise ratio - GPU
            S_squared = magnitude ** 2
            S_mean = torch.mean(S_squared, dim=1)
            S_std = torch.std(S_squared, dim=1)
            S_ratio = S_mean / (S_std + 1e-8)
            features["harmonic_ratio"] = float(torch.mean(S_ratio).item())

            # Statistical features from raw waveform - GPU
            features["audio_mean"] = float(torch.mean(audio_tensor).item())
            features["audio_std"] = float(torch.std(audio_tensor).item())

            # Chroma features - GPU using torchaudio.prototype
            chroma_transform = ChromaSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length).to(device)
            chroma = chroma_transform(audio_tensor.unsqueeze(0)).squeeze(0)
            features["chroma_mean"] = float(torch.mean(chroma).item())
            features["chroma_std"] = float(torch.std(chroma).item())

            # Store individual chroma features
            for i in range(chroma.shape[0]):
                features[f"chroma_{i + 1}_mean"] = float(torch.mean(chroma[i]).item())
                features[f"chroma_{i + 1}_std"] = float(torch.std(chroma[i]).item())

            # Statistical features - GPU implementation
            audio_mean = torch.mean(audio_tensor)
            audio_std = torch.std(audio_tensor)
            normalized = (audio_tensor - audio_mean) / (audio_std + 1e-8)

            # Skewness (third moment)
            skewness = torch.mean(normalized ** 3)
            features["audio_skew"] = float(skewness.item())

            # Kurtosis (fourth moment minus 3)
            kurtosis = torch.mean(normalized ** 4) - 3.0
            features["audio_kurtosis"] = float(kurtosis.item())

            # Clean up GPU memory
            # cleanup_tensors = [
            #     audio_tensor, zero_crossings, stft, magnitude, power, freqs,
            #     weighted_freqs, spectral_centroids, freq_diff, spectral_bandwidth,
            #     cumsum_power, total_power, rolloff_indices, rolloff_freqs,
            #     mel_transform, mel_spec, mfcc_transform, mfccs, mfcc_delta,
            #     geometric_mean, arithmetic_mean, flatness, bands, band_mask,
            #     band_mag, contrast_gpu, peak_val, valley_val, contrast,
            #     contrast_values, contrast_gpu, frames, energy, S_squared,
            #     S_mean, S_std, S_ratio, chroma_transform, chroma, audio_mean,
            #     audio_std, normalized, skewness, kurtosis
            # ]
            #
            # for tensor_name in cleanup_tensors:
            #     try:
            #         del tensor_name
            #     except Exception as e:
            #         # Log which tensor is problematic but continue cleanup
            #         print(f"Warning: Could not delete {tensor_name}: {e}")
            #         continue
            return features
