import datetime
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

ROOT_DIR = Path(__file__).resolve().parents[1]
TEXTS_DIR = ROOT_DIR / "texts"
CONVERTED_DIR = ROOT_DIR / "out" / "converted_audio"
CHUNKS_DIR = ROOT_DIR / "out" / "audio_chunks"


def convert_to_wav_mono_24k(audio_path: Path) -> Path:
    try:
        with sf.SoundFile(audio_path, "r") as f:
            if f.format != "WAV" or f.samplerate != 24000 or f.channels != 1:
                print(f"Converting {audio_path.name} to Mono Wav 24K...")
                CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
                # Create output filename with proper audio format
                converted_audio_file = Path(
                    CONVERTED_DIR / str(audio_path.stem + ".wav")
                )

                # Read the audio data
                audio_data = f.read()

                # Convert to mono if needed
                if f.channels > 1:
                    converted_audio_data = np.mean(audio_data, axis=1)
                    # print("Cenverted to Mono...")
                else:
                    converted_audio_data = audio_data

                # Resample if needed
                if f.samplerate != 24000:
                    converted_audio_data = librosa.resample(
                        converted_audio_data, orig_sr=f.samplerate, target_sr=24000
                    )
                    # print("Resampled to 24K...")

                # Save converted audio
                sf.write(
                    converted_audio_file,
                    converted_audio_data,
                    samplerate=24000,
                    format="WAV",
                )
                print(
                    f"{audio_path.name} converted to Mono WAV 24K format: {converted_audio_file}"
                )
                return converted_audio_file
            else:
                # print(f"{audio_path.name} matches Mono WAV 24K format")
                return audio_path

    except Exception as e:
        print(f"Error converting {audio_path.name}: {e}\n")


class Transcriber:
    def __init__(self):
        model_size = "large-v3"
        # print('Starting Transcriber...')
        # Run on GPU with FP16
        # model = WhisperModel(model_size, device="cuda", compute_type="float16")

        # or run on GPU with INT8
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

        # or run on CPU with INT8 !!(more than sufficient for 30s clip)!!
        # faster-whisper doesn't support MPS, use CPU instead
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path: Path):
        audio_file = audio_path
        start_time = datetime.datetime.now()

        try:
            # print(f'Loading {audio_file.name}...')
            segments, info = self.model.transcribe(str(audio_file), beam_size=5)

            print(
                "Detected language '%s' with probability %f"
                % (info.language, info.language_probability)
            )
            print(f"Transcribing {audio_file.name}...")

            transcription = ""
            for segment in segments:
                transcription += " " + segment.text.strip()

            transcription_output = Path(TEXTS_DIR / str(f"{audio_file.stem}.txt"))
            with open(str(transcription_output), "w") as file:
                file.write(f"{transcription[1:]}")

            end_time = datetime.datetime.now()
            print(f"Transcription available at ./texts/{audio_file.name[:-4]}.txt")
            print(f"{audio_file.name} Transcription:\n{transcription[1:]}")
            return transcription[1:]

        except Exception as e:
            print(f"Transcription failed for {audio_file.name} - Error: {e}")
            return

    def chunk_audio(
        self,
        audio_path: Path,
        max_chunk_duration: float = 30.0,
        split_by_sentence: bool = False,
    ) -> List[Tuple[Path, str, float, float]]:
        """
        Intelligently chunk audio file using Whisper's natural speech boundaries.

        Args:
            audio_path: Path to the audio file to chunk
            max_chunk_duration: Maximum duration in seconds for each chunk (default: 30s)
            split_by_sentence: If True, split only at sentence boundaries regardless of duration.
                             If False, respect max_chunk_duration and split at sentence boundaries when possible.

        Returns:
            List of tuples: (chunk_path, transcription, start_time, end_time)
        """
        mode_desc = (
            "sentence boundaries"
            if split_by_sentence
            else f"max duration: {max_chunk_duration}s"
        )
        print(f"Chunking {audio_path.name} by {mode_desc}...")

        try:
            # Load audio data
            audio_data, sr = sf.read(str(audio_path))

            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Transcribe with word-level timestamps
            segments, info = self.model.transcribe(
                str(audio_path), beam_size=5, word_timestamps=True
            )

            print(
                f"Detected language '{info.language}' with probability {info.language_probability:.2f}"
            )

            # Group segments into chunks
            # Strategy depends on split_by_sentence parameter
            chunks = []
            current_chunk_words = []
            current_chunk_start = None
            current_chunk_end = None

            # Sentence-ending punctuation
            sentence_enders = {".", "!", "?"}

            for segment in segments:
                # Get words from segment (if available)
                if hasattr(segment, "words") and segment.words:
                    for word in segment.words:
                        word_start = word.start
                        word_end = word.end
                        word_text = word.word.strip()

                        # Initialize chunk start time
                        if current_chunk_start is None:
                            current_chunk_start = word_start

                        # Add word to current chunk
                        current_chunk_words.append(word)
                        current_chunk_end = word_end

                        # Check if this word ends a sentence
                        is_sentence_end = any(
                            word_text.endswith(punct) for punct in sentence_enders
                        )

                        if split_by_sentence:
                            # Simple mode: split only at sentence boundaries
                            if is_sentence_end:
                                chunks.append(
                                    {
                                        "start": current_chunk_start,
                                        "end": current_chunk_end,
                                        "words": current_chunk_words,
                                        "duration": current_chunk_end
                                        - current_chunk_start,
                                    }
                                )
                                # Reset for next chunk
                                current_chunk_words = []
                                current_chunk_start = None
                                current_chunk_end = None
                        else:
                            # Duration-aware mode: respect max_chunk_duration
                            # Check current duration
                            current_duration = word_end - current_chunk_start

                            # Split strategy:
                            # 1. If we're at a sentence boundary and duration > 80% of max, split here
                            # 2. If duration exceeds max and we have words, split at last sentence or here
                            if is_sentence_end and current_duration >= (
                                0.8 * max_chunk_duration
                            ):
                                # Good place to split - at sentence boundary, near max duration
                                chunks.append(
                                    {
                                        "start": current_chunk_start,
                                        "end": current_chunk_end,
                                        "words": current_chunk_words,
                                        "duration": current_duration,
                                    }
                                )
                                # Reset for next chunk
                                current_chunk_words = []
                                current_chunk_start = None
                                current_chunk_end = None
                            elif (
                                current_duration > max_chunk_duration
                                and len(current_chunk_words) > 1
                            ):
                                # Exceeded max duration - need to split
                                # Try to find last sentence boundary in current chunk
                                split_idx = None
                                for i in range(len(current_chunk_words) - 2, -1, -1):
                                    w = current_chunk_words[i]
                                    w_text = w.word.strip()
                                    if any(
                                        w_text.endswith(punct)
                                        for punct in sentence_enders
                                    ):
                                        split_idx = i + 1
                                        break

                                if split_idx and split_idx < len(current_chunk_words):
                                    # Split at sentence boundary
                                    chunk_words = current_chunk_words[:split_idx]
                                    remaining_words = current_chunk_words[split_idx:]

                                    chunks.append(
                                        {
                                            "start": current_chunk_start,
                                            "end": chunk_words[-1].end,
                                            "words": chunk_words,
                                            "duration": chunk_words[-1].end
                                            - current_chunk_start,
                                        }
                                    )

                                    # Start new chunk with remaining words
                                    current_chunk_words = remaining_words
                                    current_chunk_start = (
                                        remaining_words[0].start
                                        if remaining_words
                                        else None
                                    )
                                    current_chunk_end = (
                                        remaining_words[-1].end
                                        if remaining_words
                                        else None
                                    )
                                else:
                                    # No sentence boundary found, split at current word
                                    chunk_words = current_chunk_words[:-1]

                                    chunks.append(
                                        {
                                            "start": current_chunk_start,
                                            "end": chunk_words[-1].end,
                                            "words": chunk_words,
                                            "duration": chunk_words[-1].end
                                            - current_chunk_start,
                                        }
                                    )

                                    # Start new chunk with current word
                                    current_chunk_words = [word]
                                    current_chunk_start = word_start
                                    current_chunk_end = word_end
                else:
                    # Fallback: no word timestamps, use segment-level
                    if current_chunk_start is None:
                        current_chunk_start = segment.start

                    potential_duration = segment.end - current_chunk_start

                    if potential_duration > max_chunk_duration and current_chunk_words:
                        # Save current chunk
                        chunks.append(
                            {
                                "start": current_chunk_start,
                                "end": current_chunk_end,
                                "text": " ".join(
                                    [
                                        w.word if hasattr(w, "word") else str(w)
                                        for w in current_chunk_words
                                    ]
                                ),
                                "duration": current_chunk_end - current_chunk_start,
                            }
                        )

                        # Start new chunk
                        current_chunk_words = [
                            {
                                "word": segment.text,
                                "start": segment.start,
                                "end": segment.end,
                            }
                        ]
                        current_chunk_start = segment.start
                        current_chunk_end = segment.end
                    else:
                        # Add segment as pseudo-word
                        current_chunk_words.append(
                            {
                                "word": segment.text,
                                "start": segment.start,
                                "end": segment.end,
                            }
                        )
                        current_chunk_end = segment.end

            # Don't forget the last chunk
            if current_chunk_words:
                chunks.append(
                    {
                        "start": current_chunk_start,
                        "end": current_chunk_end,
                        "words": current_chunk_words,
                        "duration": current_chunk_end - current_chunk_start,
                    }
                )

            # Create output directory for chunks
            CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
            chunk_info = []

            print(f"\n{'='*80}")
            print(f"Creating {len(chunks)} chunks from {audio_path.name}")
            print(f"{'='*80}")

            # Extract and save each chunk
            for i, chunk in enumerate(chunks):
                start_sample = int(chunk["start"] * sr)
                end_sample = int(chunk["end"] * sr)

                # Extract audio chunk
                chunk_audio = audio_data[start_sample:end_sample]

                # Create chunk filename
                chunk_filename = f"{audio_path.stem}_chunk_{i:03d}.wav"
                chunk_path = CHUNKS_DIR / chunk_filename

                # Save chunk
                sf.write(str(chunk_path), chunk_audio, sr)

                # Get transcription for this chunk (from word objects)
                if chunk.get("words"):
                    chunk_transcription = " ".join(
                        [
                            w.word.strip() if hasattr(w, "word") else str(w).strip()
                            for w in chunk["words"]
                        ]
                    )
                else:
                    chunk_transcription = chunk.get("text", "")

                # Store chunk info
                chunk_info.append(
                    (chunk_path, chunk_transcription, chunk["start"], chunk["end"])
                )

                print(f"\nChunk {i+1}/{len(chunks)}:")
                print(
                    f"  Duration: {chunk['duration']:.2f}s ({chunk['start']:.2f}s - {chunk['end']:.2f}s)"
                )
                print(f"  Words: {len(chunk_transcription.split())}")
                print(f"  Text: {chunk_transcription}")

            print(f"\n{'='*80}")
            print(f"Total chunks created: {len(chunk_info)}")
            print(f"Chunks saved to: {CHUNKS_DIR}")
            print(f"{'='*80}\n")
            return chunk_info

        except Exception as e:
            print(f"Chunking failed for {audio_path.name} - Error: {e}")
            return []


# TODO: Integrate into automated workflows
