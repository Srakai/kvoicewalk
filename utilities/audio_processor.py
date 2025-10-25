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
        self.model = WhisperModel(model_size, device="mps", compute_type="float16")

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
        self, audio_path: Path, max_chunk_duration: float = 30.0
    ) -> List[Tuple[Path, str, float, float]]:
        """
        Intelligently chunk audio file using Whisper's natural speech boundaries.

        Args:
            audio_path: Path to the audio file to chunk
            max_chunk_duration: Maximum duration in seconds for each chunk (default: 30s)

        Returns:
            List of tuples: (chunk_path, transcription, start_time, end_time)
        """
        print(f"Chunking {audio_path.name} with max duration: {max_chunk_duration}s...")

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

            # Group segments into chunks based on max_chunk_duration
            chunks = []
            current_chunk_segments = []
            current_chunk_start = 0.0
            current_chunk_duration = 0.0

            for segment in segments:
                segment_start = segment.start
                segment_end = segment.end
                segment_duration = segment_end - segment_start

                # If adding this segment would exceed max duration, save current chunk
                if (
                    current_chunk_segments
                    and (current_chunk_duration + segment_duration) > max_chunk_duration
                ):
                    # Save current chunk
                    chunk_end = current_chunk_segments[-1].end
                    chunks.append(
                        {
                            "start": current_chunk_start,
                            "end": chunk_end,
                            "segments": current_chunk_segments,
                            "duration": chunk_end - current_chunk_start,
                        }
                    )

                    # Start new chunk
                    current_chunk_segments = [segment]
                    current_chunk_start = segment_start
                    current_chunk_duration = segment_duration
                else:
                    # Add segment to current chunk
                    current_chunk_segments.append(segment)
                    current_chunk_duration += segment_duration

            # Don't forget the last chunk
            if current_chunk_segments:
                chunk_end = current_chunk_segments[-1].end
                chunks.append(
                    {
                        "start": current_chunk_start,
                        "end": chunk_end,
                        "segments": current_chunk_segments,
                        "duration": chunk_end - current_chunk_start,
                    }
                )

            # Create output directory for chunks
            CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
            chunk_info = []

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

                # Get transcription for this chunk
                chunk_transcription = " ".join(
                    [seg.text.strip() for seg in chunk["segments"]]
                )

                # Store chunk info
                chunk_info.append(
                    (chunk_path, chunk_transcription, chunk["start"], chunk["end"])
                )

                print(
                    f"  Chunk {i}: {chunk['duration']:.2f}s ({chunk['start']:.2f}s - {chunk['end']:.2f}s)"
                )
                print(
                    f"    Text: {chunk_transcription[:80]}..."
                    if len(chunk_transcription) > 80
                    else f"    Text: {chunk_transcription}"
                )

            print(f"\nCreated {len(chunk_info)} chunks in {CHUNKS_DIR}")
            return chunk_info

        except Exception as e:
            print(f"Chunking failed for {audio_path.name} - Error: {e}")
            return []


# TODO: Integrate into automated workflows
