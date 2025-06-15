import tempfile
import warnings

import torch
from kokoro import KPipeline


class SpeechGenerator:
    def __init__(self):
        surpressWarnings()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"DEBUG: Initializing pipeline with device: {device}")

        self.pipeline = KPipeline(lang_code="a", repo_id='hexgrad/Kokoro-82M', device=device)

        # Verify model is actually on GPU
        if self.pipeline.model:
            print(f"DEBUG: Pipeline model device: {self.pipeline.model.device}")
            print(f"DEBUG: Model parameters device: {next(self.pipeline.model.parameters()).device}")

    def generate_audio(self, text: str, voice: torch.Tensor, speed: float = 1.0) -> torch.Tensor:
        """Returns GPU tensor instead of numpy array"""

        # Ensure voice is on GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        with tempfile.NamedTemporaryFile(suffix='.pt', delete=True) as temp_file:
            voice = voice.to(device)
            torch.save(voice, temp_file.name)

            generator = self.pipeline(text, temp_file.name, speed)

            # Clean up the cached voice using the temp filename
            if temp_file.name in self.pipeline.voices:
                del self.pipeline.voices[temp_file.name]

            audio_chunks = []

            for i, (gs, ps, chunk) in enumerate(generator):
                if chunk is not None:
                    # Keep everything on GPU
                    audio_chunks.append(chunk.to(device))

            # Clean up GPU memory
            del generator  # This might hold references to GPU tensors
            del voice  # Remove local copy

            # Concatenate and return GPU tensor
            if audio_chunks:
                return torch.cat(audio_chunks, dim=0)
            else:
                return torch.tensor([], device=device)

def surpressWarnings():
    # Surpress all these warnings showing up from libraries cluttering the console
    # warnings.filterwarnings(
    #     "ignore",
    #     message=".*RNN module weights are not part of single contiguous chunk of memory.*",
    #     category=UserWarning,
    # )
    warnings.filterwarnings(
        "ignore", message=".*is deprecated in favor of*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*dropout option adds dropout after all but last recurrent layer*",
        category=UserWarning,
    )
