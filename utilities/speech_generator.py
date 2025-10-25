import re
import warnings

import torch
from kokoro import KPipeline

from utilities.util import get_device

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class SpeechGenerator:
    def __init__(self, target_text: str, other_text: str, device: str):
        self.device = device

        self.pipeline = KPipeline(
            lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device
        )

        self.target_text = target_text
        self.other_text = other_text

        # Preprocess and cache segments for both texts
        self.target_segments = self._preprocess_text(target_text)
        self.other_segments = self._preprocess_text(other_text)

    def _preprocess_text(self, text: str):
        split_pattern = r"\n+"
        text_segments = (
            re.split(split_pattern, text.strip()) if split_pattern else [text]
        )

        segments = []
        for graphemes_index, graphemes in enumerate(text_segments):
            if not graphemes.strip():
                continue

            # Do expensive g2p and tokenization once
            if self.pipeline.lang_code in "ab":  # English processing
                _, tokens = self.pipeline.g2p(graphemes)
                for gs, ps, tks in self.pipeline.en_tokenize(tokens):
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        ps = ps[:510]
                    segments.append((gs, ps, tks, graphemes_index))

        return segments

    def generate_audio(
        self, text: str, voice: torch.Tensor, speed: float = 1.0
    ) -> torch.Tensor:
        """Returns GPU tensor, optimized for both target and other text"""

        # Check which preprocessed segments to use
        if text == self.target_text:
            segments = self.target_segments
        elif text == self.other_text:
            segments = self.other_segments
        else:
            # Fallback for unexpected different text (shouldn't happen in practice)
            segments = self._preprocess_text(text)

        # use preprocessed segments with direct KPipeline.infer()
        audio_chunks = []
        for gs, ps, tks, text_index in segments:
            if not ps:
                continue

            output = KPipeline.infer(self.pipeline.model, ps, voice, speed)
            if output is not None and output.audio is not None:
                audio_chunks.append(output.audio.to(self.device))

        # Concatenate and return GPU tensor
        if audio_chunks:
            return torch.cat(audio_chunks, dim=0)
        else:
            return torch.tensor([], device=self.device)
