import torch

from utilities.pytorch_sanitizer import load_voice_safely


class VoiceGenerator:
    def __init__(self, voices: list[torch.Tensor], starting_voice: str | None):
        self.voices = voices

        self.stacked = torch.stack(voices,dim=0)
        self.mean = self.stacked.mean(dim=0)
        self.std = self.stacked.std(dim=0)
        self.min = self.stacked.min(dim=0)[0]
        self.max = self.stacked.max(dim=0)[0]

        if starting_voice:
            # TODO: Remove
            print('starting_voice = true')
            self.starting_voice = load_voice_safely(starting_voice)
        else:
            # TODO: Remove
            print('starting_voice = false')
            self.starting_voice = self.mean

    def generate_voice(self, base_tensor: torch.Tensor | None, diversity: float = 1.0, device: str = "cpu",
                       clip: bool = False):
        """Generate a new voice tensor based on the base_tensor and diversity.

        Args:
            base_tensor (torch.Tensor | None): The base tensor to generate the new voice from.
            diversity (float, optional): The diversity of the new voice. Defaults to 1.0.
            # device (str, optional): The device to generate the new voice on. Defaults to "cpu".
            clip (bool, optional): Whether to clip the new voice to the min and max values. Defaults to False.

        Returns:
            torch.Tensor: The new voice tensor.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if base_tensor is None:
            # TODO: Remove
            print('base tensor = none')
            try:
                base_tensor = self.mean.to(device=device)
            except:
                base_tensor = self.mean.to(device="cpu")
        else:
            # TODO: Remove
            print('base tensor = not none')
            try:
                base_tensor = base_tensor.clone().to(device)
            except:
                base_tensor = base_tensor.clone().to("cpu")
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"device: {device}")
        #
        # if base_tensor is None:
        #     # TODO: Remove
        #     print('base tensor = none')
        #     try:
        #         base_tensor = self.mean.to(device)
        #     except:
        #         base_tensor = self.mean.to("cpu")
        # else:
        #     # TODO: Remove
        #     print('base tensor = not none')
        #     try:
        #         base_tensor = base_tensor.clone().to(device)
        #     except:
        #         base_tensor = base_tensor.clone().to("cpu")

        # Generate random noise with same shape
        # noise = torch.randn_like(base_tensor, device=device)
        # try:
        #     noise = torch.randn_like(base_tensor, device='cuda')
        #     # TODO: Remove
        #     print("gpu enabled voice gen")
        # except:
        #     print("gpu enabled voice gen failed")
        #     noise = torch.randn_like(base_tensor, device='cpu')
        #     # TODO: Remove
        #     print("cpu enabled voice gen")
        noise = torch.randn_like(base_tensor, device='cuda:0')
        torch.save(noise, 'gpu_noise_tensor.pt')
        # print(f"noise {noise.shape}; {noise.dtype}")
        # noise = torch.randn_like(base_tensor, device='cpu')
        # torch.save(noise, 'cpu_noise_tensor.pt')
        print(f"noise {noise.shape}; {noise.dtype}")
        # TODO: Remove
        print("cpu enabled voice gen")

        # Scale noise by standard deviation and the noise_scale factor
        scaled_noise = noise * self.std.to(device) * diversity

        # Add scaled noise to base tensor
        new_tensor = base_tensor + scaled_noise

        if clip:
            new_tensor = torch.clamp(new_tensor, self.min, self.max)

        return new_tensor

    # TODO: Make more voice generation functions
