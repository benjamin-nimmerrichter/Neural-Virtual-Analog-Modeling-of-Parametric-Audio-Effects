import os
import torch
import torchaudio
import random
from torch.utils.data import Dataset
from tqdm import tqdm

class AudioPotDataset(Dataset):
    """
    Custom Dataset for loading paired audio files (input vs. processed)
    with a conditional parameter (potentiometer value).
    """
    def __init__(self, root_dir, segment_len=16384, taps_per_file=256):
        """
        Args:
            root_dir (str): Path to the folder containing .wav files.
            segment_len (int): Length of the audio chunk to be used for training.
            taps_per_file (int): How many random crops to take from one file per epoch.
        """
        self.segment_len = segment_len
        self.taps_per_file = taps_per_file
        self.data_cache = [] # Stores audio in RAM to avoid slow disk I/O during training

        # Filter and sort files to ensure deterministic pairing
        all_files = [f for f in os.listdir(root_dir) if f.endswith('.wav')]
        inputs = sorted([f for f in all_files if 'input' in f])

        print(f"Loading {len(inputs)} files into RAM...")
        for in_file in tqdm(inputs):
            # Identify matching output files based on the prefix (e.g., 'recording1_')
            prefix = in_file.split('input')[0]
            pots = [f for f in all_files if f.startswith(prefix) and 'pot' in f]

            # Load input file and force to mono [1, L]
            x_raw, _ = torchaudio.load(os.path.join(root_dir, in_file))
            if x_raw.shape[0] > 1:
                x_raw = x_raw.mean(dim=0, keepdim=True) # Downmix stereo to mono
            elif x_raw.dim() == 1:
                x_raw = x_raw.unsqueeze(0)

            # Match each input with all its corresponding 'pot' variations
            for pot_file in pots:
                y_raw, _ = torchaudio.load(os.path.join(root_dir, pot_file))
                if y_raw.shape[0] > 1:
                    y_raw = y_raw.mean(dim=0, keepdim=True)
                elif y_raw.dim() == 1:
                    y_raw = y_raw.unsqueeze(0)

                # Parse the 'pot' value from filename (e.g., 'pot07.wav' -> 0.7)
                try:
                    param_val = float(pot_file.split('pot')[-1].replace('.wav', '')) / 10.0
                except:
                    param_val = 0.5 # Default fallback

                self.data_cache.append({
                    'x': x_raw,
                    'y': y_raw,
                    'p': param_val
                })

    def __len__(self):
        """Total number of samples is defined by number of files multiplied by taps per file."""
        return len(self.data_cache) * self.taps_per_file

    def __getitem__(self, idx):
        """
        Retrieves a random segment of audio from the cached files.
        Includes a 'silence detector' to ensure the model learns from actual signal.
        """
        # Map global index to specific file in cache
        file_idx = idx // self.taps_per_file
        file_data = self.data_cache[file_idx]

        x, y = file_data['x'], file_data['y']

        # Determine the shortest length to avoid indexing errors
        min_len = min(x.shape[1], y.shape[1])

        # Define the center-point buffer
        half_req = self.segment_len // 2

        # Error handling for short files
        if min_len < self.segment_len:
            raise ValueError(
                f"Critical Error: File is too short ({min_len} samples). "
                f"TCN requires at least {self.segment_len} samples."
            )

        silence_threshold = 1e-3
        max_retries = 20

        # Attempt to find a segment that isn't silent
        for attempt in range(max_retries):
            # 1. Select a random point safely away from edges to allow for context/receptive field
            center = random.randint(half_req, min_len - half_req)

            # 2. Slice the segment around the center
            start = center - half_req
            end = start + self.segment_len

            x_seg = x[:, start:end]
            y_seg = y[:, start:end]

            # 3. Check peak amplitude of the target (y) to avoid training on empty noise
            peak_amplitude = torch.max(torch.abs(y_seg))

            # If the segment has enough energy, return it
            if peak_amplitude > silence_threshold:
                return x_seg.float(), y_seg.float(), torch.tensor([file_data['p']], dtype=torch.float32)

        # Fallback: If no loud chunk is found after all retries, return the last slice
        return x_seg.float(), y_seg.float(), torch.tensor([file_data['p']], dtype=torch.float32)
