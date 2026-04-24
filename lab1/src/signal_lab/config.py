"""Shared configuration values for the laboratory notebook."""

TARGET_SR: int = 44_100
MAX_DURATION: float = 30.0
NOISE_DURATION: float = 30.0
SUBGROUP_NUMBER: int = 6
NOISE_KIND: str = "pink"
RANDOM_SEED: int = 42

# Moderate defaults keep notebook execution responsive while preserving enough
# frequency detail for speech, music, and transient samples.
N_FFT: int = 2048
HOP_LENGTH: int = 512
FRAME_LENGTH: int = 2048
