"""Test the sampling module."""
from utils.sampling import compute_uniform_timestamps, FrameSampler
from utils.config import load_config

# Test timestamp computation
print('Testing Uniform-K timestamp computation...')
print()

for duration in [5.0, 10.0, 30.0]:
    timestamps = compute_uniform_timestamps(duration, k=10, epsilon=0.1, min_gap=0.5)
    print(f'Duration: {duration}s, K=10')
    print(f'  Timestamps: {[f"{t:.2f}" for t in timestamps]}')
    if len(timestamps) > 1:
        gaps = [timestamps[i+1]-timestamps[i] for i in range(len(timestamps)-1)]
        print(f'  Gaps: min={min(gaps):.2f}s max={max(gaps):.2f}s')
    print()

# Test from config
print('Loading sampler from config...')
config = load_config()
sampler = FrameSampler.from_config(config)
print(f'  frames_train_val: {sampler.frames_train_val}')
print(f'  frames_test: {sampler.frames_test}')
print(f'  max_per_video: {sampler.max_per_video}')
print(f'  max_per_group: {sampler.max_per_group}')
print()
print('Sampling module ready!')
