#!/usr/bin/env python3
"""
central limit theorem demonstration
alternative version with same functionality as main.py
"""

import random
import math

print("central limit theorem demonstration")
print("showing how sample means become normal regardless of original distribution")

# set seed so we get the same results each time
random.seed(42)

# let's take 1000 samples of size 30 from an exponential distribution
print("generating 1000 samples of size 30...")
sample_means = []

for i in range(1000):
    # make one sample of 30 numbers from exponential distribution
    sample = [random.expovariate(1) for _ in range(30)]
    sample_mean = sum(sample) / len(sample)
    sample_means.append(sample_mean)

# calculate some basic stats
mean_of_means = sum(sample_means) / len(sample_means)
variance = sum((x - mean_of_means)**2 for x in sample_means) / (len(sample_means) - 1)
std_dev = math.sqrt(variance)

# what we expect theoretically
pop_mean = 1.0  # exponential with rate 1 has mean 1
pop_std = 1.0   # and standard deviation 1
expected_se = pop_std / math.sqrt(30)  # standard error should be this

print("results:")
print("original distribution: exponential (right-skewed)")
print(f"population mean: {pop_mean:.4f}")
print(f"population std: {pop_std:.4f}")
print(f"expected std error: {expected_se:.4f}")

print("observed from 1000 samples:")
print(f"mean of sample means: {mean_of_means:.4f}")
print(f"std error: {std_dev:.4f}")
print(f"range: {min(sample_means):.4f} to {max(sample_means):.4f}")

# how well does it match theory?
mean_diff = abs(mean_of_means - pop_mean)
se_ratio = std_dev / expected_se

print(f"agreement with theory:")
print(f"mean difference: {mean_diff:.6f}")
print(f"std error ratio: {se_ratio:.4f}")

if 0.95 <= se_ratio <= 1.05:
    print("excellent match!")
elif 0.90 <= se_ratio <= 1.10:
    print("good match!")
else:
    print("reasonable match")

# simple histogram
print("histogram of sample means:")
min_val = min(sample_means)
max_val = max(sample_means)
num_bins = 15
bin_width = (max_val - min_val) / num_bins

bins = [0] * num_bins
for value in sample_means:
    bin_idx = min(int((value - min_val) / bin_width), num_bins - 1)
    bins[bin_idx] += 1

max_count = max(bins)
scale = 40 / max_count

for i, count in enumerate(bins):
    bin_start = min_val + i * bin_width
    bin_end = min_val + (i + 1) * bin_width
    bar_length = int(count * scale)
    bar = '█' * bar_length
    print(f"{bin_start:5.2f}-{bin_end:5.2f} |{bar} ({count})")

print("what this shows:")
print("- exponential distribution is very skewed")
print("- but sample means form a nice bell curve!")
print("- this is the central limit theorem in action")

# quick demo of sample size effect
print("effect of different sample sizes:")
for n in [5, 15, 30, 60]:
    quick_means = []
    for _ in range(200):
        sample = [random.expovariate(1) for _ in range(n)]
        quick_means.append(sum(sample) / len(sample))
    
    quick_se = math.sqrt(sum((x - 1)**2 for x in quick_means) / len(quick_means))
    expected_quick_se = 1 / math.sqrt(n)
    
    print(f"n={n:2d}: expected={expected_quick_se:.4f}, observed={quick_se:.4f}")

print("notice: bigger samples = smaller standard error")