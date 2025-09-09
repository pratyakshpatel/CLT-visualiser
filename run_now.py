#!/usr/bin/env python3

# Quick execution of the basic CLT demo
import sys
import os

# Add current directory to path
sys.path.insert(0, '/home/imjerusalem/CLT')

try:
    from basic_clt_demo import BasicCLTDemo
    
    print("🎯 CENTRAL LIMIT THEOREM DEMONSTRATION")
    print("="*50)
    print("Running a quick demo with Exponential distribution...")
    print("This shows how sample means become normal regardless of the original distribution shape!")
    
    demo = BasicCLTDemo()
    
    # Run a quick demo with exponential distribution (highly skewed)
    demo.run_clt_demo('exponential', sample_size=30, num_samples=1000, lam=1)
    
    print("\n" + "🎯 SAMPLE SIZE EFFECT")
    print("="*30)
    print("Showing how larger samples make the CLT work better...")
    
    for n in [10, 50, 100]:
        print(f"\n--- Sample Size: {n} ---")
        sample_means, stats = demo.run_clt_demo('exponential', sample_size=n, 
                                               num_samples=500, lam=1)
        expected_se = 1/n**0.5  # For exponential with lambda=1
        actual_se = stats['std_dev']
        print(f"Expected SE: {expected_se:.4f}, Actual SE: {actual_se:.4f}")
        print(f"Agreement: {(actual_se/expected_se)*100:.1f}%")

except ImportError as e:
    print(f"Import error: {e}")
    print("Running inline demo...")
    
    # Inline basic demo if import fails
    import random
    import math
    
    print("Quick CLT Demo - Exponential Distribution")
    print("="*40)
    
    # Generate 1000 samples of size 30 from exponential distribution
    sample_means = []
    for _ in range(1000):
        sample = [random.expovariate(1) for _ in range(30)]  # lambda = 1
        sample_means.append(sum(sample) / len(sample))
    
    # Calculate statistics
    mean_of_means = sum(sample_means) / len(sample_means)
    variance = sum((x - mean_of_means)**2 for x in sample_means) / len(sample_means)
    std_dev = math.sqrt(variance)
    
    print(f"Original distribution: Exponential (highly skewed)")
    print(f"Population mean: 1.0 (theoretical)")
    print(f"Population std: 1.0 (theoretical)")
    print(f"Expected standard error: {1/math.sqrt(30):.4f}")
    print(f"\nResults from 1000 samples of size 30:")
    print(f"Mean of sample means: {mean_of_means:.4f}")
    print(f"Standard error (observed): {std_dev:.4f}")
    print(f"Ratio (obs/theoretical): {std_dev/(1/math.sqrt(30)):.3f}")
    
    if 0.95 <= std_dev/(1/math.sqrt(30)) <= 1.05:
        print("✓ Excellent agreement with CLT!")
    else:
        print("~ Good demonstration of CLT")
    
    print(f"\n🎯 KEY INSIGHT:")
    print("Even though exponential distribution is highly skewed,")
    print("the sample means follow a normal distribution!")
    print("This is the power of the Central Limit Theorem.")

except Exception as e:
    print(f"Error: {e}")
    print("Please ensure you're in the correct directory and try again.")

print(f"\n{'='*50}")
print("CLT Demo Complete! 🎉")
print("The Central Limit Theorem works regardless of the original distribution shape!")
print("Try running 'python basic_clt_demo.py' for the full interactive demo.")