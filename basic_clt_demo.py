#!/usr/bin/env python3
"""
Basic CLT Demonstration without external libraries
Uses only Python standard library
"""

import random
import math
import statistics

class BasicCLTDemo:
    def __init__(self):
        self.random = random.Random()
    
    def uniform_sample(self, a=0, b=1):
        """Generate uniform random sample between a and b"""
        return self.random.uniform(a, b)
    
    def exponential_sample(self, lam=1):
        """Generate exponential random sample with rate lambda"""
        return self.random.expovariate(lam)
    
    def normal_sample(self, mu=0, sigma=1):
        """Generate normal random sample"""
        return self.random.normalvariate(mu, sigma)
    
    def binomial_sample(self, n=10, p=0.5):
        """Generate binomial random sample"""
        return sum(1 for _ in range(n) if self.random.random() < p)
    
    def generate_samples(self, distribution, sample_size, num_samples, **params):
        """Generate multiple samples and calculate their means"""
        sample_means = []
        
        for _ in range(num_samples):
            # Generate one sample of given size
            if distribution == 'uniform':
                sample = [self.uniform_sample(params.get('a', 0), params.get('b', 1)) 
                         for _ in range(sample_size)]
            elif distribution == 'exponential':
                sample = [self.exponential_sample(params.get('lam', 1)) 
                         for _ in range(sample_size)]
            elif distribution == 'binomial':
                sample = [self.binomial_sample(params.get('n', 10), params.get('p', 0.5)) 
                         for _ in range(sample_size)]
            elif distribution == 'normal':
                sample = [self.normal_sample(params.get('mu', 0), params.get('sigma', 1)) 
                         for _ in range(sample_size)]
            else:
                # Default to uniform
                sample = [self.uniform_sample(0, 1) for _ in range(sample_size)]
            
            # Calculate sample mean
            sample_means.append(sum(sample) / len(sample))
        
        return sample_means
    
    def calculate_statistics(self, data):
        """Calculate basic statistics for a dataset"""
        if not data:
            return {}
        
        n = len(data)
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / (n - 1) if n > 1 else 0
        std_dev = math.sqrt(variance)
        
        # Sort for quantiles
        sorted_data = sorted(data)
        
        return {
            'count': n,
            'mean': mean,
            'std_dev': std_dev,
            'variance': variance,
            'min': sorted_data[0],
            'max': sorted_data[-1],
            'median': sorted_data[n//2] if n % 2 == 1 else (sorted_data[n//2-1] + sorted_data[n//2]) / 2,
            'q25': sorted_data[n//4],
            'q75': sorted_data[3*n//4]
        }
    
    def create_histogram(self, data, bins=20):
        """Create a simple text-based histogram"""
        if not data:
            return "No data to plot"
        
        min_val = min(data)
        max_val = max(data)
        bin_width = (max_val - min_val) / bins
        
        # Create bins
        bin_counts = [0] * bins
        for value in data:
            bin_idx = min(int((value - min_val) / bin_width), bins - 1)
            bin_counts[bin_idx] += 1
        
        # Find max count for scaling
        max_count = max(bin_counts)
        scale = 50 / max_count if max_count > 0 else 1
        
        # Create histogram
        result = []
        for i, count in enumerate(bin_counts):
            bin_start = min_val + i * bin_width
            bin_end = min_val + (i + 1) * bin_width
            bar_length = int(count * scale)
            bar = '█' * bar_length
            result.append(f"{bin_start:6.2f}-{bin_end:6.2f} |{bar} ({count})")
        
        return '\n'.join(result)
    
    def theoretical_stats(self, distribution, sample_size, **params):
        """Calculate theoretical mean and standard error for sample means"""
        if distribution == 'uniform':
            a, b = params.get('a', 0), params.get('b', 1)
            pop_mean = (a + b) / 2
            pop_variance = (b - a) ** 2 / 12
        elif distribution == 'exponential':
            lam = params.get('lam', 1)
            pop_mean = 1 / lam
            pop_variance = 1 / (lam ** 2)
        elif distribution == 'binomial':
            n, p = params.get('n', 10), params.get('p', 0.5)
            pop_mean = n * p
            pop_variance = n * p * (1 - p)
        elif distribution == 'normal':
            pop_mean = params.get('mu', 0)
            pop_variance = params.get('sigma', 1) ** 2
        else:
            pop_mean, pop_variance = 0.5, 1/12  # Default uniform(0,1)
        
        # Standard error of the mean
        standard_error = math.sqrt(pop_variance / sample_size)
        
        return pop_mean, standard_error, pop_variance
    
    def run_clt_demo(self, distribution='uniform', sample_size=30, num_samples=1000, **params):
        """Run a complete CLT demonstration"""
        print(f"\n{'='*60}")
        print(f"CENTRAL LIMIT THEOREM DEMONSTRATION")
        print(f"Distribution: {distribution.upper()}")
        print(f"Parameters: {params}")
        print(f"Sample size: {sample_size}")
        print(f"Number of samples: {num_samples}")
        print(f"{'='*60}")
        
        # Generate sample means
        print("Generating samples...")
        sample_means = self.generate_samples(distribution, sample_size, num_samples, **params)
        
        # Calculate statistics
        stats = self.calculate_statistics(sample_means)
        
        # Theoretical predictions
        theoretical_mean, theoretical_se, pop_variance = self.theoretical_stats(
            distribution, sample_size, **params)
        
        # Display results
        print(f"\nTHEORETICAL PREDICTIONS (CLT):")
        print(f"Population mean: {theoretical_mean:.4f}")
        print(f"Population std dev: {math.sqrt(pop_variance):.4f}")
        print(f"Expected sample mean: {theoretical_mean:.4f}")
        print(f"Expected standard error: {theoretical_se:.4f}")
        
        print(f"\nOBSERVED RESULTS:")
        print(f"Sample mean of means: {stats['mean']:.4f}")
        print(f"Standard deviation of means: {stats['std_dev']:.4f}")
        print(f"Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"Median: {stats['median']:.4f}")
        
        print(f"\nCOMPARISON:")
        print(f"Mean difference: {abs(stats['mean'] - theoretical_mean):.6f}")
        print(f"Std error difference: {abs(stats['std_dev'] - theoretical_se):.6f}")
        
        # Show histogram
        print(f"\nHISTOGRAM OF SAMPLE MEANS:")
        print("Value Range    |Distribution")
        print("-" * 40)
        print(self.create_histogram(sample_means))
        
        # CLT Assessment
        ratio = stats['std_dev'] / theoretical_se
        print(f"\nCLT ASSESSMENT:")
        print(f"Observed/Theoretical SE ratio: {ratio:.3f}")
        if 0.95 <= ratio <= 1.05:
            print("✓ Excellent agreement with CLT!")
        elif 0.90 <= ratio <= 1.10:
            print("✓ Good agreement with CLT")
        elif 0.80 <= ratio <= 1.20:
            print("~ Reasonable agreement with CLT")
        else:
            print("✗ Poor agreement - may need larger sample size")
        
        return sample_means, stats

def main():
    """Main demonstration function"""
    demo = BasicCLTDemo()
    
    print("CENTRAL LIMIT THEOREM VISUALIZER")
    print("Basic Version (No External Dependencies)")
    print("="*50)
    
    demos = [
        ("Uniform Distribution", 'uniform', {'a': 0, 'b': 4}),
        ("Exponential Distribution", 'exponential', {'lam': 0.5}),
        ("Binomial Distribution", 'binomial', {'n': 20, 'p': 0.3}),
        ("Normal Distribution", 'normal', {'mu': 10, 'sigma': 2})
    ]
    
    for name, dist, params in demos:
        print(f"\n{name}")
        print("-" * len(name))
        demo.run_clt_demo(dist, sample_size=30, num_samples=1000, **params)
        
        response = input("\nPress Enter to continue to next demo (or 'q' to quit): ")
        if response.lower() == 'q':
            break
    
    print("\n" + "="*50)
    print("SAMPLE SIZE EFFECT DEMONSTRATION")
    print("="*50)
    print("Comparing different sample sizes with Exponential distribution...")
    
    for n in [5, 15, 30, 100]:
        print(f"\n--- Sample Size: {n} ---")
        sample_means, stats = demo.run_clt_demo('exponential', sample_size=n, 
                                               num_samples=500, lam=1)
        print(f"Standard error should approach: {1/math.sqrt(n):.4f}")
        print(f"Observed standard error: {stats['std_dev']:.4f}")

if __name__ == "__main__":
    main()