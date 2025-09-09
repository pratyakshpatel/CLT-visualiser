import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, expon, binom, poisson, beta, gamma, norm, chi2, lognorm
import warnings
warnings.filterwarnings('ignore')

class SimpleCLTVisualizer:
    def __init__(self):
        self.distributions = self.setup_distributions()
    
    def setup_distributions(self):
        """Define available distributions with default parameters"""
        return {
            'Uniform': {'func': uniform, 'params': {'loc': 0, 'scale': 2}},
            'Exponential': {'func': expon, 'params': {'scale': 1}},
            'Binomial': {'func': binom, 'params': {'n': 10, 'p': 0.5}},
            'Poisson': {'func': poisson, 'params': {'mu': 3}},
            'Beta': {'func': beta, 'params': {'a': 2, 'b': 5}},
            'Gamma': {'func': gamma, 'params': {'a': 2, 'scale': 1}},
            'Chi-Square': {'func': chi2, 'params': {'df': 3}},
            'Log-Normal': {'func': lognorm, 'params': {'s': 1, 'scale': 1}}
        }
    
    def generate_samples(self, dist_name, params, sample_size, num_samples):
        """Generate samples from the selected distribution"""
        dist_func = self.distributions[dist_name]['func']
        
        if dist_name == 'Binomial':
            samples = dist_func.rvs(n=params['n'], p=params['p'], 
                                   size=(num_samples, sample_size))
        elif dist_name == 'Poisson':
            samples = dist_func.rvs(mu=params['mu'], 
                                   size=(num_samples, sample_size))
        elif dist_name == 'Beta':
            samples = dist_func.rvs(a=params['a'], b=params['b'], 
                                   size=(num_samples, sample_size))
        elif dist_name == 'Gamma':
            samples = dist_func.rvs(a=params['a'], scale=params['scale'], 
                                   size=(num_samples, sample_size))
        elif dist_name == 'Uniform':
            samples = dist_func.rvs(loc=params['loc'], scale=params['scale'], 
                                   size=(num_samples, sample_size))
        elif dist_name == 'Exponential':
            samples = dist_func.rvs(scale=params['scale'], 
                                   size=(num_samples, sample_size))
        elif dist_name == 'Chi-Square':
            samples = dist_func.rvs(df=params['df'], 
                                   size=(num_samples, sample_size))
        elif dist_name == 'Log-Normal':
            samples = dist_func.rvs(s=params['s'], scale=params['scale'], 
                                   size=(num_samples, sample_size))
        
        return samples
    
    def calculate_theoretical_stats(self, dist_name, params):
        """Calculate theoretical mean and variance"""
        if dist_name == 'Uniform':
            mean = params['loc'] + params['scale'] / 2
            var = params['scale'] ** 2 / 12
        elif dist_name == 'Exponential':
            mean = params['scale']
            var = params['scale'] ** 2
        elif dist_name == 'Binomial':
            mean = params['n'] * params['p']
            var = params['n'] * params['p'] * (1 - params['p'])
        elif dist_name == 'Poisson':
            mean = params['mu']
            var = params['mu']
        elif dist_name == 'Beta':
            a, b = params['a'], params['b']
            mean = a / (a + b)
            var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        elif dist_name == 'Gamma':
            mean = params['a'] * params['scale']
            var = params['a'] * params['scale'] ** 2
        elif dist_name == 'Chi-Square':
            mean = params['df']
            var = 2 * params['df']
        elif dist_name == 'Log-Normal':
            s = params['s']
            mean = params['scale'] * np.exp(s**2 / 2)
            var = (params['scale']**2) * (np.exp(s**2) - 1) * np.exp(s**2)
        
        return mean, var
    
    def plot_clt_demo(self, dist_name, sample_size=30, num_samples=1000):
        """Create CLT demonstration plot"""
        params = self.distributions[dist_name]['params']
        
        # Generate samples
        samples = self.generate_samples(dist_name, params, sample_size, num_samples)
        sample_means = np.mean(samples, axis=1)
        
        # Calculate theoretical statistics
        pop_mean, pop_var = self.calculate_theoretical_stats(dist_name, params)
        clt_mean = pop_mean
        clt_std = np.sqrt(pop_var / sample_size)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Original distribution
        ax1 = axes[0]
        dist_func = self.distributions[dist_name]['func']
        
        if dist_name in ['Binomial', 'Poisson']:
            # Discrete distributions
            if dist_name == 'Binomial':
                x = np.arange(0, params['n'] + 1)
                y = binom.pmf(x, n=params['n'], p=params['p'])
            else:  # Poisson
                x = np.arange(0, int(params['mu'] * 3) + 1)
                y = poisson.pmf(x, mu=params['mu'])
            
            ax1.bar(x, y, alpha=0.7, color='skyblue', edgecolor='black')
        else:
            # Continuous distributions
            if dist_name == 'Beta':
                x = np.linspace(0, 1, 1000)
                y = dist_func.pdf(x, a=params['a'], b=params['b'])
            elif dist_name == 'Gamma':
                x = np.linspace(0, pop_mean + 4*np.sqrt(pop_var), 1000)
                y = dist_func.pdf(x, a=params['a'], scale=params['scale'])
            elif dist_name == 'Uniform':
                x = np.linspace(params['loc'] - 1, params['loc'] + params['scale'] + 1, 1000)
                y = dist_func.pdf(x, loc=params['loc'], scale=params['scale'])
            elif dist_name == 'Exponential':
                x = np.linspace(0, pop_mean + 4*np.sqrt(pop_var), 1000)
                y = dist_func.pdf(x, scale=params['scale'])
            elif dist_name == 'Chi-Square':
                x = np.linspace(0, pop_mean + 4*np.sqrt(pop_var), 1000)
                y = dist_func.pdf(x, df=params['df'])
            elif dist_name == 'Log-Normal':
                x = np.linspace(0.01, pop_mean + 4*np.sqrt(pop_var), 1000)
                y = dist_func.pdf(x, s=params['s'], scale=params['scale'])
            
            ax1.plot(x, y, 'b-', linewidth=2, label=f'{dist_name} Distribution')
            ax1.fill_between(x, y, alpha=0.3, color='skyblue')
        
        ax1.axvline(pop_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'True Mean = {pop_mean:.3f}')
        ax1.set_title(f'Original {dist_name} Distribution\nParameters: {params}')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Probability Density/Mass')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of sample means
        ax2 = axes[1]
        
        # Histogram of sample means
        n_bins = min(50, len(np.unique(sample_means)))
        ax2.hist(sample_means, bins=n_bins, density=True, alpha=0.7, 
                color='lightgreen', edgecolor='black',
                label=f'Sample Means (n={sample_size})')
        
        # Overlay theoretical normal distribution
        x_norm = np.linspace(sample_means.min(), sample_means.max(), 1000)
        y_norm = norm.pdf(x_norm, loc=clt_mean, scale=clt_std)
        ax2.plot(x_norm, y_norm, 'r-', linewidth=3, 
                label=f'Normal(μ={clt_mean:.3f}, σ={clt_std:.3f})')
        
        # Add vertical lines for means
        ax2.axvline(np.mean(sample_means), color='blue', linestyle='-', linewidth=2,
                   label=f'Observed Mean = {np.mean(sample_means):.3f}')
        ax2.axvline(clt_mean, color='red', linestyle='--', linewidth=2,
                   label=f'Theoretical Mean = {clt_mean:.3f}')
        
        ax2.set_title(f'Distribution of Sample Means\n(CLT with {num_samples} samples)')
        ax2.set_xlabel('Sample Mean')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        observed_std = np.std(sample_means)
        fig.suptitle(f'Central Limit Theorem Visualization\n' +
                    f'Theoretical σ_x̄ = {clt_std:.4f}, Observed σ_x̄ = {observed_std:.4f}',
                    fontsize=14, y=0.98)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"DISTRIBUTION: {dist_name}")
        print(f"Parameters: {params}")
        print(f"Sample size (n): {sample_size}")
        print(f"Number of samples: {num_samples}")
        print(f"{'='*60}")
        print(f"Population mean (μ): {pop_mean:.4f}")
        print(f"Population std (σ): {np.sqrt(pop_var):.4f}")
        print(f"Theoretical sampling mean: {clt_mean:.4f}")
        print(f"Theoretical sampling std (σ/√n): {clt_std:.4f}")
        print(f"Observed sampling mean: {np.mean(sample_means):.4f}")
        print(f"Observed sampling std: {observed_std:.4f}")
        print(f"Difference in means: {abs(np.mean(sample_means) - clt_mean):.6f}")
        print(f"Difference in stds: {abs(observed_std - clt_std):.6f}")
    
    def demo_all_distributions(self):
        """Run demo for all distributions"""
        print("Central Limit Theorem Demonstration")
        print("="*50)
        
        for dist_name in self.distributions.keys():
            print(f"\nTesting {dist_name} distribution...")
            self.plot_clt_demo(dist_name)
            input("Press Enter to continue to next distribution...")
    
    def demo_sample_size_effect(self, dist_name='Exponential'):
        """Demonstrate effect of sample size"""
        sample_sizes = [5, 15, 30, 100]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        params = self.distributions[dist_name]['params']
        pop_mean, pop_var = self.calculate_theoretical_stats(dist_name, params)
        
        for i, n in enumerate(sample_sizes):
            # Generate samples
            samples = self.generate_samples(dist_name, params, n, 1000)
            sample_means = np.mean(samples, axis=1)
            
            # Plot histogram
            axes[i].hist(sample_means, bins=30, density=True, alpha=0.7,
                        color='lightgreen', edgecolor='black')
            
            # Theoretical normal overlay
            clt_std = np.sqrt(pop_var / n)
            x_norm = np.linspace(sample_means.min(), sample_means.max(), 1000)
            y_norm = norm.pdf(x_norm, loc=pop_mean, scale=clt_std)
            axes[i].plot(x_norm, y_norm, 'r-', linewidth=2)
            
            axes[i].set_title(f'Sample Size n = {n}\nσ_x̄ = {clt_std:.3f}')
            axes[i].set_xlabel('Sample Mean')
            axes[i].set_ylabel('Density')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(f'Effect of Sample Size on CLT\n{dist_name} Distribution', fontsize=16)
        plt.tight_layout()
        plt.show()

# Run the demonstration
if __name__ == "__main__":
    visualizer = SimpleCLTVisualizer()
    
    print("CLT Visualizer - Simple Version")
    print("="*40)
    print("Available distributions:")
    for i, dist in enumerate(visualizer.distributions.keys(), 1):
        print(f"{i}. {dist}")
    
    print("\nDemonstration Options:")
    print("1. Single distribution demo")
    print("2. All distributions demo")
    print("3. Sample size effect demo")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        print("\nSelect distribution:")
        dist_list = list(visualizer.distributions.keys())
        for i, dist in enumerate(dist_list, 1):
            print(f"{i}. {dist}")
        
        try:
            dist_choice = int(input("Enter number: ")) - 1
            if 0 <= dist_choice < len(dist_list):
                selected_dist = dist_list[dist_choice]
                print(f"\nRunning CLT demo for {selected_dist}...")
                visualizer.plot_clt_demo(selected_dist)
            else:
                print("Invalid choice, using Uniform distribution")
                visualizer.plot_clt_demo('Uniform')
        except:
            print("Invalid input, using Uniform distribution")
            visualizer.plot_clt_demo('Uniform')
    
    elif choice == "2":
        visualizer.demo_all_distributions()
    
    elif choice == "3":
        print("\nSelect distribution for sample size demo:")
        dist_list = list(visualizer.distributions.keys())
        for i, dist in enumerate(dist_list, 1):
            print(f"{i}. {dist}")
        
        try:
            dist_choice = int(input("Enter number: ")) - 1
            if 0 <= dist_choice < len(dist_list):
                selected_dist = dist_list[dist_choice]
            else:
                selected_dist = 'Exponential'
        except:
            selected_dist = 'Exponential'
        
        print(f"\nRunning sample size effect demo for {selected_dist}...")
        visualizer.demo_sample_size_effect(selected_dist)
    
    else:
        print("Invalid choice, running default demo...")
        visualizer.plot_clt_demo('Uniform')