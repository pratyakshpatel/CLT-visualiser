#!/usr/bin/env python3
"""
interactive central limit theorem visualizer
requires: numpy, matplotlib, scipy, ipywidgets
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, expon, binom, poisson, beta, gamma, norm, chi2, lognorm
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings('ignore')

class CLTVisualizer:
    def __init__(self):
        self.setup_distributions()
        self.setup_widgets()
        self.setup_layout()
        
    def setup_distributions(self):
        # define all the distributions we can work with
        self.distributions = {
            'Uniform': {
                'func': uniform,
                'params': {
                    'loc': widgets.FloatSlider(value=0, min=-10, max=10, step=0.1, description='lower bound'),
                    'scale': widgets.FloatSlider(value=2, min=0.1, max=10, step=0.1, description='width')
                },
                'param_names': ['loc', 'scale']
            },
            'Exponential': {
                'func': expon,
                'params': {
                    'scale': widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description='Scale (λ⁻¹)')
                },
                'param_names': ['scale']
            },
            'Binomial': {
                'func': binom,
                'params': {
                    'n': widgets.IntSlider(value=10, min=1, max=50, step=1, description='Trials (n)'),
                    'p': widgets.FloatSlider(value=0.5, min=0.01, max=0.99, step=0.01, description='Probability (p)')
                },
                'param_names': ['n', 'p']
            },
            'Poisson': {
                'func': poisson,
                'params': {
                    'mu': widgets.FloatSlider(value=3, min=0.1, max=15, step=0.1, description='Rate (μ)')
                },
                'param_names': ['mu']
            },
            'Beta': {
                'func': beta,
                'params': {
                    'a': widgets.FloatSlider(value=2, min=0.1, max=10, step=0.1, description='Alpha (α)'),
                    'b': widgets.FloatSlider(value=5, min=0.1, max=10, step=0.1, description='Beta (β)')
                },
                'param_names': ['a', 'b']
            },
            'Gamma': {
                'func': gamma,
                'params': {
                    'a': widgets.FloatSlider(value=2, min=0.1, max=10, step=0.1, description='Shape (α)'),
                    'scale': widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description='Scale (β)')
                },
                'param_names': ['a', 'scale']
            },
            'Chi-Square': {
                'func': chi2,
                'params': {
                    'df': widgets.IntSlider(value=3, min=1, max=20, step=1, description='Degrees of freedom')
                },
                'param_names': ['df']
            },
            'Log-Normal': {
                'func': lognorm,
                'params': {
                    's': widgets.FloatSlider(value=1, min=0.1, max=3, step=0.1, description='Shape (σ)'),
                    'scale': widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description='Scale')
                },
                'param_names': ['s', 'scale']
            }
        }
    
    def setup_widgets(self):
        # setup all the interactive controls
        # distribution selector
        self.dist_selector = widgets.Dropdown(
            options=list(self.distributions.keys()),
            value='Uniform',
            description='distribution:',
            style={'description_width': 'initial'}
        )
        
        # clt parameters
        self.sample_size_slider = widgets.IntSlider(
            value=30, min=1, max=200, step=1,
            description='sample size (n):',
            style={'description_width': 'initial'}
        )
        
        self.num_samples_slider = widgets.IntSlider(
            value=1000, min=100, max=5000, step=100,
            description='number of samples:',
            style={'description_width': 'initial'}
        )
        
        # plotting options
        self.show_original = widgets.Checkbox(
            value=True,
            description='show original distribution',
            style={'description_width': 'initial'}
        )
        
        self.show_normal_overlay = widgets.Checkbox(
            value=True,
            description='show normal overlay',
            style={'description_width': 'initial'}
        )
        
        # parameter container (will be updated dynamically)
        self.param_container = widgets.VBox([])
        
        # update button
        self.update_button = widgets.Button(
            description='update plot',
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        
        # setup event handlers
        self.dist_selector.observe(self.on_distribution_change, names='value')
        self.update_button.on_click(self.update_plot)
        
        # auto-update when parameters change
        for param_widget in [self.sample_size_slider, self.num_samples_slider, 
                           self.show_original, self.show_normal_overlay]:
            param_widget.observe(self.auto_update, names='value')
    
    def setup_layout(self):
        """Setup the widget layout"""
        # Update parameter widgets for initial distribution
        self.update_parameter_widgets()
        
        # Create layout
        controls = widgets.VBox([
            widgets.HTML('<h3>CLT Visualizer Controls</h3>'),
            self.dist_selector,
            widgets.HTML('<h4>Distribution Parameters</h4>'),
            self.param_container,
            widgets.HTML('<h4>CLT Parameters</h4>'),
            self.sample_size_slider,
            self.num_samples_slider,
            widgets.HTML('<h4>Display Options</h4>'),
            self.show_original,
            self.show_normal_overlay,
            self.update_button
        ])
        
        self.layout = widgets.HBox([
            controls,
            widgets.Output(layout=widgets.Layout(width='70%'))
        ])
    
    def on_distribution_change(self, change):
        """Handle distribution selection change"""
        self.update_parameter_widgets()
        self.auto_update(None)
    
    def update_parameter_widgets(self):
        """Update parameter widgets based on selected distribution"""
        dist_name = self.dist_selector.value
        dist_info = self.distributions[dist_name]
        
        # Clear existing parameter widgets
        self.param_container.children = []
        
        # Add new parameter widgets
        param_widgets = []
        for param_name in dist_info['param_names']:
            widget = dist_info['params'][param_name]
            # Add auto-update observer
            widget.observe(self.auto_update, names='value')
            param_widgets.append(widget)
        
        self.param_container.children = param_widgets
    
    def auto_update(self, change):
        """Auto-update plot when parameters change"""
        self.update_plot(None)
    
    def get_current_params(self):
        """Get current parameter values for selected distribution"""
        dist_name = self.dist_selector.value
        dist_info = self.distributions[dist_name]
        
        params = {}
        for param_name in dist_info['param_names']:
            widget = dist_info['params'][param_name]
            params[param_name] = widget.value
        
        return params
    
    def generate_samples(self, dist_name, params, sample_size, num_samples):
        """Generate samples from the selected distribution"""
        dist_info = self.distributions[dist_name]
        dist_func = dist_info['func']
        
        # Generate samples based on distribution type
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
    
    def update_plot(self, button):
        """Update the visualization"""
        # Clear output and create new plot
        with self.layout.children[1]:
            clear_output(wait=True)
            
            # Get current parameters
            dist_name = self.dist_selector.value
            params = self.get_current_params()
            sample_size = self.sample_size_slider.value
            num_samples = self.num_samples_slider.value
            
            # Generate samples
            samples = self.generate_samples(dist_name, params, sample_size, num_samples)
            sample_means = np.mean(samples, axis=1)
            
            # Calculate theoretical statistics
            pop_mean, pop_var = self.calculate_theoretical_stats(dist_name, params)
            
            # CLT predictions
            clt_mean = pop_mean
            clt_std = np.sqrt(pop_var / sample_size)
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Original distribution (if requested)
            if self.show_original.value:
                ax1 = axes[0]
                
                # Generate data for original distribution
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
                    dist_info = self.distributions[dist_name]
                    dist_func = dist_info['func']
                    
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
                ax1.set_title(f'Original {dist_name} Distribution')
                ax1.set_xlabel('Value')
                ax1.set_ylabel('Probability Density/Mass')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                axes[0].text(0.5, 0.5, 'Original Distribution\nDisplay Disabled', 
                           ha='center', va='center', transform=axes[0].transAxes,
                           fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                axes[0].set_xticks([])
                axes[0].set_yticks([])
            
            # Plot 2: Distribution of sample means (CLT)
            ax2 = axes[1]
            
            # Histogram of sample means
            n_bins = min(50, len(np.unique(sample_means)))
            counts, bins, patches = ax2.hist(sample_means, bins=n_bins, density=True, 
                                           alpha=0.7, color='lightgreen', edgecolor='black',
                                           label=f'Sample Means (n={sample_size})')
            
            # Overlay theoretical normal distribution (if requested)
            if self.show_normal_overlay.value:
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
            
            # Add statistics text
            observed_std = np.std(sample_means)
            fig.suptitle(f'Central Limit Theorem Visualization\n' +
                        f'Theoretical σ_x̄ = {clt_std:.4f}, Observed σ_x̄ = {observed_std:.4f}',
                        fontsize=14, y=0.98)
            
            plt.tight_layout()
            plt.show()
            
            # Print summary statistics
            print(f"\n{'='*60}")
            print(f"DISTRIBUTION: {dist_name}")
            print(f"Parameters: {params}")
            print(f"Sample size (n): {sample_size}")
            print(f"Number of samples: {num_samples}")
            print(f"{'='*60}")
            print(f"Population mean (μ): {pop_mean:.4f}")
            print(f"Population std (σ): {np.sqrt(pop_var):.4f}")
            print(f"Theoretical sampling distribution mean: {clt_mean:.4f}")
            print(f"Theoretical sampling distribution std (σ/√n): {clt_std:.4f}")
            print(f"Observed sampling distribution mean: {np.mean(sample_means):.4f}")
            print(f"Observed sampling distribution std: {observed_std:.4f}")
            print(f"Difference in means: {abs(np.mean(sample_means) - clt_mean):.6f}")
            print(f"Difference in stds: {abs(observed_std - clt_std):.6f}")
    
    def display(self):
        # show the widget interface
        display(self.layout)
        # make initial plot
        self.update_plot(None)

# create and show the visualizer
if __name__ == "__main__":
    visualizer = CLTVisualizer()
    visualizer.display()