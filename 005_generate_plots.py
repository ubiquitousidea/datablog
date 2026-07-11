import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def main():
    # 1. Generate toy dataset
    # Seed for reproducibility
    np.random.seed(42)
    
    # x in the positive domain
    x = np.random.uniform(0.5, 10.0, 80)
    x = np.sort(x)
    
    # y generated from a quadratic function with IID Gaussian noise
    # Quadratic function: y = 0.5 * x^2 - 1.5 * x + 5
    # Noise: IID Gaussian noise with mean 0 and std 3.0
    true_y = 0.5 * (x ** 2) - 1.5 * x + 5.0
    noise = np.random.normal(0.0, 2.5, size=len(x))
    y = true_y + noise
    
    # 2. Fits
    # Linear Fit
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(0.5, 10.0, 200)
    y_linear_fit = slope * x_fit + intercept
    
    # Quadratic Fit
    coeffs = np.polyfit(x, y, 2)
    y_quadratic_fit = coeffs[0] * (x_fit ** 2) + coeffs[1] * x_fit + coeffs[2]
    
    # Style helper function to avoid repeating code
    def apply_style(ax):
        # White plot background
        ax.set_facecolor('white')
        ax.figure.patch.set_facecolor('white')
        
        # Light gray axis lines (spines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#cccccc')
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
        # Style tick lines and tick labels
        ax.tick_params(axis='both', which='both', colors='#888888', width=1.5)
        
        # Simple axis titles like "x" and "y"
        ax.set_xlabel('x', fontsize=12, color='#333333', labelpad=8)
        ax.set_ylabel('y', fontsize=12, color='#333333', labelpad=8)
        
        # Set background to white explicitly
        ax.grid(False)
    
    # --- Plot 1: Linear Regression Fit ---
    fig1, ax1 = plt.subplots(figsize=(6, 4.5), dpi=150)
    
    # Blue semi-transparent circular points for the data
    ax1.scatter(x, y, color='blue', alpha=0.5, marker='o', s=35, label='Data')
    
    # Orange linear regression line
    ax1.plot(x_fit, y_linear_fit, color='orange', linewidth=2.5, label='Linear Fit')
    
    apply_style(ax1)
    ax1.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0')
    fig1.tight_layout()
    
    # Save Plot 1
    plot1_path = '005_linear_fit.png'
    fig1.savefig(plot1_path, facecolor='white', edgecolor='none')
    plt.close(fig1)
    print(f"Saved linear regression plot to {plot1_path}")
    
    # --- Plot 2: Quadratic Regression Fit ---
    fig2, ax2 = plt.subplots(figsize=(6, 4.5), dpi=150)
    
    # Blue semi-transparent circular points for the data
    ax2.scatter(x, y, color='blue', alpha=0.5, marker='o', s=35, label='Data')
    
    # Green quadratic fit line
    ax2.plot(x_fit, y_quadratic_fit, color='green', linewidth=2.5, label='Quadratic Fit')
    
    apply_style(ax2)
    ax2.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0')
    fig2.tight_layout()
    
    # Save Plot 2
    plot2_path = '005_quadratic_fit.png'
    fig2.savefig(plot2_path, facecolor='white', edgecolor='none')
    plt.close(fig2)
    print(f"Saved quadratic regression plot to {plot2_path}")

    # Copy files to artifacts directory so they can be embedded/viewed
    artifact_dir = os.environ.get('ANTIGRAVITY_ARTIFACT_DIR', '')
    if artifact_dir and os.path.exists(artifact_dir):
        shutil.copy(plot1_path, os.path.join(artifact_dir, '005_linear_fit.png'))
        shutil.copy(plot2_path, os.path.join(artifact_dir, '005_quadratic_fit.png'))
        print("Copied plots to artifacts directory.")

if __name__ == '__main__':
    main()
