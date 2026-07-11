import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def main():
    # Set seed for reproducibility
    np.random.seed(10)
    
    # 1. True function and data generation
    # f(x) = a0 + a1*cos(2pi*x/10) + b1*sin(2pi*x/10) + a2*cos(4pi*x/10) + b2*sin(4pi*x/10)
    period = 10.0
    omega = 2.0 * np.pi / period
    
    # Coefficients
    a0 = 2.5
    a1, b1 = 1.2, -0.8
    a2, b2 = -0.5, 0.6
    
    def true_func(x):
        return (a0 + 
                a1 * np.cos(omega * x) + b1 * np.sin(omega * x) + 
                a2 * np.cos(2.0 * omega * x) + b2 * np.sin(2.0 * omega * x))
                
    # Training data range is [0, 10] (one full period)
    n_samples = 45
    x_train = np.random.uniform(0.0, period, n_samples)
    y_train = true_func(x_train) + np.random.normal(0, 0.35, size=n_samples)
    
    # Grid for plotting (extending from -5 to 15, showing extrapolation)
    x_grid = np.linspace(-5.0, 15.0, 400)
    y_true = true_func(x_grid)
    
    # 2. Cubic Polynomial Fit
    # Fit on training data
    poly_coeffs = np.polyfit(x_train, y_train, 3)
    y_poly_fit = np.polyval(poly_coeffs, x_grid)
    
    # 3. Fourier Regression Fit (Degree 2)
    # Feature matrix for training
    X_train_fourier = np.column_stack([
        np.ones_like(x_train),
        np.cos(omega * x_train),
        np.sin(omega * x_train),
        np.cos(2.0 * omega * x_train),
        np.sin(2.0 * omega * x_train)
    ])
    
    # Least squares solver
    fourier_coeffs, _, _, _ = np.linalg.lstsq(X_train_fourier, y_train, rcond=None)
    
    # Predict on grid
    X_grid_fourier = np.column_stack([
        np.ones_like(x_grid),
        np.cos(omega * x_grid),
        np.sin(omega * x_grid),
        np.cos(2.0 * omega * x_grid),
        np.sin(2.0 * omega * x_grid)
    ])
    y_fourier_fit = X_grid_fourier @ fourier_coeffs
    
    # Helper to style plots
    def style_ax(ax):
        ax.set_facecolor('white')
        ax.figure.patch.set_facecolor('white')
        
        # Light gray axis lines (spines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#d1d5db')
        ax.spines['left'].set_color('#d1d5db')
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
        # Style tick lines
        ax.tick_params(axis='both', which='both', colors='#6b7280', width=1.5)
        
        # Simple axis titles
        ax.set_xlabel('x', fontsize=12, color='#374151', labelpad=6)
        ax.set_ylabel('y', fontsize=12, color='#374151', labelpad=6)
        ax.grid(False)
        
        # Demarcate training range [0, 10] with vertical dashed lines and shading
        ax.axvline(0, color='#9ca3af', linestyle=':', linewidth=1.2)
        ax.axvline(10, color='#9ca3af', linestyle=':', linewidth=1.2)
        ax.axvspan(0, 10, color='#f3f4f6', alpha=0.4, label='Training Domain [0, 10]')
        
    # --- Plot 1: Cubic Polynomial ---
    fig1, ax1 = plt.subplots(figsize=(7, 5), dpi=150)
    
    # Data points
    ax1.scatter(x_train, y_train, color='#2563eb', alpha=0.55, marker='o', s=35, label='Noisy Data')
    
    # Fit line
    ax1.plot(x_grid, y_poly_fit, color='#dc2626', linewidth=2.5, label='Cubic Fit (Extrapolated)')
    
    style_ax(ax1)
    # Set y limits carefully so the polynomial explosion is visible but doesn't completely squash the plot
    ax1.set_ylim(-2.0, 9.0)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e5e7eb')
    fig1.tight_layout()
    
    poly_path = '007_cubic_extrapolation.png'
    fig1.savefig(poly_path, facecolor='white', edgecolor='none')
    plt.close(fig1)
    print(f"Saved cubic extrapolation plot to {poly_path}")
    
    # --- Plot 2: Fourier Regression ---
    fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=150)
    
    # Data points
    ax2.scatter(x_train, y_train, color='#2563eb', alpha=0.55, marker='o', s=35, label='Noisy Data')
    
    # Fit line
    ax2.plot(x_grid, y_fourier_fit, color='#16a34a', linewidth=2.5, label='Fourier Fit (Extrapolated)')
    
    style_ax(ax2)
    ax2.set_ylim(-2.0, 9.0)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#e5e7eb')
    fig2.tight_layout()
    
    fourier_path = '007_fourier_extrapolation.png'
    fig2.savefig(fourier_path, facecolor='white', edgecolor='none')
    plt.close(fig2)
    print(f"Saved fourier extrapolation plot to {fourier_path}")

    # Copy files to artifacts directory
    artifact_dir = '/Users/dsnyder/.gemini/antigravity/brain/2e99ae12-7934-4916-bb38-00adec9ec3d8'
    if os.path.exists(artifact_dir):
        shutil.copy(poly_path, os.path.join(artifact_dir, '007_cubic_extrapolation.png'))
        shutil.copy(fourier_path, os.path.join(artifact_dir, '007_fourier_extrapolation.png'))
        print("Copied plots to artifacts directory.")

if __name__ == '__main__':
    main()
