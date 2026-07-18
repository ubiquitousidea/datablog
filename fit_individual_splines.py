import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfda
from skfda.datasets import fetch_growth
from skfda.representation.basis import BSplineBasis

def main():
    print("Fetching Berkeley Growth dataset...")
    dataset = fetch_growth()
    fd = dataset.data  # FDataGrid of shape (93, 31, 1)
    gender = dataset.target  # 0 for male, 1 for female
    
    domain = fd.domain_range[0]
    t_min, t_max = domain
    
    # Calculate 4 equally spaced interior knots
    # They divide the domain [1.0, 18.0] into 5 equal subintervals
    n_interior = 4
    interior_knots = np.linspace(t_min, t_max, n_interior + 2)[1:-1]
    knots = [t_min] + list(interior_knots) + [t_max]
    
    print("Knot points sequence (including boundaries):")
    print([round(k, 4) for k in knots])
    
    # BSplineBasis of order 4 (cubic) with 4 interior knots (n_basis = order + n_interior = 8)
    basis = BSplineBasis(
        domain_range=fd.domain_range,
        n_basis=8,
        order=4,
        knots=knots
    )
    
    print("Fitting cubic splines to all 93 individuals...")
    fd_spline = fd.to_basis(basis)
    
    # Evaluate the fitted spline functions on a dense grid
    t_grid = np.linspace(t_min, t_max, 200)
    y_pred = fd_spline(t_grid)  # Shape (93, 200, 1)
    
    print("Generating visualization...")
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Style axes and spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='both', colors='#666666', labelsize=10, width=1.5)
    ax.grid(True, linestyle='--', alpha=0.3, color='#e0e0e0')
    
    # 1. Plot raw data points as a light gray background cloud
    ages = fd.grid_points[0]
    all_ages = np.tile(ages, 93)
    all_heights = fd.data_matrix.flatten()
    ax.scatter(all_ages, all_heights, color='#d9d9d9', alpha=0.4, s=3.0, label='Raw Data Points (All Subjects)')
    
    # 2. Plot individual spline curves colored by gender
    # We will track legend labels so they are added only once
    male_label_added = False
    female_label_added = False
    
    for i in range(93):
        curve = y_pred[i, :, 0]
        if gender[i] == 0:  # Male
            color = '#1f77b4'  # Cool blue
            label = 'Individual Spline (Male)' if not male_label_added else None
            male_label_added = True
        else:  # Female
            color = '#e377c2'  # Warm pink/magenta
            label = 'Individual Spline (Female)' if not female_label_added else None
            female_label_added = True
            
        ax.plot(t_grid, curve, color=color, alpha=0.3, linewidth=1.1, label=label)
        
    # 3. Add vertical lines for the four equally spaced interior knots
    for idx, k in enumerate(interior_knots):
        label = 'Interior Knots' if idx == 0 else None
        ax.axvline(k, color='#888888', linestyle=':', alpha=0.6, linewidth=1.2, label=label)
        
    # Labels, titles, and legends
    ax.set_title('Berkeley Growth Study: Individual Cubic Spline Fits (4 Interior Knots)', fontsize=14, fontweight='bold', color='#111111', pad=12)
    ax.set_xlabel('Age (Years)', fontsize=11, color='#333333', labelpad=8)
    ax.set_ylabel('Height (cm)', fontsize=11, color='#333333', labelpad=8)
    
    # Add annotation about knots
    annotation_text = (
        r"$\bf{Spline\ Specifications:}$" + "\n"
        f"Order: 4 (Cubic)\n"
        f"Interior Knots (years): {', '.join([f'{k:.2f}' for k in interior_knots])}"
    )
    ax.text(0.02, 0.95, annotation_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#fcfcfc', edgecolor='#e0e0e0', alpha=0.9))
    
    ax.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0', loc='lower right', fontsize=10)
    
    fig.tight_layout()
    
    # Save plot
    plot_filename = 'berkeley_individual_splines.png'
    fig.savefig(plot_filename, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Plot saved successfully to {plot_filename}")
    
    # Copy to artifacts directory
    artifact_dir = os.environ.get('ANTIGRAVITY_ARTIFACT_DIR', '')
    if not artifact_dir:
        artifact_dir = '/Users/dsnyder/.gemini/antigravity/brain/a8e1e0f3-7533-42b2-8bce-c9712908a576'
        
    if os.path.exists(artifact_dir):
        dest_path = os.path.join(artifact_dir, plot_filename)
        shutil.copy(plot_filename, dest_path)
        print(f"Copied plot to artifact path: {dest_path}")

if __name__ == '__main__':
    main()
