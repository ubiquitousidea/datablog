import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfda
from skfda.datasets import fetch_growth
from skfda.representation.basis import MonomialBasis, BSplineBasis

def format_poly_fda(coeffs):
    deg = len(coeffs) - 1
    expr = ""
    for i, c in enumerate(coeffs):
        p = i  # Monomial basis coefficients are in ascending order of power
        if abs(c) < 1e-5:
            continue
        
        sign = " + " if c >= 0 else " - "
        if expr == "":
            sign = "-" if c < 0 else ""
            
        # Format with 4 decimal places if the coefficient is very small, otherwise 2
        if abs(c) < 0.01:
            val = f"{abs(c):.4f}"
        else:
            val = f"{abs(c):.2f}"
            
        if p == 0:
            term = f"{val}"
        elif p == 1:
            term = f"{val}t"
        else:
            term = f"{val}t^{p}"
            
        expr += sign + term
    return expr

def format_spline_fda(coeffs):
    terms = []
    for i, c in enumerate(coeffs):
        term = f"{c:.2f}B_{{{i+1}}}(t)"
        terms.append(term)
    return " + ".join(terms)

def main():
    print("Fetching Berkeley Growth dataset...")
    dataset = fetch_growth()
    fd = dataset.data  # FDataGrid of shape (93, 31, 1)
    
    # Calculate the mean function across all 93 subjects
    print("Calculating mean growth curve...")
    fd_mean = fd.mean()
    
    domain = fd.domain_range[0]
    t_min, t_max = domain
    
    # Define dense grid for plotting predictions smoothly
    t_grid = np.linspace(t_min, t_max, 200)
    
    # Fit Models
    print("Fitting models to the mean curve...")
    
    # 1. Monomial basis (Degrees 1 to 4)
    # Degree 1: n_basis=2
    # Degree 2: n_basis=3
    # Degree 3: n_basis=4
    # Degree 4: n_basis=5
    fits_mono = {}
    coeffs_mono = {}
    for deg in [1, 2, 3, 4]:
        basis = MonomialBasis(domain_range=fd.domain_range, n_basis=deg+1)
        fd_basis = fd_mean.to_basis(basis)
        fits_mono[deg] = fd_basis(t_grid).flatten()
        coeffs_mono[deg] = fd_basis.coefficients[0]
        
    # 2. B-spline basis (Cubic Spline, order=4) with knots at 1/3 and 2/3
    k1 = t_min + (t_max - t_min) / 3.0
    k2 = t_min + 2.0 * (t_max - t_min) / 3.0
    knots = [t_min, k1, k2, t_max]
    
    basis_spline = BSplineBasis(
        domain_range=fd.domain_range,
        n_basis=6,
        order=4,
        knots=knots
    )
    fd_spline = fd_mean.to_basis(basis_spline)
    y_spline_pred = fd_spline(t_grid).flatten()
    coeffs_spline = fd_spline.coefficients[0]
    
    # Print formulas in console
    print("\nFitted Equations for Mean Curve (t is age in years, t in [1.0, 18.0]):")
    print(f"Linear:       H(t) = {format_poly_fda(coeffs_mono[1])}")
    print(f"Quadratic:    H(t) = {format_poly_fda(coeffs_mono[2])}")
    print(f"Cubic:        H(t) = {format_poly_fda(coeffs_mono[3])}")
    print(f"Quartic:      H(t) = {format_poly_fda(coeffs_mono[4])}")
    print(f"Cubic Spline: H(t) = {format_spline_fda(coeffs_spline)}")
    
    print("\nGenerating visualization...")
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=150)
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
    
    # 1. Plot raw data as small, semi-transparent points (all 93 subjects * 31 ages)
    ages = fd.grid_points[0]
    all_ages = np.tile(ages, 93)
    all_heights = fd.data_matrix.flatten()
    
    ax.scatter(all_ages, all_heights, color='#a0b2c6', alpha=0.15, s=3.5, label='Raw Heights (All Subjects)')
    
    # 2. Plot model predictions as thin lines of various colors
    ax.plot(t_grid, fits_mono[1], color='#ff7f0e', linewidth=1.5, label='Linear Fit (deg 1)')
    ax.plot(t_grid, fits_mono[2], color='#2ca02c', linewidth=1.5, label='Quadratic Fit (deg 2)')
    ax.plot(t_grid, fits_mono[3], color='#1f77b4', linewidth=1.5, label='Cubic Fit (deg 3)')
    ax.plot(t_grid, fits_mono[4], color='#9467bd', linewidth=1.5, label='Quartic Fit (deg 4)')
    ax.plot(t_grid, y_spline_pred, color='#d62728', linewidth=1.8, label='Cubic Spline Fit (2 knots)')
    
    # Add vertical lines at the knot points to help visualize the spline boundaries
    ax.axvline(k1, color='#999999', linestyle=':', alpha=0.7, linewidth=1.2, label=f'Spline Knots ({k1:.2f}, {k2:.2f})')
    ax.axvline(k2, color='#999999', linestyle=':', alpha=0.7, linewidth=1.2)
    
    # Labels and Titles
    ax.set_title('Berkeley Growth Study: Height Modeling using scikit-fda', fontsize=14, fontweight='bold', color='#111111', pad=12)
    ax.set_xlabel('Age (Years)', fontsize=11, color='#333333', labelpad=8)
    ax.set_ylabel('Height (cm)', fontsize=11, color='#333333', labelpad=8)
    
    ax.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0', loc='lower right', fontsize=9.5)
    
    # Build text for the equations
    equations_text = (
        r"$\bf{Mean\ Height\ Model\ Equations\ (t\ is\ age\ in\ years):}$" + "\n"
        f"Linear: $H(t) = {format_poly_fda(coeffs_mono[1])}$\n"
        f"Quadratic: $H(t) = {format_poly_fda(coeffs_mono[2])}$\n"
        f"Cubic: $H(t) = {format_poly_fda(coeffs_mono[3])}$\n"
        f"Quartic: $H(t) = {format_poly_fda(coeffs_mono[4])}$\n"
        f"Cubic Spline B-spline Basis representation:\n"
        r"$H(t) = \sum_{i=1}^6 c_i B_i(t)$ where coefficients $c$ are:\n"
        f"   $c = [{', '.join([f'{val:.2f}' for val in coeffs_spline])}]$\n"
        f"   and $B_i(t)$ are B-spline basis functions with knots at $[1.0, {k1:.2f}, {k2:.2f}, 18.0]$"
    )
    
    # Add textbox inside the plot at the top left
    ax.text(0.02, 0.95, equations_text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='#fcfcfc', edgecolor='#e0e0e0', alpha=0.95))
    
    fig.tight_layout()
    
    # Save visualization
    plot_filename = 'berkeley_growth_fits.png'
    fig.savefig(plot_filename, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Plot saved successfully to {plot_filename}")
    
    # Copy to artifacts directory if available
    artifact_dir = os.environ.get('ANTIGRAVITY_ARTIFACT_DIR', '')
    if not artifact_dir:
        artifact_dir = '/Users/dsnyder/.gemini/antigravity/brain/a8e1e0f3-7533-42b2-8bce-c9712908a576'
        
    if os.path.exists(artifact_dir):
        dest_path = os.path.join(artifact_dir, plot_filename)
        shutil.copy(plot_filename, dest_path)
        print(f"Copied plot to artifact path: {dest_path}")

if __name__ == '__main__':
    main()
