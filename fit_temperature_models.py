import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def format_poly(coeffs):
    deg = len(coeffs) - 1
    expr = ""
    for i, c in enumerate(coeffs):
        p = deg - i
        if abs(c) < 0.005:
            continue
        
        sign = " + " if c >= 0 else " - "
        if expr == "":
            sign = "-" if c < 0 else ""
            
        val = f"{abs(c):.2f}"
        if p == 0:
            term = f"{val}"
        elif p == 1:
            term = f"{val}t"
        else:
            term = f"{val}t^{p}"
            
        expr += sign + term
    return expr

def format_spline(beta):
    expr = ""
    terms = [
        (beta[0], ""),
        (beta[1], "t"),
        (beta[2], "t^2"),
        (beta[3], "t^3"),
        (beta[4], "(t - 1/3)_+^3"),
        (beta[5], "(t - 2/3)_+^3")
    ]
    for c, var in terms:
        if abs(c) < 0.005:
            continue
        sign = " + " if c >= 0 else " - "
        if expr == "":
            sign = "-" if c < 0 else ""
        val = f"{abs(c):.2f}"
        expr += sign + val + var
    return expr

def main():
    filepath = "air+quality/AirQualityUCI.csv"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    print("Loading and cleaning dataset...")
    df = pd.read_csv(filepath, sep=';', decimal=',')
    df = df.dropna(subset=['Date', 'Time'], how='all')
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].str.replace('.', ':'), format='%d/%m/%Y %H:%M:%S')
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    # Replace -200 with NaN and drop missing values for T
    df['T'] = df['T'].replace(-200.0, np.nan)
    df_fit = df[['Datetime', 'T']].dropna().reset_index(drop=True)
    
    t_min = df_fit['Datetime'].min()
    t_max = df_fit['Datetime'].max()
    total_duration = (t_max - t_min).total_seconds()
    
    # Normalize time to [0, 1] for numerical stability during fitting
    df_fit['t_norm'] = (df_fit['Datetime'] - t_min).dt.total_seconds() / total_duration
    t = df_fit['t_norm'].values
    y = df_fit['T'].values
    
    print("Fitting models...")
    # 1. Fit Polynomials (Degree 1 to 4)
    coeffs_lin = np.polyfit(t, y, 1)
    coeffs_quad = np.polyfit(t, y, 2)
    coeffs_cubic = np.polyfit(t, y, 3)
    coeffs_quartic = np.polyfit(t, y, 4)
    
    # 2. Fit Cubic Spline with knots at 1/3 and 2/3
    k1 = 1.0 / 3.0
    k2 = 2.0 / 3.0
    
    X_spline = np.vstack([
        np.ones_like(t),
        t,
        t**2,
        t**3,
        np.maximum(0, t - k1)**3,
        np.maximum(0, t - k2)**3
    ]).T
    beta, _, _, _ = np.linalg.lstsq(X_spline, y, rcond=None)
    
    # Generate smooth curves on a dense grid
    t_grid = np.linspace(0, 1, 1000)
    datetime_grid = t_min + pd.to_timedelta(t_grid * total_duration, unit='s')
    
    y_lin_pred = np.polyval(coeffs_lin, t_grid)
    y_quad_pred = np.polyval(coeffs_quad, t_grid)
    y_cubic_pred = np.polyval(coeffs_cubic, t_grid)
    y_quartic_pred = np.polyval(coeffs_quartic, t_grid)
    
    X_spline_grid = np.vstack([
        np.ones_like(t_grid),
        t_grid,
        t_grid**2,
        t_grid**3,
        np.maximum(0, t_grid - k1)**3,
        np.maximum(0, t_grid - k2)**3
    ]).T
    y_spline_pred = X_spline_grid @ beta
    
    # Print formulas in console
    print("\nFitted Equations (t in [0, 1] represents normalized time):")
    print(f"Linear:       T(t) = {format_poly(coeffs_lin)}")
    print(f"Quadratic:    T(t) = {format_poly(coeffs_quad)}")
    print(f"Cubic:        T(t) = {format_poly(coeffs_cubic)}")
    print(f"Quartic:      T(t) = {format_poly(coeffs_quartic)}")
    print(f"Cubic Spline: T(t) = {format_spline(beta)}")
    
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
    
    # 1. Plot raw data as small semi-transparent points
    ax.scatter(df_fit['Datetime'], y, color='#a0b2c6', alpha=0.25, s=2.5, label='Raw Data (T)')
    
    # 2. Plot model predictions as thin lines of various colors
    ax.plot(datetime_grid, y_lin_pred, color='#ff7f0e', linewidth=1.5, label='Linear Fit (deg 1)')
    ax.plot(datetime_grid, y_quad_pred, color='#2ca02c', linewidth=1.5, label='Quadratic Fit (deg 2)')
    ax.plot(datetime_grid, y_cubic_pred, color='#1f77b4', linewidth=1.5, label='Cubic Fit (deg 3)')
    ax.plot(datetime_grid, y_quartic_pred, color='#9467bd', linewidth=1.5, label='Quartic Fit (deg 4)')
    ax.plot(datetime_grid, y_spline_pred, color='#d62728', linewidth=1.8, label='Cubic Spline Fit (2 knots)')
    
    # Add vertical lines at the knot points to help visualize the spline boundaries
    knot1_datetime = t_min + pd.to_timedelta(k1 * total_duration, unit='s')
    knot2_datetime = t_min + pd.to_timedelta(k2 * total_duration, unit='s')
    ax.axvline(knot1_datetime, color='#999999', linestyle=':', alpha=0.7, linewidth=1.2, label='Spline Knots (1/3, 2/3)')
    ax.axvline(knot2_datetime, color='#999999', linestyle=':', alpha=0.7, linewidth=1.2)
    
    # Format X-axis with monthly dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    ax.set_title('UCI Temperature Modeling: Polynomials vs. Cubic Spline', fontsize=14, fontweight='bold', color='#111111', pad=12)
    ax.set_xlabel('Time (Date)', fontsize=11, color='#333333', labelpad=8)
    ax.set_ylabel('Temperature (°C)', fontsize=11, color='#333333', labelpad=8)
    
    # Beautifully aligned legend
    ax.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0', loc='upper right', fontsize=9.5)
    
    # Build text for the equations
    equations_text = (
        r"$\bf{Model\ Equations\ (t \in [0, 1]\ is\ normalized\ time):}$" + "\n"
        f"Linear: $T(t) = {format_poly(coeffs_lin)}$\n"
        f"Quadratic: $T(t) = {format_poly(coeffs_quad)}$\n"
        f"Cubic: $T(t) = {format_poly(coeffs_cubic)}$\n"
        f"Quartic: $T(t) = {format_poly(coeffs_quartic)}$\n"
        f"Cubic Spline: $T(t) = {format_spline(beta)}$"
    )
    
    # Add textbox inside the plot at the bottom left (since it's mostly empty space)
    ax.text(0.02, 0.05, equations_text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.6', facecolor='#fcfcfc', edgecolor='#e0e0e0', alpha=0.95))
    
    fig.tight_layout()
    
    # Save visualization
    plot_filename = 'temperature_fits.png'
    fig.savefig(plot_filename, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Plot saved successfully to {plot_filename}")
    
    # Copy to artifacts directory if available
    artifact_dir = os.environ.get('ANTIGRAVITY_ARTIFACT_DIR', '')
    if not artifact_dir:
        # Fallback to direct path from context
        artifact_dir = '/Users/dsnyder/.gemini/antigravity/brain/a8e1e0f3-7533-42b2-8bce-c9712908a576'
        
    if os.path.exists(artifact_dir):
        dest_path = os.path.join(artifact_dir, plot_filename)
        shutil.copy(plot_filename, dest_path)
        print(f"Copied plot to artifact path: {dest_path}")

if __name__ == '__main__':
    main()
