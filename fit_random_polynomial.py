#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Perfect Polynomial Fit on Random Data")
    parser.add_argument("num_points", type=int, help="Number of data points to generate")
    parser.add_argument("-o", "--output", type=str, default="perfect_fit.png",
                        help="Path to save the generated plot (default: perfect_fit.png)")
    parser.add_argument("-d", "--dpi", type=int, default=150,
                        help="Output resolution of the plot in DPI (default: 150)")
    parser.add_argument("--degree", type=int, default=None,
                        help="Degree of the polynomial to fit (default: num_points - 1)")
    parser.add_argument("-e", "--errors", action="store_true",
                        help="Draw vertical error lines between the points and the polynomial")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Random seed for data reproducibility (default: None)")
    args = parser.parse_args()

    n = args.num_points
    if n <= 0:
        raise ValueError("Number of points must be at least 1.")

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Determine polynomial degree
    degree = args.degree if args.degree is not None else n - 1
    if degree < 0:
        raise ValueError("Polynomial degree must be at least 0.")

    # Generate uniformly spaced X
    x = np.linspace(0, 10, n)
    # Generate randomly generated Y
    y = np.random.uniform(0, 10, n)

    # Fit a polynomial of degree
    coeffs = np.polyfit(x, y, degree)

    # Evaluate the polynomial on a fine grid for plotting a smooth curve
    x_fit = np.linspace(0, 10, 1000)
    y_fit = np.polyval(coeffs, x_fit)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Points are plotted as semi-transparent blue points
    plt.scatter(x, y, color='blue', alpha=0.5, s=80, label='Random Data Points')

    # Draw vertical error lines if requested
    if args.errors:
        y_pred = np.polyval(coeffs, x)
        plt.vlines(x, y, y_pred, colors='darkgray', linestyles=':', alpha=0.6, label='Y Errors')

    # Polynomial line is a dashed orange line, also semi-transparent
    plt.plot(x_fit, y_fit, color='orange', linestyle='--', alpha=0.5, linewidth=2, label=f'Polynomial Fit (Degree {degree})')

    # Set plot title dynamically based on polynomial degree
    if degree < n - 1:
        plt.title("Imperfect Polynomial Fit")
    else:
        plt.title("Perfect Polynomial Fit on Random Data")
    plt.xlabel("X (Uniformly Spaced)")
    plt.ylabel("Y (Random)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)

    # Save the plot
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"Successfully generated {n} random data points.")
    print(f"Fitted a degree-{degree} polynomial.")
    if args.seed is not None:
        print(f"Random seed set to {args.seed}.")
    print(f"Saved the plot to {args.output} at {args.dpi} DPI.")

if __name__ == "__main__":
    main()
