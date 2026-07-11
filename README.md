# Data Blog Repository

This repository contains code, datasets, simulations, and visualizations that support my data blog. It is organized into modular Python scripts and tools designed to generate figures, perform numerical experiments, and build interactive dashboards illustrating core mathematical and statistical concepts.

## Table of Contents
- [High-Level Purpose](#high-level-purpose)
- [Python Scripts & Modules](#python-scripts--modules)
- [Interactive Dashboard](#interactive-dashboard)
- [Installation & Running](#installation--running)

---

## High-Level Purpose

The goal of this codebase is to provide clean, reproducible implementations of data visualization, numerical modeling, regression, and signal processing. The topics currently covered include:
- **Numerical Differentiation**: Techniques for estimating derivatives from noisy time series (e.g., Simple Finite Difference, EWMA Filtering, and Least-Squares Convolution).
- **Linear Algebra & Geometry**: Interactive 2D/3D visualizations of linear regression as an orthogonal projection of target vectors onto the column space of feature matrices.
- **Regression & Curve Fitting**: Fitting linear, quadratic, and periodic models, highlighting the differences in extrapolation behavior between polynomials and Fourier methods.
- **Image Data Extraction**: A utility for programmatically reverse-engineering data points $(x, y)$ from static chart images.

---

## Python Scripts & Modules

Below is a catalog of the Python files in this repository and their respective roles in the blog posts:

### 📊 Utility & Core Modules

* **[graph_extractor.py](graph_extractor.py)**  
  A reusable module that reverse-engineers raw data coordinates $(x, y)$ from a chart image. It features:
  - Automatic plotting bounding box (`plot_bbox`) detection by scanning for grid lines.
  - Target trace color matching using Euclidean color distance.
  - Isolation of the main line trace from shaded gradient fills (by selecting the topmost matching pixels).
  
  **CLI Usage**:
  ```bash
  # Extract Beef price data (with timestamps)
  poetry run python graph_extractor.py beef_com.png --x-limits 0 11 --y-limits 290 370 --color blue --output beef_extracted.csv --date-limits 2025-08-01 2026-07-01 --verify beef_verification.png

  # Extract Crude Oil price data (with timestamps)
  poetry run python graph_extractor.py cl1_com.png --x-limits 0 11 --y-limits 50 130 --color blue --output cl1_extracted.csv --date-limits 2025-08-01 2026-07-01 --verify cl1_verification.png
  ```
  
* **[001_slope.py](001_slope.py)**  
  Implements the mathematical functions for three different derivative estimation strategies:
  1. *Type 1*: Simple finite differences divided by timestep.
  2. *Type 2*: Exponentially Weighted Moving Average (EWMA) smoothing followed by finite differences.
  3. *Type 3*: Least-squares linear slope estimation via discrete convolution (Savitzky-Golay style).

* **[002_data_generation.py](002_data_generation.py)**  
  Simulates a realistic, physical signal: a bouncing ball experiencing gravitational acceleration. It uses Euler integration, applies a Butterworth low-pass filter, and adds Gaussian noise to generate realistic test data.

---

### 🧪 Numerical Experiments & Image Generation

* **[test_graph_extractor.py](test_graph_extractor.py)**  
  Verification runner for the image extractor. It extracts $(x, y)$ points from `beef_com.png`, writes them to `extracted_data.csv`, and outputs an overlay verification plot (`verification_plot.png`).

* **[commodity_prices/run_batch_extraction.py](commodity_prices/run_batch_extraction.py)**  
  Automates the extraction of historical time-series datasets from all 18 commodity price graphs in the `commodity_prices/` directory. It uses visual scale bounding boxes and generates individual CSVs as well as a consolidated `all_commodity_prices.csv` dataset.

* **[003_test_slope.py](003_test_slope.py)**  
  Applies the derivative estimators from `001_slope.py` to the simulated bouncing ball data, plots the results comparing their noise levels, and saves `003_slope_comparison.png`.

* **[004_generate_images.py](004_generate_images.py)**  
  Generates a progression of four educational diagrams (`img_01_position_alone.png` through `img_04_position_type2_type3.png`) for a blog post explaining the evolution from raw noisy derivatives to smoothed convolution.

* **[005_generate_plots.py](005_generate_plots.py)**  
  Creates a synthetic quadratic dataset and fits linear and quadratic models, saving stylized plots (`005_linear_fit.png` and `005_quadratic_fit.png`).

* **[006_vector_projection.py](006_vector_projection.py)**  
  Generates geometric vector projection diagrams in 2D (`006_projection_2d.png`) and 3D (`006_projection_3d.png`) visualizing the linear algebra behind Least Squares ($\hat{y} = \beta X$ and residuals $e \perp \text{Col}(X)$).

* **[007_extrapolation_comparison.py](007_extrapolation_comparison.py)**  
  Fits periodic noisy data using both a cubic polynomial and a Fourier model. The plots (`007_cubic_extrapolation.png` and `007_fourier_extrapolation.png`) demonstrate the instability of polynomial extrapolation vs. the stability of Fourier extrapolation.

---

## Interactive Dashboard

* **[app.py](app.py)**  
  An interactive dashboard built with **Dash** and **Plotly** (using a dark theme). It allows readers to:
  - Dynamically adjust the bouncing ball simulation parameters (duration, sample rate, noise level).
  - Modify parameters for the EWMA span (Type 2) and Least-Squares window size (Type 3).
  - Turn estimators on/off to visualize in real-time how noise and lag affect velocity estimates.

To run the app locally:
```bash
poetry run python app.py
```

---

## Installation & Running

This project uses **Poetry** for dependency management.

1. **Install dependencies**:
   ```bash
   poetry install
   ```
2. **Run scripts**:
   Any of the generation scripts can be executed using `poetry run python <script_name>.py`.