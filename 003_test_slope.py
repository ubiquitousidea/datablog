import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Import the functions
slope_mod = load_module("slope", "001_slope.py")
estimate_slope = slope_mod.estimate_slope

data_mod = load_module("data_generation", "002_data_generation.py")
generate_bouncing_ball = data_mod.generate_bouncing_ball

def main(
    duration:float = 5.0,
    sample_rate:float = 1000.0,
    window_size:int = 51,
    type1=True, 
    type2=True,
    type3=True,
):
    # Generate data
    print("Generating bouncing ball data...")
    data = generate_bouncing_ball(duration=duration, sample_rate=sample_rate, noise_pct=0.01)
    t = np.arange(len(data)) / sample_rate

    # Calculate slopes
    print("Calculating slopes...")
    # Type 1: Simple finite differences
    slope1 = estimate_slope(data, sample_rate, estimate_type=1)
    
    # Type 2: EWMA + finite differences
    slope2 = estimate_slope(data, sample_rate, estimate_type=2, ewma_span=int(50))
    
    # Type 3: Least squares convolution
    slope3 = estimate_slope(data, sample_rate, estimate_type=3, window_size=window_size)

    # Time vectors for plotting
    t12 = t[1:]  # For diff (length N-1)
    t3 = t[window_size//2 : window_size//2 + len(slope3)] # For valid conv (length N - window_size + 1)

    # Plotting
    print("Plotting results...")
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot position on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)', color=color)
    p1, = ax1.plot(t, data, color=color, label='Position (Noisy)')
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot slopes on secondary y-axis
    ax2 = ax1.twinx()
    color = 'black'
    ax2.set_ylabel('Velocity / Slope (m/s)', color=color)
    
    p2, p3, p4 = None, None, None
    if type1: 
        p2, = ax2.plot(t12, slope1, color='tab:orange', alpha=0.4, label='Type 1 (Simple Diff)')
    if type2: 
        p3, = ax2.plot(t12, slope2, color='tab:green', alpha=0.7, label='Type 2 (EWMA)')
    if type3: 
        p4, = ax2.plot(t3, slope3, color='tab:red', linewidth=2, label='Type 3 (LS Conv)')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends
    lines = [l for l in [p1, p2, p3, p4] if l is not None]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title('Bouncing Ball Position and Velocity Estimates')
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    
    # Save the plot just in case we need it without a display
    plt.savefig('003_slope_comparison.png')
    plt.show()

if __name__ == "__main__":
    main(
        type1=False,
        type2=True,
        type3=True,
        duration=10.0,
        window_size=51,
        sample_rate=1000,
    )
