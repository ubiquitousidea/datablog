import numpy as np
import matplotlib.pyplot as plt
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

slope_mod = load_module("slope", "001_slope.py")
estimate_slope = slope_mod.estimate_slope

data_mod = load_module("data_generation", "002_data_generation.py")
generate_bouncing_ball = data_mod.generate_bouncing_ball

def plot_and_save(t, data, t_slopes, slopes, labels, colors, filename, title):
    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=130)
    
    color_pos = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)', color=color_pos)
    p1, = ax1.plot(t, data, color=color_pos, label='Position (Noisy)')
    ax1.tick_params(axis='y', labelcolor=color_pos)
    
    lines = [p1]
    
    if slopes:
        ax2 = ax1.twinx()
        color_vel = 'black'
        ax2.set_ylabel('Velocity / Slope (m/s)', color=color_vel)
        for ts, s, l, c in zip(t_slopes, slopes, labels, colors):
            p, = ax2.plot(ts, s, color=c, alpha=0.8, linewidth=2, label=l)
            lines.append(p)
        ax2.tick_params(axis='y', labelcolor=color_vel)
        
    line_labels = [l.get_label() for l in lines]
    ax1.legend(lines, line_labels, loc='upper right')
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    sample_rate = 1000.0
    duration = 5.0
    
    print("Generating data...")
    # Generate data with 0.008 noise ratio
    data = generate_bouncing_ball(duration=duration, sample_rate=sample_rate, noise_pct=0.008)
    t = np.arange(len(data)) / sample_rate
    
    print("Calculating slopes...")
    # Type 1: Simple difference
    slope1 = estimate_slope(data, sample_rate, estimate_type=1)
    
    # Type 2: EWMA with half life of 50 samples
    half_life = 50.0
    alpha = 1.0 - np.exp(-np.log(2) / half_life)
    slope2 = estimate_slope(data, sample_rate, estimate_type=2, ewma_alpha=alpha)
    
    # Type 3: Least squares convolution with 50 sample window
    window_size = 50
    slope3 = estimate_slope(data, sample_rate, estimate_type=3, window_size=window_size)
    
    # Precise time vectors for correct alignment
    t12 = (np.arange(len(slope1)) + 0.5) / sample_rate
    t3_start = (window_size - 1) / 2.0
    t3 = (np.arange(len(slope3)) + t3_start) / sample_rate
    
    print("Generating images...")
    
    # 1. Position alone
    plot_and_save(t, data, [], [], [], [], 
                  'img_01_position_alone.png', 'Bouncing Ball Position')
                  
    # 2. Position + Type 1
    plot_and_save(t, data, [t12], [slope1], ['Derivative: Type 1 (Simple Diff)'], ['tab:orange'], 
                  'img_02_position_type1.png', 'Position and Type 1 Derivative Estimate')
                  
    # 3. Position + Type 1 & Type 2
    plot_and_save(t, data, [t12, t12], [slope1, slope2], 
                  ['Derivative: Type 1 (Simple Diff)', 'Derivative: Type 2 (EWMA, half-life=50)'], 
                  ['tab:orange', 'tab:green'], 
                  'img_03_position_type1_type2.png', 'Position and Type 1 & 2 Derivative Estimates')
                  
    # 4. Position + Type 2 & Type 3
    plot_and_save(t, data, [t12, t3], [slope2, slope3], 
                  ['Derivative: Type 2 (EWMA, half-life=50)', 'Derivative: Type 3 (LS Conv, win=50)'], 
                  ['tab:green', 'tab:red'], 
                  'img_04_position_type2_type3.png', 'Position and Type 2 & 3 Derivative Estimates')

    print("All images generated successfully.")

if __name__ == "__main__":
    main()
