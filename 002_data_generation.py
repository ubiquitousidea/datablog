
import numpy as np
from scipy.signal import butter, filtfilt

def generate_bouncing_ball(duration=10.0, sample_rate=1000.0, h0=10.0, e=0.8, g=9.81, 
                           cutoff_freq=20.0, filter_order=4, noise_pct=0.02):
    """
    Generate a data vector representing the position of a bouncing ball,
    low-pass filtered, with added Gaussian noise.
    
    Parameters:
    ----------
    duration : float
        Total time of simulation in seconds.
    sample_rate : float
        Samples per second.
    h0 : float
        Initial drop height, meters.
    e : float
        Coefficient of restitution (bounciness, between 0 and 1).
    g : float
        Acceleration due to gravity, m/s^2.
    cutoff_freq : float
        Low-pass filter cutoff frequency in Hz.
    filter_order : int
        Order of the Butterworth low-pass filter.
    noise_pct : float
        Standard deviation of Gaussian noise as a percentage of initial drop height.
        
    Returns:
    -------
    y_noisy : ndarray
        The position vector of the ball after filtering and noise addition.
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate
    y = np.zeros_like(t)
    
    # Simulate bouncing ball using Euler integration
    velocity = 0.0
    dt = 1.0 / sample_rate
    y[0] = h0
    
    for i in range(1, num_samples):
        velocity -= g * dt
        y[i] = y[i-1] + velocity * dt
        
        # When hit the ground, bounce back
        if y[i] < 0:
            y[i] = -y[i]
            velocity = -velocity * e
            
    # Apply Butterworth low-pass filter
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    
    # Prevent ValueError if normal_cutoff is >= 1 (i.e. cutoff_freq >= nyquist)
    if normal_cutoff >= 1.0:
        y_filtered = y
    else:
        b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)
        
        # filtfilt needs signal to be long enough
        padlen = 3 * max(len(a), len(b))
        if len(y) > padlen:
            y_filtered = filtfilt(b, a, y)
        else:
            y_filtered = y
            
    # Add Gaussian noise
    noise_std = noise_pct * h0
    noise = np.random.normal(0, noise_std, len(y_filtered))
    y_noisy = y_filtered + noise
    
    return y_noisy


if __name__ == "__main__":
    y = generate_bouncing_ball()
    import matplotlib.pyplot as plt
    plt.plot(y)
    plt.show()