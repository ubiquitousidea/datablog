import numpy as np

def estimate_slope(data, sample_rate, estimate_type=1, ewma_span=None, ewma_alpha=None, window_size=None):
    """
    Finds the time derivative of a vector with a known sample rate.
    
    Parameters:
    -----------
    data : array_like
        The input data vector.
    sample_rate : float
        The sampling rate of the data in Hz.
    estimate_type : int, optional
        Control the type of slope estimate algorithm (1, 2, or 3). Default is 1.
        Type 1: simple differences divided by timestep.
        Type 2: EMWA smoothing followed by Type 1 difference.
        Type 3: least squares slope estimate using convolution.
    ewma_span : int, optional
        Span parameter for EWMA smoothing (only used if estimate_type=2).
    ewma_alpha : float, optional
        Alpha parameter for EWMA smoothing (only used if estimate_type=2).
    window_size : int, optional
        Window size for the least squares slope estimate (only used if estimate_type=3).
        
    Returns:
    --------
    array_like
        The estimated slope (time derivative). The length of the array depends on the estimate type.
    """
    data = np.asarray(data)
    dt = 1.0 / sample_rate
    
    if estimate_type == 1:
        # Type 1: divide the difference in value by the time step
        return np.diff(data) / dt
        
    elif estimate_type == 2:
        # Type 2: smooth with EWMA before reporting finite differences
        if ewma_span is not None:
            alpha = 2.0 / (ewma_span + 1.0)
        elif ewma_alpha is not None:
            alpha = float(ewma_alpha)
        else:
            raise ValueError("For estimate_type 2, provide either ewma_alpha or ewma_span.")
            
        smoothed = np.empty_like(data, dtype=float)
        if len(data) > 0:
            smoothed[0] = data[0]
            for i in range(1, len(data)):
                smoothed[i] = alpha * data[i] + (1.0 - alpha) * smoothed[i-1]
                
        return np.diff(smoothed) / dt
        
    elif estimate_type == 3:
        # Type 3: convolve with a slope finding kernel
        if window_size is None:
            raise ValueError("For estimate_type 3, provide a window_size.")
            
        if window_size < 2:
            raise ValueError("window_size must be at least 2.")
            
        # Time array centered at 0
        t = (np.arange(window_size) - (window_size - 1) / 2.0) * dt
        
        # Denominator for least squares slope: sum of squared differences from mean
        # Since t is centered, mean of t is 0
        denom = np.sum(t**2)
        
        # The vector whose dot product with a data window gives the least squares slope estimate
        w = t / denom
        
        # The slope finding kernel is the reverse of this vector
        kernel = w[::-1]
        
        # Convolve the data with the kernel
        return np.convolve(data, kernel, mode='valid')
        
    else:
        raise ValueError("estimate_type must be 1, 2, or 3.")
