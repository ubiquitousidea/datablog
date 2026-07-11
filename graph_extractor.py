import numpy as np
from PIL import Image
import matplotlib.colors as mcolors

def detect_plot_bbox(img_arr):
    """
    Automatically detects the plotting area bounding box (left, top, right, bottom)
    in pixels by scanning for horizontal grid/border lines.
    
    Parameters:
    -----------
    img_arr : np.ndarray
        The image converted to a numpy RGB array of shape (height, width, 3).
        
    Returns:
    --------
    tuple of (int, int, int, int)
        The detected bounding box: (left, top, right, bottom).
    """
    height, width, _ = img_arr.shape
    
    # 1. Detect horizontal lines (grid lines and top/bottom borders)
    # Grid lines are typically light grey (nearly desaturated, color values close to each other).
    # We look for rows where at least 50% of the row has a desaturated light-grey color.
    horizontal_rows = []
    for y in range(height):
        row = img_arr[y, :, :3]
        # Check for light grey gridline color range (220 to 245)
        is_light_grey = (row[:, 0] >= 220) & (row[:, 0] <= 245) & \
                        (np.abs(row[:, 0] - row[:, 1]) < 3) & \
                        (np.abs(row[:, 1] - row[:, 2]) < 3)
        if np.sum(is_light_grey) > width * 0.5:
            horizontal_rows.append(y)
            
    # Default fallback values if detection fails
    top_y = 78
    bottom_y = 910
    left_x = 16
    right_x = 1377
    
    if len(horizontal_rows) >= 2:
        top_y = horizontal_rows[0]
        bottom_y = horizontal_rows[-1]
        
    # 2. Determine left and right boundaries by analyzing where the horizontal lines start and end
    # We inspect the top and bottom detected lines to find their extent
    for y_test in [top_y, bottom_y]:
        if 0 <= y_test < height:
            row = img_arr[y_test, :, :3]
            is_grey = (row[:, 0] >= 220) & (row[:, 0] <= 245) & \
                      (np.abs(row[:, 0] - row[:, 1]) < 3) & \
                      (np.abs(row[:, 1] - row[:, 2]) < 3)
            grey_indices = np.where(is_grey)[0]
            if len(grey_indices) > 10:
                left_x = int(grey_indices[0])
                right_x = int(grey_indices[-1])
                
                # Check for a vertical axis border line on the right (often around right_x)
                # If there's a strong vertical line near right_x, use it
                for test_x in range(right_x, right_x - 15, -1):
                    if 0 <= test_x < width:
                        col = img_arr[top_y:bottom_y+1, test_x, :3]
                        # Count grey pixels in this column segment
                        col_grey = (col[:, 0] >= 220) & (col[:, 0] <= 240)
                        if np.sum(col_grey) > (bottom_y - top_y) * 0.9:
                            right_x = test_x
                            break
                break
                
    return (left_x, top_y, right_x, bottom_y)

def extract_points_from_graph(image_path, x_limits, y_limits, color="blue", plot_bbox=None):
    """
    Extracts x, y points from a graph image.
    
    Parameters:
    -----------
    image_path : str
        Path to the graph image.
    x_limits : tuple of (float, float)
        The data limits for the x-axis, corresponding to the left and right edges of the plot area.
    y_limits : tuple of (float, float)
        The data limits for the y-axis, corresponding to the bottom and top edges of the plot area.
    color : str, tuple, or list
        Color of the data trace to extract (e.g. 'blue', '#2962FF', or (41, 98, 255)).
    plot_bbox : tuple of (int, int, int, int), optional
        Pixel bounding box of the plot area: (left, top, right, bottom).
        If None, the function automatically detects the plot region.
        
    Returns:
    --------
    list of tuple of (float, float)
        List of (x, y) data points.
    """
    # 1. Load image and convert to RGB numpy array
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img)
    
    # 2. Convert color argument to 0-255 RGB values
    if isinstance(color, str):
        if color.startswith("#"):
            target_rgb = np.array(mcolors.hex2color(color)) * 255.0
        else:
            try:
                target_rgb = np.array(mcolors.to_rgb(color)) * 255.0
            except ValueError:
                # Fallback dictionary for basic names
                color_map = {
                    "blue": [0, 0, 255],
                    "red": [255, 0, 0],
                    "green": [0, 255, 0],
                    "orange": [255, 165, 0],
                    "yellow": [255, 255, 0],
                    "purple": [128, 0, 128],
                    "cyan": [0, 255, 255],
                    "magenta": [255, 0, 255],
                }
                target_rgb = np.array(color_map.get(color.lower(), [0, 0, 255]))
    elif isinstance(color, (list, tuple)) and len(color) == 3:
        target_rgb = np.array(color)
    else:
        raise ValueError("Invalid color format. Provide a color name, hex string, or (R, G, B) tuple.")
        
    # 3. Resolve plot bounding box (auto-detect if None)
    if plot_bbox is None:
        left_x, top_y, right_x, bottom_y = detect_plot_bbox(img_arr)
    else:
        left_x, top_y, right_x, bottom_y = plot_bbox
        
    extracted_points = []
    
    # Define a color distance threshold
    # Named primary colors use a larger threshold to catch variations/anti-aliasing,
    # whereas specific hexadecimal/RGB colors use a tighter threshold.
    is_named_primary = isinstance(color, str) and color.lower() in [
        "blue", "red", "green", "orange", "yellow", "purple", "cyan", "magenta"
    ]
    threshold = 130 if is_named_primary else 45
    
    # 4. Scan columns left-to-right
    for x in range(left_x, right_x + 1):
        column_pixels = img_arr[top_y:bottom_y+1, x, :3]
        # Compute Euclidean distance in RGB space to target color
        dists = np.linalg.norm(column_pixels - target_rgb, axis=1)
        matching_y_indices = np.where(dists < threshold)[0]
        
        if len(matching_y_indices) > 0:
            # Select the topmost matching pixel to extract the trace and ignore any filled area below
            top_y_idx = matching_y_indices[0]
            y_pixel = top_y + top_y_idx
            
            # Map pixel coordinates to data coordinate system
            # X axis mapping
            x_frac = (x - left_x) / (right_x - left_x)
            x_data = x_limits[0] + x_frac * (x_limits[1] - x_limits[0])
            
            # Y axis mapping (pixel origin is top-left, data origin is bottom-left)
            y_frac = (bottom_y - y_pixel) / (bottom_y - top_y)
            y_data = y_limits[0] + y_frac * (y_limits[1] - y_limits[0])
            
            extracted_points.append((x_data, y_data))
            
    return extracted_points
