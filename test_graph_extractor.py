import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from graph_extractor import extract_points_from_graph, detect_plot_bbox

def run_verification():
    image_path = "beef_com.png"
    
    # Define data limits based on the graph axis labels:
    # x-axis: Aug to Jul (12 months represented as indices 0 to 11)
    # y-axis: 290 to 370
    x_limits = (0.0, 11.0)
    y_limits = (290.0, 370.0)
    
    # 1. Extract data points using auto-detection
    print("Extracting data points using auto-detection...")
    points = extract_points_from_graph(image_path, x_limits, y_limits, color="blue")
    print(f"Successfully extracted {len(points)} points.")
    
    # 2. Save extracted points to CSV
    df = pd.DataFrame(points, columns=["x_data", "y_data"])
    csv_path = "extracted_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved extracted data to {csv_path}")
    
    # 3. Generate verification plot showing points overlayed on the original image
    print("Generating verification plot...")
    img = Image.open(image_path)
    img_arr = np.array(img)
    
    # Re-detect the bounding box to plot it
    left, top, right, bottom = detect_plot_bbox(img_arr)
    
    # Map the data points back to pixel coordinates for visual overlay
    pixel_points = []
    for x_d, y_d in points:
        # Convert x_data to x_pixel
        x_frac = (x_d - x_limits[0]) / (x_limits[1] - x_limits[0])
        x_pix = left + x_frac * (right - left)
        
        # Convert y_data to y_pixel
        y_frac = (y_d - y_limits[0]) / (y_limits[1] - y_limits[0])
        y_pix = bottom - y_frac * (bottom - top)
        
        pixel_points.append((x_pix, y_pix))
        
    pixel_points = np.array(pixel_points)
    
    # Plot the original image and overlay the extracted points
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    
    # Plot extracted points in bright red
    plt.scatter(pixel_points[:, 0], pixel_points[:, 1], color="red", s=1, label="Extracted Trace Points")
    
    # Draw detected plot boundary box
    plt.plot([left, right, right, left, left], [top, top, bottom, bottom, top],
             color="green", linestyle="--", linewidth=1.5, label="Detected Plot Area Bounding Box")
    
    plt.title("Visual Verification of Graph Data Extraction", fontsize=14, pad=15)
    plt.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    
    verification_path = "verification_plot.png"
    plt.savefig(verification_path, dpi=150)
    plt.close()
    print(f"Saved verification plot overlay to {verification_path}")

if __name__ == "__main__":
    run_verification()
