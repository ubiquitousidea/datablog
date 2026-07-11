import os
import pandas as pd
import sys

# Add parent directory to path so we can import graph_extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graph_extractor import extract_points_from_graph

# Map each image file name to its respective y_limits based on visual scale analysis
Y_LIMITS_MAP = {
    "beef_com.png": (290.0, 370.0),
    "cl1_com.png": (50.0, 130.0),
    "co1_com.png": (50.0, 130.0),
    "djm_com.png": (170.0, 260.0),
    "hg1_com.png": (4.0, 7.0),
    "ho1_com.png": (1.5, 5.0),
    "ioe_com.png": (720.0, 840.0),
    "jbp_com.png": (2950.0, 3400.0),
    "lb1_com.png": (500.0, 725.0),
    "ng1_com.png": (2.0, 6.0),
    "ngeu_com.png": (20.0, 70.0),
    "s 1_com.png": (900.0, 1300.0),
    "si_com.png": (8000.0, 10000.0),
    "w 1_com.png": (475.0, 725.0),
    "xagusd_cur.png": (30.0, 130.0),
    "xal1_com.png": (80.0, 160.0),
    "xauusd_cur.png": (3000.0, 5750.0),
    "xb1_com.png": (1.5, 4.25),
}

def main():
    dir_path = os.path.dirname(__file__)
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".png") and not f.startswith("all_")])
    
    x_limits = (0.0, 11.0)
    date_limits = ("2025-08-01", "2026-07-01")
    color = "blue"
    
    all_data_frames = []
    
    print(f"Starting batch extraction for {len(files)} files...")
    for idx, filename in enumerate(files):
        if filename not in Y_LIMITS_MAP:
            print(f"Warning: {filename} has no defined Y limits. Skipping.")
            continue
            
        y_limits = Y_LIMITS_MAP[filename]
        image_path = os.path.join(dir_path, filename)
        
        print(f"[{idx+1}/{len(files)}] Processing {filename} with y_limits={y_limits}...")
        try:
            pts = extract_points_from_graph(
                image_path,
                x_limits=x_limits,
                y_limits=y_limits,
                color=color,
                date_limits=date_limits
            )
            
            # 1. Save individual CSV
            df = pd.DataFrame(pts, columns=["x_data", "y_data", "timestamp"])
            base_name = os.path.splitext(filename)[0]
            output_csv_path = os.path.join(dir_path, f"{base_name}_extracted.csv")
            df.to_csv(output_csv_path, index=False)
            
            # 2. Append to combined dataframe
            df_combined = df.copy()
            df_combined.insert(0, "commodity", base_name)
            all_data_frames.append(df_combined)
            
            print(f"  Successfully extracted {len(pts)} points. Saved to {f'{base_name}_extracted.csv'}")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            
    # 3. Save consolidated CSV
    if all_data_frames:
        consolidated_df = pd.concat(all_data_frames, ignore_index=True)
        consolidated_csv_path = os.path.join(dir_path, "all_commodity_prices.csv")
        consolidated_df.to_csv(consolidated_csv_path, index=False)
        print(f"\nAll extractions complete! Saved consolidated file to {consolidated_csv_path}")

if __name__ == "__main__":
    main()
