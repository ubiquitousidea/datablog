import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def main():
    filepath = "air+quality/AirQualityUCI.csv"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    print("Loading dataset...")
    # Read the CSV with semicolon delimiter and comma decimal separator
    df = pd.read_csv(filepath, sep=';', decimal=',')
    
    # Drop rows where Date is null (trailing empty rows in CSV)
    df = df.dropna(subset=['Date', 'Time'], how='all')
    
    print("Parsing datetime...")
    # Combine Date and Time into Datetime. Semicolon format has times like '18.00.00'
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].str.replace('.', ':'), format='%d/%m/%Y %H:%M:%S')
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    # Replace -200 (missing value indicator) with NaN so it doesn't distort the plots
    numeric_cols = [
        'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
        'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'T', 'RH', 'AH'
    ]
    for col in numeric_cols:
        df[col] = df[col].replace(-200.0, np.nan)
        
    print("Generating plot...")
    # 5 rows x 3 columns grid
    # Column 1: Ground Truth chemical concentrations
    # Column 2: Sensor responses
    # Column 3: Meteorological parameters
    
    # We define the mapping of subplot grid positions: (row, col)
    # 0-indexed: row in [0..4], col in [0..2]
    layout = {
        'CO(GT)':       (0, 0, 'CO Concentration (mg/m³)', '#1f77b4'),
        'PT08.S1(CO)':  (0, 1, 'CO Sensor Response (PT08.S1)', '#ff7f0e'),
        'T':            (0, 2, 'Temperature (°C)', '#31a354'),
        
        'NMHC(GT)':     (1, 0, 'NMHC Concentration (μg/m³)', '#008080'),
        'PT08.S2(NMHC)':(1, 1, 'NMHC Sensor Response (PT08.S2)', '#d62728'),
        'RH':           (1, 2, 'Relative Humidity (%)', '#756bb1'),
        
        'C6H6(GT)':     (2, 0, 'Benzene Concentration (μg/m³)', '#4682b4'),
        'PT08.S5(O3)':  (2, 1, 'O₃ Sensor Response (PT08.S5)', '#e6550d'),
        'AH':           (2, 2, 'Absolute Humidity (g/m³)', '#bcbd22'),
        
        'NOx(GT)':      (3, 0, 'NOx Concentration (ppb)', '#2ca02c'),
        'PT08.S3(NOx)': (3, 1, 'NOx Sensor Response (PT08.S3)', '#e377c2'),
        
        'NO2(GT)':      (4, 0, 'NO₂ Concentration (μg/m³)', '#9467bd'),
        'PT08.S4(NO2)': (4, 1, 'NO₂ Sensor Response (PT08.S4)', '#8c564b')
    }
    
    # Create the figure with high resolution
    fig, axes = plt.subplots(5, 3, figsize=(18, 15), sharex=True, dpi=120)
    fig.patch.set_facecolor('white')
    
    # Track which subplots are used
    used_slots = set()
    
    # Plot each variable
    for col_name, (r, c, title, color) in layout.items():
        ax = axes[r, c]
        used_slots.add((r, c))
        
        # Style subplot
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#cccccc')
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.tick_params(axis='both', which='both', colors='#666666', labelsize=9)
        ax.grid(True, linestyle='--', alpha=0.3, color='#e0e0e0')
        
        # Plot raw hourly data as thin light line
        ax.plot(df['Datetime'], df[col_name], color=color, alpha=0.18, linewidth=0.5, label='Hourly Data')
        
        # Plot 24-hour rolling average for trend line
        rolling_24h = df[col_name].rolling(window=24, min_periods=6).mean()
        ax.plot(df['Datetime'], rolling_24h, color=color, alpha=0.9, linewidth=1.5, label='24h Rolling Avg')
        
        # Title and labels
        ax.set_title(title, fontsize=11, color='#333333', fontweight='semibold', pad=6)
        
        # Format x-axis dates nicely on the bottom row
        if r == 4 or (c == 2 and r == 2):
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            
    # Hide unused subplots (bottom right slots in 5x3 grid)
    for r in range(5):
        for c in range(3):
            if (r, c) not in used_slots:
                fig.delaxes(axes[r, c])
                
    # Add overarching figure titles and adjust spacing
    fig.suptitle('UCI Air Quality Dataset - Complete Time Series (2004 - 2005)', 
                 fontsize=16, fontweight='bold', color='#111111', y=0.98)
    
    # Adjust layout to make room for titles and prevent overlaps
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot locally
    plot_filename = 'air_quality_plots.png'
    fig.savefig(plot_filename, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Plot saved successfully to {plot_filename}")
    
    # Copy to artifacts directory if available
    artifact_dir = os.environ.get('ANTIGRAVITY_ARTIFACT_DIR', '')
    if artifact_dir and os.path.exists(artifact_dir):
        dest_path = os.path.join(artifact_dir, plot_filename)
        shutil.copy(plot_filename, dest_path)
        print(f"Copied plot to artifact path: {dest_path}")
    else:
        print("ANTIGRAVITY_ARTIFACT_DIR environment variable not found or does not exist. Plot saved locally only.")

if __name__ == '__main__':
    main()
