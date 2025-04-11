from pathlib import Path
from data.loader import DataLoader
from visualization.plots import AQIVisualizer
from analysis.time_series import TimeSeriesAnalyzer

def main():
    # Initialize components
    data_loader = DataLoader("data/raw/Wollongong_09042022_10042025.xlsx")
    visualizer = AQIVisualizer()
    analyzer = TimeSeriesAnalyzer()

    # Data pipeline
    print("Loading and cleaning data...")
    df = data_loader.load_and_clean()
    
    # Show summary
    print("\n--- Summary Info ---")
    print(f"Data covers: {df.index.min()} to {df.index.max()}")
    print(f"Total entries: {len(df)}")
    print(df.describe())

    # Visualization pipeline
    print("\nGenerating visualizations...")
    visualizer.plot_time_series(df)
    visualizer.plot_correlation_heatmap(df)
    visualizer.plot_monthly_boxplots(df)
    visualizer.plot_year_over_year(df)
    
    # Analysis pipeline
    print("\nPerforming time series analysis...")
    analyzer.seasonal_decomposition(df)

    print("\nAll plots saved to docs/images/")

if __name__ == "__main__":
    main()