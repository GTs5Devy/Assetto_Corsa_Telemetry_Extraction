import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

"""
UPDATED POST-SESSION ANALYSIS
Handles both old (30 columns) and new (31 columns) CSV formats
"""

def analyze_labeled_data(file_path):
    """Analyze pre-labeled telemetry data with improved error handling"""
    
    print("=" * 70)
    print("ML TRAINING DATA ANALYSIS")
    print("=" * 70)
    print()
    
    # Load data with error handling for mixed column formats
    print("Loading data...")
    try:
        # First, try to read with auto-detection
        df = pd.read_csv(file_path, sep=';', on_bad_lines='skip')
        
        # Check if we have the new format with YawGradient
        expected_cols_new = ['Lap', 'CarModel', 'Track', 'TrackPos', 'CurrentTime',
                             'YawRate', 'LateralAccel', 'LongitudinalAccel', 'VerticalAccel',
                             'SteerAngle', 'Speed', 'LocalVelX', 'LocalVelY', 'LocalVelZ',
                             'WheelSlipFL', 'WheelSlipFR', 'WheelSlipRL', 'WheelSlipRR',
                             'Throttle', 'Brake', 'Gear',
                             'SurfaceGrip', 'RoadTemp', 'AirTemp',
                             'Heading', 'Pitch', 'Roll',
                             'CarX', 'CarY', 'CarZ',
                             'SlipDiff', 'YawGradient', 'Label']
        
        expected_cols_old = ['Lap', 'CarModel', 'Track', 'TrackPos', 'CurrentTime',
                             'YawRate', 'LateralAccel', 'LongitudinalAccel', 'VerticalAccel',
                             'SteerAngle', 'Speed', 'LocalVelX', 'LocalVelY', 'LocalVelZ',
                             'WheelSlipFL', 'WheelSlipFR', 'WheelSlipRL', 'WheelSlipRR',
                             'Throttle', 'Brake', 'Gear',
                             'SurfaceGrip', 'RoadTemp', 'AirTemp',
                             'Heading', 'Pitch', 'Roll',
                             'CarX', 'CarY', 'CarZ',
                             'SlipDiff', 'Label']
        
        # Detect format
        if len(df.columns) == 31:
            print("✅ Detected new format (with YawGradient)")
            df.columns = expected_cols_new
        elif len(df.columns) == 30:
            print("✅ Detected old format (without YawGradient)")
            df.columns = expected_cols_old
            df['YawGradient'] = 0.0  # Add dummy column for compatibility
        else:
            print(f"⚠️  Warning: Found {len(df.columns)} columns, expected 30 or 31")
            print(f"   Columns found: {list(df.columns)}")
            print("   Attempting to continue with available columns...")
            
        print(f"✅ Loaded {len(df):,} data points from {df['Lap'].nunique()} laps\n")
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        print("\n💡 TIP: If you have mixed old/new data, start with a fresh CSV:")
        print("   1. Delete or rename your old log.csv")
        print("   2. Run a new session with the improved logger")
        return
    
    # === OVERALL STATISTICS ===
    print("=" * 70)
    print("OVERALL DATASET STATISTICS")
    print("=" * 70)
    
    total = len(df)
    
    # Check if Label column exists
    if 'Label' not in df.columns:
        print("❌ Error: 'Label' column not found in CSV")
        print(f"   Available columns: {list(df.columns)}")
        return
    
    label_counts = df['Label'].value_counts()
    
    print("\n📊 CLASS DISTRIBUTION:")
    for label in ['Neutral', 'Understeer', 'Oversteer']:
        count = label_counts.get(label, 0)
        pct = (count / total) * 100
        
        # Visual bar
        bar_length = int(pct / 2)
        bar = "█" * bar_length
        
        if label == 'Neutral':
            emoji = "🟢"
        elif label == 'Understeer':
            emoji = "🔵"
        else:
            emoji = "🔴"
        
        print(f"  {emoji} {label:12s}: {count:6,} ({pct:5.1f}%) {bar}")
    
    # === BALANCE CHECK ===
    print("\n" + "=" * 70)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 70)
    
    neutral_pct = (label_counts.get('Neutral', 0) / total) * 100
    understeer_pct = (label_counts.get('Understeer', 0) / total) * 100
    oversteer_pct = (label_counts.get('Oversteer', 0) / total) * 100
    
    issues = []
    
    # Check balance
    if neutral_pct > 80:
        issues.append("⚠️  Too much neutral data (>80%)")
        issues.append("   → Need 5-10 more laps of aggressive driving")
    elif neutral_pct < 50:
        issues.append("⚠️  Too little neutral data (<50%)")
        issues.append("   → Need 5 more laps of smooth, controlled driving")
    else:
        print("✅ Good neutral percentage (50-80%)")
    
    if understeer_pct < 8:
        issues.append("⚠️  Need more understeer data (<8%)")
        issues.append("   → Do 5 laps: brake late, turn hard, throttle early")
    else:
        print(f"✅ Good understeer percentage ({understeer_pct:.1f}%)")
    
    if oversteer_pct < 8:
        issues.append("⚠️  Need more oversteer data (<8%)")
        issues.append("   → Do 5 laps: trail brake, sudden throttle")
    else:
        print(f"✅ Good oversteer percentage ({oversteer_pct:.1f}%)")
    
    # === TRACK CONDITIONS ===
    print("\n" + "=" * 70)
    print("TRACK CONDITIONS")
    print("=" * 70)
    
    avg_grip = df['SurfaceGrip'].mean()
    min_grip = df['SurfaceGrip'].min()
    max_grip = df['SurfaceGrip'].max()
    grip_range = max_grip - min_grip
    
    print(f"\n🌦️  Surface Grip:")
    print(f"   Average: {avg_grip:.3f}")
    print(f"   Range:   {min_grip:.3f} - {max_grip:.3f} (Δ {grip_range:.3f})")
    
    if grip_range < 0.15:
        issues.append("⚠️  CRITICAL: Need wet track testing!")
        issues.append("   → Your RQ asks about 'varying track conditions'")
        issues.append("   → Collect 10-15 laps in WET conditions")
    else:
        print("✅ Good grip variation (tested multiple conditions)")
    
    # === LAP ANALYSIS ===
    print("\n" + "=" * 70)
    print("LAP-BY-LAP BREAKDOWN")
    print("=" * 70)
    print()
    
    lap_summary = df.groupby(['Lap', 'Label']).size().unstack(fill_value=0)
    
    # Ensure all label columns exist
    for label in ['Neutral', 'Understeer', 'Oversteer']:
        if label not in lap_summary.columns:
            lap_summary[label] = 0
    
    lap_summary['Total'] = lap_summary.sum(axis=1)
    
    # Add percentages
    for col in ['Neutral', 'Understeer', 'Oversteer']:
        lap_summary[f'{col}%'] = (lap_summary[col] / lap_summary['Total'] * 100).round(1)
    
    # Add grip info
    lap_summary['AvgGrip'] = df.groupby('Lap')['SurfaceGrip'].mean().round(3)
    
    # Identify lap types
    def classify_lap(row):
        if row.get('Neutral%', 0) > 75:
            return "🟢 Neutral"
        elif row.get('Understeer%', 0) > 30:
            return "🔵 Understeer"
        elif row.get('Oversteer%', 0) > 30:
            return "🔴 Oversteer"
        else:
            return "🟡 Mixed"
    
    lap_summary['Type'] = lap_summary.apply(classify_lap, axis=1)
    
    print(lap_summary[['Type', 'Neutral', 'Understeer', 'Oversteer', 'AvgGrip']].to_string())
    
    # === FINAL RECOMMENDATIONS ===
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    if not issues:
        print("dataset looks ready for ML training!")
        print()
        print("Next steps:")
        print("  1. Use this CSV for LSTM training")
        print("  2. Split data: 70% train, 15% validation, 15% test")
        print("  3. Consider collecting 10 more laps for robustness")
    else:
        print("Action items before ML training:\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    
    # === CREATE VISUALIZATIONS ===
    print("\n" + "=" * 70)
    print("Creating visualizations...")
    print("=" * 70)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Overall distribution
        colors = {'Neutral': 'green', 'Understeer': 'blue', 'Oversteer': 'red'}
        label_counts_sorted = label_counts.reindex(['Neutral', 'Understeer', 'Oversteer'], fill_value=0)
        bars = axes[0, 0].bar(label_counts_sorted.index, label_counts_sorted.values, 
                               color=[colors[l] for l in label_counts_sorted.index])
        axes[0, 0].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Count')
        for bar, (label, count) in zip(bars, label_counts_sorted.items()):
            pct = (count / total) * 100
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Distribution by lap
        lap_labels = df.groupby(['Lap', 'Label']).size().unstack(fill_value=0)
        lap_labels = lap_labels.reindex(columns=['Neutral', 'Understeer', 'Oversteer'], fill_value=0)
        lap_labels.plot(kind='bar', stacked=True, ax=axes[0, 1], 
                        color=colors, width=0.8)
        axes[0, 1].set_title('Class Distribution by Lap', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Lap')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend(title='Label', loc='upper left')
        axes[0, 1].tick_params(axis='x', rotation=0)
        
        # Plot 3: Slip differential distribution
        axes[1, 0].hist(df['SlipDiff'], bins=50, edgecolor='black', alpha=0.7, color='gray')
        axes[1, 0].axvline(-0.08, color='blue', linestyle='--', linewidth=2, label='Understeer threshold')
        axes[1, 0].axvline(0.08, color='red', linestyle='--', linewidth=2, label='Oversteer threshold')
        axes[1, 0].axvline(0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Neutral')
        axes[1, 0].set_title('Slip Differential Distribution\n(Improved Bergman Implementation)', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Slip Differential (Rear - Front)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Track conditions over time
        axes[1, 1].plot(df.index, df['SurfaceGrip'], alpha=0.6, linewidth=0.5, color='purple')
        axes[1, 1].set_title('Surface Grip Over Session', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Data Point Index')
        axes[1, 1].set_ylabel('Surface Grip')
        axes[1, 1].axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Dry (1.0)')
        axes[1, 1].axhline(0.7, color='orange', linestyle='--', alpha=0.5, label='Typical Wet (~0.7)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig('ml_training_analysis.png', dpi=150, bbox_inches='tight')
        print("Saved: ml_training_analysis.png\n")
        plt.show()
    except Exception as e:
        print(f"Could not create visualizations: {e}")
    
    print("=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    # Run analysis on your labeled data
    import sys
    
    # Check if file path provided as argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "third_party/log.csv"  # Default path
    
    analyze_labeled_data(file_path)