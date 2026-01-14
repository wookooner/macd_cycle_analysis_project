import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import warnings
from pathlib import Path
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# Required packages: pip install scikit-learn scipy
try:
    from sklearn.preprocessing import MinMaxScaler
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False
    print("For advanced features, install: pip install scikit-learn scipy")

warnings.filterwarnings('ignore')

# Professional dark theme with English only
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('dark_background')

# Enhanced readability settings
plt.rcParams['figure.facecolor'] = '#1A1A1A'
plt.rcParams['axes.facecolor'] = '#2D2D2D'
plt.rcParams['axes.edgecolor'] = '#606060'
plt.rcParams['grid.color'] = '#404040'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['text.color'] = '#FFFFFF'
plt.rcParams['axes.labelcolor'] = '#FFFFFF'
plt.rcParams['xtick.color'] = '#FFFFFF'
plt.rcParams['ytick.color'] = '#FFFFFF'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

class CycleFeatureVisualizer:
    """Enhanced Cycle Feature Visualizer with Improved Readability"""
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            self.data_path = "data/cycle_data/structured/cycles_4h.parquet"
        else:
            self.data_path = data_path
            
        self.df = None
        self.features_df = None
        self.available_features = {}
        self.filtered_df = None
        
        # Enhanced color palette - better contrast
        self.colors = {
            'up': '#00E676',          # Bright Green
            'down': '#FF5252',        # Bright Red
            'UP_PROFIT': '#00C853',   # Deep Green
            'UP_LOSS': '#69F0AE',     # Light Green
            'DOWN_PROFIT': '#FF6D00', # Deep Orange
            'DOWN_LOSS': '#FF1744',   # Deep Red
            'neutral': '#9E9E9E',
            'accent': '#00B0FF',
            'primary': '#00ACC1',
            'secondary': '#78909C'
        }
        
        self.category_labels = {
            'UP_PROFIT': 'Up + Profit',
            'UP_LOSS': 'Up + Loss', 
            'DOWN_PROFIT': 'Down + Profit',
            'DOWN_LOSS': 'Down + Loss'
        }
    
    def show_main_menu(self) -> int:
        """Show main menu and get user choice"""
        print("\n" + "="*75)
        print("Cycle Feature Visualizer - Main Menu")
        print("="*75)
        print("1. Start New Analysis")
        print("2. Exit Program")
        print("="*75)
        
        while True:
            try:
                choice = int(input("Enter choice (1-2): "))
                if choice in [1, 2]:
                    return choice
                else:
                    print("Please enter 1 or 2.")
            except ValueError:
                print("Please enter a number.")
            except KeyboardInterrupt:
                print("\nExiting program.")
                return 2
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess data"""
        try:
            print(f"Loading data: {self.data_path}")
            self.df = pd.read_parquet(self.data_path)
            print(f"Data loaded successfully: {len(self.df)} cycles")
            self._convert_datetime_columns()
            return self.df
        except Exception as e:
            print(f"Data loading error: {e}")
            return None
    
    def _convert_datetime_columns(self):
        """Safely convert date columns to datetime"""
        for col in ['start_date', 'end_date']:
            if col in self.df.columns:
                try:
                    if self.df[col].dtype in ['int64', 'float64']:
                        self.df[f'{col.split("_")[0]}_datetime'] = pd.to_datetime(
                            self.df[col], unit='s')
                    else:
                        self.df[f'{col.split("_")[0]}_datetime'] = pd.to_datetime(
                            self.df[col])
                except Exception as e:
                    print(f"Error converting {col}: {e}")
                    self.df[f'{col.split("_")[0]}_datetime'] = pd.date_range(
                        start='2015-01-01', periods=len(self.df), freq='4H')
    
    def flatten_features(self) -> pd.DataFrame:
        """Flatten nested cycle_features"""
        if self.df is None:
            print("Please load data first.")
            return None
        
        print("Flattening feature data...")
        flattened_records = []
        
        for idx, row in self.df.iterrows():
            record = {
                'cycle_id': row['cycle_id'],
                'timeframe': row['timeframe'],
                'start_datetime': row['start_datetime'],
                'end_datetime': row['end_datetime'],
                'cycle_type': row['cycle_type'],
                'duration_candles': row['duration_candles'],
                'category': row['category']
            }
            
            if isinstance(row['cycle_features'], dict):
                features = row['cycle_features']
                for category, feature_dict in features.items():
                    if isinstance(feature_dict, dict):
                        for feature_name, value in feature_dict.items():
                            flattened_key = f"{category}_{feature_name}"
                            record[flattened_key] = value
                            
            flattened_records.append(record)
        
        self.features_df = pd.DataFrame(flattened_records)
        self._create_composite_category()
        self._categorize_features()
        
        print(f"Feature flattening completed: {len(self.features_df)} rows, "
              f"{len(self.features_df.columns)} columns")
        
        if 'composite_category' in self.features_df.columns:
            print("4-Way category distribution:")
            for cat, count in self.features_df['composite_category'].value_counts().items():
                print(f"  - {self.category_labels.get(cat, cat)}: {count} cycles")
        
        return self.features_df
    
    def _create_composite_category(self):
        """Create 4-way composite categories"""
        if 'change_price_pct' not in self.features_df.columns:
            return
        
        def get_composite_category(row):
            cycle_type = row['cycle_type'].upper()
            price_change = row['change_price_pct']
            if pd.isna(price_change):
                return f"{cycle_type}_UNKNOWN"
            profit_suffix = "PROFIT" if price_change > 0 else "LOSS"
            return f"{cycle_type}_{profit_suffix}"
        
        self.features_df['composite_category'] = self.features_df.apply(
            get_composite_category, axis=1)
    
    def _categorize_features(self):
        """Categorize features by type"""
        self.available_features = {
            'shape': [], 'strength': [], 'start': [], 'end': [],
            'change': [], 'volatility': [], 'aggregate': []
        }
        
        for col in self.features_df.columns:
            if col.startswith(tuple(self.available_features.keys())):
                category = col.split('_')[0]
                if category in self.available_features:
                    self.available_features[category].append(col)
    
    def show_available_features(self):
        """Display available features"""
        print("\nAvailable Features:")
        print("=" * 50)
        
        for category, features in self.available_features.items():
            if features:
                print(f"\n{category.upper()} ({len(features)} features):")
                for i, feature in enumerate(features, 1):
                    print(f"   {i:2d}. {feature}")
    
    def get_numeric_features(self) -> List[str]:
        """Get list of numeric features"""
        if self.features_df is None:
            return []
        
        exclude_cols = ['cycle_id', 'timeframe', 'start_datetime', 'end_datetime', 
                       'cycle_type', 'category', 'composite_category']
        
        numeric_features = [col for col in self.features_df.columns 
                          if self.features_df[col].dtype in ['int64', 'float64'] 
                          and col not in exclude_cols]
        
        return sorted(numeric_features)
    
    def parse_filter_expression(self, filter_str):
        """Parse filter expression"""
        filters = []
        if not filter_str or not filter_str.strip():
            return filters
        
        parts = [part.strip() for part in filter_str.split(',') if part.strip()]
        
        for part in parts:
            if not part:
                continue
            
            if part.lower() in ['up', 'down']:
                filters.append(('cycle_type', '==', part.lower()))
                continue
            
            range_match = re.match(
                r'([+-]?\d*\.?\d+)\s*([<>]=?)\s*(\w+)\s*([<>]=?)\s*([+-]?\d*\.?\d+)', 
                part)
            if range_match:
                val1, op1, feature, op2, val2 = range_match.groups()
                val1, val2 = float(val1), float(val2)
                if '<' in op1 and '<' in op2:
                    filters.append((feature, '>', val1))
                    filters.append((feature, '<', val2))
                elif '>' in op1 and '>' in op2:
                    filters.append((feature, '>', val2))
                    filters.append((feature, '<', val1))
                continue
            
            comp_match = re.match(
                r'(\w+)\s*([<>]=?|==|!=)\s*([+-]?\d*\.?\d+)', part)
            if comp_match:
                feature, op, value = comp_match.groups()
                filters.append((feature, op, float(value)))
                continue
            
            comp_match2 = re.match(
                r'([+-]?\d*\.?\d+)\s*([<>]=?)\s*(\w+)', part)
            if comp_match2:
                value, op, feature = comp_match2.groups()
                if '<' in op:
                    op = '>' if '<' == op else '>='
                elif '>' in op:
                    op = '<' if '>' == op else '<='
                filters.append((feature, op, float(value)))
                continue
        
        return filters
    
    def apply_filters(self, df, filters):
        """Apply filters to dataframe"""
        if not filters:
            return df
        
        result_df = df.copy()
        
        for feature, op, value in filters:
            if feature not in result_df.columns:
                possible_names = [col for col in result_df.columns if feature in col]
                if possible_names:
                    feature = possible_names[0]
                else:
                    print(f"Warning: '{feature}' column not found.")
                    continue
            
            if op == '==':
                result_df = result_df[result_df[feature] == value]
            elif op == '!=':
                result_df = result_df[result_df[feature] != value]
            elif op == '>':
                result_df = result_df[result_df[feature] > value]
            elif op == '>=':
                result_df = result_df[result_df[feature] >= value]
            elif op == '<':
                result_df = result_df[result_df[feature] < value]
            elif op == '<=':
                result_df = result_df[result_df[feature] <= value]
        
        return result_df
    
    def get_user_feature_choice(self) -> str:
        """Get user's feature choice"""
        print("\n" + "="*60)
        print("Select a feature to analyze")
        print("="*60)
        
        all_features = []
        for category, features in self.available_features.items():
            if features:
                print(f"\n{category.upper()} Category:")
                for feature in features:
                    all_features.append(feature)
                    print(f"   {len(all_features):2d}. {feature}")
        
        if not all_features:
            print("No available features.")
            return None
        
        while True:
            try:
                print(f"\nEnter number (1-{len(all_features)}): ", end="")
                choice = int(input()) - 1
                
                if 0 <= choice < len(all_features):
                    selected_feature = all_features[choice]
                    print(f"Selected feature: {selected_feature}")
                    return selected_feature
                else:
                    print(f"Please enter a number between 1 and {len(all_features)}.")
            except ValueError:
                print("Please enter a number.")
            except KeyboardInterrupt:
                print("\nReturning to main menu.")
                return None
    
    def get_user_correlation_target_choice(self, main_feature: str) -> str:
        """Get user's correlation target feature choice"""
        print("\n" + "="*60)
        print("Select target feature for correlation analysis")
        print("="*60)
        
        numeric_features = self.get_numeric_features()
        if main_feature in numeric_features:
            numeric_features.remove(main_feature)
        
        if not numeric_features:
            print("No available numeric features for correlation analysis.")
            return None
        
        categorized_features = {}
        for feature in numeric_features:
            category = feature.split('_')[0] if '_' in feature else 'other'
            if category not in categorized_features:
                categorized_features[category] = []
            categorized_features[category].append(feature)
        
        all_features = []
        for category in sorted(categorized_features.keys()):
            features = categorized_features[category]
            if features:
                print(f"\n{category.upper()} Category:")
                for feature in features:
                    all_features.append(feature)
                    print(f"   {len(all_features):2d}. {feature}")
        
        while True:
            try:
                print(f"\nEnter number (1-{len(all_features)}): ", end="")
                choice = int(input()) - 1
                
                if 0 <= choice < len(all_features):
                    selected_feature = all_features[choice]
                    print(f"Selected correlation target: {selected_feature}")
                    return selected_feature
                else:
                    print(f"Please enter a number between 1 and {len(all_features)}.")
            except ValueError:
                print("Please enter a number.")
            except KeyboardInterrupt:
                print("\nReturning to main menu.")
                return None
    
    def get_user_filter(self) -> list:
        """Get user filter input"""
        print("\n" + "="*50)
        print("Filter Settings (Enter=All cycles)")
        print("="*50)
        print("Examples: up, change_price_pct>0, 10<duration_candles<50")
        
        filter_str = input("Enter filter conditions: ").strip()
        
        if filter_str:
            filters = self.parse_filter_expression(filter_str)
            print(f"Applied filters: {len(filters)} conditions")
            return filters
        else:
            print("No filters - analyzing all cycles")
            return None
    
    def get_user_cycle_filter(self) -> str:
        """Get cycle filtering choice"""
        print("\n" + "="*40)
        print("Select cycle filtering")
        print("="*40)
        
        filters = {
            1: ("all", "All cycles"),
            2: ("up", "Up cycles only"),
            3: ("down", "Down cycles only"),
            4: ("4way", "4-Way categories"),
            5: ("profit_only", "Profit cycles only"),
            6: ("loss_only", "Loss cycles only")
        }
        
        for num, (filter_type, desc) in filters.items():
            print(f"   {num}. {desc}")
        
        while True:
            try:
                print(f"\nEnter number (1-{len(filters)}): ", end="")
                choice = int(input())
                
                if choice in filters:
                    filter_type, desc = filters[choice]
                    print(f"Selected filter: {desc}")
                    return filter_type
                else:
                    print(f"Please enter a number between 1 and {len(filters)}.")
            except ValueError:
                print("Please enter a number.")
            except KeyboardInterrupt:
                print("\nReturning to main menu.")
                return None
    
    def get_user_normalization_choice(self) -> str:
        """Get normalization method choice"""
        print("\n" + "="*50)
        print("Select normalization method")
        print("="*50)
        
        methods = {
            1: ("minmax", "Min-Max Scaling (0~1)"),
            2: ("zscore", "Z-Score Standardization"),
            3: ("robust", "Robust Scaling"),
            4: ("none", "No normalization")
        }
        
        for num, (method, desc) in methods.items():
            print(f"   {num}. {desc}")
        
        while True:
            try:
                print(f"\nEnter number (1-{len(methods)}): ", end="")
                choice = int(input())
                
                if choice in methods:
                    method, desc = methods[choice]
                    print(f"Selected normalization: {desc}")
                    return method
                else:
                    print(f"Please enter a number between 1 and {len(methods)}.")
            except ValueError:
                print("Please enter a number.")
            except KeyboardInterrupt:
                print("\nReturning to main menu.")
                return None
    
    def create_all_visualizations(self, 
                                 feature_name: str,
                                 cycle_filter: str = "all",
                                 normalization: str = "minmax",
                                 filters: list = None,
                                 correlation_target: str = None) -> None:
        """Create all visualizations for selected feature"""
        
        if self.features_df is None:
            print("Please run flatten_features() first.")
            return
        
        if feature_name not in self.features_df.columns:
            print(f"'{feature_name}' feature not found.")
            return
        
        working_df = self.features_df.copy()
        
        if filters:
            print(f"\nApplying filters...")
            working_df = self.apply_filters(working_df, filters)
            print(f"After filtering: {len(working_df)} cycles")
        
        if cycle_filter == "up":
            working_df = working_df[working_df['cycle_type'] == 'up']
        elif cycle_filter == "down":
            working_df = working_df[working_df['cycle_type'] == 'down']
        elif cycle_filter == "profit_only":
            working_df = working_df[working_df['change_price_pct'] > 0]
        elif cycle_filter == "loss_only":
            working_df = working_df[working_df['change_price_pct'] <= 0]
        
        cols_to_check = [feature_name]
        if correlation_target:
            cols_to_check.append(correlation_target)
            
        plot_df = working_df.dropna(subset=cols_to_check).sort_values('start_datetime')
        
        if len(plot_df) == 0:
            print(f"No valid data for '{feature_name}' feature.")
            return
        
        self.filtered_df = plot_df
        
        print(f"\nGenerating visualizations: {feature_name}")
        print(f"Data count: {len(plot_df)}")
        
        use_4way = (cycle_filter == "4way" and 'composite_category' in plot_df.columns)
        
        # Generate visualizations
        self._create_normalized_timeseries(plot_df, feature_name, normalization, use_4way)
        self._create_distribution_analysis(plot_df, feature_name, use_4way)
        
        if correlation_target and correlation_target in plot_df.columns:
            self._create_correlation_analysis(plot_df, feature_name, correlation_target, use_4way)
        
        print("\nAll visualizations completed!")
    
    def _create_normalized_timeseries(self, plot_df, feature_name, normalization, use_4way=False):
        """Original time series visualization with fixed date formatting"""
        try:
            feature_data = plot_df[feature_name].values
            
            # Normalization
            if normalization == "minmax":
                if ADVANCED_FEATURES:
                    scaler = MinMaxScaler()
                    normalized_data = scaler.fit_transform(
                        feature_data.reshape(-1, 1)).flatten()
                else:
                    min_val, max_val = feature_data.min(), feature_data.max()
                    normalized_data = ((feature_data - min_val) / (max_val - min_val) 
                                     if max_val != min_val else np.zeros_like(feature_data))
                norm_label = "Min-Max"
            elif normalization == "zscore":
                normalized_data = ((feature_data - np.mean(feature_data)) / 
                                 np.std(feature_data))
                norm_label = "Z-Score"
            elif normalization == "robust":
                median = np.median(feature_data)
                mad = np.median(np.abs(feature_data - median))
                normalized_data = ((feature_data - median) / (mad * 1.4826) 
                                 if mad != 0 else np.zeros_like(feature_data))
                norm_label = "Robust"
            else:
                normalized_data = feature_data
                norm_label = "Original"
            
            plot_df = plot_df.copy()
            plot_df['normalized_feature'] = normalized_data
            
            # Create figure with better spacing
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), dpi=100)
            fig.subplots_adjust(hspace=0.35, top=0.94, bottom=0.10, left=0.08, right=0.96)
            
            # Plot 1: Original data
            if use_4way and 'composite_category' in plot_df.columns:
                for cat in sorted(plot_df['composite_category'].unique()):
                    cat_data = plot_df[plot_df['composite_category'] == cat]
                    if len(cat_data) > 0:
                        ax1.scatter(cat_data['start_datetime'], cat_data[feature_name],
                                  c=self.colors.get(cat, self.colors['neutral']),
                                  label=f'{self.category_labels.get(cat, cat)} (n={len(cat_data)})',
                                  alpha=0.5, s=30, edgecolors='none')
            else:
                for cycle_type in sorted(plot_df['cycle_type'].unique()):
                    type_data = plot_df[plot_df['cycle_type'] == cycle_type]
                    ax1.scatter(type_data['start_datetime'], type_data[feature_name],
                              c=self.colors[cycle_type],
                              label=f'{cycle_type.upper()} (n={len(type_data)})',
                              alpha=0.5, s=30, edgecolors='none')
            
            ax1.set_ylabel(f'{feature_name.replace("_", " ").title()}', 
                          fontsize=13, fontweight='bold')
            ax1.set_title(f'Original Data Time Series', 
                         fontsize=14, fontweight='bold', pad=15)
            ax1.legend(loc='best', framealpha=0.9, edgecolor='white', ncol=2)
            ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax1.tick_params(axis='both', labelsize=11)
            
            # Format x-axis dates for ax1
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Plot 2: Normalized data
            if use_4way and 'composite_category' in plot_df.columns:
                for cat in sorted(plot_df['composite_category'].unique()):
                    cat_data = plot_df[plot_df['composite_category'] == cat]
                    if len(cat_data) > 0:
                        ax2.plot(cat_data['start_datetime'], cat_data['normalized_feature'],
                               color=self.colors.get(cat, self.colors['neutral']),
                               alpha=0.3, linewidth=1)
                        ax2.scatter(cat_data['start_datetime'], cat_data['normalized_feature'],
                                  c=self.colors.get(cat, self.colors['neutral']),
                                  label=self.category_labels.get(cat, cat),
                                  alpha=0.5, s=30, edgecolors='none')
            else:
                for cycle_type in sorted(plot_df['cycle_type'].unique()):
                    type_data = plot_df[plot_df['cycle_type'] == cycle_type]
                    ax2.plot(type_data['start_datetime'], type_data['normalized_feature'],
                           color=self.colors[cycle_type], alpha=0.3, linewidth=1)
                    ax2.scatter(type_data['start_datetime'], type_data['normalized_feature'],
                              c=self.colors[cycle_type],
                              label=f'{cycle_type.upper()}',
                              alpha=0.5, s=30, edgecolors='none')
            
            # Trend line
            if len(plot_df) > 2:
                z = np.polyfit(range(len(plot_df)), plot_df['normalized_feature'], 1)
                trend_line = np.poly1d(z)
                ax2.plot(plot_df['start_datetime'], trend_line(range(len(plot_df))),
                        color='#FFD700', linestyle='--', linewidth=2.5, alpha=0.8,
                        label=f'Trend (slope={z[0]:.4f})')
            
            ax2.set_xlabel('Time', fontsize=13, fontweight='bold')
            ax2.set_ylabel(f'{feature_name.replace("_", " ").title()} ({norm_label})', 
                          fontsize=13, fontweight='bold')
            ax2.set_title(f'Normalized Data Time Series', 
                         fontsize=14, fontweight='bold', pad=15)
            ax2.legend(loc='best', framealpha=0.9, edgecolor='white', ncol=2)
            ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax2.tick_params(axis='both', labelsize=11)
            
            # Format x-axis dates for ax2
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            fig.suptitle(f'Time Series: {feature_name.replace("_", " ").title()}',
                        fontsize=16, fontweight='bold', y=0.98)
            
            plt.show()
            
        except Exception as e:
            print(f"Time series visualization error: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_distribution_analysis(self, plot_df, feature_name, use_4way=False):
        """Distribution: Histogram + Pie Chart + Binned Distribution"""
        try:
            feature_data = plot_df[feature_name].dropna()
            
            if len(feature_data) == 0:
                print(f"No data available for {feature_name}")
                return
            
            # Determine categories
            if use_4way and 'composite_category' in plot_df.columns:
                categories = sorted(plot_df['composite_category'].unique())
                cat_labels = [self.category_labels.get(cat, cat) for cat in categories]
                cat_column = 'composite_category'
            else:
                categories = sorted(plot_df['cycle_type'].unique())
                cat_labels = [cat.upper() for cat in categories]
                cat_column = 'cycle_type'
            
            # Create figure with 3 subplots
            fig = plt.figure(figsize=(16, 5), dpi=100)
            fig.subplots_adjust(hspace=0.25, wspace=0.3, top=0.9, 
                              bottom=0.12, left=0.06, right=0.97)
            
            ax1 = plt.subplot(1, 3, 1)
            ax2 = plt.subplot(1, 3, 2)
            ax3 = plt.subplot(1, 3, 3)
            
            # Plot 1: Histogram
            for cat, label in zip(categories, cat_labels):
                cat_data = plot_df[plot_df[cat_column] == cat][feature_name].dropna()
                if len(cat_data) > 0:
                    color = (self.colors.get(cat, self.colors['neutral']) if use_4way 
                            else self.colors[cat])
                    ax1.hist(cat_data, alpha=0.6, bins=25, color=color,
                            label=label, edgecolor='white', linewidth=0.8)
            
            ax1.set_xlabel(feature_name.replace('_', ' ').title(), 
                          fontsize=12, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax1.set_title('Histogram Distribution', fontsize=13, fontweight='bold', pad=12)
            ax1.legend(loc='upper right', framealpha=0.9, edgecolor='white')
            ax1.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
            ax1.tick_params(axis='both', labelsize=10)
            
            # Plot 2: KDE (Kernel Density Estimation) Plot
            for cat, label in zip(categories, cat_labels):
                cat_data = plot_df[plot_df[cat_column] == cat][feature_name].dropna()
                if len(cat_data) > 5:  # Need enough data for KDE
                    color = (self.colors.get(cat, self.colors['neutral']) if use_4way 
                            else self.colors[cat])
                    
                    # Calculate KDE
                    from scipy.stats import gaussian_kde
                    try:
                        kde = gaussian_kde(cat_data)
                        x_range = np.linspace(cat_data.min(), cat_data.max(), 200)
                        density = kde(x_range)
                        
                        # Plot KDE curve
                        ax2.plot(x_range, density, color=color, linewidth=2.5, 
                                label=label, alpha=0.8)
                        ax2.fill_between(x_range, density, alpha=0.3, color=color)
                        
                        # Add mean and median lines
                        mean_val = cat_data.mean()
                        median_val = cat_data.median()
                        
                        ax2.axvline(mean_val, color=color, linestyle='--', 
                                   linewidth=1.5, alpha=0.6)
                        ax2.axvline(median_val, color=color, linestyle=':', 
                                   linewidth=1.5, alpha=0.6)
                    except:
                        pass
            
            ax2.set_xlabel(feature_name.replace('_', ' ').title(), 
                          fontsize=12, fontweight='bold')
            ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax2.set_title('Probability Density (KDE)\n-- Mean | : Median', 
                         fontsize=13, fontweight='bold', pad=12)
            ax2.legend(loc='upper right', framealpha=0.9, edgecolor='white', fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
            ax2.tick_params(axis='both', labelsize=10)
            
            # Plot 3: Binned Distribution Heatmap
            n_bins = min(10, int(np.sqrt(len(feature_data))))
            bins = np.linspace(feature_data.min(), feature_data.max(), n_bins + 1)
            bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(n_bins)]
            
            dist_matrix = np.zeros((len(categories), n_bins))
            
            for cat_idx, cat in enumerate(categories):
                cat_data = plot_df[plot_df[cat_column] == cat][feature_name].dropna()
                if len(cat_data) > 0:
                    hist, _ = np.histogram(cat_data, bins=bins)
                    dist_matrix[cat_idx, :] = (hist / len(cat_data)) * 100
            
            im = ax3.imshow(dist_matrix, cmap='YlOrRd', aspect='auto', 
                           interpolation='nearest')
            
            ax3.set_xticks(np.arange(n_bins))
            ax3.set_yticks(np.arange(len(categories)))
            ax3.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
            ax3.set_yticklabels(cat_labels, fontsize=10)
            
            # Add text with adaptive color (black on bright, white on dark)
            # YlOrRd colormap: low values = bright yellow, high values = dark red
            max_val = dist_matrix.max()
            threshold = max_val * 0.4  # Adjust threshold for better contrast
            
            for i in range(len(categories)):
                for j in range(n_bins):
                    if dist_matrix[i, j] > 0:
                        # Low values (bright yellow) = black text
                        # High values (dark red) = white text
                        text_color = 'white' if dist_matrix[i, j] > threshold else 'black'
                        text = ax3.text(j, i, f'{dist_matrix[i, j]:.1f}%',
                                       ha="center", va="center", color=text_color,
                                       fontsize=8, fontweight='bold')
            
            ax3.set_xlabel('Value Range', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Category', fontsize=12, fontweight='bold')
            ax3.set_title('Binned Distribution (%)', fontsize=13, fontweight='bold', pad=12)
            
            cbar = plt.colorbar(im, ax=ax3, pad=0.02)
            cbar.set_label('Percentage (%)', fontsize=10, fontweight='bold')
            cbar.ax.tick_params(labelsize=9)
            
            fig.suptitle(f'Distribution Analysis: {feature_name.replace("_", " ").title()}',
                        fontsize=15, fontweight='bold', y=0.98)
            
            plt.show()
            
        except Exception as e:
            print(f"Distribution analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_correlation_analysis(self, plot_df, feature_x, feature_y, use_4way=False):
        """Correlation analysis (3 plots)"""
        try:
            clean_df = plot_df.dropna(subset=[feature_x, feature_y])
            
            if len(clean_df) < 3:
                print(f"Not enough data for correlation analysis")
                return
            
            x_data = clean_df[feature_x].values
            y_data = clean_df[feature_y].values
            
            if ADVANCED_FEATURES:
                pearson_r, pearson_p = pearsonr(x_data, y_data)
                spearman_r, spearman_p = spearmanr(x_data, y_data)
            else:
                pearson_r = np.corrcoef(x_data, y_data)[0, 1]
                pearson_p = 0.0
                spearman_r = np.nan
                spearman_p = np.nan
            
            # Determine categories
            if use_4way and 'composite_category' in clean_df.columns:
                categories = sorted(clean_df['composite_category'].unique())
                cat_labels = [self.category_labels.get(cat, cat) for cat in categories]
                cat_column = 'composite_category'
            else:
                categories = sorted(clean_df['cycle_type'].unique())
                cat_labels = [cat.upper() for cat in categories]
                cat_column = 'cycle_type'
            
            # Create figure with 3 subplots
            fig = plt.figure(figsize=(16, 5), dpi=100)
            fig.subplots_adjust(hspace=0.25, wspace=0.3, top=0.9, 
                              bottom=0.12, left=0.06, right=0.97)
            
            ax1 = plt.subplot(1, 3, 1)
            ax2 = plt.subplot(1, 3, 2)
            ax3 = plt.subplot(1, 3, 3)
            
            # Plot 1: Scatter plot
            for cat, label in zip(categories, cat_labels):
                cat_data = clean_df[clean_df[cat_column] == cat]
                if len(cat_data) > 0:
                    color = (self.colors.get(cat, self.colors['neutral']) if use_4way 
                            else self.colors[cat])
                    ax1.scatter(cat_data[feature_x], cat_data[feature_y],
                              c=color, label=label,
                              alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
            
            # Regression line
            if len(clean_df) > 1:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_data.min(), x_data.max(), 100)
                y_line = p(x_line)
                ax1.plot(x_line, y_line, color='#FFD700', linestyle='--', 
                        linewidth=2.5, alpha=0.9, label=f'Regression (R={pearson_r:.3f})')
            
            ax1.set_xlabel(feature_x.replace('_', ' ').title(), 
                          fontsize=12, fontweight='bold')
            ax1.set_ylabel(feature_y.replace('_', ' ').title(), 
                          fontsize=12, fontweight='bold')
            ax1.set_title(f'Scatter Plot\nPearson R = {pearson_r:.4f}', 
                         fontsize=13, fontweight='bold', pad=12)
            ax1.legend(loc='best', framealpha=0.9, edgecolor='white', fontsize=9)
            ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax1.tick_params(axis='both', labelsize=10)
            
            # Plot 2: Residual plot
            if len(clean_df) > 1:
                predicted = p(x_data)
                residuals = y_data - predicted
                
                for cat, label in zip(categories, cat_labels):
                    cat_mask = clean_df[cat_column] == cat
                    if cat_mask.any():
                        color = (self.colors.get(cat, self.colors['neutral']) if use_4way 
                                else self.colors[cat])
                        ax2.scatter(predicted[cat_mask], residuals[cat_mask],
                                  c=color, alpha=0.7, s=40, 
                                  edgecolors='white', linewidth=0.5)
                
                ax2.axhline(y=0, color='#FFD700', linestyle='--', linewidth=2.5, alpha=0.9)
                ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
                ax2.set_title('Residual Plot', fontsize=13, fontweight='bold', pad=12)
                ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
                ax2.tick_params(axis='both', labelsize=10)
            
            # Plot 3: Joint distribution (hexbin)
            ax3.hexbin(clean_df[feature_x], clean_df[feature_y],
                      gridsize=25, cmap='YlOrRd', alpha=0.7, mincnt=1)
            
            ax3.set_xlabel(feature_x.replace('_', ' ').title(), 
                          fontsize=12, fontweight='bold')
            ax3.set_ylabel(feature_y.replace('_', ' ').title(), 
                          fontsize=12, fontweight='bold')
            ax3.set_title('Density Plot', fontsize=13, fontweight='bold', pad=12)
            ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax3.tick_params(axis='both', labelsize=10)
            
            fig.suptitle(f'Correlation: {feature_x} vs {feature_y}',
                        fontsize=15, fontweight='bold', y=0.98)
            
            plt.show()
            
        except Exception as e:
            print(f"Correlation analysis error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main execution function"""
    
    print("Enhanced Cycle Feature Visualizer")
    print("="*75)
    
    visualizer = CycleFeatureVisualizer()
    
    print("\nLoading data...")
    if visualizer.load_data() is None:
        print("Data loading failed.")
        return
    
    if visualizer.flatten_features() is None:
        print("Feature preprocessing failed.")
        return
    
    print(f"Total {len(visualizer.features_df)} cycle data prepared.")
    
    while True:
        try:
            menu_choice = visualizer.show_main_menu()
            
            if menu_choice == 2:
                print("\nThank you for using Enhanced Cycle Feature Visualizer!")
                break
            
            print("\n" + "="*75)
            print("Starting New Analysis")
            print("="*75)
            
            feature_name = visualizer.get_user_feature_choice()
            if feature_name is None:
                continue
            
            print("\n" + "="*60)
            print("Correlation analysis?")
            print("="*60)
            print("1. Yes - Select correlation target")
            print("2. No - Skip")
            
            correlation_target = None
            while True:
                try:
                    print(f"\nEnter choice (1-2): ", end="")
                    corr_choice = int(input())
                    
                    if corr_choice == 1:
                        correlation_target = visualizer.get_user_correlation_target_choice(
                            feature_name)
                        break
                    elif corr_choice == 2:
                        print("Skipping correlation analysis.")
                        break
                    else:
                        print("Please enter 1 or 2.")
                        
                except ValueError:
                    print("Please enter a number.")
                except KeyboardInterrupt:
                    correlation_target = None
                    break
            
            filters = visualizer.get_user_filter()
            cycle_filter = visualizer.get_user_cycle_filter()
            if cycle_filter is None:
                continue
            
            normalization = visualizer.get_user_normalization_choice()
            if normalization is None:
                continue
            
            print(f"\nGenerating visualizations...")
            
            visualizer.create_all_visualizations(
                feature_name=feature_name,
                cycle_filter=cycle_filter,
                normalization=normalization,
                filters=filters,
                correlation_target=correlation_target
            )
            
            print("\nVisualizations completed!")
                    
        except KeyboardInterrupt:
            print("\n\nThank you!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()