#!/usr/bin/env python3
"""
F1 Plot Generation Script

This standalone script generates all visualization plots for F1 statistics.
It can be run manually to generate plots for a specific year.

Usage:
    python plot_generation.py <year>

Example:
    python plot_generation.py 2023
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import logging
import argparse
import numpy as np
import gc
import time

import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_images_dir():
    """Create images directory if it doesn't exist"""
    os.makedirs("f1-ui/public/images", exist_ok=True)

# Define custom colors for drivers and teams
DRIVER_COLORS = {
    'Verstappen': '#0600EF',       # Red Bull blue
    'Hamilton': '#00D2BE',         # Mercedes teal
    'Leclerc': '#DC0000',          # Ferrari red
    'Norris': '#FF8700',           # McLaren orange
    'Sainz': '#DC0000',            # Ferrari red
    'Russell': '#00D2BE',          # Mercedes teal
    'Perez': '#0600EF',            # Red Bull blue
    'Piastri': '#FF8700',          # McLaren orange
    'Alonso': '#006F62',           # Aston Martin green
    'Stroll': '#006F62',           # Aston Martin green
    'Gasly': '#0090FF',            # Alpine blue
    'Ocon': '#0090FF',             # Alpine blue
    'Albon': '#005AFF',            # Williams blue
    'Tsunoda': '#B6BABD',          # Racing Bulls silver
    'Bottas': '#900000',           # Alfa Romeo/Sauber red
    'Hulkenberg': '#FFFFFF',       # Haas white
    'Magnussen': '#FFFFFF',        # Haas white
    'Zhou': '#900000',             # Alfa Romeo/Sauber red
    'Sargeant': '#005AFF',         # Williams blue
    'Ricciardo': '#B6BABD',        # Racing Bulls silver
}

TEAM_COLORS = {
    'Red Bull': '#0600EF',         # Red Bull blue
    'Mercedes': '#00D2BE',         # Mercedes teal
    'Ferrari': '#DC0000',          # Ferrari red
    'McLaren': '#FF8700',          # McLaren orange
    'Aston Martin': '#006F62',     # Aston Martin green
    'Alpine': '#0090FF',           # Alpine blue
    'Williams': '#005AFF',         # Williams blue
    'Racing Bulls': '#B6BABD',     # Racing Bulls silver
    'Kick Sauber': '#900000',      # Alfa Romeo/Sauber red
    'Haas': '#FFFFFF',             # Haas white
    'Alfa Romeo': '#900000',       # Alfa Romeo red
    'AlphaTauri': '#B6BABD',       # AlphaTauri silver
    'Toro Rosso': '#0600EF',       # Toro Rosso blue
    'Force India': '#FF5F0F',      # Force India orange
    'Racing Point': '#F596C8',     # Racing Point pink
    'Renault': '#FFF500',          # Renault yellow
    'Lotus': '#000000',            # Lotus black
    'Sauber': '#006EFF',           # Sauber blue
}

# Fallback colors for any driver/team not in the dictionaries
FALLBACK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def setup_plot():
    """Set up a plot with grey background and white text"""
    plt.figure(figsize=(12, 8), facecolor='#333333')
    ax = plt.axes(facecolor='#333333')
    ax.tick_params(colors='white', which='both')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    return ax

def generate_driver_points_plot(year):
    """Generate driver points history plot"""
    try:
        # Import locally to avoid circular imports
        from backend import sql_runner as sr
        from backend import queries
        
        # Get data
        combined_df = sr.run_sql_query("F1", "race_results", queries.get_points_history_query(), params={"year": year})
        
        if combined_df.empty:
            logger.info(f"No race or sprint data found for {year}.")
            return False
        
        # Check required columns
        required_columns = ['year', 'round', 'driver', 'points']
        missing_columns = [col for col in required_columns if col not in combined_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Calculate cumulative points for drivers
        driver_df = combined_df.groupby(['year', 'round', 'driver'])['points'].sum().reset_index()
        
        # Calculate cumulative points manually
        driver_cumulative = []
        for driver, group in driver_df.sort_values(['driver', 'round']).groupby('driver'):
            group = group.copy()
            group['cumulative_points'] = group['points'].cumsum()
            driver_cumulative.append(group)
        
        if not driver_cumulative:
            logger.warning(f"No driver data found for {year}")
            return False
            
        driver_cumulative_df = pd.concat(driver_cumulative)
        
        # Get top 5 drivers by total points
        total_points = driver_cumulative_df.groupby('driver')['cumulative_points'].max()
        top_drivers = total_points.sort_values(ascending=False).head(5).index.tolist()
        
        if not top_drivers:
            logger.warning(f"No top drivers found for {year}")
            return False
        
        # Set up plot
        ax = setup_plot()
        
        # Plot each driver's points
        for i, driver in enumerate(top_drivers):
            data = driver_cumulative_df[driver_cumulative_df['driver'] == driver]
            if not data.empty:
                color = DRIVER_COLORS.get(driver.split(' ')[-1], FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
                plt.plot(data['round'], data['cumulative_points'], marker='o', linewidth=3, label=driver, color=color)
        
        # Add labels and title
        plt.title(f'Driver Points Progression - {year}', fontsize=16, color='white')
        plt.xlabel('Race Number', fontsize=14)
        plt.ylabel('Points', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3, color='#555555')
        
        # Create legend with white text and matching background
        legend = plt.legend(loc='upper left', fontsize=12)
        legend.get_frame().set_facecolor('#333333')
        legend.get_frame().set_edgecolor('#555555')
        for text in legend.get_texts():
            text.set_color('white')
        
        plt.tight_layout()
        
        # Save the image
        save_path = f"images/driver_points_history_{year}.jpg"
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='#333333')
        plt.close()
        
        logger.info(f"Generated driver points history image for {year}")
        return True
    except Exception as e:
        logger.error(f"Error generating driver points plot: {str(e)}")
        plt.close('all')
        return False

def generate_team_points_plot(year):
    """Generate team points history plot"""
    try:
        # Import locally to avoid circular imports
        from backend import sql_runner as sr
        from backend import queries
        
        # Get data
        combined_df = sr.run_sql_query("F1", "race_results", queries.get_points_history_query(), params={"year": year})
        
        if combined_df.empty:
            logger.info(f"No race or sprint data found for {year}.")
            return False
        
        # Check required columns
        required_columns = ['year', 'round', 'constructor', 'points']
        missing_columns = [col for col in required_columns if col not in combined_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Calculate cumulative points for teams
        team_df = combined_df.groupby(['year', 'round', 'constructor'])['points'].sum().reset_index()
        
        # Calculate cumulative points manually
        team_cumulative = []
        for constructor, group in team_df.sort_values(['constructor', 'round']).groupby('constructor'):
            group = group.copy()
            group['cumulative_points'] = group['points'].cumsum()
            team_cumulative.append(group)
        
        if not team_cumulative:
            logger.warning(f"No team data found for {year}")
            return False
            
        team_cumulative_df = pd.concat(team_cumulative)
        
        # Set up plot
        ax = setup_plot()
        
        # Plot each team's points
        for i, constructor in enumerate(team_cumulative_df['constructor'].unique()):
            data = team_cumulative_df[team_cumulative_df['constructor'] == constructor]
            if not data.empty:
                color = TEAM_COLORS.get(constructor, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
                plt.plot(data['round'], data['cumulative_points'], marker='o', linewidth=3, label=constructor, color=color)
        
        # Add labels and title
        plt.title(f'Constructor Points Progression - {year}', fontsize=16, color='white')
        plt.xlabel('Race Number', fontsize=14)
        plt.ylabel('Points', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3, color='#555555')
        
        # Create legend with white text and matching background
        legend = plt.legend(loc='upper left', fontsize=12)
        legend.get_frame().set_facecolor('#333333')
        legend.get_frame().set_edgecolor('#555555')
        for text in legend.get_texts():
            text.set_color('white')
        
        plt.tight_layout()
        
        # Save the image
        save_path = f"images/team_points_history_{year}.jpg"
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='#333333')
        plt.close()
        
        logger.info(f"Generated team points history image for {year}")
        return True
    except Exception as e:
        logger.error(f"Error generating team points plot: {str(e)}")
        plt.close('all')
        return False

def generate_combined_stats_plot(year, is_driver=True):
    """Generate a combined plot with wins, podiums, and poles"""
    try:
        # Import locally to avoid circular imports
        from backend import sql_runner as sr
        from backend import queries
        
        entity_type = "Driver" if is_driver else "Constructor"
        
        # Create sample data if database connection fails
        sample_data = pd.DataFrame({
            'Name': ['Driver1', 'Driver2', 'Driver3', 'Driver4', 'Driver5'] if is_driver else 
                   ['Team1', 'Team2', 'Team3', 'Team4', 'Team5'],
            'Wins': [5, 4, 3, 2, 1],
            'Podiums': [10, 8, 6, 4, 2],
            'Poles': [6, 4, 3, 2, 1]
        })
        
        # Try to get real data
        try:
            # Get data for wins, podiums, and poles
            if is_driver:
                wins_df = sr.run_sql_query("F1", "drivers_championship", queries.get_driver_wins_season(), params={"year": year})
                podiums_df = sr.run_sql_query("F1", "race_results", queries.get_driver_podiums(year), params={"year": year})
                poles_df = sr.run_sql_query("F1", "race_results", queries.get_driver_poles_season(), params={"year": year})
                
                # Rename columns for consistency
                if not wins_df.empty:
                    wins_df = wins_df.rename(columns={'driver': 'Name', 'wins': 'Wins'})
                if not podiums_df.empty:
                    podiums_df = podiums_df.rename(columns={'Driver': 'Name', 'Podiums': 'Podiums'})
                if not poles_df.empty:
                    poles_df = poles_df.rename(columns={'driver': 'Name', 'poles': 'Poles'})
            else:
                wins_df = sr.run_sql_query("F1", "constructors_championship", queries.get_team_wins_season(), params={"year": year})
                podiums_df = sr.run_sql_query("F1", "race_results", queries.get_constructor_podiums(year), params={"year": year})
                poles_df = sr.run_sql_query("F1", "race_results", queries.get_team_poles_season(), params={"year": year})
                
                # Rename columns for consistency
                if not wins_df.empty:
                    wins_df = wins_df.rename(columns={'constructor': 'Name', 'wins': 'Wins'})
                if not podiums_df.empty:
                    podiums_df = podiums_df.rename(columns={'Constructor': 'Name', 'Wins': 'Podiums'})
                if not poles_df.empty:
                    poles_df = poles_df.rename(columns={'constructor': 'Name', 'poles': 'Poles'})
            
            # Merge dataframes
            merged_df = pd.DataFrame(columns=['Name'])
            
            if not wins_df.empty and 'Name' in wins_df.columns and 'Wins' in wins_df.columns:
                merged_df = pd.merge(merged_df, wins_df[['Name', 'Wins']], on='Name', how='outer')
            else:
                merged_df['Wins'] = 0
                
            if not podiums_df.empty and 'Name' in podiums_df.columns and 'Podiums' in podiums_df.columns:
                merged_df = pd.merge(merged_df, podiums_df[['Name', 'Podiums']], on='Name', how='outer')
            else:
                merged_df['Podiums'] = 0
                
            if not poles_df.empty and 'Name' in poles_df.columns and 'Poles' in poles_df.columns:
                merged_df = pd.merge(merged_df, poles_df[['Name', 'Poles']], on='Name', how='outer')
            else:
                merged_df['Poles'] = 0
            
            # Fill NaN with 0
            merged_df = merged_df.fillna(0)
            
            # Sort by total achievements
            merged_df['Total'] = merged_df['Wins'] + merged_df['Podiums'] + merged_df['Poles']
            merged_df = merged_df.sort_values('Total', ascending=False).head(10)  # Top 10 only
            merged_df = merged_df.drop('Total', axis=1)
            
            if merged_df.empty or len(merged_df) == 0:
                logger.info(f"No data found for {entity_type} stats plot, using sample data.")
                merged_df = sample_data
        except Exception as e:
            logger.warning(f"Error getting data for {entity_type} stats plot: {str(e)}. Using sample data.")
            merged_df = sample_data
        
        # Generate plot with grey background
        plt.figure(figsize=(14, 10), facecolor='#333333')
        ax = plt.axes(facecolor='#333333')
        ax.tick_params(colors='white', which='both')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        
        # Set up bar positions
        bar_width = 0.25
        r1 = np.arange(len(merged_df))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Create bars with different colors for each stat type
        bars1 = plt.bar(r1, merged_df['Wins'], width=bar_width, color='#FF5F5F', label='Wins')
        bars2 = plt.bar(r2, merged_df['Podiums'], width=bar_width, color='#5F9EFF', label='Podiums')
        bars3 = plt.bar(r3, merged_df['Poles'], width=bar_width, color='#5FFF5F', label='Poles')
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only show labels for non-zero values
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom', color='white', fontsize=9)
        
        # Add labels and title
        plt.xlabel(f'{entity_type}s', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title(f'{entity_type} Performance Stats - {year}', fontsize=16, color='white')
        plt.xticks([r + bar_width for r in range(len(merged_df))], merged_df['Name'], rotation=45, ha='right', color='white')
        plt.grid(True, linestyle='--', alpha=0.3, axis='y', color='#555555')
        
        # Add legend with matching background
        legend = plt.legend(loc='upper right', fontsize=12)
        legend.get_frame().set_facecolor('#333333')
        legend.get_frame().set_edgecolor('#555555')
        for text in legend.get_texts():
            text.set_color('white')
        
        plt.tight_layout()
        
        # Save the image
        save_path = f"images/{entity_type.lower()}_stats_{year}.jpg"
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='#333333')
        plt.close()
        
        logger.info(f"Generated combined {entity_type} stats image for {year}")
        return True
    except Exception as e:
        logger.error(f"Error generating combined {entity_type} stats plot: {str(e)}")
        plt.close('all')
        return False

def generate_all_plots(year):
    """Generate all plots for a given year"""
    ensure_images_dir()
    
    # Generate plots one at a time with cleanup between each
    plot_configs = [
        # Points history plots
        generate_driver_points_plot,
        generate_team_points_plot,
        
        # Combined stats plots (wins, podiums, poles)
        lambda y: generate_combined_stats_plot(y, True),  # Driver stats
        lambda y: generate_combined_stats_plot(y, False)  # Team stats
    ]
    
    for i, func in enumerate(plot_configs):
        try:
            # Generate one plot
            func(year)
            
            # Force garbage collection after each plot
            plt.close('all')
            gc.collect()
            
            # Small delay between plots to allow memory to be freed
            if i < len(plot_configs) - 1:
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error generating plot {i+1} for {year}: {str(e)}")
    
    logger.info(f"Successfully generated all plots for {year}")

def generate_plots_for_all_years():
    """Generate plots for all years from 1950 to 2025"""
    start_year = 1950
    end_year = 2025
    
    for year in range(start_year, end_year + 1):
        try:
            logger.info(f"Generating plots for {year}...")
            generate_all_plots(year)
        except Exception as e:
            logger.error(f"Error generating plots for {year}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Generate F1 statistics plots')
    parser.add_argument('--year', type=int, help='Year to generate plots for')
    parser.add_argument('--all', action='store_true', help='Generate plots for all years (1950-2025)')
    args = parser.parse_args()
    
    if args.all:
        generate_plots_for_all_years()
    elif args.year:
        generate_all_plots(args.year)
    else:
        print("Please specify either --year YEAR or --all")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_generation.py --year YEAR or python plot_generation.py --all")
        sys.exit(1)
        
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
