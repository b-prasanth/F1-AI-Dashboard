import pandas as pd
import os
import plotly.express as px
from . import queries
from . import sql_runner as sr
import fastf1
from fastf1 import get_event_schedule, get_session
import logging
import matplotlib.pyplot as plt
from datetime import datetime
# from ML_Model import quali_df
# from ML_Model import race_df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def gen_points_plot(df, title,type, save_path):
    import os
    if os.path.exists(save_path):
        os.remove(save_path)

    fig = px.line(
        df,
        x='round',
        y='cumulative_points',
        color=f'{type}',
        title=f'{title}',
        labels={'round': 'Round', 'cumulative_points': 'Points'},
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Plotly  # Dynamic color palette
    )
    if type == 'driver':
        legend_title = 'Driver'
    elif type == 'constructor':
        legend_title = 'Constructor'

    # Update layout for white background and better visualization
    fig.update_layout(
        xaxis=dict(tickmode='linear', title='Round'),
        yaxis=dict(title='Points'),
        legend_title_text=f'{legend_title}',
        template='plotly_white',  # White background template
        plot_bgcolor='white',  # Ensure plot background is white
        paper_bgcolor='white',  # Ensure paper background is white
        font=dict(color='#333333')  # Darker text for contrast
    )

    fig.write_image(save_path, scale=2)  # Higher resolution for images
    logger.info(f"Plot saved as image to {save_path}")

def gen_bar_plot(df, title, x_col, y_col, save_path, color_palette='viridis'):
    """Generate a bar plot and save it to the specified path"""
    try:
        import os
        if os.path.exists(save_path):
            os.remove(save_path)
        
        # Check if DataFrame is empty
        if df.empty:
            logger.warning(f"Empty DataFrame provided for {title} plot")
            return False
        
        # Check if required columns exist
        if x_col not in df.columns or y_col not in df.columns:
            logger.warning(f"Required columns {x_col} or {y_col} not in DataFrame for {title} plot")
            return False
        
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=title,
            color=x_col,  # Color by category
            color_discrete_sequence=px.colors.qualitative.Plotly,
            text=y_col  # Display values on bars
        )
        
        # Update layout for better visualization
        fig.update_layout(
            xaxis=dict(title=x_col.capitalize()),
            yaxis=dict(title=y_col.capitalize()),
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#333333')
        )
        
        # Show values on top of bars
        fig.update_traces(textposition='outside')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Use matplotlib as a fallback if plotly fails
        try:
            fig.write_image(save_path, scale=2)  # Higher resolution for images
        except Exception as e:
            logger.warning(f"Plotly image export failed: {str(e)}. Trying matplotlib fallback.")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.bar(df[x_col], df[y_col])
            plt.title(title)
            plt.xlabel(x_col.capitalize())
            plt.ylabel(y_col.capitalize())
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        
        logger.info(f"Bar plot saved as image to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Error generating bar plot: {str(e)}")
        return False

def save_points_history_plot(history_df, team_df, selected_year):
    try:
        # Check if DataFrame is empty or missing required columns
        team_save_path = f"/Users/prasanthbalaji/Desktop/Java Projects/F1_V3/pythonProject/images/team_points_history_{selected_year}.jpg"
        driver_save_path = f"/Users/prasanthbalaji/Desktop/Java Projects/F1_V3/pythonProject/images/driver_points_history_{selected_year}.jpg"

        if history_df.empty:
            logger.error(f"Empty DataFrame provided for {selected_year} points history plot")
            return

        required_columns = ['round', 'cumulative_points', 'driver']
        team_columns = ['round', 'cumulative_points', 'constructor']
        if not all(col in history_df.columns for col in required_columns):
            logger.error(f"DataFrame missing required columns: {required_columns}")
            return
        if not all(col in team_df.columns for col in team_columns):
            logger.error(f"DataFrame missing required columns: {required_columns}")
            return
        gen_points_plot(history_df, f'Top 5 Drivers Points {selected_year}', 'driver', driver_save_path)
        gen_points_plot(team_df, f'Constructors Points {selected_year}', 'constructor', team_save_path)

    except Exception as e:
        logger.error(f"Error generating/saving points history plot: {str(e)}")

def generate_stats_plots(year):
    """Generate all statistics plots for a given year"""
    try:
        # Use relative imports to avoid path issues
        import queries
        import sql_runner as sr
        
        # Generate driver wins plot
        try:
            driver_wins_df = sr.run_sql_query("F1", "drivers_championship", queries.get_driver_wins_season(), params={"year": year})
            if not driver_wins_df.empty:
                driver_wins_path = f"/Users/prasanthbalaji/Desktop/Java Projects/F1_V3/pythonProject/images/driver_wins_{year}.jpg"
                gen_bar_plot(driver_wins_df, f'Driver Wins {year}', 'driver', 'wins', driver_wins_path)
        except Exception as e:
            logger.error(f"Error generating driver wins plot: {str(e)}")
        
        # Generate constructor wins plot
        try:
            team_wins_df = sr.run_sql_query("F1", "constructors_championship", queries.get_team_wins_season(), params={"year": year})
            if not team_wins_df.empty:
                team_wins_path = f"/Users/prasanthbalaji/Desktop/Java Projects/F1_V3/pythonProject/images/team_wins_{year}.jpg"
                gen_bar_plot(team_wins_df, f'Constructor Wins {year}', 'constructor', 'wins', team_wins_path)
        except Exception as e:
            logger.error(f"Error generating team wins plot: {str(e)}")
        
        # Generate driver podiums plot
        try:
            driver_podiums_df = sr.run_sql_query("F1", "race_results", queries.get_driver_podiums(year), params={"year": year})
            if not driver_podiums_df.empty:
                driver_podiums_path = f"/Users/prasanthbalaji/Desktop/Java Projects/F1_V3/pythonProject/images/driver_podiums_{year}.jpg"
                gen_bar_plot(driver_podiums_df, f'Driver Podiums {year}', 'Driver', 'Podiums', driver_podiums_path)
        except Exception as e:
            logger.error(f"Error generating driver podiums plot: {str(e)}")
        
        # Generate constructor podiums plot
        try:
            team_podiums_df = sr.run_sql_query("F1", "race_results", queries.get_constructor_podiums(year), params={"year": year})
            if not team_podiums_df.empty:
                team_podiums_path = f"/Users/prasanthbalaji/Desktop/Java Projects/F1_V3/pythonProject/images/team_podiums_{year}.jpg"
                gen_bar_plot(team_podiums_df, f'Constructor Podiums {year}', 'Constructor', 'Wins', team_podiums_path)
        except Exception as e:
            logger.error(f"Error generating team podiums plot: {str(e)}")
        
        # Generate driver poles plot
        try:
            driver_poles_df = sr.run_sql_query("F1", "race_results", queries.get_driver_poles_season(), params={"year": year})
            if not driver_poles_df.empty:
                driver_poles_path = f"/Users/prasanthbalaji/Desktop/Java Projects/F1_V3/pythonProject/images/driver_poles_{year}.jpg"
                gen_bar_plot(driver_poles_df, f'Driver Pole Positions {year}', 'driver', 'poles', driver_poles_path)
        except Exception as e:
            logger.error(f"Error generating driver poles plot: {str(e)}")
        
        # Generate constructor poles plot
        try:
            team_poles_df = sr.run_sql_query("F1", "race_results", queries.get_team_poles_season(), params={"year": year})
            if not team_poles_df.empty:
                team_poles_path = f"/Users/prasanthbalaji/Desktop/Java Projects/F1_V3/pythonProject/images/team_poles_{year}.jpg"
                gen_bar_plot(team_poles_df, f'Constructor Pole Positions {year}', 'constructor', 'poles', team_poles_path)
        except Exception as e:
            logger.error(f"Error generating team poles plot: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error generating statistics plots: {str(e)}")
        return False

def get_final_championship_standings(table_name, year, is_constructor=False):
    """Extract final championship standings for a specific year"""
    try:
        if is_constructor:
            year_data = sr.run_sql_query("F1", "constructors_championship", queries.get_constructors_championship_query(), params={"year": year})
        else:
            year_data = sr.run_sql_query("F1", "drivers_championship", queries.get_drivers_championship_query(), params={"year": year})

        if year_data.empty:
            logger.info(f"No data found for {year} {'Constructors' if is_constructor else 'Drivers'} Championship")
            return None

        final_standings = year_data

        if is_constructor:
            display_columns = {'Position': 'Position', 'Team': 'Team', 'Points': 'Points', 'Wins': 'Wins'}
        else:
            display_columns = {'Position': 'Position', 'Driver': 'Driver', 'Team': 'Team', 'Points': 'Points', 'Wins': 'Wins'}

        available_columns = {k: v for k, v in display_columns.items() if k in final_standings.columns}
        final_standings = final_standings[list(available_columns.keys())].rename(columns=available_columns)

        return final_standings
    except Exception as e:
        logger.info(f"Error processing {year} championship data: {str(e)}")
        return None

def get_current_year_calendar(year):
    try:
        df = sr.run_sql_query("F1", "f1_calendar", queries.get_calendar_query(), params={"selected_year": year})
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.info(f"No calendar data found for {year}.")
            return pd.DataFrame(columns=['Round', 'GrandPrix', 'Circuit', 'Date']), None

        current_year_schedule = df.rename(columns={
            'grand_prix': 'GrandPrix',
            'circuit': 'Circuit',
            'date': 'Date',
            'round': 'Round'
        })

        current_year_schedule['Round'] = pd.to_numeric(current_year_schedule['Round'], errors='coerce').astype('Int64')
        current_year_schedule['Date'] = pd.to_datetime(current_year_schedule['Date'])

        # Get current date
        today = pd.Timestamp.today().normalize()

        # Initialize next_race_round
        next_race_round = None

        # Update status, winner, and pole sitter based on race date
        for index, race in current_year_schedule.iterrows():
            if race['Date'].date() > today.date():
                current_year_schedule.at[index, 'Status'] = 'Scheduled'
                current_year_schedule.at[index, 'Winner'] = 'TBA'
                current_year_schedule.at[index, 'PoleSitter'] = 'TBA'

                # Identify the next race
                if next_race_round is None:
                    next_race_round = race['Round']

        # If no future races, set next_race_round to the last round
        if next_race_round is None and not current_year_schedule.empty:
            next_race_round = current_year_schedule['Round'].iloc[-1]

        # Add next_race_round to the DataFrame
        current_year_schedule['next_race_round'] = next_race_round

        # Return next race information
        if not current_year_schedule.empty:
            next_race = current_year_schedule[current_year_schedule['Round'] == next_race_round].iloc[0]
            next_race_info = {
                'GrandPrix': next_race['GrandPrix'],
                'Date': next_race['Date'].strftime('%Y-%m-%d'),
                'Circuit': next_race['Circuit'],
                'Round': next_race['Round']
            }
            return current_year_schedule, next_race_info
        else:
            return current_year_schedule, None
    except Exception as e:
        logger.info(f"Error fetching {year} calendar: {e}")
        logger.error(f"Error in get_current_year_calendar: {e}")
        return pd.DataFrame(columns=['Round', 'GrandPrix', 'Circuit', 'Date']), None

# def get_pivoted_race_results(year, type):
#     try:
#         calendar_df, _ = get_current_year_calendar(year)
#         if calendar_df.empty:
#             return pd.DataFrame()

#         grand_prix_list = calendar_df['GrandPrix'].tolist()

#         if type == 'Race':
#             race_results_df = sr.run_sql_query("F1", "race_results", queries.get_race_results_query(), params={"year": year})
#             value_column = 'finish_position'
#             display_column = 'FinishPosition'
#         else:  # Quali
#             race_results_df = sr.run_sql_query("F1", "qualifying_results", queries.get_qualifying_results_query(), params={"year": year})
#             value_column = 'position'
#             display_column = 'QualifyingPosition'

#         if not isinstance(race_results_df, pd.DataFrame) or race_results_df.empty:
#             logger.info(f"No {type.lower()} results available for {year}")
#             return pd.DataFrame()

#         race_results_df = race_results_df.rename(columns={
#             'driver': 'Driver',
#             'constructor': 'Team',
#             'grand_prix': 'GrandPrix',
#             value_column: display_column
#         })

#         pivot_df = race_results_df.pivot_table(
#             index=['Driver', 'Team'],
#             columns='GrandPrix',
#             values=display_column,
#             aggfunc='first'
#         ).reset_index()

#         # Add Championship Position
#         standings_df = sr.run_sql_query("F1", "drivers_championship", queries.get_drivers_championship_query(), params={"year": year})
#         if standings_df.empty:
#             logger.info(f"No championship standings available for {year}")
#             return pd.DataFrame()

#         standings_df = standings_df[['Driver', 'Position']].rename(columns={'Position': 'Championship position'})

#         final_df = pd.merge(pivot_df, standings_df, on='Driver', how='left')

#         # Order columns
#         ordered_columns = ['Driver', 'Team'] + grand_prix_list + ['Championship position']
#         final_df = final_df.reindex(columns=ordered_columns)

#         # Set Championship position as index and sort
#         final_df = final_df.set_index('Championship position').sort_index()

#         return final_df

#     except Exception as e:
#         logger.info(f"Error preparing pivoted {type.lower()} results: {e}")
#         logger.error(f"Error in get_pivoted_race_results: {e}")
#         return pd.DataFrame()
def get_pivoted_race_results(year, type):
    try:
        # Fetch calendar data
        calendar_data = get_current_year_calendar(year)
        if not isinstance(calendar_data, tuple) or len(calendar_data) != 2:
            logger.info(f"Invalid calendar data returned for {year}")
            return pd.DataFrame()

        calendar_df, _ = calendar_data
        if not isinstance(calendar_df, pd.DataFrame) or calendar_df.empty:
            logger.info(f"No valid calendar data found for {year}")
            return pd.DataFrame()

        grand_prix_list = calendar_df['GrandPrix'].tolist()

        if type == 'Race':
            race_results_df = sr.run_sql_query("F1", "race_results", queries.get_race_results_query(), params={"year": year})
            value_column = 'finish_position'
            display_column = 'FinishPosition'
        else:  # Quali
            race_results_df = sr.run_sql_query("F1", "qualifying_results", queries.get_qualifying_results_query(), params={"year": year})
            value_column = 'position'
            display_column = 'QualifyingPosition'

        if not isinstance(race_results_df, pd.DataFrame) or race_results_df.empty:
            logger.info(f"No {type.lower()} results available for {year}")
            return pd.DataFrame()

        race_results_df = race_results_df.rename(columns={
            'driver': 'Driver',
            'constructor': 'Team',
            'grand_prix': 'GrandPrix',
            value_column: display_column
        })

        pivot_df = race_results_df.pivot_table(
            index=['Driver', 'Team'],
            columns='GrandPrix',
            values=display_column,
            aggfunc='first'
        ).reset_index()

        # Add Championship Position
        standings_df = sr.run_sql_query("F1", "drivers_championship", queries.get_drivers_championship_query(), params={"year": year})
        if not isinstance(standings_df, pd.DataFrame) or standings_df.empty:
            logger.info(f"No championship standings available for {year}")
            return pd.DataFrame()

        standings_df = standings_df[['Driver', 'Position']].rename(columns={'Position': 'Championship position'})

        final_df = pd.merge(pivot_df, standings_df, on='Driver', how='left')

        # Order columns
        ordered_columns = ['Driver', 'Team'] + grand_prix_list + ['Championship position']
        final_df = final_df.reindex(columns=ordered_columns)

        # Set Championship position as index and sort
        final_df = final_df.set_index('Championship position').sort_index()

        return final_df

    except Exception as e:
        logger.info(f"Error preparing pivoted {type.lower()} results: {e}")
        logger.error(f"Error in get_pivoted_race_results: {e}")
        return pd.DataFrame()

def get_driver_points_history(year):
    """Fetch and calculate cumulative points for drivers across rounds"""
    try:
        combined_df = sr.run_sql_query("F1", "race_results", queries.get_points_history_query(), params={"year": year})

        if combined_df.empty:
            logger.info(f"No race or sprint data found for {year}.")
            return pd.DataFrame(), pd.DataFrame()

        # Calculate cumulative points for drivers
        driver_df = combined_df.groupby(['year', 'round', 'driver']).sum().reset_index()
        driver_cumulative = driver_df.sort_values('round').groupby('driver', include_groups=False).apply(
            lambda x: x.assign(cumulative_points=x['points'].cumsum())
        ).reset_index(drop=True)
        total_driver_points = driver_cumulative.groupby('driver')['cumulative_points'].last().sort_values(
            ascending=False)
        top_drivers = total_driver_points.head(5).index
        driver_history = driver_cumulative[driver_cumulative['driver'].isin(top_drivers)].sort_values(
            ['driver', 'round'])

        # Calculate cumulative points for teams
        team_df = combined_df.groupby(['year', 'round', 'constructor']).sum().reset_index()
        team_cumulative = team_df.sort_values('round').groupby('constructor', include_groups=False).apply(
            lambda x: x.assign(cumulative_points=x['points'].cumsum())
        ).reset_index(drop=True)

        return driver_history, team_cumulative
    except Exception as e:
        logger.info(f"Error calculating points history for {year}: {str(e)}")
        logger.error(f"Error in get_points_history: {e}")
        return pd.DataFrame(), pd.DataFrame()


def get_team_drivers(year):
    team_colors = {
        'Red Bull Racing': '#1E3A8A',  # Blue
        'Red Bull': '#1E3A8A',  # Blue
        'Ferrari': '#DC2626',  # Red
        'Mercedes': '#06B6D4',  # Cyan
        'McLaren': '#F97316',  # Orange
        'Alpine': '#FFC1CC',  # Light Blue
        'Racing Bulls': '#D3D3D3',  # White
        'Aston Martin': '#006F62',  # Green
        'Williams': '#005AFF',  # Bright Blue
        'Haas': '#7E2B14',  # Maroon
        'Kick Sauber': '#00FF00'  # Bright Green
    }

    df = sr.run_sql_query("F1", "drivers_championship", queries.get_team_drivers(), params={"year": year})

    if df.empty:
        logger.info(f"No driver data found for {year} in drivers_championship.")
        return []

    # Group drivers by team
    teams = {}
    for _, row in df.iterrows():
        team = row['Team']
        driver = row['Driver']
        if team not in teams:
            teams[team] = []
        teams[team].append(driver)

    # Convert to list of dictionaries
    teams_info = [{"Team": team, "Drivers": ", ".join(drivers), "Color": team_colors.get(team, '#808080')} for
                  team, drivers in teams.items()]

    return teams_info

def get_gp_results(year, grand_prix):
    race_data = sr.run_sql_query("F1", "race_results", queries.get_gp_race_results_query(),
                                 params={"year": year, "grand_prix": grand_prix})
    # Ensure race_data is a DataFrame; if empty, return empty list
    if race_data is None or (hasattr(race_data, 'empty') and race_data.empty):
        return []
    return race_data

# selected_year = 2025
# final_drivers = get_final_championship_standings("drivers_championship", selected_year, is_constructor=False)
# final_constructors = get_final_championship_standings("constructors_championship", selected_year, is_constructor=True)
# print(final_drivers)
# print(final_constructors)
