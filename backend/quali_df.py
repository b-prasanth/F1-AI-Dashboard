from datetime import datetime
import pandas as pd
from . import queries
from . import sql_runner as sr
import logging
import fastf1 as f1api

# from ML_Model.model import model_runner
# from ML_Model.quali_xbg_model import model_runner
from .ensemble import model_runner

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Adjust width

def merge_f1_quali_dataframes(team_quali_df, driver_quali_df, summary_df, prev_year_result, year):

    try:
        # Define desired columns for the final DataFrame
        final_columns = [
            'Year', 'driver', 'team', 'championship_position', 'championship_points',
            'avg_qualifying_position', 'avg_qualifying_gap_vs_teammate',
            'qualifying_consistency', 'times_ahead', 'driver_quali_index',
            'overall_driver_index', 'avg_team_qualifying_position',
            'best_team_grid_position', 'team_qualifying_consistency',
            'team_front_row_lockouts', 'team_q3_appearances', 'team_q3_percentage',
            'car_quali_index'
        ]

        # Log input DataFrame columns for debugging
        logger.info(f"Team Quali DF columns: {team_quali_df.columns.tolist()}")
        logger.info(f"Driver Quali DF columns: {driver_quali_df.columns.tolist()}")
        logger.info(f"Summary DF columns: {summary_df.columns.tolist()}")

        # Select and rename columns from team_quali_df
        team_quali_cols = [
            'year', 'constructor', 'avg_qualifying_position', 'best_grid_position',
            'qualifying_consistency', 'front_row_lockouts', 'q3_appearances',
            'q3_percentage', 'car_quali_index'
        ]
        team_quali_cols = [col for col in team_quali_cols if col in team_quali_df.columns]
        team_quali_selected = team_quali_df[team_quali_cols].rename(columns={
            'year': 'Year',
            'constructor': 'team',
            'avg_qualifying_position': 'avg_team_qualifying_position',
            'best_grid_position': 'best_team_grid_position',
            'qualifying_consistency': 'team_qualifying_consistency',
            'front_row_lockouts': 'team_front_row_lockouts',
            'q3_appearances': 'team_q3_appearances',
            'q3_percentage': 'team_q3_percentage'
        })

        # Select and rename columns from driver_quali_df
        driver_quali_cols = [
            'year', 'driver', 'constructor', 'avg_qualifying_gap_vs_teammate',
            'qualifying_consistency', 'times_ahead', 'driver_quali_index'
        ]
        driver_quali_cols = [col for col in driver_quali_cols if col in driver_quali_df.columns]
        driver_quali_selected = driver_quali_df[driver_quali_cols].rename(columns={
            'year': 'Year',
            'constructor': 'team_driver'  # Rename to avoid conflict
        })

        # Select and rename columns from summary_df
        summary_cols = [
            'year', 'driver', 'team', 'championship_position', 'championship_points',
            'avg_qualifying_position', 'overall_driver_index'
        ]
        summary_cols = [col for col in summary_cols if col in summary_df.columns]
        summary_selected = summary_df[summary_cols].rename(columns={'year': 'Year'})

        # Merge DataFrames
        # Start with summary_df as the base
        merged_df = summary_selected.copy()

        # Merge with driver_quali_df on 'driver' and 'Year'
        if 'driver' in merged_df.columns and 'driver' in driver_quali_selected.columns:
            merged_df = pd.merge(
                merged_df,
                driver_quali_selected,
                on=['driver', 'Year'],
                how='left',
                suffixes=('_summary', '_driver_quali')
            )
            # Ensure 'team' from summary_df is retained
            if 'team_summary' in merged_df.columns:
                merged_df['team'] = merged_df['team_summary']
                merged_df = merged_df.drop(columns=['team_summary', 'team_driver_quali'], errors='ignore')
            elif 'team' not in merged_df.columns:
                logger.error("No 'team' column after first merge")
                raise KeyError("No 'team' column after first merge")
        else:
            logger.error("Missing 'driver' column in summary_df or driver_quali_df")
            raise KeyError("Missing 'driver' column in summary_df or driver_quali_df")

        # Merge with team_quali_df on 'team' and 'Year'
        if 'team' in merged_df.columns and 'team' in team_quali_selected.columns:
            merged_df = pd.merge(
                merged_df,
                team_quali_selected,
                on=['team', 'Year'],
                how='left'
            )
        else:
            logger.error("Missing 'team' column in merged_df or team_quali_df")
            raise KeyError("Missing 'team' column in merged_df or team_quali_df")

        # Reorder columns to match the requested order
        available_columns = [col for col in final_columns if col in merged_df.columns]
        merged_df = merged_df[available_columns]

        # Log success
        logger.info(f"Merged DataFrame created with columns: {merged_df.columns.tolist()}")
        today = pd.Timestamp.today().normalize()
        current_year = today.year

        if current_year == year:
            final_merged_df = merged_df
            final_merged_df['Rainfall'] = 0
        else:
            final_merged_df = pd.merge(merged_df, prev_year_result, on=['driver'], how='left')

        return merged_df, final_merged_df.dropna()

    except Exception as e:
        logger.error(f"Error merging DataFrames: {str(e)}")
        return pd.DataFrame(columns=final_columns)


def get_next_grandprix(year, calendar_df):

    try:
        # Ensure Date column is in datetime format
        calendar_df['Date'] = pd.to_datetime(calendar_df['Date'])

        # Current date
        today = pd.Timestamp.today().normalize()

        # Filter for upcoming races
        upcoming_races = calendar_df[calendar_df['Date'].dt.date >= today.date()]

        # Get the next race (earliest upcoming date)
        next_race = upcoming_races.sort_values('Date').head(1)

        if not next_race.empty:
            next_race_info = {
                'GrandPrix': next_race['GrandPrix'].iloc[0],
                'Date': next_race['Date'].iloc[0].strftime('%B %d, %Y'),
                'Circuit': next_race['Circuit'].iloc[0],
                'Round': next_race['Round'].iloc[0]
            }
            return next_race_info
        else:
            return None

    except Exception as e:
        print(f"Error finding next Grand Prix: {str(e)}")
        return None

def get_quali_results(year, grand_prix):

    session = f1api.get_session(year, grand_prix, 'Qualifying')
    logger.info(f"Loading qualifying data for {year} {grand_prix}")

    # Load session data
    session.load()
    logger.info(f"Qualifying data loaded for {session.event['EventName']}")

    # Extract qualifying results
    quali_results = session.results[[
        'FullName', 'Position', 'Q1', 'Q2', 'Q3'
    ]].rename(columns={
        'FullName': 'driver',
        'Position': 'qualifying_position'
    })

    # Assign qualifying times based on position
    quali_results['qualifying_time'] = None
    quali_results['session_phase'] = None

    def format_time(td):
        if pd.isna(td):
            return None
        time_str = str(td)
        return time_str.replace('0 days 00:', '')

    quali_results['Q1'] = quali_results['Q1'].apply(format_time)
    quali_results['Q2'] = quali_results['Q2'].apply(format_time)
    quali_results['Q3'] = quali_results['Q3'].apply(format_time)

    # Assign times based on position
    for idx, row in quali_results.iterrows():
        position = row['qualifying_position']
        if pd.notna(position):
            position = int(position)
            if 1 <= position <= 10 and pd.notna(row['Q3']):
                quali_results.at[idx, 'qualifying_time'] = row['Q3']
                quali_results.at[idx, 'session_phase'] = 'Q3'
            elif 11 <= position <= 15 and pd.notna(row['Q2']):
                quali_results.at[idx, 'qualifying_time'] = row['Q2']
                quali_results.at[idx, 'session_phase'] = 'Q2'
            elif 16 <= position <= 20 and pd.notna(row['Q1']):
                quali_results.at[idx, 'qualifying_time'] = row['Q1']
                quali_results.at[idx, 'session_phase'] = 'Q1'

    # Check if the session was wet
    weather_data = session.weather_data
    is_wet = weather_data['Rainfall'].any() if 'Rainfall' in weather_data.columns else False
    quali_results['Rainfall'] = 1 if is_wet else 0

    # Select final columns
    quali_results = quali_results[[
        'driver', 'qualifying_position', 'qualifying_time', 'session_phase', 'Rainfall'
    ]]

    return quali_results

def get_perf_metrics(training_year):
    quali_df = sr.run_sql_query("F1", "custom", queries.get_quali_perf_data(), params={'year': training_year})
    race_df = sr.run_sql_query("F1", "custom", queries.get_race_perf_data(), params={'year': training_year})
    driver_quali_df = sr.run_sql_query("F1", "custom", queries.get_driver_quali_perf(), params={'year': training_year})
    driver_race_df = sr.run_sql_query("F1", "custom", queries.get_driver_race_perf(), params={'year': training_year})
    driver_summary_df = sr.run_sql_query("F1", "drivers_championship", queries.get_driver_summary(),
                                         params={'year': training_year})
    return quali_df, race_df, driver_quali_df, driver_race_df, driver_summary_df


def custom_merge_df(old_df, new_df):
    """
    Concatenate old_df and new_df instead of merging, as we're accumulating data over years.
    """
    try:
        # If old_df is empty, return new_df as the starting point
        if old_df.empty:
            return new_df
        # Concatenate DataFrames vertically
        new_merged_df = pd.concat([old_df, new_df], ignore_index=True)
        return new_merged_df
    except Exception as e:
        logger.error(f"Error in custom_merge_df: {str(e)}")
        return old_df  # Return old_df to avoid losing data


def prep_quali_df(start_year, current_year, next_race, new_df=None):
    """
    Recursively prepare qualifying DataFrame for multiple years.
    """
    # Initialize new_df as empty DataFrame with expected columns if None
    if new_df is None:
        final_columns = [
            'Year', 'driver', 'team', 'championship_position', 'championship_points',
            'avg_qualifying_position', 'avg_qualifying_gap_vs_teammate',
            'qualifying_consistency', 'times_ahead', 'driver_quali_index',
            'overall_driver_index', 'avg_team_qualifying_position',
            'best_team_grid_position', 'team_qualifying_consistency',
            'team_front_row_lockouts', 'team_q3_appearances', 'team_q3_percentage',
            'car_quali_index', 'Rainfall'
        ]
        new_df = pd.DataFrame(columns=final_columns)

    if start_year > current_year:
        return new_df

    try:
        logger.info(f"Processing year: {start_year}")
        quali_df, race_df, driver_quali_df, driver_race_df, driver_summary_df = get_perf_metrics(start_year)

        if start_year != current_year:
            # For past years, include qualifying results
            prev_year_result = get_quali_results(start_year, next_race)
            # prev_year_result = get_quali_results(start_year, next_race['GrandPrix'])
            merged_quali_df, final_quali_df = merge_f1_quali_dataframes(
                quali_df, driver_quali_df, driver_summary_df, prev_year_result, start_year
            )
        else:
            # For current year, use empty DataFrame as prev_year_result
            empty_list = pd.DataFrame()
            merged_quali_df, final_quali_df = merge_f1_quali_dataframes(
                quali_df, driver_quali_df, driver_summary_df, empty_list, start_year
            )

        # Append to cumulative DataFrame
        new_df = custom_merge_df(new_df, final_quali_df)

        # Recursive call for the next year
        return prep_quali_df(start_year + 1, current_year, next_race, new_df)

    except Exception as e:
        logger.error(f"Error processing year {start_year}: {str(e)}")
        return new_df


# Function to convert time string (MM:SS.ssssss) to formatted time (MM:SS.sss)
def format_time_to_thousandths(time_str):
    if pd.isna(time_str):
        return time_str
    try:
        # Split minutes and seconds
        minutes, seconds = time_str.split(':')
        seconds = float(seconds)
        # Format to three decimal places
        minutes = int(minutes)
        seconds_formatted = f"{seconds:.3f}"
        return f"{minutes:02d}:{seconds_formatted.zfill(6)}"
    except (ValueError, AttributeError):
        return time_str
# Main execution

def quali_main(next_race):
    today = pd.Timestamp.today().normalize()
    current_year = today.year
    start_year = current_year - 6
    # calendar_df = f1.get_current_year_calendar(current_year)
    # global next_race
    # next_race = get_next_grandprix(current_year, calendar_df)
    # print(f"Next Grand Prix: {next_race['GrandPrix']}")

    # Get qualifying results for previous year (for verification)
    # print(get_quali_results(current_year - 1, next_race['GrandPrix']))
    print(get_quali_results(current_year - 1, next_race))

    # Prepare the final DataFrame
    final_df = prep_quali_df(start_year, current_year, next_race)
    print(final_df)

    quali_result, model, feature_importance = model_runner(final_df)
    quali_pred = quali_result[['qualifying_position', 'driver', 'team', 'qualifying_time', 'session_phase']]
    quali_pred = quali_pred.rename(columns={
        'qualifying_position': 'Position',
        'qualifying_time': 'Time',
        'session_phase': 'Session',
        'driver': 'Driver',
        'team': 'Constructor'
    })

    # Format the 'Time' column to thousandths of a second
    quali_pred['Time'] = quali_pred['Time'].apply(format_time_to_thousandths)

    quali_pred = quali_pred.set_index('Position')
    print(quali_pred)
    return quali_pred

# df_quali_pred = quali_main('Belgian Grand Prix')