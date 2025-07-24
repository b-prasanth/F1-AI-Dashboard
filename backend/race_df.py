import pandas as pd
import numpy as np
import logging
import fastf1 as f1api
from . import queries
from . import sql_runner as sr
from .race_xbg_model import runner

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Adjust width

def get_current_year_calendar(year):
    try:
        df = sr.run_sql_query("F1", "f1_calendar", queries.get_calendar_query(), params={"selected_year": year})
        if df.empty:
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

def merge_f1_race_dataframes(team_quali_df, driver_quali_df, summary_df, prev_year_result, year, circuit):
    try:
        # Define desired columns for the final DataFrame (removed duplicates)
        final_columns = [
            'Year', 'driver', 'team', 'circuit', 'championship_position', 'championship_points',
            'avg_race_position', 'points_finishes', 'wins', 'reliability_rate', 'driver_race_index',
            'overall_driver_index', 'avg_team_finish_position', 'team_wins', 'team_points_finishes',
            'total_team_points', 'team_podiums', 'team_reliability_rate', 'car_race_index',
            'Rainfall', 'LapTime', 'LapNumber', 'Stint', 'Compound', 'TyreLife', 'Team',
            'Current Position', 'PitIn', 'PitOut', 'TeamName', 'Finish Position', 'GridPosition'
        ]

        # Log input DataFrame columns and shapes for debugging
        logger.info(f"Team Quali DF columns: {team_quali_df.columns.tolist()}, shape: {team_quali_df.shape}")
        logger.info(f"Driver Quali DF columns: {driver_quali_df.columns.tolist()}, shape: {driver_quali_df.shape}")
        logger.info(f"Summary DF columns: {summary_df.columns.tolist()}, shape: {summary_df.shape}")
        logger.info(f"Prev Year Result DF columns: {prev_year_result.columns.tolist()}, shape: {prev_year_result.shape}")

        # Select and rename columns from team_quali_df
        team_race_cols = [
            'year', 'constructor', 'avg_finish_position', 'wins',
            'podiums', 'points_finishes', 'reliability_rate',
            'total_points', 'car_race_index'
        ]
        team_race_cols = [col for col in team_race_cols if col in team_quali_df.columns]
        team_race_selected = team_quali_df[team_race_cols].rename(columns={
            'year': 'Year',
            'constructor': 'team',
            'avg_finish_position': 'avg_team_finish_position',
            'wins': 'team_wins',
            'podiums': 'team_podiums',
            'points_finishes': 'team_points_finishes',
            'reliability_rate': 'team_reliability_rate',
            'total_points': 'total_team_points'
        })

        # Select and rename columns from driver_quali_df
        driver_race_cols = [
            'year', 'driver', 'constructor', 'avg_positions_gained',
            'points_finishes', 'wins', 'reliability_rate', 'driver_race_index'
        ]
        driver_race_cols = [col for col in driver_race_cols if col in driver_quali_df.columns]
        driver_race_selected = driver_quali_df[driver_race_cols].rename(columns={
            'year': 'Year',
            'constructor': 'team_driver'  # Rename to avoid conflict
        })

        # Select and rename columns from summary_df
        summary_cols = [
            'year', 'driver', 'team', 'championship_position', 'championship_points',
            'avg_race_position', 'overall_driver_index'
        ]
        summary_cols = [col for col in summary_cols if col in summary_df.columns]
        summary_selected = summary_df[summary_cols].rename(columns={'year': 'Year'})

        # Merge DataFrames
        merged_df = summary_selected.copy()

        # Merge with driver_quali_df on 'driver' and 'Year'
        if 'driver' in merged_df.columns and 'driver' in driver_race_selected.columns:
            merged_df = pd.merge(
                merged_df,
                driver_race_selected,
                on=['driver', 'Year'],
                how='left',
                suffixes=('_summary', '_driver_quali')
            )
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
        if 'team' in merged_df.columns and 'team' in team_race_selected.columns:
            merged_df = pd.merge(
                merged_df,
                team_race_selected,
                on=['team', 'Year'],
                how='left'
            )
        else:
            logger.error("Missing 'team' column in merged_df or team_quali_df")
            raise KeyError("Missing 'team' column in merged_df or team_quali_df")

        # Add circuit column
        merged_df['circuit'] = circuit

        # Remove duplicates based on Year, driver, team, circuit
        merged_df = merged_df.drop_duplicates(subset=['Year', 'driver', 'team', 'circuit'], keep='last')

        # Reorder columns to match the requested order
        available_columns = [col for col in final_columns if col in merged_df.columns]
        merged_df = merged_df[available_columns]

        # Log success
        logger.info(f"Merged DataFrame created with columns: {merged_df.columns.tolist()}, shape: {merged_df.shape}")

        if year == current_year:
            merged_df['Rainfall'] = 0
            final_merged_df = merged_df
        else:
            # Drop 'circuit' from prev_year_result to avoid duplication
            prev_year_result = prev_year_result.drop(columns=['circuit'], errors='ignore')
            final_merged_df = pd.merge(
                merged_df,
                prev_year_result,
                on=['driver'],
                how='left'
            )

        # Log final merged DataFrame
        logger.info(f"Final Merged DF columns: {final_merged_df.columns.tolist()}, shape: {final_merged_df.shape}")

        return merged_df, final_merged_df.dropna()

    except Exception as e:
        logger.error(f"Error merging DataFrames: {str(e)}")
        return pd.DataFrame(columns=final_columns), pd.DataFrame(columns=final_columns)

def get_perf_metrics(training_year):
    quali_df = sr.run_sql_query("F1", "custom", queries.get_quali_perf_data(), params={'year': training_year})
    race_df = sr.run_sql_query("F1", "custom", queries.get_race_perf_data(), params={'year': training_year})
    driver_quali_df = sr.run_sql_query("F1", "custom", queries.get_driver_quali_perf(), params={'year': training_year})
    driver_race_df = sr.run_sql_query("F1", "custom", queries.get_driver_race_perf(), params={'year': training_year})
    driver_summary_df = sr.run_sql_query("F1", "drivers_championship", queries.get_driver_summary(),
                                         params={'year': training_year})
    return quali_df, race_df, driver_quali_df, driver_race_df, driver_summary_df

def get_next_grandprix(year, calendar_df):
    try:
        calendar_df['Date'] = pd.to_datetime(calendar_df['Date'])
        today = pd.Timestamp.today().normalize()
        upcoming_races = calendar_df[calendar_df['Date'].dt.date >= today.date()]
        next_race = upcoming_races.sort_values('Date').head(1)
        if not next_race.empty:
            next_race_info = {
                'GrandPrix': next_race['GrandPrix'].iloc[0],
                'Date': next_race['Date'].iloc[0].strftime('%B %d, %Y'),
                'Circuit': next_race['Circuit'].iloc[0],
                'Round': next_race['Round'].iloc[0]
            }
            return next_race_info
        return None
    except Exception as e:
        logger.error(f"Error finding next Grand Prix: {str(e)}")
        return None

def merge_f1_api_output(year, grandprix):
    session = f1api.get_session(year, grandprix, 'Race')
    session.load()
    results_df = session.results
    weather_df = session.weather_data
    laps_df = session.laps

    result_temp_df = results_df[['Abbreviation', 'TeamName', 'Position', 'GridPosition']].rename(
        columns={'Position': 'Finish Position'})
    laps_temp_df = laps_df[
        ['Driver', 'LapTime', 'LapNumber', 'Stint', 'Compound', 'TyreLife', 'Team', 'Position', 'PitInTime',
         'PitOutTime']].rename(columns={'Position': 'Current Position', 'Driver': 'Abbreviation'})

    # Create PitIn and PitOut columns (1 if not NaT, 0 if NaT)
    laps_temp_df['PitIn'] = laps_temp_df['PitInTime'].notna().astype(int)
    laps_temp_df['PitOut'] = laps_temp_df['PitOutTime'].notna().astype(int)
    laps_temp_df = laps_temp_df.drop(['PitInTime', 'PitOutTime'], axis=1)

    is_wet = weather_df['Rainfall'].any() if 'Rainfall' in weather_df.columns else False
    result_temp_df['Rainfall'] = 1 if is_wet else 0
    driver_mapping = results_df[['Abbreviation', 'FullName']].set_index('Abbreviation')['FullName'].to_dict()
    new_df = pd.merge(laps_temp_df, result_temp_df, how='left', on='Abbreviation')
    new_df['Abbreviation'] = new_df['Abbreviation'].map(driver_mapping)
    new_df = new_df.rename(columns={'Abbreviation': 'driver'})
    # Drop redundant TeamName column
    new_df = new_df.drop(columns=['TeamName'], errors='ignore')
    return new_df, driver_mapping

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

def custom_merge_df(old_df, new_df):
    try:
        if old_df.empty:
            return new_df
        new_merged_df = pd.concat([old_df, new_df], ignore_index=True)
        return new_merged_df
    except Exception as e:
        logger.error(f"Error in custom_merge_df: {str(e)}")
        return old_df

def prep_race_df(start_year, current_year, new_df=None):
    if new_df is None:
        final_columns = [
            'Year', 'driver', 'team', 'circuit', 'championship_position', 'championship_points',
            'avg_race_position', 'points_finishes', 'wins', 'reliability_rate', 'driver_race_index',
            'overall_driver_index', 'avg_team_finish_position', 'team_wins', 'team_points_finishes',
            'total_team_points', 'team_podiums', 'team_reliability_rate', 'car_race_index',
            'Rainfall', 'LapTime', 'LapNumber', 'Stint', 'Compound', 'TyreLife', 'Team',
            'Current Position', 'PitIn', 'PitOut', 'TeamName', 'Finish Position', 'GridPosition'
        ]
        new_df = pd.DataFrame(columns=final_columns)

    if start_year > current_year:
        return new_df

    try:
        logger.info(f"Processing year: {start_year}")
        quali_df, race_df, driver_quali_df, driver_race_df, driver_summary_df = get_perf_metrics(start_year)

        if start_year != current_year:
            prev_year_result_df, driver_mapping = merge_f1_api_output(start_year, next_race['GrandPrix'])
            merged_race_df, final_race_df = merge_f1_race_dataframes(
                race_df, driver_race_df, driver_summary_df, prev_year_result_df, start_year, next_race['Circuit']
            )
        else:
            empty_df = pd.DataFrame()
            merged_race_df, final_race_df = merge_f1_race_dataframes(
                race_df, driver_race_df, driver_summary_df, empty_df, start_year, next_race['Circuit']
            )

        new_df = custom_merge_df(new_df, final_race_df)
        return prep_race_df(start_year + 1, current_year, new_df)

    except Exception as e:
        logger.error(f"Error processing year {start_year}: {str(e)}")
        return new_df

def format_laptime_to_mmss(laptime):
    if pd.isna(laptime):
        return np.nan
    try:
        total_seconds = laptime.total_seconds()
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"
    except:
        return np.nan

def format_racetime_to_hhmmss(total_seconds):
    if pd.isna(total_seconds):
        return np.nan
    try:
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    except:
        return np.nan

def process_race_data(df):
    df_processed = df.copy()
    df_processed['LapTime_seconds'] = df_processed['LapTime'].apply(
        lambda x: x.total_seconds() if pd.notna(x) else np.nan
    )
    df_processed['racetime_seconds'] = df_processed.groupby(['driver', 'Year'])['LapTime_seconds'].transform('sum')
    df_processed['LapTime'] = df_processed['LapTime'].apply(format_laptime_to_mmss)
    df_processed['racetime'] = df_processed['racetime_seconds'].apply(format_racetime_to_hhmmss)
    df_processed = df_processed.drop(['LapTime_seconds', 'racetime_seconds'], axis=1)
    return df_processed

def race_main(grand_prix=None):
    today = pd.Timestamp.today().normalize()
    global current_year
    current_year = today.year
    start_year = current_year - 6
    calendar_df, next_race_info = get_current_year_calendar(current_year)
    global next_race
    next_race = next_race_info if next_race_info else get_next_grandprix(current_year, calendar_df)
    if next_race:
        print(f"Next Grand Prix: {next_race['GrandPrix']}")
    else:
        print("No upcoming Grand Prix found")
        next_race = {'GrandPrix': grand_prix or 'Unknown', 'Circuit': 'Unknown', 'Round': None, 'Date': None}
    
    final_df = prep_race_df(start_year, current_year)
    df_with_racetime = process_race_data(final_df)
    
    df_race_pred = runner(df_with_racetime)
    print("Type of df_race_pred:", type(df_race_pred))
    print("Content of df_race_pred:", df_race_pred)

    if not isinstance(df_race_pred, pd.DataFrame):
        try:
            df_race_pred = pd.DataFrame(df_race_pred)
        except Exception as e:
            raise ValueError(f"Failed to convert runner output to DataFrame: {e}")

    expected_columns = ['finish_position', 'driver', 'team', 'total_time', 'points']
    if not all(col in df_race_pred.columns for col in expected_columns):
        print("Available columns:", df_race_pred.columns)
        raise ValueError("Some expected columns are missing in df_race_pred")

    df_race_pred = df_race_pred[expected_columns]
    df_race_pred = df_race_pred.set_index('finish_position')
    df_race_pred = df_race_pred.rename(columns={
        'driver': 'Driver',
        'team': 'Constructor',
        'total_time': 'Race Time',
        'points': 'Points'
    })

    def seconds_to_hms(seconds):
        if not np.isfinite(seconds):
            return 'DNF'
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

    df_race_pred['Race Time'] = df_race_pred['Race Time'].apply(seconds_to_hms)
    print(df_race_pred)
    return df_race_pred


# df_race_pred = race_main()
