import pandas as pd
import mysql.connector
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import fastf1
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable FastF1 caching
fastf1.Cache.enable_cache('/Users/prasanthbalaji/Desktop/F1_Project/backend/Cache')

# Points systems
RACE_POINTS_SYSTEM = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_RACE_POINTS_SYSTEM = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
SPRINT_OLD_POINTS_SYSTEM = {1: 3, 2: 2, 3: 1}


def set_db_config():
    load_dotenv()
    DB_CONFIG = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': 'F1'  # Matches your database name
    }
    return DB_CONFIG


def get_db_engine():
    DB_CONFIG = set_db_config()
    connection_string = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    return create_engine(connection_string)


def get_last_updated():
    engine = get_db_engine()
    try:
        queries = {
            'race': "SELECT MAX(year) as max_year, MAX(round) as max_round FROM race_results WHERE year = (SELECT MAX(year) FROM race_results)",
            'qualifying': "SELECT MAX(year) as max_year, MAX(round) as max_round FROM qualifying_results WHERE year = (SELECT MAX(year) FROM qualifying_results)",
            'sprint_race': "SELECT MAX(year) as max_year, MAX(round) as max_round FROM sprint_race_results WHERE year = (SELECT MAX(year) FROM sprint_race_results)",
            'sprint_qualifying': "SELECT MAX(year) as max_year, MAX(round) as max_round FROM sprint_qualifying_results WHERE year = (SELECT MAX(year) FROM sprint_qualifying_results)"
        }
        last_updated = {}
        with engine.connect() as conn:
            for key, query in queries.items():
                result = pd.read_sql(text(query), conn)
                year, round_num = result.iloc[0]['max_year'], result.iloc[0]['max_round']
                last_updated[key] = {
                    'year': int(year) if pd.notnull(year) else 2018,
                    'round': int(round_num) if pd.notnull(round_num) else 0
                }
        return last_updated
    except Exception as e:
        logger.error(f"Error fetching last updated details: {e}")
        return {
            'race': {'year': 2018, 'round': 0},
            'qualifying': {'year': 2018, 'round': 0},
            'sprint_race': {'year': 2018, 'round': 0},
            'sprint_qualifying': {'year': 2018, 'round': 0}
        }


def fetch_race_calendar(year):
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        return schedule
    except Exception as e:
        logger.error(f"Error fetching race calendar for {year}: {e}")
        return pd.DataFrame()


def is_session_completed(event, session_type):
    try:
        session = event.get_session(session_type)
        session.load(telemetry=False, laps=False, weather=False)
        return True
    except Exception as e:
        event_name = getattr(event, 'EventName', 'Unknown Event')
        event_year = getattr(event, 'year', getattr(event, 'Year', 'Unknown Year'))
        logger.warning(f"Session {session_type} not available for {event_name} {event_year}: {e}")
        return False


def fetch_race_results(session, event, is_sprint=False):
    try:
        session.load()
        results = session.results
        race_data = []
        points_system = SPRINT_RACE_POINTS_SYSTEM if is_sprint else RACE_POINTS_SYSTEM

        for idx, row in results.iterrows():
            finish_position = int(row['Position']) if pd.notnull(row['Position']) else None
            points = points_system.get(finish_position, 0)

            race_data.append({
                'year': event.year,
                'round': event.RoundNumber,
                'grand_prix': event.EventName,
                'circuit': event.Location,
                'date': event.EventDate.date() if pd.notnull(event.EventDate) else None,
                'driver': row['FullName'],
                'driver_id': row['Abbreviation'],
                'constructor': row['TeamName'],
                'grid_position': int(row['GridPosition']) if pd.notnull(row['GridPosition']) else None,
                'finish_position': finish_position,
                'status': row['Status'],
                'points': float(points),
                'fastest_lap_rank': int(row['FastestLap']) if 'FastestLap' in row and pd.notnull(
                    row['FastestLap']) else None,
                'fastest_lap_time': pd.Timedelta(
                    row['FastestLapTime']).total_seconds() / 1000000 if 'FastestLapTime' in row and pd.notnull(
                    row['FastestLapTime']) else None,
                'fastest_lap_speed_kph': float(row['FastestLapSpeed']) if 'FastestLapSpeed' in row and pd.notnull(
                    row['FastestLapSpeed']) else None
            })

        return pd.DataFrame(race_data)
    except Exception as e:
        event_name = getattr(event, 'EventName', 'Unknown Event')
        event_year = getattr(event, 'year', getattr(event, 'Year', 'Unknown Year'))
        logger.error(f"Error fetching {'sprint ' if is_sprint else ''}race results for {event_name} {event_year}: {e}")
        return pd.DataFrame()


def fetch_qualifying_results(session, event, is_sprint=False):
    try:
        session.load()
        results = session.results
        qual_data = []

        for idx, row in results.iterrows():
            qual_data.append({
                'year': event.year,
                'round': event.RoundNumber,
                'grand_prix': event.EventName,
                'circuit': event.Location,
                'date': event.EventDate.date() if pd.notnull(event.EventDate) else None,
                'driver': row['FullName'],
                'driver_id': row['Abbreviation'],
                'constructor': row['TeamName'],
                'position': int(row['Position']) if pd.notnull(row['Position']) else None,
                'q1_time': pd.Timedelta(row['Q1']).total_seconds() / 1000000 if pd.notnull(row['Q1']) else None,
                'q2_time': pd.Timedelta(row['Q2']).total_seconds() / 1000000 if pd.notnull(row['Q2']) else None,
                'q3_time': pd.Timedelta(row['Q3']).total_seconds() / 1000000 if pd.notnull(row['Q3']) else None
            })

        return pd.DataFrame(qual_data)
    except Exception as e:
        event_name = getattr(event, 'EventName', 'Unknown Event')
        event_year = getattr(event, 'year', getattr(event, 'Year', 'Unknown Year'))
        logger.error(
            f"Error fetching {'sprint ' if is_sprint else ''}qualifying results for {event_name} {event_year}: {e}")
        return pd.DataFrame()


def update_table(df, table_name):
    if df.empty:
        logger.warning(f"No data to update for {table_name}")
        return
    engine = get_db_engine()
    try:
        df.to_sql(table_name, engine, if_exists='append', index=False)
        logger.info(f"Successfully updated {table_name} with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error updating {table_name}: {e}")


def main():
    last_updated = get_last_updated()
    current_year = datetime.now().year
    current_date = datetime.now().date()

    for year in range(
            min(last_updated['race']['year'], last_updated['qualifying']['year'], last_updated['sprint_race']['year'],
                last_updated['sprint_qualifying']['year']), current_year + 1):
        if year != 2025:  # Only process 2025
            continue

        schedule = fetch_race_calendar(year)
        if schedule.empty:
            continue

        for idx, event in schedule.iterrows():
            try:
                round_num = event['RoundNumber']
                event_date = pd.to_datetime(event['EventDate']).date() if pd.notnull(event['EventDate']) else None

                # Skip non-race events, invalid rounds, or future events
                if pd.isna(round_num) or event['EventFormat'] == 'testing' or (
                        event_date and event_date >= current_date):
                    event_name = event.get('EventName', 'Unknown Event')
                    logger.info(
                        f"Skipping event: {event_name} {year} (Reason: {'Future event' if event_date and event_date >= current_date else 'Non-race or invalid round'})")
                    continue

                # Process Qualifying
                if year > last_updated['qualifying']['year'] or (
                        year == last_updated['qualifying']['year'] and round_num > last_updated['qualifying']['round']):
                    if is_session_completed(event, 'Qualifying'):
                        logger.info(f"Fetching qualifying results for {event['EventName']} {year}")
                        qual_session = event.get_session('Qualifying')
                        qual_df = fetch_qualifying_results(qual_session, event)
                        update_table(qual_df, 'qualifying_results')

                # Process Sprint-related sessions only if the event format is a sprint format
                if event['EventFormat'] in ['sprint', 'sprint_shootout']:
                    # Process Sprint Qualifying
                    if year > last_updated['sprint_qualifying']['year'] or (
                            year == last_updated['sprint_qualifying']['year'] and round_num >
                            last_updated['sprint_qualifying']['round']):
                        if is_session_completed(event, 'Sprint Qualifying'):
                            logger.info(f"Fetching sprint qualifying results for {event['EventName']} {year}")
                            sprint_qual_session = event.get_session('Sprint Qualifying')
                            sprint_qual_df = fetch_qualifying_results(sprint_qual_session, event, is_sprint=True)
                            update_table(sprint_qual_df, 'sprint_qualifying_results')

                    # Process Sprint Race
                    if year > last_updated['sprint_race']['year'] or (
                            year == last_updated['sprint_race']['year'] and round_num > last_updated['sprint_race'][
                        'round']):
                        if is_session_completed(event, 'Sprint'):
                            logger.info(f"Fetching sprint race results for {event['EventName']} {year}")
                            sprint_session = event.get_session('Sprint')
                            sprint_df = fetch_race_results(sprint_session, event, is_sprint=True)
                            update_table(sprint_df, 'sprint_race_results')

                # Process Race
                if year > last_updated['race']['year'] or (
                        year == last_updated['race']['year'] and round_num > last_updated['race']['round']):
                    if is_session_completed(event, 'Race'):
                        logger.info(f"Fetching race results for {event['EventName']} {year}")
                        race_session = event.get_session('Race')
                        race_df = fetch_race_results(race_session, event)
                        update_table(race_df, 'race_results')

            except Exception as e:
                event_name = event.get('EventName', 'Unknown Event')
                logger.error(f"Error processing event {event_name} {year}: {e}")
                continue


if __name__ == "__main__":
    main()
