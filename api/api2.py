import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import List, Dict
import logging
import traceback
from backend import f1_func, queries, sql_runner as sr
import numpy as np

# Import datetime for fallback in get_years
import datetime

# Local fallback for preprocess_params if not available in sql_runner
def preprocess_params(params):
    if params:
        converted_params = {}
        for key, value in params.items():
            if isinstance(value, np.integer):
                converted_params[key] = int(value)
            else:
                converted_params[key] = value
        return converted_params
    return params

# Function to convert NumPy types to native Python types for serialization
def convert_numpy_types(data):
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    return data

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_year(year: int) -> None:
    logger.debug(f"Validating year: {year}")
    try:
        available_years = sr.get_available_years(queries.get_available_years_query())
        if year not in available_years:
            logger.warning(f"Year {year} is not available.")
            raise HTTPException(status_code=400, detail=f"Year {year} is not available.")
    except Exception as e:
        logger.error(f"Error validating year {year}: {traceback.format_exc()}")
        pass
    logger.debug(f"Year {year} validated successfully.")

@app.get("/api/years")
def get_years() -> Dict[str, List[int]]:
    logger.info("Fetching available years.")
    try:
        years = sr.get_available_years(queries.get_available_years_query())
        logger.debug(f"Available years: {years}")
        return {"years": sorted(years, reverse=True)}
    except Exception as e:
        logger.error(f"Error fetching years: {traceback.format_exc()}")
        return {"years": [datetime.now().year]}

@app.get("/api/calendar/{year}")
def get_calendar(year: int) -> List[Dict]:
    logger.info(f"Fetching calendar for year: {year}")
    try:
        df = sr.run_sql_query("F1", "f1_calendar", queries.get_calendar_query(), params={"selected_year": year})
        if df is not None and not df.empty:
            df = df.rename(columns={
                'round': 'Round',
                'grand_prix': 'GrandPrix',
                'circuit': 'Circuit',
                'date': 'Date',
                'PoleSitter': 'PoleSitter',
                'PoleSitterId': 'PoleSitterId',
                'Winner': 'Winner',
                'WinnerId': 'WinnerId',
                'status': 'Status'
            })
            df['Status'] = df['Status'].apply(lambda x: '✅ Completed' if x == 'Completed' else 'Scheduled')
            today = pd.Timestamp.today().normalize().date()
            for index, race in df.iterrows():
                race_date = pd.to_datetime(race['Date']).date()
                if race_date > today:
                    df.at[index, 'Status'] = 'Scheduled'
                    df.at[index, 'Winner'] = 'TBA'
                    df.at[index, 'PoleSitter'] = 'TBA'
            logger.info(f"Successfully fetched calendar for {year} from database.")
            return df.to_dict(orient="records")
        else:
            logger.warning(f"No calendar data in database for {year}.")
            return []
    except Exception as e:
        logger.error(f"Error fetching calendar for year {year}: {traceback.format_exc()}")
        return []

@app.get("/api/standings/drivers/{year}")
def get_drivers_standings(year: int) -> List[Dict]:
    logger.info(f"Fetching drivers standings for year: {year}")
    try:
        df = sr.run_sql_query("F1", "drivers_championship", queries.get_drivers_championship_query(), params={"year": year})
        if df is not None and not df.empty:
            logger.info(f"Successfully fetched drivers standings for {year}.")
            return df.to_dict(orient="records")
        else:
            logger.warning(f"No drivers standings found for {year}.")
            return []
    except Exception as e:
        logger.error(f"Error fetching drivers standings for {year}: {traceback.format_exc()}")
        return []

@app.get("/api/standings/constructors/{year}")
def get_constructors_standings(year: int) -> List[Dict]:
    logger.info(f"Fetching constructors standings for year: {year}")
    try:
        df = sr.run_sql_query("F1", "constructors_championship", queries.get_constructors_championship_query(), params={"year": year})
        if df is not None and not df.empty:
            logger.info(f"Successfully fetched constructors standings for {year}.")
            return df.to_dict(orient="records")
        else:
            logger.warning(f"No constructors standings found for {year}.")
            return []
    except Exception as e:
        logger.error(f"Error fetching constructors standings for {year}: {traceback.format_exc()}")
        return []

@app.get("/api/results/race/{year}")
def get_race_results(year: int) -> List[Dict]:
    logger.info(f"Fetching race results for year: {year}")
    try:
        df = f1_func.get_pivoted_race_results(year, 'Race')
        if df is not None and not df.empty:
            df_reset = df.reset_index()
            driver_ids = sr.run_sql_query("F1", "drivers_championship", "SELECT driver, driver_id FROM drivers_championship WHERE year = :year", params={"year": year})
            driver_id_dict = dict(zip(driver_ids['driver'], driver_ids['driver_id']))
            df_reset['driver_id'] = df_reset['Driver'].map(driver_id_dict)
            logger.info(f"Successfully fetched race results for {year}.")
            return df_reset.to_dict(orient="records")
        else:
            logger.warning(f"No race results found for {year}.")
            return []
    except Exception as e:
        logger.error(f"Error fetching race results for {year}: {traceback.format_exc()}")
        return []

@app.get("/api/results/qualifying/{year}")
def get_quali_results(year: int) -> List[Dict]:
    logger.info(f"Fetching qualifying results for year: {year}")
    try:
        df = f1_func.get_pivoted_race_results(year, 'Quali')
        if df is not None and not df.empty:
            df_reset = df.reset_index()
            driver_ids = sr.run_sql_query("F1", "drivers_championship", "SELECT driver, driver_id FROM drivers_championship WHERE year = :year", params={"year": year})
            driver_id_dict = dict(zip(driver_ids['driver'], driver_ids['driver_id']))
            df_reset['driver_id'] = df_reset['Driver'].map(driver_id_dict)
            logger.info(f"Successfully fetched qualifying results for {year}.")
            return df_reset.to_dict(orient="records")
        else:
            logger.warning(f"No qualifying results found for {year}.")
            return []
    except Exception as e:
        logger.error(f"Error fetching qualifying results for {year}: {traceback.format_exc()}")
        return []

@app.get("/api/teams/{year}")
def get_team_drivers(year: int) -> List[Dict]:
    logger.info(f"Fetching team drivers for year: {year}")
    try:
        teams = f1_func.get_team_drivers(year)
        logger.info(f"Successfully fetched team drivers for {year}.")
        return teams
    except Exception as e:
        logger.error(f"Error fetching team drivers for {year}: {traceback.format_exc()}")
        return []

@app.get("/api/race_results/{year}/{grand_prix}")
def get_race_results_by_gp(year: int, grand_prix: str) -> List[Dict]:
    logger.info(f"Fetching race results for year: {year}, grand_prix: {grand_prix}")
    try:
        df = sr.run_sql_query("F1", "race_results", queries.get_race_results_by_gp(), params={"year": year, "grand_prix": grand_prix})
        if df is not None and not df.empty:
            return df.to_dict(orient="records")
        else:
            return []
    except Exception as e:
        logger.error(f"Error fetching race results: {traceback.format_exc()}")
        return []

@app.get("/api/driver_stats/{driver_id}")
def get_driver_stats(driver_id: str) -> Dict:
    logger.info(f"Fetching driver stats for driver_id: {driver_id}")
    try:
        stats = {}
        
        # Total entries (all-time)
        query = "SELECT COUNT(*) as total_entries FROM race_results WHERE driver_id = :driver_id"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        stats['total_entries'] = result.iloc[0]['total_entries'] if not result.empty else 0
        
        # First race (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE driver_id = :driver_id ORDER BY date ASC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        stats['first_race'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # Last race (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE driver_id = :driver_id ORDER BY date DESC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        stats['last_race'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # Best grid position (all-time, excluding 0)
        query = "SELECT MIN(grid_position) as best_grid FROM race_results WHERE driver_id = :driver_id AND grid_position IS NOT NULL AND grid_position > 0"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        if not result.empty and result.iloc[0]['best_grid'] is not None:
            best_grid = result.iloc[0]['best_grid']
            stats['best_grid_position'] = best_grid
            query = "SELECT COUNT(*) as count FROM race_results WHERE driver_id = :driver_id AND grid_position = :best_grid"
            count_result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id, "best_grid": best_grid}))
            stats['best_grid_count'] = count_result.iloc[0]['count'] if not count_result.empty else 0
        else:
            stats['best_grid_position'] = 'N/A'
            stats['best_grid_count'] = 0
        
        # Best race result (all-time, excluding 0)
        query = "SELECT MIN(finish_position) as best_finish FROM race_results WHERE driver_id = :driver_id AND finish_position IS NOT NULL AND finish_position > 0"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        if not result.empty and result.iloc[0]['best_finish'] is not None:
            best_finish = result.iloc[0]['best_finish']
            stats['best_race_result'] = best_finish
            query = "SELECT COUNT(*) as count FROM race_results WHERE driver_id = :driver_id AND finish_position = :best_finish"
            count_result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id, "best_finish": best_finish}))
            stats['best_race_count'] = count_result.iloc[0]['count'] if not count_result.empty else 0
        else:
            stats['best_race_result'] = 'N/A'
            stats['best_race_count'] = 0
        
        # Best championship position (all-time)
        query = "SELECT MIN(position) as best_position FROM drivers_championship WHERE driver_id = :driver_id"
        result = sr.run_sql_query("F1", "drivers_championship", query, params=preprocess_params({"driver_id": driver_id}))
        if not result.empty and result.iloc[0]['best_position'] is not None:
            best_position = result.iloc[0]['best_position']
            stats['best_championship_position'] = best_position
            query = "SELECT GROUP_CONCAT(year) as years FROM drivers_championship WHERE driver_id = :driver_id AND position = :best_position"
            years_result = sr.run_sql_query("F1", "drivers_championship", query, params=preprocess_params({"driver_id": driver_id, "best_position": best_position}))
            stats['best_championship_years'] = years_result.iloc[0]['years'] if not years_result.empty else 'N/A'
        else:
            stats['best_championship_position'] = 'N/A'
            stats['best_championship_years'] = 'N/A'
        
        # Number of race wins (all-time)
        query = "SELECT COUNT(*) as wins FROM race_results WHERE driver_id = :driver_id AND finish_position = 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        stats['race_wins'] = result.iloc[0]['wins'] if not result.empty else 0
        
        # Number of pole positions (all-time)
        query = "SELECT COUNT(*) as poles FROM race_results WHERE driver_id = :driver_id AND grid_position = 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        stats['pole_positions'] = result.iloc[0]['poles'] if not result.empty else 0
        
        # Number of podiums (all-time)
        query = "SELECT COUNT(*) as podiums FROM race_results WHERE driver_id = :driver_id AND finish_position <= 3"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        stats['podiums'] = result.iloc[0]['podiums'] if not result.empty else 0
        
        # First podium (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE driver_id = :driver_id AND finish_position <= 3 ORDER BY date ASC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        stats['first_podium'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # Last podium (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE driver_id = :driver_id AND finish_position <= 3 ORDER BY date DESC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        stats['last_podium'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # First win (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE driver_id = :driver_id AND finish_position = 1 ORDER BY date ASC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        stats['first_win'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # Last win (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE driver_id = :driver_id AND finish_position = 1 ORDER BY date DESC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        stats['last_win'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # First pole (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE driver_id = :driver_id AND grid_position = 1 ORDER BY date ASC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        stats['first_pole'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # Last pole (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE driver_id = :driver_id AND grid_position = 1 ORDER BY date DESC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        stats['last_pole'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # Teams and years (all-time)
        query = "SELECT constructor, GROUP_CONCAT(DISTINCT year) as years FROM race_results WHERE driver_id = :driver_id GROUP BY constructor"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        stats['teams'] = result.to_dict(orient="records") if not result.empty else []
        
        # Get driver name (all-time)
        query = "SELECT driver FROM race_results WHERE driver_id = :driver_id LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"driver_id": driver_id}))
        stats['driver'] = result.iloc[0]['driver'] if not result.empty else 'Unknown'
        
        # Convert NumPy types to native Python types for serialization
        return convert_numpy_types(stats)
    except Exception as e:
        logger.error(f"Error fetching driver stats: {traceback.format_exc()}")
        return {}

@app.get("/api/team_stats/{constructor}")
def get_team_stats(constructor: str) -> Dict:
    logger.info(f"Fetching team stats for constructor: {constructor}")
    try:
        stats = {}
        
        # Add wildcards for partial matching
        constructor_with_wildcards = f"{constructor}%"
        
        # Total entries (all-time)
        query = "SELECT COUNT(DISTINCT grand_prix, year) as total_entries FROM race_results WHERE constructor LIKE :constructor"
        # query = "SELECT COUNT(*) as total_entries FROM race_results WHERE constructor LIKE :constructor"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        stats['total_entries'] = result.iloc[0]['total_entries'] if not result.empty else 0
        
        # First race (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE constructor LIKE :constructor ORDER BY date ASC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        stats['first_race'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # Last race (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE constructor LIKE :constructor ORDER BY date DESC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        stats['last_race'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # Best grid position (all-time, excluding 0)
        query = "SELECT MIN(grid_position) as best_grid FROM race_results WHERE constructor LIKE :constructor AND grid_position IS NOT NULL AND grid_position > 0"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        if not result.empty and result.iloc[0]['best_grid'] is not None:
            best_grid = result.iloc[0]['best_grid']
            stats['best_grid_position'] = best_grid
            query = "SELECT COUNT(*) as count FROM race_results WHERE constructor LIKE :constructor AND grid_position = :best_grid"
            count_result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards, "best_grid": best_grid}))
            stats['best_grid_count'] = count_result.iloc[0]['count'] if not count_result.empty else 0
        else:
            stats['best_grid_position'] = 'N/A'
            stats['best_grid_count'] = 0
        
        # Best race result (all-time, excluding 0)
        query = "SELECT MIN(finish_position) as best_finish FROM race_results WHERE constructor LIKE :constructor AND finish_position IS NOT NULL AND finish_position > 0"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        if not result.empty and result.iloc[0]['best_finish'] is not None:
            best_finish = result.iloc[0]['best_finish']
            stats['best_race_result'] = best_finish
            query = "SELECT COUNT(*) as count FROM race_results WHERE constructor LIKE :constructor AND finish_position = :best_finish"
            count_result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards, "best_finish": best_finish}))
            stats['best_race_count'] = count_result.iloc[0]['count'] if not count_result.empty else 0
        else:
            stats['best_race_result'] = 'N/A'
            stats['best_race_count'] = 0
        
        # Best championship position (all-time)
        query = "SELECT MIN(position) as best_position FROM constructors_championship WHERE constructor LIKE :constructor"
        result = sr.run_sql_query("F1", "constructors_championship", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        if not result.empty and result.iloc[0]['best_position'] is not None:
            best_position = result.iloc[0]['best_position']
            stats['best_championship_position'] = best_position
            query = "SELECT GROUP_CONCAT(year) as years FROM constructors_championship WHERE constructor LIKE :constructor AND position = :best_position"
            years_result = sr.run_sql_query("F1", "constructors_championship", query, params=preprocess_params({"constructor": constructor_with_wildcards, "best_position": best_position}))
            stats['best_championship_years'] = years_result.iloc[0]['years'] if not years_result.empty else 'N/A'
        else:
            stats['best_championship_position'] = 'N/A'
            stats['best_championship_years'] = 'N/A'
        
        # Number of race wins (all-time)
        query = "SELECT COUNT(*) as wins FROM race_results WHERE constructor LIKE :constructor AND finish_position = 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        stats['race_wins'] = result.iloc[0]['wins'] if not result.empty else 0
        
        # Number of pole positions (all-time)
        query = "SELECT COUNT(*) as poles FROM race_results WHERE constructor LIKE :constructor AND grid_position = 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        stats['pole_positions'] = result.iloc[0]['poles'] if not result.empty else 0
        
        # Number of podiums (all-time)
        query = "SELECT COUNT(*) as podiums FROM race_results WHERE constructor LIKE :constructor AND finish_position <= 3"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        stats['podiums'] = result.iloc[0]['podiums'] if not result.empty else 0
        
        # First podium (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE constructor LIKE :constructor AND finish_position <= 3 ORDER BY date ASC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        stats['first_podium'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # Last podium (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE constructor LIKE :constructor AND finish_position <= 3 ORDER BY date DESC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        stats['last_podium'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # First win (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE constructor LIKE :constructor AND finish_position = 1 ORDER BY date ASC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        stats['first_win'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # Last win (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE constructor LIKE :constructor AND finish_position = 1 ORDER BY date DESC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        stats['last_win'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # First pole (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE constructor LIKE :constructor AND grid_position = 1 ORDER BY date ASC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        stats['first_pole'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # Last pole (all-time)
        query = "SELECT year, grand_prix FROM race_results WHERE constructor LIKE :constructor AND grid_position = 1 ORDER BY date DESC LIMIT 1"
        result = sr.run_sql_query("F1", "race_results", query, params=preprocess_params({"constructor": constructor_with_wildcards}))
        stats['last_pole'] = result.iloc[0].to_dict() if not result.empty else {'year': 'N/A', 'grand_prix': 'N/A'}
        
        # Convert NumPy types to native Python types for serialization
        return convert_numpy_types(stats)
    except Exception as e:
        logger.error(f"Error fetching team stats: {traceback.format_exc()}")
        return {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
