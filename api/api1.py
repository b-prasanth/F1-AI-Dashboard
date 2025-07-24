import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import pandas as pd
from datetime import datetime
from typing import List, Dict
import logging
import traceback
import importlib.util
import aiofiles
import asyncio
import os
from starlette.responses import FileResponse

from backend import f1_func, queries, sql_runner as sr

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS to allow React frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
from fastapi.staticfiles import StaticFiles
import os

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Mount the images directory
app.mount("/static/images", StaticFiles(directory="images"), name="images")

# Helper function to validate year
def validate_year(year: int) -> None:
    logger.debug(f"Validating year: {year}")
    try:
        available_years = sr.get_available_years(queries.get_available_years_query())
        if year not in available_years and year != datetime.now().year:
            logger.warning(f"Year {year} is not available.")
            raise HTTPException(status_code=400, detail=f"Year {year} is not available.")
    except Exception as e:
        logger.error(f"Error validating year {year}: {traceback.format_exc()}")
        # Don't raise error for validation, just log it
        pass
    logger.debug(f"Year {year} validated successfully.")

# Endpoint to get available years
@app.get("/api/years")
def get_years() -> Dict[str, List[int]]:
    logger.info("Fetching available years.")
    try:
        years = sr.get_available_years(queries.get_available_years_query())
        logger.debug(f"Available years: {years}")
        return {"years": sorted(years, reverse=True)}
    except Exception as e:
        logger.error(f"Error fetching years: {traceback.format_exc()}")
        # Return current year as fallback
        return {"years": [datetime.now().year]}

# Endpoint for race calendar
# @app.get("/api/calendar/{year}")
# def get_calendar(year: int) -> List[Dict]:
#     logger.info(f"Fetching calendar for year: {year}")
#     try:
#         validate_year(year)

#         # Fetch calendar data from the database
#         logger.debug(f"Fetching calendar for year {year} from database.")
#         try:
#             df = sr.run_sql_query("F1", "f1_calendar", queries.get_calendar_query(), params={"selected_year": year})
#             if df is not None and not df.empty:
#                 df = df.rename(columns={
#                     'round': 'Round',
#                     'grand_prix': 'GrandPrix',
#                     'circuit': 'Circuit',
#                     'date': 'Date',
#                     'pole_driver': 'PoleSitter',
#                     'win_driver': 'Winner'
#                 })
#                 today = pd.Timestamp.today().normalize().date()
#                 df['Status'] = '✅ Completed'
#                 next_race_round = None  # Initialize next_race_round
#                 for index, race in df.iterrows():
#                     race_date = race['Date']
#                     if race_date > today:
#                         df.at[index, 'Status'] = 'Scheduled'
#                         df.at[index, 'Winner'] = 'TBA'
#                         df.at[index, 'PoleSitter'] = 'TBA'  # Fixed column name
#                         if next_race_round is None:
#                             next_race_round = race['Round']
            
#                 # Identify the next race
#                 if next_race_round is None:
#                     next_race_round = race['Round']
#                 logger.info(f"Successfully fetched calendar for {year} from database.")
#                 return df.to_dict(orient="records")
            
#             else:
#                 logger.warning(f"No calendar data in database for {year}.")
#                 return []
#         except Exception as e:
#             logger.error(f"Error fetching calendar data from database: {traceback.format_exc()}")
#             return []

#     except Exception as e:
#         logger.error(f"Error fetching calendar for year {year}: {traceback.format_exc()}")
#         return []

# # Endpoint for drivers' championship standings
# @app.get("/api/standings/drivers/{year}")
# def get_drivers_standings(year: int) -> List[Dict]:
#     logger.info(f"Fetching drivers standings for year: {year}")
#     try:
#         validate_year(year)
#         df = f1_func.get_final_championship_standings("drivers_championship", year, is_constructor=False)
#         if df is not None and not df.empty:
#             logger.info(f"Successfully fetched drivers standings for {year}.")
#             return df.to_dict(orient="records")
#         else:
#             logger.warning(f"No drivers standings found for {year}.")
#             return []
#     except Exception as e:
#         logger.error(f"Error fetching drivers standings for {year}: {traceback.format_exc()}")
#         return []

# # Endpoint for constructors' championship standings
# @app.get("/api/standings/constructors/{year}")
# def get_constructors_standings(year: int) -> List[Dict]:
#     logger.info(f"Fetching constructors standings for year: {year}")
#     try:
#         validate_year(year)
#         df = f1_func.get_final_championship_standings("constructors_championship", year, is_constructor=True)
#         if df is not None and not df.empty:
#             logger.info(f"Successfully fetched constructors standings for {year}.")
#             return df.to_dict(orient="records")
#         else:
#             logger.warning(f"No constructors standings found for {year}.")
#             return []
#     except Exception as e:
#         logger.error(f"Error fetching constructors standings for {year}: {traceback.format_exc()}")
#         return []

# Check if required ML dependencies are available
def check_ml_dependencies():
    required_packages = ['lightgbm', 'xgboost', 'sklearn']
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'lightgbm':
                import lightgbm
            elif package == 'xgboost':
                import xgboost
        except ImportError:
            missing.append(package)
    return missing

# Endpoint for qualifying predictions
@app.get("/api/predictions/quali")
def get_quali_predictions() -> List[Dict]:
    logger.info("Fetching qualifying predictions.")
    try:
        # Check ML dependencies first
        missing_deps = check_ml_dependencies()
        if missing_deps:
            logger.warning(f"Missing ML dependencies for qualifying predictions: {missing_deps}")
            return []

        # Check if ML modules can be imported safely
        spec = importlib.util.find_spec("backend.quali_df")
        if spec is None:
            logger.warning("backend.quali_df not found for qualifying predictions.")
            return []

        from backend import quali_df
        
        # Get the current year calendar and next race information
        current_year = datetime.now().year
        calendar_df, next_race_info = f1_func.get_current_year_calendar(current_year)
        
        # If there is a next race, pass the Grand Prix to the prediction model
        if next_race_info:
            predictions = quali_df.quali_main(next_race_info['GrandPrix'])
            if predictions is not None and not predictions.empty:
                logger.info(f"Successfully fetched qualifying predictions for {next_race_info['GrandPrix']}.")
                return predictions.reset_index().to_dict(orient="records")
            else:
                logger.warning(f"No qualifying predictions returned for {next_race_info['GrandPrix']}.")
                return []
        else:
            logger.warning("No next race found, cannot fetch qualifying predictions.")
            return []
    except ImportError as e:
        logger.error(f"Import error for quali predictions: {traceback.format_exc()}")
        return []
    except Exception as e:
        logger.error(f"Error fetching quali predictions: {traceback.format_exc()}")
        return []

# Endpoint for race predictions
@app.get("/api/predictions/race")
def get_race_predictions() -> List[Dict]:
    logger.info("Fetching race predictions.")
    try:
        # Check ML dependencies first
        missing_deps = check_ml_dependencies()
        if missing_deps:
            logger.warning(f"Missing ML dependencies for race predictions: {missing_deps}")
            return []

        # Check if ML modules can be imported safely
        spec = importlib.util.find_spec("backend.race_df")
        if spec is None:
            logger.warning("backend.race_df not found for race predictions.")
            return []

        from backend import race_df
        
        # Get the current year calendar and next race information
        current_year = datetime.now().year
        calendar_df, next_race_info = f1_func.get_current_year_calendar(current_year)
        
        # If there is a next race, pass the Grand Prix to the prediction model
        if next_race_info:
            predictions = race_df.race_main()
            if predictions is not None and not predictions.empty:
                logger.info(f"Successfully fetched race predictions for {next_race_info['GrandPrix']}.")
                return predictions.reset_index().to_dict(orient="records")
            else:
                logger.warning(f"No race predictions returned for {next_race_info['GrandPrix']}.")
                return []
        else:
            logger.warning("No next race found, cannot fetch race predictions.")
            return []
    except ImportError as e:
        logger.error(f"Import error for race predictions: {traceback.format_exc()}")
        return []
    except Exception as e:
        logger.error(f"Error fetching race predictions: {traceback.format_exc()}")
        return []

# Endpoint for team drivers
@app.get("/api/teams/{year}")
def get_team_drivers(year: int) -> List[Dict]:
    logger.info(f"Fetching team drivers for year: {year}")
    try:
        validate_year(year)
        teams = f1_func.get_team_drivers(year)
        logger.info(f"Successfully fetched team drivers for {year}.")
        return teams
    except Exception as e:
        logger.error(f"Error fetching team drivers for {year}: {traceback.format_exc()}")
        return []

# Endpoint to get next grand prix details
@app.get("/api/next_grand_prix")
def get_next_grand_prix():
    logger.info("Fetching next grand prix details.")
    try:
        current_year = datetime.now().year
        calendar_df, next_race_info = f1_func.get_current_year_calendar(current_year)
        if next_race_info:
            # Convert np.int64 to int
            next_race_info['Round'] = int(next_race_info['Round'])
            logger.info(f"Successfully fetched next grand prix details: {next_race_info}")
            return next_race_info
        else:
            logger.warning("No next race found.")
            return {}
    except Exception as e:
        logger.error(f"Error fetching next grand prix details: {traceback.format_exc()}")
        return {}

# # Endpoint for points history
# @app.get("/api/stats/points_history/{year}")
# def get_points_history(year: int) -> List[Dict]:
#     logger.info(f"Fetching points history for year: {year}")
#     try:
#         validate_year(year)
#         driver_history, team_history = f1_func.get_driver_points_history(year)
#         if driver_history is not None and not driver_history.empty and team_history is not None and not team_history.empty:
#             logger.info(f"Successfully fetched points history for {year}.")
#             return {"drivers": driver_history.to_dict(orient="records"), "teams": team_history.to_dict(orient="records")}
#         else:
#             logger.warning(f"No points history found for {year}.")
#             return {"drivers": [], "teams": []}
#     except Exception as e:
#         logger.error(f"Error fetching points history for {year}: {traceback.format_exc()}")
#         return {"drivers": [], "teams": []}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "F1 API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
