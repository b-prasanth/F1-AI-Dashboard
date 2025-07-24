import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
# from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')


class F1RacePredictor:
    def __init__(self, random_state=42):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'LapTime_seconds'
        self.random_state = random_state

        # Set random seeds for reproducibility
        np.random.seed(self.random_state)

    def parse_laptime_to_seconds(self, laptime_str):
        """Convert lap time string to seconds"""
        if pd.isna(laptime_str) or laptime_str == 'NaN':
            return np.nan

        try:
            # Handle format like '01:48.082'
            if ':' in str(laptime_str):
                parts = str(laptime_str).split(':')
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(laptime_str)
        except:
            return np.nan

    def seconds_to_laptime(self, seconds):
        """Convert seconds back to lap time format"""
        if pd.isna(seconds):
            return "NaN"

        minutes = int(seconds // 60)
        sec = seconds % 60
        return f"{minutes:02d}:{sec:06.3f}"

    def preprocess_data(self, df):
        """Preprocess the data for training"""
        # Remove Jack Doohan from the dataset entirely
        df = df[df['driver'] != 'Jack Doohan'].copy()

        # Convert lap times to seconds
        df['LapTime_seconds'] = df['LapTime'].apply(self.parse_laptime_to_seconds)

        # Remove rows with missing lap times (these are the 2025 data we want to predict)
        training_data = df[df['LapTime_seconds'].notna()].copy()
        prediction_data = df[df['LapTime_seconds'].isna()].copy()

        # Feature engineering
        for data in [training_data, prediction_data]:
            # Encode categorical variables
            categorical_cols = ['driver', 'team', 'circuit', 'Compound']
            for col in categorical_cols:
                if col in data.columns:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        # Fit on all unique values from both datasets
                        all_values = pd.concat([training_data[col], prediction_data[col]]).unique()
                        self.label_encoders[col].fit(all_values)

                    data[f'{col}_encoded'] = self.label_encoders[col].transform(data[col])

            # Create additional features
            data['TyreLife_squared'] = data['TyreLife'] ** 2
            data['driver_performance_index'] = data['overall_driver_index'] * data['reliability_rate']
            data['team_performance_index'] = data['car_race_index'] * data['team_reliability_rate']

            # Handle missing values
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

        # Define feature columns
        self.feature_columns = [
            'LapNumber', 'Stint', 'TyreLife', 'TyreLife_squared',
            'championship_position', 'championship_points', 'avg_race_position',
            'wins', 'reliability_rate', 'driver_race_index', 'overall_driver_index',
            'avg_team_finish_position', 'team_wins', 'team_points_finishes',
            'total_team_points', 'team_podiums', 'team_reliability_rate',
            'car_race_index', 'GridPosition', 'driver_performance_index',
            'team_performance_index', 'driver_encoded', 'team_encoded',
            'circuit_encoded', 'Compound_encoded'
        ]

        # Filter features that exist in the data
        available_features = [col for col in self.feature_columns if col in training_data.columns]
        self.feature_columns = available_features

        return training_data, prediction_data

    def train_models(self, X_train, y_train, X_val, y_val):
        """Train ensemble models"""
        print("Training ensemble models...")

        # XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgb'] = xgb_model

        # LightGBM
        print("Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        self.models['lgb'] = lgb_model

        # Gradient Boosting
        print("Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state
        )
        gb_model.fit(X_train, y_train)
        self.models['gb'] = gb_model

        # CatBoost
        # print("Training CatBoost...")
        # cat_model = CatBoostRegressor(
        #     iterations=200,
        #     depth=6,
        #     learning_rate=0.1,
        #     random_state=self.random_state,
        #     verbose=False
        # )
        # cat_model.fit(X_train, y_train)
        # self.models['catboost'] = cat_model

        # Evaluate models
        print("\nModel Performance on Validation Set:")
        ensemble_preds = []

        for name, model in self.models.items():
            val_pred = model.predict(X_val)
            ensemble_preds.append(val_pred)
            mae = mean_absolute_error(y_val, val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}")

        # Ensemble prediction
        ensemble_pred = np.mean(ensemble_preds, axis=0)
        mae = mean_absolute_error(y_val, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        print(f"Ensemble: MAE={mae:.3f}, RMSE={rmse:.3f}")

    def predict_lap_times(self, X):
        """Make ensemble predictions"""
        predictions = []

        for model in self.models.values():
            pred = model.predict(X)
            predictions.append(pred)

        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred

    def simulate_race(self, prediction_data, total_laps=44):
        """Simulate race results"""
        print(f"\nSimulating race with {total_laps} laps...")

        # Set random seed for reproducible race simulation
        np.random.seed(self.random_state)

        # Get unique drivers for 2025, excluding Jack Doohan
        drivers_2025 = prediction_data[prediction_data['Year'] == 2025]['driver'].unique()
        drivers_2025 = [driver for driver in drivers_2025 if driver != 'Jack Doohan']

        print(f"Predicting for {len(drivers_2025)} drivers (excluding Jack Doohan)")

        race_results = []

        for driver in drivers_2025:
            driver_data = prediction_data[prediction_data['driver'] == driver].iloc[0]

            # Simulate lap times for the race
            total_time = 0

            # Estimate race based on driver's historical performance
            base_lap_time = 108.0  # Base lap time in seconds for Spa

            # Adjust based on driver and team performance
            driver_factor = (25 - driver_data['championship_position']) / 25
            team_factor = driver_data['car_race_index'] / 1000
            reliability_factor = driver_data['reliability_rate'] / 100

            # Calculate average lap time
            avg_lap_time = base_lap_time * (1 - driver_factor * 0.05) * (1 - team_factor * 0.03)

            # Add some randomness for pit stops and race conditions
            pit_stop_time = np.random.uniform(20, 25)  # Pit stop time
            num_pit_stops = np.random.choice([1, 2], p=[0.3, 0.7])

            # Calculate total race time
            race_time = (avg_lap_time * total_laps) + (pit_stop_time * num_pit_stops)

            # Add reliability factor (chance of DNF)
            if np.random.random() > reliability_factor:
                race_time = np.inf  # DNF
                finish_position = 21  # DNF position
            else:
                finish_position = 0  # Will be calculated after sorting

            race_results.append({
                'driver': driver,
                'team': driver_data['team'],
                'total_time': race_time,
                'avg_lap_time': avg_lap_time,
                'finish_position': finish_position,
                'championship_position': driver_data['championship_position']
            })

        # Sort by total time and assign positions
        race_results.sort(key=lambda x: x['total_time'])

        # F1 points system
        points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 10

        for i, result in enumerate(race_results):
            if result['total_time'] != np.inf:
                result['finish_position'] = i + 1
                result['points'] = points_system[i] if i < len(points_system) else 0
            else:
                result['finish_position'] = 'DNF'
                result['points'] = 0

        return race_results

    def format_time(self, seconds):
        """Format time for display"""
        if seconds == np.inf:
            return "DNF"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:06.3f}"
        else:
            return f"{minutes:02d}:{secs:06.3f}"

    def display_results(self, race_results):
        """Display race results"""
        print("\n" + "=" * 80)
        print("2025 BELGIAN GRAND PRIX - RACE RESULTS")
        print("=" * 80)
        print(f"{'Pos':<4} {'Driver':<18} {'Team':<15} {'Race Time':<12} {'Points':<6}")
        print("-" * 80)

        for result in race_results:
            if result['finish_position'] != 'DNF':
                pos = f"{result['finish_position']}"
                time_str = self.format_time(result['total_time'])

                # Calculate gap to leader
                if result['finish_position'] == 1:
                    gap = ""
                else:
                    leader_time = race_results[0]['total_time']
                    gap_seconds = result['total_time'] - leader_time
                    gap = f"+{gap_seconds:.3f}s"

                print(f"{pos:<4} {result['driver']:<18} {result['team']:<15} {time_str:<12} {result['points']:<6}")
            else:
                print(f"DNF  {result['driver']:<18} {result['team']:<15} {'DNF':<12} {result['points']:<6}")

    def run_prediction(self, df):
        """Main prediction pipeline"""
        print("F1 Race Prediction System - Ensemble Model")
        print("=" * 50)

        # Preprocess data
        training_data, prediction_data = self.preprocess_data(df)

        print(f"Training data shape: {training_data.shape}")
        print(f"Prediction data shape: {prediction_data.shape}")

        # Prepare training data
        X = training_data[self.feature_columns]
        y = training_data[self.target_column]

        # Split for validation with fixed random state
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Scale features
        self.scalers['features'] = StandardScaler()
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_val_scaled = self.scalers['features'].transform(X_val)

        # Train models
        self.train_models(X_train_scaled, y_train, X_val_scaled, y_val)

        # Simulate race
        race_results = self.simulate_race(prediction_data)

        # Display results
        self.display_results(race_results)

        return race_results


def runner(df, random_state=42):
    # Create DataFrame
    # df = pd.DataFrame(sample_data)

    # Initialize and run predictor with fixed random state
    predictor = F1RacePredictor(random_state=random_state)
    results = predictor.run_prediction(df)

    print("\nPrediction completed successfully!")
    print("Note: This is a simulation based on driver/team performance metrics.")
    print("Actual race results may vary due to real-time factors like weather, strategy, etc.")
    return results