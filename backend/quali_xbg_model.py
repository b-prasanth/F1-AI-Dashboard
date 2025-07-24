import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_time_to_seconds(time_str):
    """Convert qualifying_time from MM:SS.ssssss to seconds."""
    if pd.isna(time_str):
        return np.nan
    try:
        minutes, seconds = time_str.split(':')
        seconds = float(seconds) + int(minutes) * 60
        return seconds
    except:
        return np.nan


def convert_seconds_to_time(seconds):
    """Convert seconds back to MM:SS.ssssss format."""
    if pd.isna(seconds):
        return np.nan
    minutes = int(seconds // 60)
    seconds_part = seconds % 60
    return f"{minutes:02d}:{seconds_part:09.6f}"


def assign_session_phases(df):
    """Assign session phases based on qualifying position with F1 rules."""
    df = df.copy()

    # Sort by qualifying position to ensure correct assignment
    df = df.sort_values('qualifying_position').reset_index(drop=True)

    # Assign session phases based on F1 qualifying rules
    session_phases = []
    for i, pos in enumerate(df['qualifying_position']):
        if pos <= 10:  # Top 10 make it to Q3
            session_phases.append('Q3')
        elif pos <= 15:  # Positions 11-15 eliminated in Q2
            session_phases.append('Q2')
        else:  # Positions 16-20 eliminated in Q1
            session_phases.append('Q1')

    df['session_phase'] = session_phases
    return df


def derive_positions_from_times(df):
    """Derive qualifying positions and session phases from predicted times."""
    df = df.copy()

    # Sort by qualifying time (fastest to slowest)
    df = df.sort_values('qualifying_time_seconds').reset_index(drop=True)

    # Assign positions 1-20 based on time ranking
    df['qualifying_position'] = range(1, len(df) + 1)

    # Assign session phases based on positions
    df = assign_session_phases(df)

    return df


def preprocess_data(df):
    """Preprocess the dataset for XGBoost modeling."""
    # Drop Jack Doohan
    df = df[df['driver'] != 'Jack Doohan'].copy()

    # Convert qualifying_time to seconds
    df['qualifying_time_seconds'] = df['qualifying_time'].apply(convert_time_to_seconds)

    # Features for XGBoost
    features = [
        'Year', 'driver', 'team', 'championship_position', 'championship_points',
        'avg_qualifying_position', 'avg_qualifying_gap_vs_teammate',
        'qualifying_consistency', 'times_ahead', 'driver_quali_index',
        'overall_driver_index', 'avg_team_qualifying_position',
        'best_team_grid_position', 'team_qualifying_consistency',
        'team_front_row_lockouts', 'team_q3_appearances', 'team_q3_percentage',
        'car_quali_index', 'Rainfall'
    ]

    # Split data into training (2022-2024) and test (2025)
    train_df = df[df['Year'] < 2025].copy()
    # train_df = df[(df['Year'] < 2025) & (df['Rainfall'] == 0)].copy()
    test_df = df[df['Year'] == 2025].copy()

    # Store original test data for later use
    original_test_df = test_df.copy()

    # Encode categorical features
    le_driver = LabelEncoder()
    le_team = LabelEncoder()

    # Fit encoders on training data
    train_df['driver_encoded'] = le_driver.fit_transform(train_df['driver'])
    train_df['team_encoded'] = le_team.fit_transform(train_df['team'])

    # Transform test data, handling unseen categories
    test_df['driver_encoded'] = test_df['driver'].map(
        lambda x: le_driver.transform([x])[0] if x in le_driver.classes_ else -1
    )
    test_df['team_encoded'] = test_df['team'].map(
        lambda x: le_team.transform([x])[0] if x in le_team.classes_ else -1
    )

    # Numerical features
    numerical_features = [
        'Year', 'championship_position', 'championship_points',
        'avg_qualifying_position', 'avg_qualifying_gap_vs_teammate',
        'qualifying_consistency', 'times_ahead', 'driver_quali_index',
        'overall_driver_index', 'avg_team_qualifying_position',
        'best_team_grid_position', 'team_qualifying_consistency',
        'team_front_row_lockouts', 'team_q3_appearances', 'team_q3_percentage',
        'car_quali_index', 'Rainfall'
    ]

    # For XGBoost, we can use minimal preprocessing
    # Just handle any missing values and keep categorical encodings
    train_df[numerical_features] = train_df[numerical_features].fillna(train_df[numerical_features].mean())
    test_df[numerical_features] = test_df[numerical_features].fillna(train_df[numerical_features].mean())

    # Final feature set for XGBoost
    final_features = numerical_features + ['driver_encoded', 'team_encoded']

    return train_df, test_df, original_test_df, final_features


def train_xgboost_model(train_df, test_df, original_test_df, features):
    """Train XGBoost model for qualifying time prediction and derive all other predictions."""

    # Prepare training data
    X_train_full = train_df[features]
    y_train_full = train_df['qualifying_time_seconds']

    # Remove rows with missing target values
    mask = ~y_train_full.isna()
    X_train_full = X_train_full[mask]
    y_train_full = y_train_full[mask]

    # X_train_full = X_train_full.drop('Year', axis=1)

    # Train-validation split for evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    # XGBoost parameters optimized for time prediction
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 20
    }

    # Train XGBoost model
    logger.info("Training XGBoost model for qualifying time prediction...")
    xgb_model = xgb.XGBRegressor(**xgb_params)

    # Fit with evaluation set
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate model
    y_pred_val = xgb_model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred_val)
    mse = mean_squared_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    logger.info(f"XGBoost Qualifying Time Metrics:")
    logger.info(f"MAE: {mae:.3f} seconds")
    logger.info(f"MSE: {mse:.3f} seconds²")
    logger.info(f"R²: {r2:.3f}")
    logger.info(f"RMSE: {np.sqrt(mse):.3f} seconds")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info("Top 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        logger.info(f"{i + 1:2d}. {row['feature']:<30} {row['importance']:.4f}")

    # Predict for 2025
    logger.info("Generating 2025 predictions...")
    X_test = test_df[features]
    predicted_times = xgb_model.predict(X_test)

    # Create results dataframe
    results = original_test_df.copy()
    results['qualifying_time_seconds'] = predicted_times
    results['qualifying_time'] = results['qualifying_time_seconds'].apply(convert_seconds_to_time)

    # Derive positions and session phases from predicted times
    results = derive_positions_from_times(results)

    # Ensure Year is displayed correctly
    results['Year'] = 2025

    # Select final columns and sort by qualifying position
    # KEEP qualifying_time_seconds for later use in model_runner
    final_columns = [
        'Year', 'driver', 'team', 'qualifying_position', 'qualifying_time', 'qualifying_time_seconds', 'session_phase'
    ]

    results = results[final_columns].sort_values('qualifying_position').reset_index(drop=True)

    # Validate results
    logger.info("Validation Results:")
    logger.info(f"Total drivers: {len(results)}")
    logger.info(f"Q3 drivers: {len(results[results['session_phase'] == 'Q3'])}")
    logger.info(f"Q2 drivers: {len(results[results['session_phase'] == 'Q2'])}")
    logger.info(f"Q1 drivers: {len(results[results['session_phase'] == 'Q1'])}")
    logger.info(f"Unique positions: {len(results['qualifying_position'].unique())}")
    logger.info(f"Position range: {results['qualifying_position'].min()} to {results['qualifying_position'].max()}")
    logger.info(f"Time range: {results['qualifying_time'].min()} to {results['qualifying_time'].max()}")

    return results, xgb_model, feature_importance


def model_runner(df):
    """Main function to run XGBoost F1 qualifying prediction pipeline."""
    logger.info("Starting XGBoost F1 Qualifying Prediction Pipeline...")

    # Preprocess data
    train_df, test_df, original_test_df, features = preprocess_data(df)

    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    logger.info(f"Features: {len(features)}")
    logger.info(f"Training samples with valid qualifying times: {train_df['qualifying_time_seconds'].notna().sum()}")

    # Train XGBoost model and predict
    results, model, feature_importance = train_xgboost_model(
        train_df, test_df, original_test_df, features
    )

    # Print results (excluding qualifying_time_seconds for clean display)
    display_results = results.drop('qualifying_time_seconds', axis=1)
    print("\n" + "=" * 90)
    print("🏎️  F1 2025 QUALIFYING PREDICTIONS - XGBoost Model")
    print("=" * 90)
    print(display_results.to_string(index=False))
    print("=" * 90)

    # Additional summary statistics
    print(f"\n📊 SUMMARY:")
    print(f"Total Drivers: {len(results)}")
    print(f"🏆 Q3 (Positions 1-10): {len(results[results['session_phase'] == 'Q3'])} drivers")
    print(f"🥈 Q2 (Positions 11-15): {len(results[results['session_phase'] == 'Q2'])} drivers")
    print(f"🥉 Q1 (Positions 16-20): {len(results[results['session_phase'] == 'Q1'])} drivers")
    print(f"⏱️  Fastest Time: {results.iloc[0]['qualifying_time']}")
    print(f"🏁 Pole Position: {results.iloc[0]['driver']} ({results.iloc[0]['team']})")

    # Time gaps analysis - NOW using the correct column
    pole_time = results.iloc[0]['qualifying_time_seconds']
    print(f"\n⏱️  TIME GAPS FROM POLE:")
    for i in range(min(5, len(results))):
        gap = results.iloc[i]['qualifying_time_seconds'] - pole_time
        print(f"P{i + 1}: {results.iloc[i]['driver']:<15} +{gap:.3f}s")

    # Save results (without qualifying_time_seconds for clean CSV)
    display_results.to_csv('f1_2025_qualifying_predictions_xgboost.csv', index=False)
    logger.info("Predictions saved to 'f1_2025_qualifying_predictions_xgboost.csv'")

    # Save feature importance
    feature_importance.to_csv('xgboost_feature_importance.csv', index=False)
    logger.info("Feature importance saved to 'xgboost_feature_importance.csv'")

    return results, model, feature_importance

# Example usage:
# df = pd.read_csv('your_f1_dataset.csv')
# predictions, model, feature_importance = model_runner(df)