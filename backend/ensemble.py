from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_xgboost_model(train_df, test_df, features, target='qualifying_time_seconds'):
    X = train_df[features]
    y = train_df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("Train/Val Split Shapes - Train: %s | Val: %s", X_train.shape, X_val.shape)

    models = {
        'XGBoost': xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    }

    predictions_val = {}
    predictions_test = {}
    validation_maes = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(test_df[features])

        predictions_val[name] = y_pred_val
        predictions_test[name] = y_pred_test

        mae = mean_absolute_error(y_val, y_pred_val)
        mse = mean_squared_error(y_val, y_pred_val)
        r2 = r2_score(y_val, y_pred_val)

        validation_maes[name] = mae
        logger.info(f"{name} - MAE: {mae:.3f}, RMSE: {np.sqrt(mse):.3f}, R²: {r2:.3f}")

    # Compute weighted average
    mae_values = np.array(list(validation_maes.values()))
    inverse_weights = 1 / mae_values
    weights = inverse_weights / inverse_weights.sum()

    logger.info("Ensemble Weights: %s", dict(zip(models.keys(), weights.round(3))))

    model_names = list(models.keys())
    test_preds_array = np.array([predictions_test[name] for name in model_names])
    ensemble_test_pred = np.average(test_preds_array, axis=0, weights=weights)

    # Return full test set with predictions
    test_df = test_df.copy()
    test_df['qualifying_time_seconds'] = ensemble_test_pred
    test_df['qualifying_time'] = test_df['qualifying_time_seconds'].apply(convert_seconds_to_time)

    results = derive_positions_from_times(test_df)

    # Feature importance from XGBoost
    booster = models['XGBoost']
    importance = booster.feature_importances_
    feature_importance = pd.DataFrame({'feature': features, 'importance': importance}).sort_values(by='importance', ascending=False)

    return results, models, feature_importance

def convert_time_to_seconds(time_str):
    if pd.isna(time_str):
        return np.nan
    try:
        minutes, seconds = time_str.split(':')
        return int(minutes) * 60 + float(seconds)
    except:
        return np.nan

def convert_seconds_to_time(seconds):
    if pd.isna(seconds):
        return np.nan
    minutes = int(seconds // 60)
    seconds_part = seconds % 60
    return f"{minutes:02d}:{seconds_part:09.6f}"

def assign_session_phases(df):
    df = df.sort_values('qualifying_position').reset_index(drop=True)
    session_phases = []
    for pos in df['qualifying_position']:
        if pos <= 10:
            session_phases.append('Q3')
        elif pos <= 15:
            session_phases.append('Q2')
        else:
            session_phases.append('Q1')
    df['session_phase'] = session_phases
    return df

def derive_positions_from_times(df):
    df = df.sort_values('qualifying_time_seconds').reset_index(drop=True)
    df['qualifying_position'] = range(1, len(df) + 1)
    return assign_session_phases(df)

def preprocess_data(df):
    df = df[df['driver'] != 'Jack Doohan'].copy()
    df['qualifying_time_seconds'] = df['qualifying_time'].apply(convert_time_to_seconds)

    # Categorical encoders
    le_driver = LabelEncoder()
    le_team = LabelEncoder()

    df['driver_encoded'] = le_driver.fit_transform(df['driver'])
    df['team_encoded'] = le_team.fit_transform(df['team'])

    numerical_features = [
        'Year', 'championship_position', 'championship_points',
        'avg_qualifying_position', 'avg_qualifying_gap_vs_teammate',
        'qualifying_consistency', 'times_ahead', 'driver_quali_index',
        'overall_driver_index', 'avg_team_qualifying_position',
        'best_team_grid_position', 'team_qualifying_consistency',
        'team_front_row_lockouts', 'team_q3_appearances', 'team_q3_percentage',
        'car_quali_index', 'Rainfall'
    ]

    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].mean())

    final_features = numerical_features + ['driver_encoded', 'team_encoded']
    train_df = df[df['Year'] < 2025].copy()
    test_df = df[df['Year'] == 2025].copy()

    return train_df, test_df, df.copy(), final_features

def model_runner(df):
    logger.info("Starting F1 Ensemble Qualifying Prediction...")

    train_df, test_df, original_test_df, features = preprocess_data(df)

    logger.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
    results, model, feature_importance = train_xgboost_model(train_df, test_df, features)

    display_results = results.drop(columns=['qualifying_time_seconds'])
    print("\n" + "=" * 90)
    print("🏎️  F1 2025 QUALIFYING PREDICTIONS - Ensemble")
    print("=" * 90)
    print(display_results.to_string(index=False))
    print("=" * 90)

    print(f"\n📊 SUMMARY:")
    print(f"Total Drivers: {len(results)}")
    print(f"Q3: {len(results[results['session_phase'] == 'Q3'])} | Q2: {len(results[results['session_phase'] == 'Q2'])} | Q1: {len(results[results['session_phase'] == 'Q1'])}")
    print(f"⏱️  Fastest Time: {results.iloc[0]['qualifying_time']}")
    print(f"🏁 Pole Position: {results.iloc[0]['driver']} ({results.iloc[0]['team']})")

    pole_time = results.iloc[0]['qualifying_time_seconds']
    print(f"\n⏱️  TIME GAPS FROM POLE:")
    for i in range(min(5, len(results))):
        gap = results.iloc[i]['qualifying_time_seconds'] - pole_time
        print(f"P{i + 1}: {results.iloc[i]['driver']:<15} +{gap:.3f}s")

    display_results.to_csv('f1_2025_qualifying_predictions_ensemble.csv', index=False)
    feature_importance.to_csv('feature_importance_ensemble.csv', index=False)

    return results, model, feature_importance
