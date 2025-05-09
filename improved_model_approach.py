import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
def load_and_preprocess(data_path):
    # Load data
    data = pd.read_csv(data_path)
    
    # Convert Date and Time to datetime
    data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')
    
    # Handle missing values
    data['City'].fillna('Unknown', inplace=True)
    data['Direction'].fillna('Unknown', inplace=True)
    data['Location'].fillna('Unknown', inplace=True)
    data['Involved'].fillna('Unknown', inplace=True)
    data['Type'].fillna('Unknown', inplace=True)
    
    # Feature engineering
    # Time features
    data['Hour'] = data['DateTime'].dt.hour
    data['Minute'] = data['DateTime'].dt.minute
    data['DayOfWeek'] = data['DateTime'].dt.dayofweek
    data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
    data['IsRushHour'] = (((data['Hour'] >= 7) & (data['Hour'] <= 9)) | 
                         ((data['Hour'] >= 16) & (data['Hour'] <= 19))).astype(int)
    data['Month'] = data['DateTime'].dt.month
    data['TimeOfDay'] = pd.cut(
        data['Hour'].clip(0, 23.99),
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
        include_lowest=True
    )
    
    # Advanced features
    data['DayOfMonth'] = data['DateTime'].dt.day
    data['Season'] = pd.cut(
        data['Month'],
        bins=[0, 3, 6, 9, 12],
        labels=['Winter', 'Spring', 'Summer', 'Fall'],
        include_lowest=True
    )
    data['IsHoliday'] = 0  # You could add a proper holiday checker here
    
    # Location features
    # Create interaction features
    data['CityDirection'] = data['City'] + '_' + data['Direction']
    
    # Create simplified target by categorizing incident types
    def simplify_incident_type(incident_type):
        incident_type = str(incident_type).upper()
        if 'ACCIDENT' in incident_type or 'COLLISION' in incident_type:
            return 'ACCIDENT'
        elif 'STALLED' in incident_type or 'MECHANICAL' in incident_type or 'BREAKDOWN' in incident_type:
            return 'VEHICLE_ISSUE'
        elif 'FIRE' in incident_type:
            return 'FIRE'
        elif 'FLOOD' in incident_type or 'WATER' in incident_type:
            return 'FLOOD'
        elif 'ROAD' in incident_type or 'CONSTRUCTION' in incident_type or 'WORK' in incident_type:
            return 'ROADWORK'
        else:
            return 'OTHER'
    
    data['IncidentCategory'] = data['Type'].apply(simplify_incident_type)
    
    return data

# Feature selection and engineering
def prepare_features(data):
    # Select relevant features
    categorical_features = ['City', 'Direction', 'TimeOfDay', 'Season', 'CityDirection']
    numeric_features = ['Latitude', 'Longitude', 'Hour', 'Month', 'DayOfWeek', 
                         'IsWeekend', 'IsRushHour', 'DayOfMonth']
    
    # Create feature matrix
    X = data[categorical_features + numeric_features]
    y = data['IncidentCategory']
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded, label_encoder, categorical_features, numeric_features

# Train advanced models
def train_advanced_models(X, y, categorical_features, numeric_features):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(pd.get_dummies(X_train), y_train)
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Define base models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
        ('catboost', CatBoostClassifier(iterations=200, learning_rate=0.1, depth=5, random_seed=42, verbose=0))
    ]
    
    # Create a stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        cv=5
    )
    
    # Create a voting classifier
    voting_clf = VotingClassifier(
        estimators=base_models,
        voting='soft'
    )
    
    # Train models
    models = {
        'XGBoost': Pipeline([('preprocessor', preprocessor), ('classifier', base_models[2][1])]),
        'LightGBM': Pipeline([('preprocessor', preprocessor), ('classifier', base_models[3][1])]),
        'CatBoost': Pipeline([('preprocessor', preprocessor), ('classifier', base_models[4][1])]), 
        'Stacking': Pipeline([('preprocessor', preprocessor), ('classifier', stacking_clf)]),
        'Voting': Pipeline([('preprocessor', preprocessor), ('classifier', voting_clf)])
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)  # Using original data for some models
        
        # For complex ensemble models, use the SMOTE-resampled data
        if name in ['Stacking', 'Voting']:
            model.fit(X_train_res, y_train_res)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'model': model
        }
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    return results, X_test, y_test

# Perform hyperparameter tuning for the best model
def tune_best_model(best_model_name, best_model, X, y, categorical_features, numeric_features):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Define parameter grid based on the best model
    if best_model_name == 'XGBoost':
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.8, 0.9, 1.0]
        }
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(random_state=42))
        ])
    elif best_model_name == 'LightGBM':
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__num_leaves': [31, 50, 70],
            'classifier__subsample': [0.8, 0.9, 1.0]
        }
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LGBMClassifier(random_state=42))
        ])
    elif best_model_name == 'CatBoost':
        param_grid = {
            'classifier__iterations': [100, 200, 300],
            'classifier__depth': [4, 6, 8],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__l2_leaf_reg': [1, 3, 5, 7]
        }
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', CatBoostClassifier(random_seed=42, verbose=0))
        ])
    else:  # For Stacking or Voting
        # Use RandomizedSearchCV with reduced parameters
        param_grid = {
            'classifier__rf__n_estimators': [100, 200],
            'classifier__rf__max_depth': [5, 10],
            'classifier__gb__n_estimators': [100, 200],
            'classifier__gb__learning_rate': [0.05, 0.1]
        }
        model = best_model
    
    # Use RandomizedSearchCV instead of GridSearchCV for efficiency
    random_search = RandomizedSearchCV(
        model, param_distributions=param_grid,
        n_iter=10, cv=5, scoring='accuracy', 
        n_jobs=-1, random_state=42, verbose=1
    )
    
    print(f"Tuning {best_model_name}...")
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = random_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Tuned {best_model_name} Test Accuracy: {accuracy:.4f}")
    
    return random_search.best_estimator_, accuracy, y_pred

# Main function
def main():
    # Load and preprocess data
    data = load_and_preprocess('data_mmda_traffic_spatial.csv')
    
    # Prepare features
    X, y, label_encoder, categorical_features, numeric_features = prepare_features(data)
    
    # Train models
    results, X_test, y_test = train_advanced_models(X, y, categorical_features, numeric_features)
    
    # Get best model
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\nBest Model: {best_model_name} with accuracy {best_accuracy:.4f}")
    
    # Tune best model if needed
    if best_accuracy < 0.8:  # If accuracy is still below 80%, tune the model
        print("Accuracy is below 80%, performing hyperparameter tuning...")
        tuned_model, tuned_accuracy, y_tuned_pred = tune_best_model(
            best_model_name, best_model, X, y, categorical_features, numeric_features
        )
        
        # Check if tuning improved the model
        if tuned_accuracy > best_accuracy:
            best_model = tuned_model
            best_accuracy = tuned_accuracy
            y_pred = y_tuned_pred
            print(f"Tuned model improved accuracy to {best_accuracy:.4f}")
        else:
            y_pred = results[best_model_name]['predictions']
            print("Tuning did not improve model accuracy.")
    else:
        y_pred = results[best_model_name]['predictions']
    
    # Generate detailed classification report
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Create and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('improved_confusion_matrix.png')
    
    # Save the best model
    joblib.dump(best_model, 'improved_traffic_incident_model.pkl')
    joblib.dump(label_encoder, 'improved_label_encoder.pkl')
    
    print(f"\nImproved model saved with accuracy: {best_accuracy:.4f}")
    
    return best_model, label_encoder, best_accuracy

if __name__ == "__main__":
    main()
