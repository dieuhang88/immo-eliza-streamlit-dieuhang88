# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso
from scipy.stats import randint, uniform
import joblib
import os
import warnings

df = pd.read_csv("../data/Kangaroo.csv")
# mapping epcScore for each province
def map_epc_score(df):
    epc_invalid = ['C_A', 'F_C', 'G_C', 'D_C', 'F_D', 'E_C', 'G_E', 'E_D', 'C_B', 'X', 'G_F']
    df = df[~df['epcScore'].isin(epc_invalid)].copy()

    wallonia_provinces = ['Liège', 'Walloon Brabant', 'Namur', 'Hainaut', 'Luxembourg']
    flanders_provinces = ['Antwerp', 'Flemish Brabant', 'East Flanders', 'West Flanders', 'Limburg']

    epc_maps = {
        "Wallonia": {'A++': 0, 'A+': 30, 'A': 65, 'B': 125, 'C': 200, 'D': 300, 'E': 375, 'F': 450, 'G': 510},
        "Flanders": {'A++': 0, 'A+': 0, 'A': 50, 'B': 150, 'C': 250, 'D': 350, 'E': 450, 'F': 500, 'G': 510},
        "Brussels": {'A++': 0, 'A+': 0, 'A': 45, 'B': 75, 'C': 125, 'D': 175, 'E': 250, 'F': 300, 'G': 350}
    }

    def map_score(row):
        if row['province'] in wallonia_provinces:
            return epc_maps['Wallonia'].get(row['epcScore'], None)
        elif row['province'] in flanders_provinces:
            return epc_maps['Flanders'].get(row['epcScore'], None)
        elif row['province'] == 'Brussels':
            return epc_maps['Brussels'].get(row['epcScore'], None)
        return None

    df.loc[:, 'epc_enum'] = df.apply(map_score, axis=1)
    return df

# Cleaning Function
def cleaning(df):
    # Drop columns missing 100 % values and unnecessary columns
    df = df.drop(columns=["Unnamed: 0", "url", "id", "monthlyCost", "accessibleDisabledPeople", "hasBalcony"])
    
    # Drop columns not important
    drop_cols = [
        'roomCount', 'diningRoomSurface', 'streetFacadeWidth', 'kitchenSurface', 'hasBasement', 'hasArmoredDoor',
        'floorCount', 'hasDiningRoom', 'hasDressingRoom', 'gardenSurface', 'terraceSurface', 'livingRoomSurface',
        'gardenOrientation', 'heatingType', 'kitchenType', 'terraceOrientation'
    ]
    df = df.drop(columns=drop_cols)
    
    # Drop rows with missing target or essential features
    df = df.dropna(axis=0, subset=['price','habitableSurface','bedroomCount','bathroomCount','epcScore'])

    # Convert boolean columns to binary (0/1)
    binary_cols = [
        'hasLift', 'hasHeatPump', 'hasPhotovoltaicPanels', 'hasAirConditioning', 'hasVisiophone', 'hasOffice',
        'hasSwimmingPool', 'hasFireplace', 'hasAttic', 'parkingCountIndoor', 'parkingCountOutdoor'
    ]
    for col in binary_cols:
        df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)


    # For facadeCount, rename and fill missing values based on subtype and remove rows with more than 4 facades
    df = df[df['facedeCount'] <= 4]  # Remove rows with more than 4 facades
    df['facadeCount'] = df['facedeCount']
    df = df.drop(columns='facedeCount')

    apartment_subtypes = ['APARTMENT', 'FLAT_STUDIO', 'GROUND_FLOOR', 'PENTHOUSE', 'APARTMENT_BLOCK']
    df.loc[df['subtype'].isin(apartment_subtypes), 'facadeCount'] = df.loc[df['subtype'].isin(apartment_subtypes), 'facadeCount'].fillna(1)

    house_subtypes = ['HOUSE', 'VILLA', 'DUPLEX', 'TOWN_HOUSE', 'MANSION']
    df.loc[df['subtype'].isin(house_subtypes), 'facadeCount'] = df.loc[df['subtype'].isin(house_subtypes), 'facadeCount'].fillna(3)

    larger_house_subtypes = ['EXCEPTIONAL_PROPERTY', 'BUNGALOW', 'COUNTRY_COTTAGE', 'TRIPLEX', 'CHALET', 'CASTLE', 'MANOR_HOUSE']
    df.loc[df['subtype'].isin(larger_house_subtypes), 'facadeCount'] = df.loc[df['subtype'].isin(larger_house_subtypes), 'facadeCount'].fillna(4)

    other_subtypes = ['MIXED_USE_BUILDING', 'SERVICE_FLAT', 'KOT', 'FARMHOUSE', 'LOFT', 'OTHER_PROPERTY']
    df.loc[df['subtype'].isin(other_subtypes), 'facadeCount'] = df.loc[df['subtype'].isin(other_subtypes), 'facadeCount'].fillna(2)

    condition_mapping = {
        'AS_NEW': 0, 'JUST_RENOVATED': 1, 'GOOD': 2,
        'TO_RENOVATE': 3, 'TO_RESTORE': 4, 'TO_BE_DONE_UP': 5
    }
    df['buildingCondition_mapping'] = df['buildingCondition'].map(condition_mapping)

    flood_mapping = {
        "NON_FLOOD_ZONE": 0,
        "POSSIBLE_N_CIRCUMSCRIBED_WATERSIDE_ZONE": 1,
        "CIRCUMSCRIBED_WATERSIDE_ZONE": 2,
        "POSSIBLE_N_CIRCUMSCRIBED_FLOOD_ZONE": 3,
        "POSSIBLE_FLOOD_ZONE": 4,
        "CIRCUMSCRIBED_FLOOD_ZONE": 5,
        "RECOGNIZED_FLOOD_ZONE": 6,
        "RECOGNIZED_N_CIRCUMSCRIBED_WATERSIDE_FLOOD_ZONE": 7,
        "RECOGNIZED_N_CIRCUMSCRIBED_FLOOD_ZONE": 8
    }
    df['floodZoneType_mapping'] = df['floodZoneType'].map(flood_mapping)

    return df

# # Add postcode mean price feature
# def add_postcode_price_mean(df):
#     postcode_mean = df.groupby('postCode')['price'].mean().to_dict()
#     df['postcode_price_mean'] = df['postCode'].map(postcode_mean)
#     return df

# ======= ADD price_per_m2 and postcode average price per m2 =======
def add_postcode_price_mean(df):
    # Calculate price per m2
    df['price_per_m2'] = df['price'] / df['habitableSurface']
    
    # Compute average price_per_m2 per postcode
    postcode_price = df.groupby('postCode')['price_per_m2'].mean().reset_index()
    postcode_price.rename(columns={'price_per_m2': 'postcode_avg_price_per_m2'}, inplace=True)
    
    # Merge the average back to original df
    df = df.merge(postcode_price, on='postCode', how='left')
    
    return df
# Apply all preprocessing
df = map_epc_score(df)
df = cleaning(df)
df = add_postcode_price_mean(df)
# --- Feature limits for manual outlier removal ---
feature_limits = {
    'price': (100000, 800000),
    'habitableSurface': (20, 500),
    'bathroomCount': (1, 10),
    'bedroomCount': (1, 10),
    #'landSurface': (0, 10000),  # Only for houses
}

def apply_feature_limits(df, limits):
    df_filtered = df.copy()
    for feature, (min_val, max_val) in limits.items():
        df_filtered = df_filtered[(df_filtered[feature] >= min_val) & (df_filtered[feature] <= max_val)]
    return df_filtered

# --- Apply limits to remove outliers ---
df = apply_feature_limits(df, feature_limits)
# --- Data Preprocessing ---
# Ensure boolean features are binary
boolean_features = [
    'hasLift', 'hasHeatPump', 'hasPhotovoltaicPanels', 'hasAirConditioning', 
    'hasVisiophone', 'hasOffice', 'hasSwimmingPool', 'hasFireplace', 'hasAttic', 
    'parkingCountIndoor', 'parkingCountOutdoor'
]
df[boolean_features] = df[boolean_features].astype(int)

# Define feature groups
numerical_features = [
    'habitableSurface', 'bedroomCount', 'bathroomCount', 'facadeCount',
    'landSurface', 'buildingConstructionYear', 'epc_enum', 'floodZoneType_mapping', 'buildingCondition_mapping', 
    'postcode_avg_price_per_m2'
]
categorical_features = [
    'type', 'subtype', 'province'
]
# Define all features and target
all_features = numerical_features + categorical_features + boolean_features
X = df[all_features]
y = df['price']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features),
    ('bool', 'passthrough', boolean_features)
])
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(random_state=42, eval_metric='rmse'))
])

# Define hyperparameter search space
xgb_param_dist = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(3, 15),
    'model__learning_rate': uniform(0.01, 0.3),
    'model__subsample': uniform(0.6, 0.4),
    'model__colsample_bytree': uniform(0.6, 0.4),
    'model__gamma': uniform(0, 5)
}

# Randomized Search
xgb_search = RandomizedSearchCV(
    xgb_pipeline,
    param_distributions=xgb_param_dist,
    n_iter=50,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Train
print("🔍 Tuning XGBoost...")
xgb_search.fit(X_train, y_train)
print("✅ Best XGBoost Params:", xgb_search.best_params_)
print(f"✅ Best CV MAE: {-xgb_search.best_score_:.2f}")

# Evaluate on test set
best_xgb_model = xgb_search.best_estimator_
y_pred = best_xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 Test Results | MAE = {mae:.2f} | RMSE = {rmse:.2f} | R² = {r2:.4f}")

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(best_xgb_model, 'models/xgb_model.pkl')
print("💾 Model saved to models/xgb_model.pkl")