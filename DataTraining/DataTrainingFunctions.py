from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
import joblib
from datetime import datetime
import pandas as pd
from path.path import TRANFORMED_DATA_DIR_TEMP, MODEL_TEMP_DIR, MODEL_DIR





def read_data():
    filename = list(TRANFORMED_DATA_DIR_TEMP.glob('*.csv'))[0]
    print(f"Runnung on file: {filename}")
    players = pd.read_csv(filename,sep =";", encoding='Windows-1252')
    if len(players.columns) == 1:
        players = pd.read_csv(filename,sep =",", encoding='Windows-1252')
    players.sort_values(by=['Player'], ascending=True)
    assert_not_null(players)
    return players

def assert_not_null(df):
    assert sum(df.isnull().sum()) < 30, "There are not null values in the dataset"


def transformer_pipeline():
    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Age']),
            #('cat', OneHotEncoder(), ['Pos', 'Tm'])
            
        ],
        remainder='passthrough')
    # Define model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        #('regressor', RandomForestRegressor())
        #('oversampler', RandomOverSampler(random_state=42)),
        ('classifier', LogisticRegression())

    ])
    return model

def training(players):
    X = players[["EFF", "PTS", "Age"]]
    y_futur_star = players['future_star']
    X_train_futur_star, X_test_futur_star, y_train_futur_star, y_test_futur_star = train_test_split(X, y_futur_star, test_size=0.2, random_state=42)
    model = transformer_pipeline()
    model.fit(X_train_futur_star, y_train_futur_star)
    pts_preds = model.predict(X_test_futur_star)
    print(f'RMSE for Player prediction: {mean_squared_error(y_test_futur_star, pts_preds, squared=False)}')

    # Save the model to a file
    return mean_squared_error(y_test_futur_star, pts_preds, squared=False), model

def retrain_model(original_model, new_players):
    # Assuming new_players has the same structure as the original players DataFrame
    X = new_players[["EFF", "PTS", "Age"]]
    y_futur_star = new_players['future_star']
    
    X_train_futur_star, X_test_futur_star, y_train_futur_star, y_test_futur_star = train_test_split(X, y_futur_star, test_size=0.2, random_state=42)
    
    # Retrain the model using the original_model as a starting point
    updated_model = joblib.load(original_model)  # Start with the original_model
    updated_model.fit(X_train_futur_star, y_train_futur_star)
    
    # Make predictions on the test set
    pts_preds = updated_model.predict(X_test_futur_star)
    
    # Evaluate the updated model
    rmse = mean_squared_error(y_test_futur_star, pts_preds, squared=False)
    print(f'RMSE for Player prediction: {rmse}')

    # Return the updated model and evaluation metric
    return rmse, updated_model







def dump_model(model):
    joblib.dump(model, f'{MODEL_DIR}/model_{int(datetime.timestamp(datetime.now()))}.joblib')

def dump_model_temp(model):
    joblib.dump(model, f'{MODEL_TEMP_DIR}/model_temp_{int(datetime.timestamp(datetime.now()))}.joblib')




def main_training():
    df = read_data()
    model_file = list(TRANFORMED_DATA_DIR_TEMP.glob('*.joblib'))
    if len(model_file) >= 1:
        print("adjustement")
        _, model = retrain_model(model_file[0],df)
    else :
        _, model = training(df)
    #dump_model_temp(model)
    return model