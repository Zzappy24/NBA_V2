from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    log_loss, mean_absolute_error, r2_score, mean_squared_error
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
import joblib
from datetime import datetime
import pandas as pd
from path.path import TRANFORMED_DATA_DIR_TEMP, MODEL_TEMP_DIR, MODEL_DIR
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt





def read_data():
    filename = list(TRANFORMED_DATA_DIR_TEMP.glob('*.csv'))[0]
    try :
        players = pd.read_csv(filename,sep =";", encoding='Windows-1252')
        if len(players.columns) == 1:
            players = pd.read_csv(filename,sep =",", encoding='Windows-1252')
    except Exception:
        players = pd.read_csv(filename,sep =";", encoding='utf-8')
        if len(players.columns) == 1:
            players = pd.read_csv(filename,sep =",", encoding='utf-8')
    assert_not_null(players)
    return players

def assert_not_null(df):
    assert sum(df.isnull().sum()) < 1000, "There are not null values in the dataset"


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
    rmse = mean_squared_error(y_test_futur_star, pts_preds, squared=False)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test_futur_star, pts_preds)
    
    # Calculate F1 score
    f1 = f1_score(y_test_futur_star, pts_preds)
    # Calculate accuracy
    accuracy = accuracy_score(y_test_futur_star, pts_preds)
    
    # Calculate precision
    precision = precision_score(y_test_futur_star, pts_preds)
    
    # Calculate recall
    recall = recall_score(y_test_futur_star, pts_preds)
    
    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test_futur_star, pts_preds)
    
    # Calculate log loss
    log_loss_val = log_loss(y_test_futur_star, pts_preds)
    
    # Calculate mean absolute error (MAE)
    mae = mean_absolute_error(y_test_futur_star, pts_preds)
    
    # Calculate R-squared
    r_squared = r2_score(y_test_futur_star, pts_preds)
    return rmse, cm, f1, accuracy, precision, recall, roc_auc, log_loss_val, mae, r_squared, model

def retrain_model(original_model, new_players, ):
    # Assuming new_players has the same structure as the original players DataFrame
    X = new_players[["EFF", "PTS", "Age"]]
    y_futur_star = new_players['future_star']
    
    X_train_futur_star, X_test_futur_star, y_train_futur_star, y_test_futur_star = train_test_split(X, y_futur_star, test_size=0.2, random_state=42)
    
    # Retrain the model using the original_model as a starting point
    updated_model = joblib.load(original_model)  # Start with the original_model
    updated_model.fit(X_train_futur_star, y_train_futur_star)
    
    # Make predictions on the test set
    pts_preds = updated_model.predict(X_test_futur_star)

    rmse = mean_squared_error(y_test_futur_star, pts_preds, squared=False)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test_futur_star, pts_preds)
    
    # Calculate F1 score
    f1 = f1_score(y_test_futur_star, pts_preds)
    # Calculate accuracy
    accuracy = accuracy_score(y_test_futur_star, pts_preds)
    
    # Calculate precision
    precision = precision_score(y_test_futur_star, pts_preds)
    
    # Calculate recall
    recall = recall_score(y_test_futur_star, pts_preds)
    
    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test_futur_star, pts_preds)
    
    # Calculate log loss
    log_loss_val = log_loss(y_test_futur_star, pts_preds)
    
    # Calculate mean absolute error (MAE)
    mae = mean_absolute_error(y_test_futur_star, pts_preds)
    
    # Calculate R-squared
    r_squared = r2_score(y_test_futur_star, pts_preds)
    return rmse, cm, f1, accuracy, precision, recall, roc_auc, log_loss_val, mae, r_squared, updated_model







def dump_model(model, timestamp):
    joblib.dump(model, f'{MODEL_DIR}/model_{timestamp}.joblib')

def dump_model_temp(model, timestamp):
    joblib.dump(model, f'{MODEL_TEMP_DIR}/model_temp_{timestamp}.joblib')

def update_statistics_csv(rmse, cm, f1, accuracy, precision, recall, roc_auc, log_loss_val, mae, r_squared, model):
    # Create the line to append to the text file
    new_line = f"{model};{rmse};[{cm[0]}{cm[1]}];{f1};{accuracy};{precision};{recall};{roc_auc};{log_loss_val};{mae};{r_squared}\n"
    # Append the new line to the text file
    with open('data/statistics/statistics.csv', 'a',encoding="utf8") as file:
        file.write(new_line)


def main_training():
    df = read_data()
    model_file = list(TRANFORMED_DATA_DIR_TEMP.glob('*.joblib'))
    if len(model_file) >= 1:
        print("adjustement")
        rmse, cm, f1, accuracy, precision, recall, roc_auc, log_loss_val, mae, r_squared, model = retrain_model(model_file[0],df)
    else :
        rmse, cm, f1, accuracy, precision, recall, roc_auc, log_loss_val, mae, r_squared, model = training(df)
    #dump_model_temp(model)
    return rmse, cm, f1, accuracy, precision, recall, roc_auc, log_loss_val, mae, r_squared, model