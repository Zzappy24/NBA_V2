
import sys

from path.path import CURATED_DATA_DIR_TEMP, TRANFORMED_DATA_DIR_TEMP, MODEL_TEMP_DIR, RAW_DATA_DIR_TEMP, RAW_DATA_DIR

from pathlib import Path
#from path.path import CURATED_DATA_DIR_TEMP, TRANFORMED_DATA_DIR_TEMP, MODEL_TEMP_DIR, RAW_DATA_DIR_TEMP
from DataCleaning.DataCleaningFunctions import write_csv_cleaned, write_csv_cleaned_temp, main_cleaning
from DataTransforming.DataTransformingFunctions import write_csv_tranformed, write_csv_transformed_temp, main_transforming
from DataTraining.DataTrainingFunctions import dump_model, dump_model_temp, main_training, update_statistics_csv
import os
import logging
import time
import shutil
from RaiseError.Error import DuplicateTrainingDataError
from datetime import datetime

log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='Log/pipeline_log.txt', level=logging.INFO, format=log_format)

def date_timestamp():
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    return int(ts)

def remove_file_temp():
    try :
        os.remove(f"dataTemp/raw_temp/{[f for f in os.listdir(RAW_DATA_DIR_TEMP) if '.csv' in f.lower()][0]}")
    except Exception:
        pass
    try :
        os.remove(f"dataTemp/curated_temp/{[f for f in os.listdir(CURATED_DATA_DIR_TEMP) if '.csv' in f.lower()][0]}")
    except Exception:
        pass
    try :
        os.remove(f"dataTemp/training_temp/{[f for f in os.listdir(TRANFORMED_DATA_DIR_TEMP) if '.csv' in f.lower()][0]}")
    except Exception:
        pass
    try :
        os.remove(f"dataTemp/model_temp/{[f for f in os.listdir(MODEL_TEMP_DIR) if '.joblib' in f.lower()][0]}")
    except Exception:
        pass

if __name__ == "__main__":
    timestamp = date_timestamp()
    succes = True
    step : str
    raws = [f[0:-15] for f in os.listdir(RAW_DATA_DIR) if '.csv' in f.lower()]

    raw_temp = [f[0:-4] for f in os.listdir(RAW_DATA_DIR_TEMP) if '.csv' in f.lower()][0]
    if raw_temp in raws:
        os.remove(f"dataTemp/raw_temp/{[f for f in os.listdir(RAW_DATA_DIR_TEMP) if '.csv' in f.lower()][0]}")
        raise DuplicateTrainingDataError("model already train on this data, be carefull with the choice of the data")
    
    try :
        logging.info("clean :")
        step = "clean"
        print("clean :")
        df_clean = main_cleaning()
        write_csv_cleaned_temp(df_clean, timestamp)
        logging.info("cleaning successful")

        logging.info("transform :")
        step = "transform"
        print("transform :")
        df_transformed = main_transforming()
        write_csv_transformed_temp(df_transformed, timestamp)
        logging.info("transforming successful")
        
        logging.info("train :")
        step = "train"
        print("train :")
        rmse, cm, f1, accuracy, precision, recall, roc_auc, log_loss_val, mae, r_squared, model = main_training()
        dump_model_temp(model, timestamp)
        logging.info("training successful")

        #logging.info("verification :")
        #logging.info("succes")

    except Exception as e:
        logging.error(f"error in step {step} : {e}")
        succes = False
        print(e)
        remove_file_temp()
    
    #time.sleep(10)
    if succes == True:
        file = [f for f in os.listdir(RAW_DATA_DIR_TEMP) if '.csv' in f.lower()][0]
        shutil.move(f"{RAW_DATA_DIR_TEMP}/{file}", f"{RAW_DATA_DIR}/{file[0:-4]}_{timestamp}.csv")
        #Path(f"./dataTemp/raw/{file}").rename(f"./data/raw/{file}")
        write_csv_cleaned(df_clean, timestamp)
        write_csv_tranformed(df_transformed, timestamp)
        dump_model(model, timestamp)
        update_statistics_csv(rmse, cm, f1, accuracy, precision, recall, roc_auc, log_loss_val, mae, r_squared,f"model_{timestamp}")

    
    remove_file_temp()
    


